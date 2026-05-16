import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math

def reduce_word(w):
    """
    Riduce una parola usando le relazioni di idempotenza dei proiettori:
    Non elimina qui i prodotti ortogonali M_b M_b' = 0 o sigma_n sigma_m = 0
    per b != b' o n != m. Quelli sono gestiti da is_zero_word().
    """
    w = list(w)
    changed = True

    while changed:
        changed = False
        out = []
        i = 0
        while i < len(w):
            if i + 1 < len(w):
                a, b = w[i], w[i + 1]

                # M_b M_b = M_b
                if a == b and a.startswith("M"):
                    out.append(a)
                    i += 2
                    changed = True
                    continue
                # sigma_n sigma_n = sigma_n
                if a == b and a.startswith("s"):
                    out.append(a)
                    i += 2
                    changed = True
                    continue
            out.append(w[i])
            i += 1
        w = out

    return tuple(w)


def is_zero_word(w):
    """
    Riconosce parole che valgono zero per ortogonalità:
    M_b M_b' = 0 se b != b'
    sigma_n sigma_m = 0 se n != m
    """
    for a, b in zip(w, w[1:]):

        # M_b M_b' = 0 per b != b'
        if a.startswith("M") and b.startswith("M") and a != b:
            return True
        # sigma_n sigma_m = 0 per n != m
        if a.startswith("s") and b.startswith("s") and a != b:
            return True

    return False


def rotations(w):
    """Restituisce tutte le rotazioni cicliche della parola w."""
    w = tuple(w)
    if len(w) == 0:
        return [()]
    return [w[i:] + w[:i] for i in range(len(w))]


def canonical_trace_word(w):
    """
    Forma canonica di una parola dentro una traccia.
    Usa:
    - riduzione M_b^2=M_b e sigma_n^2=sigma_n;
    - ciclicità della traccia;
    - equivalenza sotto inversione della parola, assumendo operatori hermitiani.
    """
    w = reduce_word(tuple(w))
    candidates = rotations(w) + rotations(tuple(reversed(w)))
    candidates = [reduce_word(c) for c in candidates]
    return min(candidates)


class TracialSDP:
    def __init__(self):
        self.vars = {}

    def T(self, w):
        """Restituisce la variabile CVXPY associata a Tr(w), oppure 0 se la parola è nulla."""
        w = tuple(w)

        if is_zero_word(w):
            return 0

        key = canonical_trace_word(w)
        if is_zero_word(key):
            return 0

        if key not in self.vars:
            name = "T_I" if len(key) == 0 else "T_" + "_".join(key)
            self.vars[key] = cp.Variable(name=name)

        return self.vars[key]

    def moment_matrix(self, words):
        """Costruisce Gamma con Gamma[u,v] = Tr(u^dagger v)."""
        n = len(words)
        G = [[None for _ in range(n)] for _ in range(n)]

        for i, u in enumerate(words):
            for j, v in enumerate(words):
                u_dagger = tuple(reversed(u))
                G[i][j] = self.T(u_dagger + tuple(v))

        return cp.bmat(G)

    def localizing_matrix(self, words, rho):
        """
        Costruisce la localizing matrix associata a rho - rho^2 >= 0.s
        Elemento [u,v] = Tr(u^dagger (rho-rho^2) v).
        """
        n = len(words)
        L = [[None for _ in range(n)] for _ in range(n)]

        for i, u in enumerate(words):
            for j, v in enumerate(words):
                left = tuple(reversed(u))
                right = tuple(v)
                L[i][j] = self.T(left + (rho,) + right) - self.T(left + (rho, rho) + right)

        return cp.bmat(L)
    
def build_operators(n_x, n_trunc):
    rhos = [f"r{x}" for x in range(n_x)]
    measurements = [f"M{b}" for b in range(n_x)]
    sigmas = [f"s{n}" for n in range(n_trunc + 1)]
    return rhos, measurements, sigmas


def build_words(n_x, n_trunc, include_extra=True):
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)

    words = [()]

    # Parole di lunghezza 1
    words += [(r,) for r in rhos]
    words += [(M,) for M in measurements]
    words += [(s,) for s in sigmas]

    # Parole di lunghezza 2 rilevanti
    words += [(r, M) for r in rhos for M in measurements]
    words += [(r, s) for r in rhos for s in sigmas]
    words += [(s, M) for s in sigmas for M in measurements]

    if include_extra:
        # Alcune parole più forti (da usare per n_x > 2)
        words += [(r, r) for r in rhos]
        words += [(s, r) for s in sigmas for r in rhos]
        words += [(r, M, s) for r in rhos for M in measurements for s in sigmas]
        words += [(M, r, s) for M in measurements for r in rhos for s in sigmas]

    # Rimuovi duplicati dopo canonizzazione, scartando parole nulle
    unique = []
    seen = set()

    for w in words:
        key = canonical_trace_word(w)

        if is_zero_word(key):
            continue

        if key not in seen:
            seen.add(key)
            unique.append(key)

    return unique


def build_localizing_words(n_x, n_trunc):
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)

    words = [()]
    words += [(r,) for r in rhos]
    words += [(M,) for M in measurements]
    words += [(s,) for s in sigmas]

    return words

def prepare_omega(omega, n_x, n_trunc):
    if np.isscalar(omega):
        return np.full((n_x, n_trunc + 1), float(omega))

    omega = np.asarray(omega, dtype=float)

    expected_shape = (n_x, n_trunc + 1)
    if omega.shape != expected_shape:
        raise ValueError(f"omega deve avere shape {expected_shape}, ricevuto {omega.shape}")

    return omega


def solve_n_state_discrimination(n_x, n_trunc, omega, solver="CLARABEL", verbose=False, include_extra_words=True):
    """
    Risolve un rilassamento SDP per discriminazione di n_x stati
    con vincoli sulle componenti fotoniche fino a n_trunc.
    """
    omega = prepare_omega(omega, n_x, n_trunc)

    sdp = TracialSDP()
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)

    words = build_words(n_x, n_trunc, include_extra=include_extra_words)
    loc_words = build_localizing_words(n_x, n_trunc)

    Gamma = sdp.moment_matrix(words)
    constraints = [Gamma >> 0]

    # Localizing matrices: rho_x - rho_x^2 >= 0
    for r in rhos:
        constraints.append(sdp.localizing_matrix(loc_words, r) >> 0)

    # Normalizzazione degli stati: Tr(rho_x)=1
    for r in rhos:
        constraints.append(sdp.T((r,)) == 1)

    # Normalizzazione dei proiettori: Tr(sigma_n)=1
    for s in sigmas:
        constraints.append(sdp.T((s,)) == 1)

    # Vincoli fotonici: Tr(rho_x sigma_n) >= 1 - omega[x,n]
    for x, r in enumerate(rhos):
        for n, s in enumerate(sigmas):
            constraints.append(sdp.T((r, s)) >= 1 - omega[x, n])

    # Completezza della misura sul supporto degli stati:
    # sum_b Tr(rho_x M_b) = Tr(rho_x) = 1
    for r in rhos:
        constraints.append(sum(sdp.T((r, M)) for M in measurements) == 1)

    # Witness di n-state discrimination
    W = sum(sdp.T((rhos[x], measurements[x])) for x in range(n_x)) / n_x

    problem = cp.Problem(cp.Maximize(W), constraints)
    problem.solve(solver=solver, verbose=verbose)

    return {
        "n_x": n_x,
        "n_trunc": n_trunc,
        "omega": omega,
        "sdp_upper_bound": problem.value,
        "status": problem.status,
        "num_moment_variables": len(sdp.vars),
        "num_words": len(words),
        "words": words,
        "moment_variables": sdp.vars,
    }


def analytic_two_state_vacuum(omega):
    return 0.5 + np.sqrt(omega * (1 - omega))


def analytic_three_state_vacuum(omega):
    return (1 + omega) / 3 + (2 * np.sqrt(2) / 3) * np.sqrt(omega * (1 - omega))


def poisson_photon_weights(N, n_trunc):
    return np.array([np.exp(-N) * N**n / math.factorial(n) for n in range(n_trunc + 1)])


def poisson_omega(N, n_x, n_trunc):
    probn = poisson_photon_weights(N, n_trunc)
    omega_row = 1 - probn
    return np.tile(omega_row, (n_x, 1)) # Assumendo che tutti gli stati abbiano stessa energia media/distribuzione 


# =========================
# MAIN
# =========================

'''
omegas = [0.01, 0.05, 0.10, 0.25, 0.50]
results = []

for omega in omegas:
    res = solve_n_state_discrimination(
        n_x=3,
        n_trunc=0,
        omega=omega,
        solver="CLARABEL",
        include_extra_words=True,
    )
    analytic = analytic_three_state_vacuum(omega)
    res["analytic"] = analytic
    res["absolute_error"] = None if res["sdp_upper_bound"] is None else abs(res["sdp_upper_bound"] - analytic)
    results.append(res)

for res in results:
    print(
        f"omega={res['omega'][0,0]:.3f} | "
        f"SDP={res['sdp_upper_bound']:.10f} | "
        f"reference={res['analytic']:.10f} | "
        f"diff={res['absolute_error']:.2e} | "
        f"status={res['status']} | "
        f"vars={res['num_moment_variables']}"
    )

'''


#N_values = [0.01, 0.05, 0.10, 0.20]
N_values = np.linspace(0.01, 0.8, 50)
n_x = 3
n_trunc_values = [0,1,2]
# Attenzione: può richiedere tempo aumentando n_x, n_trunc o il numero di punti.

plt.figure(figsize=(7, 5))

for n_trunc in n_trunc_values:

    print(f"\n===== n_trunc = {n_trunc} =====")
    sdp_values = []
    for N in N_values:
        omega = poisson_omega(N, n_x=n_x, n_trunc=n_trunc)
        res = solve_n_state_discrimination(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            solver="CLARABEL",
            include_extra_words=True,
        )
        sdp_values.append(res["sdp_upper_bound"])

        print(
            f"N={N:.3f} | "
            f"SDP={res['sdp_upper_bound']:.10f} | "
            f"status={res['status']} | "
            #f"words={res['num_words']} | "
            #f"vars={res['num_moment_variables']}"
        )

    plt.plot(N_values, sdp_values, "--", label=fr"$n_{{\mathrm{{trunc}}}}={n_trunc}$")

plt.xlabel(r"$N$")
plt.ylabel(r"$W_{3\mathrm{disc}}$")
plt.title(r"3-state discrimination$")
plt.grid(True)
plt.legend()
plt.show()