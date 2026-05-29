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
    #candidates = rotations(w)
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
    words += [(r, r) for r in rhos]
    words += [(s, r) for s in sigmas for r in rhos]
    
    if include_extra:
        # Alcune parole più forti (da usare per n_x > 2)
        words += [(r, M, s) for r in rhos for M in measurements for s in sigmas]
        words += [(M, r, s) for M in measurements for r in rhos for s in sigmas]

    # Applica le regole di proiezione riducendo le parole
    unique = []
    seen = set()
    for w in words:
        w = reduce_word(tuple(w))

        if is_zero_word(w):
            continue

        if w not in seen:
            seen.add(w)
            unique.append(w)

    return unique


def build_localizing_words(n_x, n_trunc):
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)

    words = [()]
    words += [(r,) for r in rhos]
    words += [(M,) for M in measurements]
    words += [(s,) for s in sigmas]

    return words

def solve_n_state_discrimination(n_x, n_trunc, omega, solver="CLARABEL", verbose=False, include_extra_words=True):
    """
    Risolve un rilassamento SDP per discriminazione di n_x stati
    con vincoli sulle componenti fotoniche fino a n_trunc.
    """

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
            constraints.append(sdp.T((r, s)) <= 1)


    # Completezza della POVM: sum_b M_b = I
    # sum_b Tr(u^dagger M_b v) = Tr(u^dagger v)
    for u in loc_words:
        for v in loc_words:

            lhs = sum(
                sdp.T(tuple(reversed(u)) + (M,) + tuple(v))
                for M in measurements
            )
            rhs = sdp.T(tuple(reversed(u)) + tuple(v))
            constraints.append(lhs == rhs)
    
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

def poisson_photon_weights(N, n_trunc):
    return np.array([np.exp(-N) * N**n / math.factorial(n) for n in range(n_trunc + 1)])


def poisson_omega(N, n_x, n_trunc):
    probn = poisson_photon_weights(N, n_trunc)
    omega_row = 1 - probn
    return np.tile(omega_row, (n_x, 1)) # Assumendo che tutti gli stati abbiano stessa energia media/distribuzione 



# =========================================================
# MIN ENTROPY LINEARIZZATA DAL TRICK l=b
# =========================================================

def cvx_sum(exprs):
    """Somma CVXPY più stabile di sum([...]) per liste lunghe."""
    exprs = list(exprs)
    if len(exprs) == 0:
        return 0
    return cp.sum(cp.hstack(exprs))


def solve_min_entropy_randomness(
    n_x,
    n_trunc,
    omega,
    W_obs=None,
    p_obs=None,
    x_star=0,
    solver="MOSEK",
    verbose=False,
    include_extra_words=True,
    tol=1e-5,
):
    """
    SDP per certificare H_min(B|X=x_star,Lambda) nel task di n-state discrimination.

    Note:
      - q_l è una variabile classica esterna.
      - una copia della moment matrix per ogni guess/outcome l.
      - vincoli fotonici imposti solo in media su l.
      - obiettivo pg = sum_l T_l(rho_xstar M_l), poi H_min = -log2(pg).

    Convenzione:
      omega[x,n] = 1 - P_n(x)
    quindi il lower bound fotonico è:
      photon_lb[x,n] = 1 - omega[x,n].
    """

    if p_obs is not None:
        p_obs = np.asarray(p_obs, dtype=float)
        if p_obs.shape != (n_x, n_x):
            raise ValueError(f"p_obs deve avere shape {(n_x, n_x)}, ricevuta {p_obs.shape}")

    # Se non passi né W_obs né p_obs, usa il massimo witness come nel grafico del paper.
    if W_obs is None and p_obs is None:
        tmp = solve_n_state_discrimination(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            solver=solver,
            verbose=verbose,
            include_extra_words=include_extra_words,
        )
        W_obs = tmp["sdp_upper_bound"]

    rhos, measurements, sigmas = build_operators(n_x, n_trunc)
    words = build_words(n_x, n_trunc, include_extra=include_extra_words)
    loc_words = build_localizing_words(n_x, n_trunc)

    sdps = []
    q_list = []
    W_list = []
    constraints = []

    # Una hidden strategy lambda=l per ogni possibile guess b=l.
    for l in range(n_x):
        sdp_l = TracialSDP()
        sdps.append(sdp_l)

        q_l = cp.Variable(nonneg=True, name=f"q_{l}")
        q_list.append(q_l)

        Gamma_l = sdp_l.moment_matrix(words)
        constraints.append(Gamma_l >> 0)

        # Localizing: rho_x - rho_x^2 >= 0, in forma omogeneizzata.
        for r in rhos:
            constraints.append(sdp_l.localizing_matrix(loc_words, r) >> 0)

        # Normalizzazioni pesate: Tr_l(rho_x)=q_l e Tr_l(sigma_n)=q_l.
        for r in rhos:
            constraints.append(sdp_l.T((r,)) == q_l)
        for s in sigmas:
            constraints.append(sdp_l.T((s,)) == q_l)

        # Completezza POVM per ogni blocco l.
        for u in loc_words:
            for v in loc_words:
                lhs = cvx_sum(
                    sdp_l.T(tuple(reversed(u)) + (M,) + tuple(v))
                    for M in measurements
                )
                rhs = sdp_l.T(tuple(reversed(u)) + tuple(v))
                constraints.append(lhs == rhs)

        W_l = cvx_sum(
            sdp_l.T((rhos[x], measurements[x])) for x in range(n_x)
        ) / n_x
        W_list.append(W_l)

    # Distribuzione classica delle hidden strategies.
    constraints.append(cvx_sum(q_list) == 1)

    # Vincoli fotonici medi. Con la tua convenzione omega=1-P_n, il bound è 1-omega.
    photon_lb = 1.0 - np.asarray(omega, dtype=float)
    for x, r in enumerate(rhos):
        for n, s in enumerate(sigmas):
            constraints.append(
                cvx_sum(sdps[l].T((r, s)) for l in range(n_x))
                >= photon_lb[x, n]
            )

    # Dati osservati: o distribuzione completa, o solo witness.
    if p_obs is not None:
        for x, r in enumerate(rhos):
            for b, M in enumerate(measurements):
                constraints.append(
                    cvx_sum(sdps[l].T((r, M)) for l in range(n_x))
                    == p_obs[x, b]
                )

    if W_obs is not None:
        W_total = cvx_sum(W_list)
        constraints.append(W_total >= W_obs - tol)
    else:
        W_total = None

    # Guessing probability.
    pg = cvx_sum(
        sdps[l].T((rhos[x_star], measurements[l]))
        for l in range(n_x)
    )

    problem = cp.Problem(cp.Maximize(pg), constraints)
    problem.solve(
                    solver="MOSEK",
                    verbose=verbose,
                    mosek_params={
                        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1,
                        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-7,
                        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-7,
                    },)

    pg_value = problem.value
    if problem.status not in ["optimal", "optimal_inaccurate"] or pg_value is None or pg_value <= 0:
        H_min = None
        pg_clip = None
    else:
        pg_clip = min(max(float(pg_value), 0.0), 1.0)
        H_min = -np.log2(pg_clip)

    q_values = None
    if problem.status in ["optimal", "optimal_inaccurate"]:
        q_values = [None if q.value is None else float(q.value) for q in q_list]

    return {
        "n_x": n_x,
        "n_trunc": n_trunc,
        "omega": omega,
        "photon_lb": photon_lb,
        "x_star": x_star,
        "W_obs": W_obs,
        "p_obs": p_obs,
        "guessing_probability": pg_value,
        "guessing_probability_clipped": pg_clip,
        "H_min_bits": H_min,
        "status": problem.status,
        "q_values": q_values,
        "W_total": None if W_total is None else W_total.value,
        "num_blocks": n_x,
        "num_constraints": len(constraints),
        "num_words_per_block": len(words),
        "num_moment_variables_per_block": [len(sdp.vars) for sdp in sdps],
    }


# =========================
# MAIN
# =========================

#N_values = [0.005, 0.1, 0.2, 0.5]
N_values = np.linspace(0.01, 2.0, 30)
n_x = 2
n_trunc_values = [0,1,2]

plt.figure(figsize=(7, 5))

for n_trunc in n_trunc_values:
    print(f"\n===== n_trunc = {n_trunc} =====")
    hmin_values = []
    pg_values = []
    W_values = []

    for N in N_values:
        omega = poisson_omega(N, n_x=n_x, n_trunc=n_trunc)

        # 1) Calcolo il massimo witness compatibile coi vincoli fotonici
        res_W = solve_n_state_discrimination(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            solver="MOSEK",
            include_extra_words=True,
        )
        W_obs = res_W["sdp_upper_bound"]

        # 2) Calcolo la guessing probability massima compatibile con
        # lo stesso witness osservato, quindi H_min = -log2(pg)
        res_H = solve_min_entropy_randomness(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            W_obs=W_obs,
            x_star=0,
            solver="MOSEK",
            include_extra_words=True,
            tol=1e-5,
        )

        hmin_values.append(res_H["H_min_bits"])
        pg_values.append(res_H["guessing_probability"])
        W_values.append(W_obs)

        print(
            f"N={N:.3f} | "
            f"W_obs={W_obs:.10f} | "
            f"pg={res_H['guessing_probability']:.10f} | "
            f"H_min={res_H['H_min_bits']:.10f} bits | "
            f"status={res_H['status']} | "
            f"W_total={res_H['W_total']} | "
            f"q={res_H['q_values']}"
        )

    plt.plot(N_values, hmin_values, "--", label=fr"$n_{{\mathrm{{trunc}}}}={n_trunc}$")

plt.xlabel(r"$N$")
plt.ylabel(r"$H_{\min}$ [bits]")
plt.title(fr"Min-entropy from {n_x}-states discrimination")
plt.grid(True)
plt.legend()
plt.show()
