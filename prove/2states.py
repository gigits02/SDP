import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 2-state discrimination 
#
# Regole da applicare per la riduzione delle parole:
# - regole proiettive (idempotenza e ortogonalità)
# - ciclicità della traccia 
# - inversione delle parole (tutti gli op sono hermitiani)
#
#
# Vincoli:
# - rho_0, rho_1 sono stati normalizzati: Tr(rho)=1
# - rho_0, rho_1 sono, in generale, misti: 0 <= rho_x <= I (vincolo rho_x-rho_x^2 >= 0)
# - Normalizzazione del proiettore sul vuoto: Tr(S)=1
# - M0, M1 formano una POVM completa: M0 + M1 = I
# - Vincolo sulla componente di vuoto: Tr(rho_x S) >= 1 - omega
#
# Funzione obiettivo:
# - Witness: W = 1/2 [Tr(rho_0 M0) + Tr(rho_1 M1)]
#
# ============================================================

projs = {"M0", "M1", "S"}


def reduce_word(w):
    """Riduce una parola usando idempotenza dei proiettori: M_b^2=M_b, S^2=S."""
    w = list(w)
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(w):
            if i + 1 < len(w) and w[i] == w[i + 1] and w[i] in projs:
                out.append(w[i])
                i += 2
                changed = True
            else:
                out.append(w[i])
                i += 1
        w = out
    return tuple(w)


def is_zero_word(w):
    """
    Riconosce prodotti nulli per ortogonalità della misura: M0 M1 = M1 M0 = 0.
    """
    for a, b in zip(w, w[1:]):
        if a.startswith("M") and b.startswith("M") and a != b:
            return True
    return False


def rotations(w):
    """Restituisce tutte le rotazioni cicliche della parola w."""
    if len(w) == 0:
        return [()]
    return [w[i:] + w[:i] for i in range(len(w))]


def canonical_trace_word(w):
    """
    Forma canonica di una parola dentro una traccia.
    Usa ciclicità della traccia e inversione, assumendo operatori hermitiani.
    """
    w = reduce_word(tuple(w))
    candidates = rotations(w) + rotations(tuple(reversed(w)))
    candidates = [reduce_word(c) for c in candidates]
    return min(candidates)


class TracialSDP:
    def __init__(self):
        self.vars = {}

    def T(self, w):
        """Restituisce la variabile associata a Tr(w), oppure 0 se la parola è nulla (per ortogonalità)."""
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
        Localizing matrix associata a rho - rho^2 >= 0.
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


def build_words(include_extra=False):
    """
    Costruisce le parole necessarie per il rilassamento SDP.
    include_extra=False è il default per un confronto più pulito.
    Se True, aggiunge alcune parole di lunghezza 3 (ispirate a task più complessi).
    """
    rhos = ["r0", "r1"]
    measurements = ["M0", "M1"]

    words = [()]
    words += [(r,) for r in rhos]
    words += [(M,) for M in measurements]
    words += [("S",)]

    words += [(r, M) for r in rhos for M in measurements]
    words += [(r, "S") for r in rhos]
    words += [("S", M) for M in measurements]
    words += [(r, r) for r in rhos]
    words += [("S", r) for r in rhos]

    if include_extra:
        words += [(r, M, "S") for r in rhos for M in measurements]
        words += [(M, r, "S") for M in measurements for r in rhos]

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


def build_localizing_words():
    return [(), ("r0",), ("r1",), ("M0",), ("M1",), ("S",)]


def solve_two_state_discrimination(
    omega,
    solver="CLARABEL",
    verbose=False,
    include_extra_words=False,
):
    """
    Risolve il rilassamento SDP per discriminazione di 2 stati con POVM completa.
    Parametri:
    - omega: Vincolo Tr(rho_x S) >= 1 - omega.
    - include_extra_words: Se True aggiunge parole più forti di lunghezza 3.
    """
    sdp = TracialSDP()

    rhos = ["r0", "r1"]
    measurements = ["M0", "M1"]
    words = build_words(include_extra=include_extra_words)
    loc_words = build_localizing_words()

    Gamma = sdp.moment_matrix(words)
    constraints = [Gamma >> 0]

    for r in rhos:
        constraints.append(sdp.localizing_matrix(loc_words, r) >> 0)

    # Normalizzazione degli stati
    for r in rhos:
        constraints.append(sdp.T((r,)) == 1)

    # Normalizzazione del proiettore del vuoto S=sigma_0
    constraints.append(sdp.T(("S",)) == 1)

    # Vincolo sul peso del vuoto
    for r in rhos:
        constraints.append(sdp.T((r, "S")) >= 1 - omega)
        constraints.append(sdp.T((r, "S")) <= 1)

    # Completezza POVM: M0 + M1 = I dentro le tracce testate
    for u in loc_words:
        for v in loc_words:
            lhs = sum(
                sdp.T(tuple(reversed(u)) + (M,) + tuple(v))
                for M in measurements
            )
            rhs = sdp.T(tuple(reversed(u)) + tuple(v))
            constraints.append(lhs == rhs)

    # Witness con POVM a due outcome
    W = 0.5 * (sdp.T(("r0", "M0")) + sdp.T(("r1", "M1")))

    problem = cp.Problem(cp.Maximize(W), constraints)
    problem.solve(solver=solver, verbose=verbose)

    analytic = 0.5 + np.sqrt(omega * (1 - omega))

    return {
        "omega": omega,
        "sdp_upper_bound": problem.value,
        "analytic": analytic,
        "absolute_error": None if problem.value is None else abs(problem.value - analytic),
        "status": problem.status,
        "num_moment_variables": len(sdp.vars),
        "num_words": len(words),
        "words": words,
        "localizing_words": loc_words,
        "moment_variables": sdp.vars,
    }


# ========================
# MAIN
# ========================
omegas = [0.01, 0.05, 0.10, 0.25, 0.50]
results = []

for omega in omegas:
    res = solve_two_state_discrimination(
        omega,
        solver="MOSEK",
        include_extra_words=False,
    )
    results.append(res)

for res in results:
    print(
        f"omega={res['omega']:.3f} | "
        f"SDP={res['sdp_upper_bound']:.10f} | "
        f"analytic={res['analytic']:.10f} | "
        f"err={res['absolute_error']:.2e} | "
        f"status={res['status']} | "
        f"words={res['num_words']} | "
        f"vars={res['num_moment_variables']}"
    )

#Plot opzionale
omegas = np.linspace(0.001, 0.999, 50)
sdp_values = []
analytic_values = []
for omega in omegas:
    res = solve_two_state_discrimination(omega)
    sdp_values.append(res["sdp_upper_bound"])
    analytic_values.append(res["analytic"])

plt.figure(figsize=(7, 5))
plt.plot(omegas, sdp_values, "o", label="SDP upper bound")
plt.plot(omegas, analytic_values, "-", label="Analytic")
plt.xlabel(r"$\omega$")
plt.ylabel(r"$W_{2\mathrm{disc}}$")
plt.title("2-state discrimination witness with complete POVM")
plt.legend()
plt.grid(True)
plt.show()
