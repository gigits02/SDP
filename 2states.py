import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# M e S sono proiettori: M^2=M, S^2=S.
# rho_0 e rho_1 sono, in generale, stati misti.
projs = {"M", "S"}

def reduce_word(w):
    """Riduce una parola usando M^2=M e S^2=S."""
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

# Esempio
print(reduce_word(("M", "M", "r0", "S", "S")))

def rotations(w):
    """Restituisce tutte le rotazioni cicliche della parola w"""
    if len(w) == 0:
        return [()]
    return [w[i:] + w[:i] for i in range(len(w))]

def canonical_trace_word(w):
    """Forma canonica di una parola dentro una traccia."""
    w = reduce_word(tuple(w))
    candidates = rotations(w) + rotations(tuple(reversed(w)))
    #candidates = rotations(w)
    candidates = [reduce_word(c) for c in candidates]
    return min(candidates)

# Esempio
print("Esempio tracce canoniche di parole:\n")
for word in [("r0", "M", "S"), ("M", "S", "r0"), ("S", "r0", "M"), ("S", "S", "r0", "r0"), ("S", "r0", "M", "S")]:
    print(word, "->", canonical_trace_word(word))

class TracialSDP:
    def __init__(self):
        self.vars = {}

    def T(self, w):
        """Restituisce la variabile CVXPY associata a Tr(w)."""
        key = canonical_trace_word(tuple(w))
        if key not in self.vars:
            name = "T_I" if len(key) == 0 else "T_" + "".join(key)
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
        Costruisce la localizing matrix associata a rho - rho^2 >= 0.
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

#Esempio variabili di momento
small_words = [(), ("r0",), ("M",), ("S",)]
small_sdp = TracialSDP()
small_G = small_sdp.moment_matrix(small_words)

print("Variabili create:")
for key in sorted(small_sdp.vars.keys()):
    var = small_sdp.vars[key]
    print(f"{key}  -->  {var.name()}")

print("Matrice dei momenti:\n")
print(small_G)

def solve_two_state_discrimination(omega, solver="CLARABEL", verbose=False):
    '''
    Risolve un rilassamento SDP per il problema di discriminazione di 2 stati.
    omega controlla il vincolo Tr(rho_x S) >= 1 - omega.
    Piccolo omega = stati molto vicini al vuoto => più difficili da distinguere.
    '''
    sdp = TracialSDP()

    # Lista di parole per Gamma.
    # Più parole -> rilassamento più forte (ma SDP più pesante).
    words = [
        (),
        ("r0",), ("r1",), ("M",), ("S",),
        ("r0", "M"), ("r1", "M"),
        ("r0", "r0"), ("r1", "r1"),
        ("S", "r0"), ("S", "r1"), ("S", "M"),
    ]

    # Parole per le localizing matrices.
    loc_words_r0 = [(), ("r0",), ("M",), ("S",)]
    loc_words_r1 = [(), ("r1",), ("M",), ("S",)]

    Gamma = sdp.moment_matrix(words)
    L0 = sdp.localizing_matrix(loc_words_r0, "r0")
    L1 = sdp.localizing_matrix(loc_words_r1, "r1")

    constraints = [Gamma >> 0, L0 >> 0, L1 >> 0]

    # Normalizzazione degli stati
    constraints += [sdp.T(("r0",)) == 1]
    constraints += [sdp.T(("r1",)) == 1]

    # Normalizzazione del proiettore del vuoto S=sigma_0
    constraints += [sdp.T(("S",)) == 1]

    # Vincolo sul peso del vuoto
    constraints += [sdp.T(("r0", "S")) >= 1 - omega]
    constraints += [sdp.T(("r1", "S")) >= 1 - omega]

    # Obiettivo: W = 1/2 + 1/2 Tr(r0 M) - 1/2 Tr(r1 M)
    W = 0.5 + 0.5 * sdp.T(("r0", "M")) - 0.5 * sdp.T(("r1", "M"))

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
        "moment_variables": sdp.vars,
    }


# ========================
# MAIN
# ========================

omegas = [0.01, 0.05, 0.10, 0.25, 0.50]
results = []

for omega in omegas:
    res = solve_two_state_discrimination(omega, solver="CLARABEL") #solver come CLARABEL o SCS (veloci) per maggiore accuratezza MOSEK
    results.append(res)

for res in results:
    print(
        f"omega={res['omega']:.3f} | "
        f"SDP={res['sdp_upper_bound']:.10f} | "
        f"analytic={res['analytic']:.10f} | "
        f"err={res['absolute_error']:.2e} | "
        f"status={res['status']}"
    )


# =========================
# PLOTS
# =========================
'''
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
plt.title("2-state discrimination witness")
plt.legend()
plt.grid(True)
plt.show()
'''