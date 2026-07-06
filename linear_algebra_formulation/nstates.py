import cvxpy as cp
import numpy as np
import math
from dataclasses import dataclass
import time
from pathlib import Path
import pandas as pd
from datetime import datetime

# ========================
# WORD REDUCTION FUNCTIONS
# ========================
def reduce_word(w):
    """
    Riduce una parola usando idempotenza di POVM proiettive e sigma_n.
    Non mette a zero i prodotti ortogonali: quello lo fa is_zero_word.
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

                if a == b and a.startswith("M"):
                    out.append(a)
                    i += 2
                    changed = True
                    continue

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
    Riconosce parole nulle per ortogonalità:
    M_b M_b' = 0 se b != b',
    sigma_n sigma_m = 0 se n != m.
    """
    w = tuple(w)

    for a, b in zip(w, w[1:]):
        if a.startswith("M") and b.startswith("M") and a != b:
            return True
        if a.startswith("s") and b.startswith("s") and a != b:
            return True

    return False


def rotations(w):
    w = tuple(w)
    if len(w) == 0:
        return [()]
    return [w[i:] + w[:i] for i in range(len(w))]


def canonical_word(w):
    """
    Forma canonica per Tr(w), usando:
    - riduzione algebrica;
    - ciclicità della traccia;
    - inversione della parola.
    """
    w = reduce_word(tuple(w))

    if is_zero_word(w):
        return None

    candidates = rotations(w) + rotations(tuple(reversed(w)))
    candidates = [reduce_word(c) for c in candidates]
    candidates = [c for c in candidates if not is_zero_word(c)]

    if not candidates:
        return None

    return min(candidates)


def dagger_word(w):
    """Gli operatori sono hermitiani, quindi u^dagger è la parola invertita."""
    return tuple(reversed(tuple(w)))


# =========================
# OPERATOR/WORDS BUILDING FUNCTIONS
# =========================
def build_operators(n_x, n_trunc):
    rhos = [f"r{x}" for x in range(n_x)]
    measurements = [f"M{b}" for b in range(n_x)]
    sigmas = [f"s{n}" for n in range(n_trunc + 1)]
    return rhos, measurements, sigmas


def unique_reduced_words(words):
    out = []
    seen = set()

    for w in words:
        w = reduce_word(tuple(w))
        if is_zero_word(w):
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)

    return out


def build_words(n_x, n_trunc, include_extra=False):
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)
    generators = rhos + measurements + sigmas

    words = [()]
    words += [(A,) for A in generators]
    words += [(A, B) for A in generators for B in generators]

    if include_extra:
        words += [(r, M, s) for r in rhos for M in measurements for s in sigmas]
        words += [(M, r, s) for M in measurements for r in rhos for s in sigmas]

    return unique_reduced_words(words)


def build_localizing_words(n_x, n_trunc):
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)

    words = [()]
    words += [(r,) for r in rhos]
    words += [(M,) for M in measurements]
    words += [(s,) for s in sigmas]

    return unique_reduced_words(words)


def build_completeness_words(n_x, n_trunc, mode="loc"):
    """
    mode='loc': usa le stesse parole della localizing matrix.
    mode='extended': aggiunge parole più aggressive.
    """
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)

    if mode == "loc":
        return build_localizing_words(n_x, n_trunc)

    if mode == "extended":
        words = build_localizing_words(n_x, n_trunc)
        words += [(r, M) for r in rhos for M in measurements]
        words += [(r, s) for r in rhos for s in sigmas]
        words += [(s, M) for s in sigmas for M in measurements]
        return unique_reduced_words(words)

    raise ValueError("mode deve essere 'loc' oppure 'extended'.")


# =========================
# MOMENT BASIS: class that keeps track of canonical words and their indices
# =========================
class MomentBasis:
    """
    Dizionario parola canonica -> indice alpha.
    """
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = []

    def add(self, w):
        key = canonical_word(w)
        if key is None:
            return None

        if key not in self.word_to_idx:
            self.word_to_idx[key] = len(self.idx_to_word)
            self.idx_to_word.append(key)

        return self.word_to_idx[key]

    def __len__(self):
        return len(self.idx_to_word)

def add_entry_to_row(row, basis, w, coeff):
    """
    Aggiunge coeff * alpha[idx(w)] a una riga sparsa rappresentata da row.
    Se la parola è nulla, non fa nulla.
    Row è un dizionario che associa indici di parola canonica a coefficienti:
    se l'indice non è presente, il coefficiente è 0, se l'indice è presente, i coefficienti si sommano.
    Servirà per la costruzione delle matrici dei vincoli, prima sparse e poi dense per CVXPY.
    """
    idx = basis.add(w)
    if idx is None:
        return row

    row[idx] = row.get(idx, 0.0) + float(coeff)
    return row


def dense_row_from_dict(row_dict, d):
    """
    Questa funzione serve perché bisogna passare a CVXPY un array. Dunque dal dizionario row_dict, che rappresenta una riga sparsa,
    (indice parola canonica -> coefficiente), si costruisce un array denso di dimensione d.
    """
    row = np.zeros(d)
    for k, v in row_dict.items():
        row[k] += v
    return row


# =========================
# SYMBOLIC MODEL: COLLECTING ALL THE CANONICAL MOMENTS AND THEIR INDICES
# ========================

@dataclass #Per non scrivere il costruttore
class SymbolicDiscriminationModel:
    n_x: int
    n_trunc: int
    words: list
    loc_words: list
    completeness_words: list
    rhos: list
    measurements: list
    sigmas: list
    basis: MomentBasis
    Gamma_idx: np.ndarray
    Lplus_idx: dict
    Lminus_idx: dict
    Aeq: np.ndarray
    beq: np.ndarray
    photon_rows: list
    objective_row: np.ndarray


def matrix_indices_from_words(basis, row_words, middle=()):
    """
    Restituisce una matrice di indici K tale che:
    K[i,j] = indice di alpha associato a Tr(u_i^dagger middle v_j),
    oppure -1 se la parola è nulla. (-1 serve solo come flag poi per non salvare in alpha la traccia nulla)
    """
    n = len(row_words)
    K = -np.ones((n, n), dtype=int)

    for i, u in enumerate(row_words):
        for j, v in enumerate(row_words):
            w = dagger_word(u) + tuple(middle) + tuple(v)
            idx = basis.add(w)
            if idx is not None:
                K[i, j] = idx

    return K


def collect_symbolic_model(
    n_x,
    n_trunc,
    include_extra=False,
    completeness_mode="loc",
):
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)
    words = build_words(n_x, n_trunc, include_extra=include_extra)
    loc_words = build_localizing_words(n_x, n_trunc)
    completeness_words = build_completeness_words(n_x, n_trunc, mode=completeness_mode)

    basis = MomentBasis()

    # 1) Gamma
    Gamma_idx = matrix_indices_from_words(basis, words)

    # 2) Localizing matrices:
    # L_x[u,v] = Tr(u^dagger rho_x v) - Tr(u^dagger rho_x rho_x v)
    # Conviene trattare separatamente i due termini, in due dizionari che associano a ciascun rho_x la matrice di indici corrispondente.
    Lplus_idx = {}
    Lminus_idx = {}

    for r in rhos:
        Lplus_idx[r] = matrix_indices_from_words(basis, loc_words, middle=(r,))
        Lminus_idx[r] = matrix_indices_from_words(basis, loc_words, middle=(r, r))

    # 3) Normalizzazioni e completezza: prima come righe sparse
    eq_rows = []
    eq_vals = []

    # Tr(rho_x)=1
    for r in rhos:
        row = {}
        add_entry_to_row(row, basis, (r,), +1.0)
        eq_rows.append(row)
        eq_vals.append(1.0)

    # Tr(sigma_n)=1
    for s in sigmas:
        row = {}
        add_entry_to_row(row, basis, (s,), +1.0)
        eq_rows.append(row)
        eq_vals.append(1.0)

    # Completezza: sum_b Tr(u^dagger M_b v) = Tr(u^dagger v)
    for u in completeness_words:
        for v in completeness_words:
            row = {}

            for M in measurements:
                add_entry_to_row(row, basis, dagger_word(u) + (M,) + tuple(v), +1.0)

            add_entry_to_row(row, basis, dagger_word(u) + tuple(v), -1.0)

            # Scarta righe identicamente nulle
            row = {k: val for k, val in row.items() if abs(val) > 1e-12}
            if row:
                eq_rows.append(row)
                eq_vals.append(0.0)

    # 4) Photon rows: Tr(rho_x sigma_n) >= 1 - omega[x,n]
    photon_rows_sparse = []
    for x, r in enumerate(rhos):
        for n, s in enumerate(sigmas):
            row = {}
            add_entry_to_row(row, basis, (r, s), +1.0)
            photon_rows_sparse.append((x, n, row))

    # 5) Objective W = 1/n_x sum_x Tr(rho_x M_x)
    obj_sparse = {}
    for x, r in enumerate(rhos):
        M = measurements[x]
        add_entry_to_row(obj_sparse, basis, (r, M), 1.0 / n_x)

    # Ora conosciamo d = numero totale di momenti e creiamo le matrici / vettori densi per CVXPY
    d = len(basis)

    Aeq = np.vstack([dense_row_from_dict(row, d) for row in eq_rows]) if eq_rows else np.zeros((0, d))
    beq = np.asarray(eq_vals, dtype=float)

    photon_rows = []
    for x, n, row in photon_rows_sparse:
        photon_rows.append((x, n, dense_row_from_dict(row, d)))

    objective_row = dense_row_from_dict(obj_sparse, d)

    return SymbolicDiscriminationModel(
        n_x=n_x,
        n_trunc=n_trunc,
        words=words,
        loc_words=loc_words,
        completeness_words=completeness_words,
        rhos=rhos,
        measurements=measurements,
        sigmas=sigmas,
        basis=basis,
        Gamma_idx=Gamma_idx,
        Lplus_idx=Lplus_idx,
        Lminus_idx=Lminus_idx,
        Aeq=Aeq,
        beq=beq,
        photon_rows=photon_rows,
        objective_row=objective_row,
    )


# =========================
# CVXPY EXPRESSIONS FROM INDEX MATRICES (ASSIGNING ALPHA VARIABLES TO MOMENT AND LOCALIZING MATRIX)
# =========================

def expr_from_index_matrix(K, alpha_expr):
    """
    Converte una matrice K di indici in una matrice CVXPY.
    Usato per costruire la matrice dei momenti Gamma e i due termini per la localizing a partire dagli indici associati alle parole canoniche.
    K[i,j] = k  -> alpha_expr[k]
    K[i,j] = -1 -> 0
    """
    rows = []

    for i in range(K.shape[0]):
        row = []
        for j in range(K.shape[1]):
            k = int(K[i, j])
            if k < 0:
                row.append(0.0)
            else:
                row.append(alpha_expr[k])
        rows.append(row)

    return cp.bmat(rows)


def localizing_expr(model, r, alpha_expr):
    Lplus = expr_from_index_matrix(model.Lplus_idx[r], alpha_expr)
    Lminus = expr_from_index_matrix(model.Lminus_idx[r], alpha_expr)
    return Lplus - Lminus



# ========================
# SDP FOR N-STATE DISCRIMINATION
# =========================


def poisson_photon_weights(N, n_trunc):
    return np.array([np.exp(-N) * N**n / math.factorial(n) for n in range(n_trunc + 1)])


def poisson_omega(N, n_x, n_trunc):
    probn = poisson_photon_weights(N, n_trunc)
    omega_row = 1 - probn
    return np.tile(omega_row, (n_x, 1))


def solve_discrimination_witness(
    n_x,
    n_trunc,
    omega,
    solver="MOSEK",
    include_extra=False,
    completeness_mode="loc",
    eliminate_equalities=True,
    verbose=False,
    psd_tol=0.0,
):
    """
    SDP per n-state discrimination con formulazione:
    max c^T alpha
    s.t. Gamma(alpha) >= 0
         L_x(alpha) >= 0
         A alpha = b (normalizzazioni e completezza POVMs)
         photon constraints
    """
    omega = np.asarray(omega, dtype=float)
    assert omega.shape == (n_x, n_trunc + 1)
    print(f"Preparo il modello simbolico...")
    model = collect_symbolic_model(
        n_x=n_x,
        n_trunc=n_trunc,
        include_extra=include_extra,
        completeness_mode=completeness_mode,
    )
    print(f"FATTO")
    d = len(model.basis)
    constraints = []

    alpha = cp.Variable(d, name="alpha")
    alpha_expr = alpha
    print(f"Aggiungendo i constraint 'model.Aeq @ alpha == model.beq'...")
    if model.Aeq.shape[0] > 0:
        constraints.append(model.Aeq @ alpha == model.beq)
    print(f"FATTO")

    info = {
        "eliminated": False,
        "d_alpha_original": d,
        "d_beta": d,
        "num_equalities": model.Aeq.shape[0],
        "equality_residual_norm": None,
    }

    # PSD Gamma
    print(f"Aggiungendo i constraint 'Gamma >> 0'...")
    Gamma = expr_from_index_matrix(model.Gamma_idx, alpha_expr)
    nG = len(model.words)
    constraints.append(Gamma >> psd_tol * np.eye(nG))
    print(f"FATTO")
    print(f"Aggiungendo i constraint 'localizing >> 0'...")
    # PSD localizing matrices
    for r in model.rhos:
        Lr = localizing_expr(model, r, alpha_expr)
        nL = len(model.loc_words)
        constraints.append(Lr >> psd_tol * np.eye(nL))
    print(f"FATTO")
    print(f"Aggiungendo i photon constraint...")
    # Photon constraints
    for x, n, row in model.photon_rows:
        constraints.append(row @ alpha_expr >= 1.0 - omega[x, n])
        constraints.append(row @ alpha_expr <= 1.0)
    print(f"FATTO")
    print(f"Preparando l'obiettivo...")
    objective = model.objective_row @ alpha_expr
    print(f"FATTO")
    print(f"Risolvendo l' SDP...")
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver, verbose=verbose)
    print(f"FATTO")

    alpha_value = None
    if problem.status in ("optimal", "optimal_inaccurate"):
        try:
            alpha_value = np.asarray(alpha_expr.value, dtype=float).reshape(-1)
        except Exception:
            alpha_value = None

    return {
        "problem": problem,
        "model": model,
        "status": problem.status,
        "sdp_upper_bound": problem.value,
        "alpha_value": alpha_value,
        "solver": solver,
        **info,
        "num_words": len(model.words),
        "num_loc_words": len(model.loc_words),
        "num_moment_variables": d,
        "num_constraints": len(constraints),
    }


# =========================
# MAIN
# =========================

#N_values = [0.005, 0.1, 0.2, 0.5]
N_values = np.linspace(0.01, 1.0, 10)
n_x = 4
n_trunc_values = [0,1,2]

# False uses words until level k=2 full, True extends the set over partial words of lenght 3, namely (r, M, s) and (M, r, s) for all r, M, s.
include_extra_words = True
# "extended" if you want to use the completeness constraints over the extended set of words, "loc" for the localizing set of words
completeness_mode = "loc" 
eliminate_equalities = False

#crea o entra nel path e salva la configurazione con tutte le info necessarie
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"nx{n_x}_N{N_values[0]:.2f}_{N_values[-1]:.2f}_ntrunc{'-'.join(map(str, n_trunc_values))}_{timestamp}"
outdir = Path("results") / "nstates" / run_name
outdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "nstates_discrimination_results.csv"
config_path = outdir / "nstates_config.txt"

with open(config_path, "w") as f:
    f.write(f"script = nstates.py\n")
    f.write(f"n_x = {n_x}\n")
    f.write(f"N_values = {N_values}\n")
    f.write(f"n_trunc_values = {n_trunc_values}\n")
    f.write(f"solver = MOSEK\n")
    f.write(f"include_extra_words = {include_extra_words}\n")
    f.write(f"completeness_mode = {completeness_mode}\n")
    f.write(f"eliminate_equalities = {eliminate_equalities}\n")

rows = []
#simulazione
for n_trunc in n_trunc_values:

    print(f"\n===== n_trunc = {n_trunc} =====")
    sdp_values = []
    for N in N_values:
        t0 = time.perf_counter()
        
        omega = poisson_omega(N, n_x=n_x, n_trunc=n_trunc)
        res = solve_discrimination_witness(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            solver="MOSEK",
            include_extra=include_extra_words,
            completeness_mode=completeness_mode,
            eliminate_equalities=eliminate_equalities,
            verbose=False,
        )
        sdp_values.append(res["sdp_upper_bound"])

        runtime = time.perf_counter() - t0
        '''
        print(
            f"N={N:.3f} | "
            f"SDP={res['sdp_upper_bound']:.10f} | "
            f"status={res['status']} | "
            #f"words={res['num_words']} | "
            #f"num_moment_variables={res['num_moment_variables']}"
        )
        '''
        rows.append({
            "N": float(N),
            "n_x": n_x,
            "n_trunc": n_trunc,
            "status": res["status"],
            "value": res["sdp_upper_bound"],
            "runtime_sec": runtime,
            "num_constraints": res["num_constraints"],
            "num_words": res["num_words"],
            "num_loc_words": res["num_loc_words"],
            "num_moment_variables": res["num_moment_variables"],
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)


