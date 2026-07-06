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
    Riconosce parole nulle per ortogonalita':
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
    - ciclicita' della traccia;
    - inversione della parola, assumendo operatori hermitiani.
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
    """Gli operatori sono hermitiani, quindi u^dagger e' la parola invertita."""
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
    mode='extended': aggiunge parole piu' forti.
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
# MOMENT BASIS
# =========================

class MomentBasis:
    """Dizionario parola canonica -> indice alpha."""

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
    Funzione che crea l'algebra per lavorare sui coefficienti del vettore di base dei momenti.
    'Row' è un dizionario che associa indici di parola canonica a coefficienti:
    se l'indice non è presente, il coefficiente è 0, se l'indice è presente, i coefficienti si sommano.
    """
    idx = basis.add(w)
    if idx is None:
        return row

    row[idx] = row.get(idx, 0.0) + float(coeff)
    return row


def dense_row_from_dict(row_dict, d):
    """
    Questa funzione serve perché bisogna passare a CVXPY un array. Dunque dal dizionario row_dict,
    che rappresenta una riga sparsa (indice parola canonica -> coefficiente),
    si costruisce un array denso di dimensione d.
    """
    row = np.zeros(d)
    for k, v in row_dict.items():
        row[k] += v
    return row


# =========================
# SYMBOLIC MODEL
# =========================

@dataclass
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
    completeness_Aeq: np.ndarray
    rho_norm_rows: dict
    sigma_norm_rows: dict
    photon_rows: list
    probability_rows: dict


def matrix_indices_from_words(basis, row_words, middle=()):
    """
    Restituisce una matrice di indici K tale che:
    K[i,j] = indice di alpha associato a Tr(u_i^dagger middle v_j),
    oppure -1 se la parola e' nulla.
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


def collect_symbolic_model(n_x, n_trunc, include_extra=False, completeness_mode="loc"):
    """
    Costruisce la rappresentazione algebrica del modello simbolico
    per l'SDP di min-entropy.

    Per la min-entropy servono:
    - Gamma_idx e localizing matrices per ogni blocco lambda;
    - rho_norm_rows e sigma_norm_rows per imporre
      Tr_l(rho_x)=q_l e Tr_l(sigma_n)=q_l;
    - completeness_Aeq, che contiene solo uguaglianze omogenee;
    - photon_rows per i vincoli fotonici medi;
    - probability_rows per p_obs, witness totale e guessing probability.
    """

    rhos, measurements, sigmas = build_operators(n_x, n_trunc)
    words = build_words(n_x, n_trunc, include_extra=include_extra)
    loc_words = build_localizing_words(n_x, n_trunc)
    completeness_words = build_completeness_words(n_x, n_trunc, mode=completeness_mode)

    basis = MomentBasis()

    # 1) Moment matrix Gamma
    Gamma_idx = matrix_indices_from_words(basis, words)

    # 2) Localizing matrices L_r = Lplus_r - Lminus_r
    Lplus_idx = {}
    Lminus_idx = {}

    for r in rhos:
        Lplus_idx[r] = matrix_indices_from_words(basis, loc_words, middle=(r,))
        Lminus_idx[r] = matrix_indices_from_words(basis, loc_words, middle=(r, r))

    # 3) Normalizzazioni pesate nei blocchi lambda
    rho_norm_sparse = {}
    sigma_norm_sparse = {}

    for r in rhos:
        row = {}
        add_entry_to_row(row, basis, (r,), +1.0)
        rho_norm_sparse[r] = row

    for s in sigmas:
        row = {}
        add_entry_to_row(row, basis, (s,), +1.0)
        sigma_norm_sparse[s] = row

    # 4) Completezza POVM: solo uguaglianze omogenee
    completeness_rows = []

    for u in completeness_words:
        for v in completeness_words:
            row = {}
            for M in measurements:
                add_entry_to_row(row, basis, dagger_word(u) + (M,) + tuple(v), +1.0,)
            add_entry_to_row(row, basis, dagger_word(u) + tuple(v), -1.0,)

            row = {k: val for k, val in row.items() if abs(val) > 1e-12}
            if row:
                completeness_rows.append(row)

    # 5) Photon rows: Tr(rho_x sigma_n)
    photon_rows_sparse = []
    for x, r in enumerate(rhos):
        for n, s in enumerate(sigmas):
            row = {}
            add_entry_to_row(row, basis, (r, s), +1.0)
            photon_rows_sparse.append((x, n, row))

    # 6) Probability rows: Tr(rho_x M_b)
    probability_rows_sparse = {}
    for x, r in enumerate(rhos):
        for b, M in enumerate(measurements):
            row = {}
            add_entry_to_row(row, basis, (r, M), +1.0)
            probability_rows_sparse[(x, b)] = row

    # Ora conosciamo d = numero totale di momenti e creiamo le matrici / vettori densi per CVXPY
    d = len(basis)

    rho_norm_rows = {r: dense_row_from_dict(row, d) for r, row in rho_norm_sparse.items()}
    sigma_norm_rows = {s: dense_row_from_dict(row, d) for s, row in sigma_norm_sparse.items()}
    completeness_Aeq = (np.vstack([dense_row_from_dict(row, d) for row in completeness_rows]) if completeness_rows else np.zeros((0, d)))

    photon_rows = []
    for x, n, row in photon_rows_sparse:
        photon_rows.append((x, n, dense_row_from_dict(row, d)))

    probability_rows = {key: dense_row_from_dict(row, d) for key, row in probability_rows_sparse.items()}

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
        completeness_Aeq=completeness_Aeq,
        rho_norm_rows=rho_norm_rows,
        sigma_norm_rows=sigma_norm_rows,
        photon_rows=photon_rows,
        probability_rows=probability_rows,
    )


# =========================
# CVXPY EXPRESSIONS FROM INDEX MATRICES
# =========================

def scalar_as_block(x):
    """Converte uno scalare CVXPY in un blocco 1x1 per cp.bmat."""
    if isinstance(x, (int, float, np.number)):
        return cp.Constant([[float(x)]])
    return cp.reshape(x, (1, 1), order="F")


def expr_from_index_matrix(K, alpha_expr):
    """
    Converte una matrice K di indici in una matrice CVXPY.
    K[i,j] = k  -> alpha_expr[k]
    K[i,j] = -1 -> 0
    """
    rows = []
    for i in range(K.shape[0]):
        row = []
        for j in range(K.shape[1]):
            k = int(K[i, j])
            if k < 0:
                row.append(scalar_as_block(0.0))
            else:
                row.append(scalar_as_block(alpha_expr[k]))
        rows.append(row)
    return cp.bmat(rows)


def localizing_expr(model, r, alpha_expr):
    Lplus = expr_from_index_matrix(model.Lplus_idx[r], alpha_expr)
    Lminus = expr_from_index_matrix(model.Lminus_idx[r], alpha_expr)
    return Lplus - Lminus


def cvx_sum(exprs):
    exprs = list(exprs)
    if len(exprs) == 0:
        return 0
    return cp.sum(cp.hstack(exprs))


# =========================
# PHOTON WEIGHTS AND WITNESS CSV LOAD
# =========================

def poisson_photon_weights(N, n_trunc):
    return np.array([np.exp(-N) * N**n / math.factorial(n) for n in range(n_trunc + 1)])


def poisson_omega(N, n_x, n_trunc):
    probn = poisson_photon_weights(N, n_trunc)
    omega_row = 1 - probn
    return np.tile(omega_row, (n_x, 1))


def load_witness_from_csv(n_x, n_trunc, N, csv_path, value_column="value"):
    """
    Legge il witness W_obs dal file CSV dei risultati della discrimination.
    """
    df = pd.read_csv(csv_path)
    
    row = df[(df["n_x"] == n_x) & (df["n_trunc"] == n_trunc) & (df["N"] == N)].iloc[0]

    return float(row[value_column])


# =========================
# SDP FOR MIN-ENTROPY
# =========================

def solve_min_entropy(
    n_x,
    n_trunc,
    omega,
    W_obs=None,
    p_obs=None,
    x_star=0,
    solver="MOSEK",
    include_extra=False,
    completeness_mode="loc",
    eliminate_equalities=False,
    verbose=False,
    tol=1e-5,
    psd_tol=0.0,
):
    """
    SDP per la min-entropy con nuova formulazione.

    Linearizing trick l=b:
    - un blocco alpha_l per ogni hidden variable lambda=l=b, cioe' per ogni possibile outcome;
    - una variabile q_l >= 0 \sum_l q_l == 1 per ogni blocco;
    - normalizzazioni pesate: Tr_l(rho_x)=q_l e Tr_l(sigma_n)=q_l;
    - completezza POVM omogenea in ogni blocco;
    - vincoli fotonici medi sulla somma dei blocchi;
    - dati osservati: o p_obs completa, o witness W_obs;
    - obiettivo: massimizzare P_g = sum_l Tr_l(rho_{x_star} M_l).
    """
    omega = np.asarray(omega, dtype=float)
    assert omega.shape == (n_x, n_trunc + 1), f"omega shape {omega.shape}, atteso {(n_x, n_trunc + 1)}"

    if p_obs is not None:
        p_obs = np.asarray(p_obs, dtype=float)
        if p_obs.shape != (n_x, n_x):
            raise ValueError(f"p_obs deve avere shape {(n_x, n_x)}, ricevuta {p_obs.shape}")

    #print(f"Preparo il modello simbolico...")
    model = collect_symbolic_model(
        n_x=n_x,
        n_trunc=n_trunc,
        include_extra=include_extra,
        completeness_mode=completeness_mode,
    )
    #print(f"FATTO")

    d = len(model.basis)
    n_blocks = n_x
    nG = len(model.words)
    nL = len(model.loc_words)

    alpha_blocks = [cp.Variable(d, name=f"alpha_lambda_{l}") for l in range(n_blocks)]
    q = cp.Variable(n_blocks, nonneg=True, name="q")

    constraints = []

    # Un blocco SDP per ogni lambda=l.
    for l, alpha_l in enumerate(alpha_blocks):
        
        #print(f"Aggiungendo i constraint 'Gamma_{l} >> 0'...")
        Gamma_l = expr_from_index_matrix(model.Gamma_idx, alpha_l)
        constraints.append(Gamma_l >> psd_tol * np.eye(nG))
        #print(f"FATTO")

        #print(f"Aggiungendo i constraint 'localizing_{l} >> 0'...")
        for r in model.rhos:
            Lr_l = localizing_expr(model, r, alpha_l)
            constraints.append(Lr_l >> psd_tol * np.eye(nL))
        #print(f"FATTO")

        #print(f"Aggiungendo i constraint normalizzazioni pesate...")
        # Normalizzazioni pesate: Tr_l(rho_x)=q_l, Tr_l(sigma_n)=q_l.
        for r in model.rhos:
            constraints.append(model.rho_norm_rows[r] @ alpha_l == q[l])
        for s in model.sigmas:
            constraints.append(model.sigma_norm_rows[s] @ alpha_l == q[l])
        #print(f"FATTO")

        #print(f"Aggiungendo i constraint completezza POVMs...")
        # Completezza POVM nel blocco lambda: solo parte omogenea.
        if model.completeness_Aeq.shape[0] > 0:
            constraints.append(model.completeness_Aeq @ alpha_l == 0)
        #print(f"FATTO")

    #print(f"Aggiungendo i constraint sulla normalizzazione di q...")
    # Distribuzione classica delle hidden variables.
    constraints.append(cp.sum(q) == 1)
    #print(f"FATTO")

    #print(f"Aggiungendo i photon constraint...")
    # Vincoli fotonici medi: somma sui blocchi lambda.
    photon_lb = 1.0 - omega
    for x, n, row in model.photon_rows:
        mean_photon_expr = cvx_sum(row @ alpha_l for alpha_l in alpha_blocks)
        constraints.append(mean_photon_expr >= photon_lb[x, n])
        constraints.append(mean_photon_expr <= 1.0)
    #print(f"FATTO")

    #print(f"Aggiungendo constraint sulla witness/statisiche...")
    # Dati osservati: distribuzione completa p(b|x), se data.
    if p_obs is not None:
        for x in range(n_x):
            for b in range(n_x):
                prob_expr = cvx_sum(
                    model.probability_rows[(x, b)] @ alpha_l
                    for alpha_l in alpha_blocks
                )
                constraints.append(prob_expr == p_obs[x, b])

    # Dato osservato alternativo: witness.
    W_total = 0
    for l in range(n_x):
        alpha_l = alpha_blocks[l]
        for x in range(n_x):
            row = model.probability_rows[(x, x)]
            W_total += (row @ alpha_l) / n_x
    
    if W_obs is not None:
        constraints.append(W_total >= W_obs - tol)
    #print(f"FATTO")

    #print(f"Preparando l'obiettivo...")
    # Guessing probability: lambda=l prova a indovinare outcome b=l.
    pg = cvx_sum(model.probability_rows[(x_star, l)] @ alpha_blocks[l] for l in range(n_blocks))
    #print(f"FATTO")

    #print(f"Risolvendo l' SDP...")
    problem = cp.Problem(cp.Maximize(pg), constraints)
    problem.solve(solver=solver, verbose=verbose)
    #print(f"FATTO")

    pg_value = problem.value
    if problem.status not in ("optimal", "optimal_inaccurate") or pg_value is None or pg_value <= 0:
        pg_clip = None
        H_min = None
    else:
        pg_clip = min(max(float(pg_value), 0.0), 1.0)
        H_min = -np.log2(pg_clip)

    q_values = None
    if problem.status in ("optimal", "optimal_inaccurate") and q.value is not None:
        q_values = [float(v) for v in np.asarray(q.value).reshape(-1)]

    alpha_values = None
    if problem.status in ("optimal", "optimal_inaccurate"):
        alpha_values = []
        for alpha_l in alpha_blocks:
            if alpha_l.value is None:
                alpha_values.append(None)
            else:
                alpha_values.append(np.asarray(alpha_l.value, dtype=float).reshape(-1))

    return {
        "problem": problem,
        "model": model,
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
        "alpha_values": alpha_values,
        "W_total": None if W_total is None else W_total.value,
        "solver": solver,
        "num_blocks": n_blocks,
        "num_constraints": len(constraints),
        "num_words_per_block": len(model.words),
        "num_loc_words_per_block": len(model.loc_words),
        "num_moment_variables_per_block": d,
    }


# =========================
# MAIN
# =========================

N_values = np.linspace(0.01, 1.0, 10)
n_x = 4
n_trunc_values = [0, 1, 2]

# False uses words until level k=2 full, True extends the set over partial words of lenght 3, namely (r, M, s) and (M, r, s) for all r, M, s.
include_extra_words = True
# "extended" if you want to use the completeness constraints over the extended set of words, "loc" for the localizing set of words
completeness_mode = "loc" 
eliminate_equalities = False
solver = "MOSEK" 

# W_obs viene letto dal CSV della run witness già calcolata.
witness_csv_path = "./results/nstates/nx4_N0.01_1.00_ntrunc0-1-2_20260704_141320/nstates_discrimination_results.csv"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"nx{n_x}_N{N_values[0]:.2f}_{N_values[-1]:.2f}_ntrunc{'-'.join(map(str, n_trunc_values))}_{timestamp}"
outdir = Path("results") / "minEntropy" / run_name
outdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "minEntropy_results.csv"
config_path = outdir / "minEntropy_config.txt"

with open(config_path, "w") as f:
    f.write("script = minentropy2.py\n")
    f.write(f"n_x = {n_x}\n")
    f.write(f"N_values = {N_values}\n")
    f.write(f"n_trunc_values = {n_trunc_values}\n")
    f.write(f"solver = {solver}\n")
    f.write(f"include_extra_words = {include_extra_words}\n")
    f.write(f"completeness_mode = {completeness_mode}\n")
    f.write(f"eliminate_equalities = {eliminate_equalities}\n")


rows = []

for n_trunc in n_trunc_values:
    print(f"\n===== n_trunc = {n_trunc} =====")

    for N in N_values:
        t0 = time.perf_counter()
        omega = poisson_omega(N, n_x=n_x, n_trunc=n_trunc)

        # 1) Witness osservato letto da CSV.
        W_obs = load_witness_from_csv(
            n_x=n_x,
            n_trunc=n_trunc,
            N=N,
            csv_path=witness_csv_path,
            value_column="value",
        )

        # 2) Min-entropy con vincolo sul witness osservato.
        res_H = solve_min_entropy(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            W_obs=W_obs,
            p_obs=None,
            x_star=0,
            solver=solver,
            include_extra=include_extra_words,
            completeness_mode=completeness_mode,
            eliminate_equalities=eliminate_equalities,
            verbose=False,
            tol=1e-5,
        )

        runtime = time.perf_counter() - t0
        '''
        print(
            f"N={N:.3f} | "
            f"W_obs={W_obs:.10f} | "
            f"pg={res_H['guessing_probability_clipped']} | "
            f"H_min={res_H['H_min_bits']} | "
            f"status={res_H['status']} | "
            f"q={res_H['q_values']}"
        )
        '''

        rows.append({
            "N": float(N),
            "n_x": n_x,
            "n_trunc": n_trunc,
            "W_obs": W_obs,
            "W_total": res_H["W_total"],
            "pg": res_H["guessing_probability_clipped"],
            "status": res_H["status"],
            "H_min_bits": res_H["H_min_bits"],
            "runtime_sec": runtime,
            "num_constraints": res_H["num_constraints"],
            "num_blocks": res_H["num_blocks"],
            "num_words_per_block": res_H["num_words_per_block"],
            "num_loc_words_per_block": res_H["num_loc_words_per_block"],
            "num_moment_variables_per_block": res_H["num_moment_variables_per_block"],
            "q_values": res_H["q_values"],
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
