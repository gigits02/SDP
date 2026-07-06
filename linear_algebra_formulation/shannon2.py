import cvxpy as cp
import numpy as np
import math
import chaospy
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
    Non riduce le variabili ausiliarie Z_b: Z_b non e' un proiettore.
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


def split_z_and_rest(w):
    z_part = []
    rest = []

    for a in w:
        if a.startswith("Z"):
            z_part.append(a)
        else:
            rest.append(a)

    z_part = tuple(sorted(z_part))
    rest = tuple(rest)

    return z_part, rest

def canonical_word(w):
    w = reduce_word(tuple(w))

    if is_zero_word(w):
        return None

    z_part, rest = split_z_and_rest(w)

    rest = reduce_word(rest)

    if is_zero_word(rest):
        return None

    candidates = rotations(rest) + rotations(tuple(reversed(rest)))
    candidates = [reduce_word(c) for c in candidates]
    candidates = [c for c in candidates if not is_zero_word(c)]

    if not candidates:
        return None

    return z_part + min(candidates)


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


def build_z_operators(n_x):
    return [f"Z{b}" for b in range(n_x)]


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


def build_base_words(n_x, n_trunc, include_extra=False):
    """
    Monomi senza Z, analoghi al livello k=2 usato nel witness/min-entropy.
    """
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
    mode='extended': aggiunge parole piu' aggressive.
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


def lift_with_Z(words, z_ops):
    """
    Estende le parole alle variabili dipendenti dal nodo i: S_i = {1, Z_b} x monomi.
    """
    lifted = list(words)
    for Z in z_ops:
        for w in words:
            lifted.append((Z,) + tuple(w))
    return unique_reduced_words(lifted)


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

    def idx(self, w):
        """Restituisce l'indice se la parola e' gia' nella base; non aggiunge nuove parole."""
        key = canonical_word(w)
        if key is None:
            return None
        return self.word_to_idx.get(key, None)

    def __len__(self):
        return len(self.idx_to_word)


def add_entry_to_row(row, basis, w, coeff):
    idx = basis.add(w)
    if idx is None:
        return row

    row[idx] = row.get(idx, 0.0) + float(coeff)
    return row


def dense_row_from_dict(row_dict, d):
    row = np.zeros(d)
    for k, v in row_dict.items():
        row[k] += v
    return row


def dense_row_for_word(basis, w, d, coeff=1.0):
    row = np.zeros(d)
    idx = basis.idx(w)
    if idx is not None:
        row[idx] = float(coeff)
    return row


# =========================
# SYMBOLIC MODEL FOR SHANNON/BFF
# =========================

@dataclass
class SymbolicShannonModel:
    n_x: int
    n_trunc: int
    base_words: list
    lifted_words: list
    loc_words: list
    lifted_loc_words: list
    completeness_words: list
    lifted_completeness_words: list
    rhos: list
    measurements: list
    sigmas: list
    z_ops: list
    basis: MomentBasis
    Gamma_idx: np.ndarray
    Lplus_idx: dict
    Lminus_idx: dict
    zG_idx: dict
    norm_rows: dict
    photon_rows: list
    probability_rows: dict
    z_probability_rows: dict
    h_probability_rows: dict
    h_rho_rows: dict
    z_rho_rows: dict
    h_rho_norm_rows: dict
    z_sigma_rows: dict
    h_sigma_rows: dict
    hG_idx: dict
    completeness_Aeq: np.ndarray


def matrix_indices_from_words(basis, row_words, middle=()):
    n = len(row_words)
    K = -np.ones((n, n), dtype=int)

    for i, u in enumerate(row_words):
        for j, v in enumerate(row_words):
            w = dagger_word(u) + tuple(middle) + tuple(v)
            idx = basis.add(w)
            if idx is not None:
                K[i, j] = idx

    return K


def collect_symbolic_shannon_model(
    n_x,
    n_trunc,
    include_extra=False,
    completeness_mode="loc",
):
    """
    Costruisce il modello simbolico per un singolo nodo di quadratura BFF.

    Qui la matrice dei momenti e' indicizzata dalle parole lifted
    {1, Z_b} x base_words.
    In questo modo le quantita' Tr(Z w) e Tr(Z^2 w)
    sono normali componenti dello stesso vettore alpha.
    """
    rhos, measurements, sigmas = build_operators(n_x, n_trunc)
    z_ops = build_z_operators(n_x)

    base_words = build_base_words(n_x, n_trunc, include_extra=include_extra)
    loc_words = build_localizing_words(n_x, n_trunc)
    completeness_words = build_completeness_words(n_x, n_trunc, mode=completeness_mode)

    lifted_words = lift_with_Z(base_words, z_ops)
    lifted_loc_words = lift_with_Z(loc_words, z_ops)
    lifted_completeness_words = lift_with_Z(completeness_words, z_ops)

    basis = MomentBasis()

    # 1) Full lifted moment matrix
    Gamma_idx = matrix_indices_from_words(basis, lifted_words)

    # 2) Localizing matrices on lifted localizing words
    Lplus_idx = {}
    Lminus_idx = {}
    for r in rhos:
        Lplus_idx[r] = matrix_indices_from_words(basis, lifted_loc_words, middle=(r,))
        Lminus_idx[r] = matrix_indices_from_words(basis, lifted_loc_words, middle=(r, r))

    # 3) Negativita' di Z_b e positività di Z_b^2:
    zG_idx = {}
    hG_idx = {}
    for Z in z_ops:
        zG_idx[Z] = matrix_indices_from_words(basis, base_words, middle=(Z,))
        hG_idx[Z] = matrix_indices_from_words(basis, base_words, middle=(Z, Z))

    # 4) Normalizzazioni base: Tr(rho_x)=1, Tr(sigma_n)=1
    norm_sparse = {}
    for r in rhos:
        row = {}
        add_entry_to_row(row, basis, (r,), +1.0)
        norm_sparse[r] = row

    for s in sigmas:
        row = {}
        add_entry_to_row(row, basis, (s,), +1.0)
        norm_sparse[s] = row

    # 5) Completezza POVM sulle parole lifted:
    #    sum_b Tr(u^dagger M_b v) = Tr(u^dagger v)
    completeness_rows = []
    for u in lifted_completeness_words:
        for v in lifted_completeness_words:
            row = {}
            for M in measurements:
                add_entry_to_row(row, basis, dagger_word(u) + (M,) + tuple(v), +1.0)
            add_entry_to_row(row, basis, dagger_word(u) + tuple(v), -1.0)

            row = {k: val for k, val in row.items() if abs(val) > 1e-12}
            if row:
                completeness_rows.append(row)

    # 6) Photon rows: Tr(rho_x sigma_n)
    photon_sparse = []
    for x, r in enumerate(rhos):
        for n, s in enumerate(sigmas):
            row = {}
            add_entry_to_row(row, basis, (r, s), +1.0)
            photon_sparse.append((x, n, row))

    # 7) Probability rows: Tr(rho_x M_b), per p_obs e witness
    probability_sparse = {}
    for x, r in enumerate(rhos):
        for b, M in enumerate(measurements):
            row = {}
            add_entry_to_row(row, basis, (r, M), +1.0)
            probability_sparse[(x, b)] = row


    z_rho_sparse = {}
    h_rho_norm_sparse = {}
    z_sigma_sparse = {}
    h_sigma_sparse = {}

    for b, Z in enumerate(z_ops):

        for r in rhos:
            row = {}
            add_entry_to_row(row, basis, (Z, r), +1.0)
            z_rho_sparse[(b, r)] = row

            row = {}
            add_entry_to_row(row, basis, (Z, Z, r), +1.0)
            h_rho_norm_sparse[(b, r)] = row

        for s in sigmas:
            row = {}
            add_entry_to_row(row, basis, (Z, s), +1.0)
            z_sigma_sparse[(b, s)] = row

            row = {}
            add_entry_to_row(row, basis, (Z, Z, s), +1.0)
            h_sigma_sparse[(b, s)] = row

    # 8) BFF objective rows:
    #    p_z  = Tr(Z_b rho_x* M_b)
    #    p_h  = Tr(Z_b^2 rho_x* M_b)
    #    rho_h = Tr(Z_b^2 rho_x*)
    z_probability_sparse = {}
    h_probability_sparse = {}
    h_rho_sparse = {}


    for b, M in enumerate(measurements):
        Z = z_ops[b]

        for x, r in enumerate(rhos):
            row_z = {}
            add_entry_to_row(row_z, basis, (Z, r, M), +1.0)
            z_probability_sparse[(x, b)] = row_z

            row_h = {}
            add_entry_to_row(row_h, basis, (Z, Z, r, M), +1.0)
            h_probability_sparse[(x, b)] = row_h

            row_hr = {}
            add_entry_to_row(row_hr, basis, (Z, Z, r), +1.0)
            h_rho_sparse[(x, b)] = row_hr

    # Ora conosciamo d = numero totale di momenti e creiamo le matrici / vettori densi per CVXPY
    d = len(basis)

    norm_rows = {key: dense_row_from_dict(row, d) for key, row in norm_sparse.items()}

    photon_rows = []
    for x, n, row in photon_sparse:
        photon_rows.append((x, n, dense_row_from_dict(row, d)))

    probability_rows = {key: dense_row_from_dict(row, d) for key, row in probability_sparse.items()}
    z_probability_rows = {key: dense_row_from_dict(row, d) for key, row in z_probability_sparse.items()}
    h_probability_rows = {key: dense_row_from_dict(row, d) for key, row in h_probability_sparse.items()}
    h_rho_rows = {key: dense_row_from_dict(row, d) for key, row in h_rho_sparse.items()}

    completeness_Aeq = (np.vstack([dense_row_from_dict(row, d) for row in completeness_rows]) if completeness_rows else np.zeros((0, d)))

    z_rho_rows = {
        key: dense_row_from_dict(row, d)
        for key, row in z_rho_sparse.items()
    }

    h_rho_norm_rows = {
        key: dense_row_from_dict(row, d)
        for key, row in h_rho_norm_sparse.items()
    }

    z_sigma_rows = {
        key: dense_row_from_dict(row, d)
        for key, row in z_sigma_sparse.items()
    }

    h_sigma_rows = {
        key: dense_row_from_dict(row, d)
        for key, row in h_sigma_sparse.items()
    }
    return SymbolicShannonModel(
        n_x=n_x,
        n_trunc=n_trunc,
        base_words=base_words,
        lifted_words=lifted_words,
        loc_words=loc_words,
        lifted_loc_words=lifted_loc_words,
        completeness_words=completeness_words,
        lifted_completeness_words=lifted_completeness_words,
        rhos=rhos,
        measurements=measurements,
        sigmas=sigmas,
        z_ops=z_ops,
        basis=basis,
        Gamma_idx=Gamma_idx,
        Lplus_idx=Lplus_idx,
        Lminus_idx=Lminus_idx,
        zG_idx=zG_idx,
        norm_rows=norm_rows,
        photon_rows=photon_rows,
        probability_rows=probability_rows,
        z_probability_rows=z_probability_rows,
        h_probability_rows=h_probability_rows,
        h_rho_rows=h_rho_rows,
        completeness_Aeq=completeness_Aeq,
        hG_idx=hG_idx,
        z_rho_rows=z_rho_rows,
        h_rho_norm_rows=h_rho_norm_rows,
        z_sigma_rows=z_sigma_rows,
        h_sigma_rows=h_sigma_rows,
    )


# =========================
# CVXPY EXPRESSIONS FROM INDEX MATRICES
# =========================

def _scalar_as_block(x):
    """Converte uno scalare/Expression CVXPY in un blocco 1x1 per cp.bmat."""
    if isinstance(x, (int, float, np.number)):
        return cp.Constant([[float(x)]])
    return cp.reshape(x, (1, 1), order="F")


def expr_from_index_matrix(K, alpha_expr):
    rows = []
    for i in range(K.shape[0]):
        row = []
        for j in range(K.shape[1]):
            k = int(K[i, j])
            if k < 0:
                row.append(_scalar_as_block(0.0))
            else:
                row.append(_scalar_as_block(alpha_expr[k]))
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
# QUADRATURE / PHOTON WEIGHTS / WITNESS CSV LOAD
# =========================

def make_bff_quadrature(m_in=4, eps=1e-3):
    """
    Gauss-Radau su [eps,1] con nodo fisso 1.
    Nota: m_in=4 produce il numero effettivo di nodi
    restituito da chaospy (m=8).
    """
    distribution = chaospy.Uniform(lower=eps, upper=1.0)
    nodes, weights = chaospy.quadrature.radau(m_in, distribution, fixed_point=1.0)

    t = np.asarray(nodes).reshape(-1)
    w = np.asarray(weights).reshape(-1)

    order = np.argsort(t)
    return t[order], w[order]


def poisson_photon_weights(N, n_trunc):
    return np.array([np.exp(-N) * N**n / math.factorial(n) for n in range(n_trunc + 1)])


def poisson_omega(N, n_x, n_trunc):
    probn = poisson_photon_weights(N, n_trunc)
    omega_row = 1 - probn
    return np.tile(omega_row, (n_x, 1))


def load_witness_from_csv(n_x, n_trunc, N, csv_path, value_column="value"):
    df = pd.read_csv(csv_path)
    row = df[(df["n_x"] == n_x) & (df["n_trunc"] == n_trunc) & (df["N"] == N)].iloc[0]
    return float(row[value_column])


# =========================
# SDP FOR SHANNON ENTROPY
# =========================

def solve_shannon_entropy(
    n_x,
    n_trunc,
    omega,
    W_obs=None,
    p_obs=None,
    x_star=0,
    t=None,
    w=None,
    m=4,
    solver="MOSEK",
    include_extra=False,
    completeness_mode="loc",
    eliminate_equalities=False,
    verbose=False,
    W_tol=1e-5,
    psd_tol=1e-8,
):
    """
    SDP per la Shannon entropy con nuova formulazione.

    Per ogni nodo di quadratura si risolve un SDP:
        min H_i(alpha)
    con vincoli base + lifted moment matrix + negativita' di Z_b.

    Il risultato finale e':
        H >= c_m + sum_i H_i.
    """
    omega = np.asarray(omega, dtype=float)
    assert omega.shape == (n_x, n_trunc + 1), f"omega shape {omega.shape}, atteso {(n_x, n_trunc + 1)}"

    if p_obs is not None:
        p_obs = np.asarray(p_obs, dtype=float)
        if p_obs.shape != (n_x, n_x):
            raise ValueError(f"p_obs deve avere shape {(n_x, n_x)}, ricevuta {p_obs.shape}")

    if t is None or w is None:
        t, w = make_bff_quadrature(m_in=m, eps=1e-3)
    else:
        t = np.asarray(t, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)

    if len(t) != len(w):
        raise ValueError("t e w devono avere la stessa lunghezza")
    if np.any(t <= 0):
        raise ValueError("i nodi t devono essere strettamente positivi.")

    tau = w / (t * np.log(2.0))
    c_m = float(np.sum(tau))
    m_eff = len(t)

    print("Preparo il modello simbolico Shannon/BFF...")
    model = collect_symbolic_shannon_model(
        n_x=n_x,
        n_trunc=n_trunc,
        include_extra=include_extra,
        completeness_mode=completeness_mode,
    )
    print("FATTO")

    d = len(model.basis)
    nG = len(model.lifted_words)
    nL = len(model.lifted_loc_words)
    nZ = len(model.base_words)
    photon_lb = 1.0 - omega

    H_total = c_m
    node_values = []
    statuses = []
    num_constraints = []

    for i in range(m_eff):
        print(f"\n--- Nodo BFF {i+1}/{m_eff} ---")

        alpha = cp.Variable(d, name=f"alpha_node_{i}")
        constraints = []

        print(f"Aggiungendo constraint 'Gamma_{i+1} >> 0'...")
        # Full lifted moment matrix
        Gamma = expr_from_index_matrix(model.Gamma_idx, alpha)
        constraints.append(Gamma >> -psd_tol * np.eye(nG))
        print(f"FATTO")

        print(f"Aggiungendo i constraint 'localizing_{i+1} >> 0'...")
        # Localizing matrices rho-rho^2 on lifted loc words
        for r in model.rhos:
            Lr = localizing_expr(model, r, alpha)
            constraints.append(Lr >> -psd_tol * np.eye(nL))
        print(f"FATTO")

        print(f"Aggiungendo i constraint normalizzazioni base...")
        # Normalizzazioni base
        for r in model.rhos:
            constraints.append(model.norm_rows[r] @ alpha == 1.0)
        for s in model.sigmas:
            constraints.append(model.norm_rows[s] @ alpha == 1.0)
        print(f"FATTO")

        print(f"Aggiungendo i constraint completezza lifted POVMs...")
        # Completezza POVM sulle parole lifted
        if model.completeness_Aeq.shape[0] > 0:
            constraints.append(model.completeness_Aeq @ alpha == 0)
        print(f"FATTO")

        print(f"Aggiungendo i photon constraint...")
        # Photon constraints
        for x, n, row in model.photon_rows:
            expr = row @ alpha
            constraints.append(expr >= photon_lb[x, n])
            constraints.append(expr <= 1.0)
        print(f"FATTO")

        print(f"Aggiungendo constraint sulla witness/statisiche...")
        # Dati osservati: p_obs oppure witness
        if p_obs is not None:
            for x in range(n_x):
                for b in range(n_x):
                    constraints.append(model.probability_rows[(x, b)] @ alpha == p_obs[x, b])

        W_total = 0
        for x in range(n_x):
            W_total += (model.probability_rows[(x, x)] @ alpha) / n_x

        if W_obs is not None:
            constraints.append(W_total >= W_obs - W_tol)
        print(f"FATTO")

        print(f"Aggiungendo i constraint negativita' Z_b...")
        # Negativita' di ogni Z_b e positività di Z_b^2
        for Z in model.z_ops:
            zG = expr_from_index_matrix(model.zG_idx[Z], alpha)
            hG = expr_from_index_matrix(model.hG_idx[Z], alpha)
            constraints.append(zG << psd_tol * np.eye(nZ))
            constraints.append(hG >> -psd_tol * np.eye(nZ))
        print(f"FATTO")


        print(f"Aggiungendo constraint z scalare...")
        # Z_b scalare: Tr(Z_b rho_x) = Z_b, Tr(Z_b^2 rho_x) = Z_b^2,
        # Tr(Z_b sigma_n) = Z_b, Tr(Z_b^2 sigma_n) = Z_b^2
        z_scalar = cp.Variable(n_x, nonpos=True, name=f"z_scalar_node_{i}")
        h_scalar = cp.Variable(n_x, nonneg=True, name=f"h_scalar_node_{i}")
        for b, Z in enumerate(model.z_ops):

            for r in model.rhos:
                constraints.append(
                    model.z_rho_rows[(b, r)] @ alpha == z_scalar[b]
                )
                constraints.append(
                    model.h_rho_norm_rows[(b, r)] @ alpha == h_scalar[b]
                )

            for s in model.sigmas:
                constraints.append(
                    model.z_sigma_rows[(b, s)] @ alpha == z_scalar[b]
                )
                constraints.append(
                    model.h_sigma_rows[(b, s)] @ alpha == h_scalar[b]
                )
        print(f"FATTO")

        print(f"Preparando l'obiettivo per nodo_{i+1}...")
        # BFF objective for this node
        H_i = 0
        for b in range(n_x):
            p_z = model.z_probability_rows[(x_star, b)] @ alpha
            p_h = model.h_probability_rows[(x_star, b)] @ alpha
            rho_h = model.h_rho_rows[(x_star, b)] @ alpha
            H_i += tau[i] * (2.0 * p_z + (1.0 - t[i]) * p_h + t[i] * rho_h)
        print(f"FATTO")

        print(f"Risolvendo il {i+1}° SDP ...")
        problem = cp.Problem(cp.Minimize(H_i), constraints)
        problem.solve(solver=solver, verbose=verbose)
        print(f"FATTO")

        statuses.append(problem.status)
        num_constraints.append(len(constraints))

        if problem.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
            return {
                "n_x": n_x,
                "n_trunc": n_trunc,
                "omega": omega,
                "photon_lb": photon_lb,
                "x_star": x_star,
                "W_obs": W_obs,
                "p_obs": p_obs,
                "H_shannon_bits": None,
                "c_m": c_m,
                "tau": tau,
                "t": t,
                "w": w,
                "node_values": node_values,
                "statuses": statuses,
                "failed_node": i,
                "num_constraints_per_node": num_constraints,
                "num_words": len(model.lifted_words),
                "num_base_words": len(model.base_words),
                "num_moment_variables": d,
                "m_eff": m_eff,
            }

        node_values.append(float(problem.value))
        H_total += float(problem.value)

    return {
        "n_x": n_x,
        "n_trunc": n_trunc,
        "omega": omega,
        "photon_lb": photon_lb,
        "x_star": x_star,
        "W_obs": W_obs,
        "p_obs": p_obs,
        "H_shannon_bits": max(float(H_total), 0.0),
        "c_m": c_m,
        "tau": tau,
        "t": t,
        "w": w,
        "node_values": node_values,
        "statuses": statuses,
        "failed_node": None,
        "num_constraints_per_node": num_constraints,
        "num_words": len(model.lifted_words),
        "num_base_words": len(model.base_words),
        "num_moment_variables": d,
        "m_eff": m_eff,
    }


# =========================
# MAIN
# =========================

N_values = np.linspace(0.01, 1.0, 10)
n_x = 3
n_trunc_values = [0, 1, 2]


# False uses words until level k=2 full, True extends the set over partial words of lenght 3, namely (r, M, s) and (M, r, s) for all r, M, s.
include_extra_words = True
# "extended" if you want to use the completeness constraints over the extended set of words, "loc" for the localizing set of words
completeness_mode = "loc" 
eliminate_equalities = False
solver = "MOSEK" 
mode = "witness"
# mode = "full_distribution"

# Se mode == "witness", W_obs viene letto dal CSV della run witness gia' calcolata.
witness_csv_path = "./results/nstates/nx3_N0.01_1.00_ntrunc0-1-2_20260704_002747/nstates_discrimination_results.csv"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"nx{n_x}_N{N_values[0]:.2f}_{N_values[-1]:.2f}_ntrunc{'-'.join(map(str, n_trunc_values))}_{timestamp}"
outdir = Path("results") / "shannon" / run_name
outdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "shannon_results.csv"
config_path = outdir / "shannon_config.txt"

with open(config_path, "w") as f:
    f.write("script = shannon2.py\n")
    f.write(f"n_x = {n_x}\n")
    f.write(f"N_values = {N_values}\n")
    f.write(f"n_trunc_values = {n_trunc_values}\n")
    f.write(f"solver = {solver}\n")
    f.write(f"include_extra_words = {include_extra_words}\n")
    f.write(f"completeness_mode = {completeness_mode}\n")
    f.write(f"eliminate_equalities = {eliminate_equalities}\n")
    f.write(f"mode = {mode}\n")

rows = []

for n_trunc in n_trunc_values:
    print(f"\n===== n_trunc = {n_trunc} =====")

    for N in N_values:
        t0 = time.perf_counter()
        omega = poisson_omega(N, n_x=n_x, n_trunc=n_trunc)

        if mode == "witness":
            W_obs = load_witness_from_csv(
                n_x=n_x,
                n_trunc=n_trunc,
                N=N,
                csv_path=witness_csv_path,
                value_column="value",
            )
            p_obs = None

        elif mode == "full_distribution":
            W_obs = None
            p_obs = np.eye(n_x)

        else:
            raise ValueError("'mode' deve essere 'witness' oppure 'full_distribution'")

        H_result = solve_shannon_entropy(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            W_obs=W_obs,
            p_obs=p_obs,
            x_star=0,
            m=4,
            solver=solver,
            include_extra=include_extra_words,
            completeness_mode=completeness_mode,
            eliminate_equalities=eliminate_equalities,
            verbose=False,
            W_tol=1e-5,
        )

        runtime = time.perf_counter() - t0

        rows.append({
            "N": float(N),
            "n_x": n_x,
            "n_trunc": n_trunc,
            "W_obs": H_result["W_obs"],
            "statuses": str(H_result["statuses"]),
            "failed_node": H_result["failed_node"],
            "H_shannon_bits": H_result["H_shannon_bits"],
            "runtime_sec": runtime,
            "c_m": H_result["c_m"],
            "m_eff": H_result["m_eff"],
            "num_constraints": str(H_result["num_constraints_per_node"]),
            "num_words": H_result["num_words"],
            "num_base_words": H_result["num_base_words"],
            "num_moment_variables": H_result["num_moment_variables"],
            "node_values": str(H_result["node_values"]),
        })

        pd.DataFrame(rows).to_csv(csv_path, index=False)
