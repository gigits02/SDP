import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
import chaospy
from math import erf, sqrt, inf
from scipy.special import erfinv
import time
from pathlib import Path
import pandas as pd
from datetime import datetime

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
    
def build_operators(n_x, n_b, n_trunc):
    rhos = [f"r{x}" for x in range(n_x)]
    measurements = [f"M{b}" for b in range(n_b)]
    sigmas = [f"s{n}" for n in range(n_trunc + 1)]
    return rhos, measurements, sigmas

def build_words(n_x, n_b, n_trunc, include_extra=True, bpsk_level=True):
    rhos, measurements, sigmas = build_operators(n_x, n_b, n_trunc)
    generators = rhos + measurements + sigmas
    words = [()]

    # livello 1
    words += [(A,) for A in generators]

    # livello 2 completo: tutti i prodotti AB
    words += [(A, B) for A in generators for B in generators]

    if include_extra:
        # Alcune parole più forti (da usare per n_x > 2)
        words += [(r, M, s) for r in rhos for M in measurements for s in sigmas]
        words += [(M, r, s) for M in measurements for r in rhos for s in sigmas]

    if bpsk_level:
        # extra indicati dagli autori: rho rho rho + sigma M rho
        words += [(r1, r2, r3) for r1 in rhos for r2 in rhos for r3 in rhos]
        words += [(s, M, r) for s in sigmas for M in measurements for r in rhos]

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


def build_localizing_words(n_x, n_b, n_trunc):
    rhos, measurements, sigmas = build_operators(n_x, n_b, n_trunc)

    words = [()]
    words += [(r,) for r in rhos]
    words += [(M,) for M in measurements]
    words += [(s,) for s in sigmas]

    return words

def solve_n_state_discrimination(n_x, n_b, n_trunc, omega, solver="CLARABEL", verbose=False, include_extra_words=True):
    """
    Risolve un rilassamento SDP per discriminazione di n_x stati
    con vincoli sulle componenti fotoniche fino a n_trunc.
    """

    sdp = TracialSDP()
    rhos, measurements, sigmas = build_operators(n_x, n_b, n_trunc)

    words = build_words(n_x, n_b, n_trunc, include_extra=include_extra_words, bpsk_level=False)
    loc_words = build_localizing_words(n_x, n_b, n_trunc)

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
        "n_b": n_b,
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

def solve_min_entropy_randomness(n_x, n_b, n_trunc, omega, W_obs=None, p_obs=None, x_star=0, solver="MOSEK", verbose=False, include_extra_words=True, bpsk_level=True, tol=1e-5):
    """
    SDP per certificare H_min(B|X=x_star,Lambda) nel task di n-state discrimination.
    """

    if p_obs is not None:
        p_obs = np.asarray(p_obs, dtype=float)
        if p_obs.shape != (n_x, n_b):
            raise ValueError(f"p_obs deve avere shape {(n_x, n_b)}, ricevuta {p_obs.shape}")


    if W_obs is None and p_obs is None:
        tmp = solve_n_state_discrimination(
            n_x=n_x,
            n_b=n_b,
            n_trunc=n_trunc,
            omega=omega,
            solver=solver,
            verbose=verbose,
            include_extra_words=include_extra_words,
        )
        W_obs = tmp["sdp_upper_bound"]

    rhos, measurements, sigmas = build_operators(n_x, n_b, n_trunc)
    words = build_words(n_x, n_b, n_trunc, include_extra=include_extra_words, bpsk_level=bpsk_level)
    loc_words = build_localizing_words(n_x, n_b, n_trunc)

    sdps = []
    q_list = []
    W_list = []
    constraints = []

    # Una hidden strategy lambda=l per ogni possibile guess (b=l).
    for l in range(n_b):
        sdp_l = TracialSDP()
        sdps.append(sdp_l)

        q_l = cp.Variable(nonneg=True, name=f"q_{l}")
        q_list.append(q_l)

        Gamma_l = sdp_l.moment_matrix(words)
        constraints.append(Gamma_l >> 0)

        # Localizing: rho_x - rho_x^2 >= 0.
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

        if n_b >= n_x:
            W_l = cvx_sum(
                sdp_l.T((rhos[x], measurements[x])) for x in range(n_x)
            ) / n_x
        else:
            W_l = 0
        W_list.append(W_l)

    # Distribuzione classica delle hidden strategies.
    constraints.append(cvx_sum(q_list) == 1)

    # Vincoli fotonici medi.
    photon_lb = 1.0 - np.asarray(omega, dtype=float)
    for x, r in enumerate(rhos):
        for n, s in enumerate(sigmas):
            constraints.append(
                cvx_sum(sdps[l].T((r, s)) for l in range(n_b))
                >= photon_lb[x, n]
            )

    # Dati osservati: o distribuzione completa, o solo witness.
    if p_obs is not None:
        for x, r in enumerate(rhos):
            for b, M in enumerate(measurements):
                constraints.append(
                    cvx_sum(sdps[l].T((r, M)) for l in range(n_b))
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
        for l in range(n_b)
    )

    problem = cp.Problem(cp.Maximize(pg), constraints)
    problem.solve(
                    solver="MOSEK",
                    verbose=verbose,
                    mosek_params={
                        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1,
                        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-7,
                        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-7,
                    }
    )

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
        "num_blocks": n_b,
        "num_constraints": len(constraints),
        "num_words_per_block": len(words),
        "num_moment_variables_per_block": [len(sdp.vars) for sdp in sdps],
    }

# =========================================================
# SHANNON ENTROPY LINEARIZZATA CON BFF / GAUSS-RADAU
# =========================================================

def cvx_sum(exprs):
    """Somma CVXPY più stabile di sum([...]) per liste lunghe."""
    exprs = list(exprs)
    if len(exprs) == 0:
        return 0
    return cp.sum(cp.hstack(exprs))


def make_bff_quadrature(m_in=4, eps=1e-3):
    """
    Restituisce nodi t e pesi w della Gauss-Radau su [0,1] usando chaospy.

    il numero di quadrature m=8 è restituito da chaospy.radau(m_in, Uniform(eps,1), fixed_point=1)
    quando m_in è uguale a 4.
    """
    distribution = chaospy.Uniform(lower=eps, upper=1.0)
    nodes, weights = chaospy.quadrature.radau(m_in, distribution, fixed_point=1.0)

    t = np.asarray(nodes).reshape(-1)
    w = np.asarray(weights).reshape(-1)

    order = np.argsort(t)
    return t[order], w[order]


class LiftedTraceBlock:
    """
    Blocco BFF per un singolo paio (i,b). Contiene variabili lineari
    per Tr(w), Tr(Z w), Tr(Z^2 w). Z è trattato come z I
    e la convessita' è imposta tramite il vincolo generale
    [[G,zG],[zG,hG]] >= 0.
    """
    def __init__(self, words, base):
        self.words = words
        self.base = base
        self.z_vars = {}
        self.h_vars = {}
        self.z_scalar = cp.Variable(nonpos=True)
        self.h_scalar = cp.Variable(nonneg=True)

    def T(self, w):
        return self.base.T(w)

    def ZT(self, w):
        w = tuple(w)
        if is_zero_word(w):
            return 0
        key = canonical_trace_word(w)
        if is_zero_word(key):
            return 0
        if key not in self.z_vars:
            name = "ZT_I" if len(key) == 0 else "ZT_" + "_".join(key)
            self.z_vars[key] = cp.Variable(name=name)
        return self.z_vars[key]

    def HT(self, w):
        w = tuple(w)
        if is_zero_word(w):
            return 0
        key = canonical_trace_word(w)
        if is_zero_word(key):
            return 0
        if key not in self.h_vars:
            name = "HT_I" if len(key) == 0 else "HT_" + "_".join(key)
            self.h_vars[key] = cp.Variable(name=name)
        return self.h_vars[key]

    def GammaMatrix_from(self, func, words=None):
        if words is None:
            words = self.words
        return cp.bmat([
            [func(tuple(reversed(u)) + tuple(v)) for v in words]
            for u in words
        ])

    def localizing_from(self, func, loc_words, rho):
        return cp.bmat([
            [
                func(tuple(reversed(u)) + (rho,) + tuple(v))
                - func(tuple(reversed(u)) + (rho, rho) + tuple(v))
                for v in loc_words
            ]
            for u in loc_words
        ])

def solve_shannon_entropy_bff_randomness(
    n_x,
    n_b,
    n_trunc,
    omega,
    W_obs=None,
    p_obs=None,
    x_star=0,
    t=None,
    w=None,
    m=8,
    solver="MOSEK",
    verbose=False,
    include_extra_words=True,
    bpsk_level=True,
    W_tol=1e-5,
    psd_tol=1e-8,
):
    """
    SDP per certificare H(b|x=x_star,lambda) con Shannon entropy
    linearizzata via Brown-Fawzi-Fawzi (metodo delle quadrature di Gauss-Radau).

    Convenzioni:
      - scenario n-state discrimination: n_x stati e n_x outcome;
      - una sola measurement y=0, quindi M_b := M_{b|0};
      - omega[x,n] = 1 - P_n(x), quindi Tr(rho_x sigma_n) >= 1-omega[x,n];
      - se p_obs e' dato, viene fissata tutta la distribuzione p(b|x),
      altrimenti si fissa il witness W >= W_obs - tol.

    La forma implementata è:
      H >= c_m + sum_i min_Gamma tau_i sum_b [
            2 Tr(Z_b rho_x* M_b)
          + (1-t_i) Tr(Z_b rho_x* Z_b M_b)
          + t_i Tr(Z_b rho_x* Z_b)
        ].
    In forma rilassata usiamo variabili Tr(w), Tr(Zw), Tr(Z^2w),
    il vincolo zG <= 0 e PSD [[G,zG],[zG,hG]] >= 0.
    """

    omega = np.asarray(omega, dtype=float)
    if omega.shape != (n_x, n_trunc + 1):
        raise ValueError(f"omega deve avere shape {(n_x, n_trunc + 1)}, ricevuta {omega.shape}")

    if p_obs is not None:
        p_obs = np.asarray(p_obs, dtype=float)
        if p_obs.shape != (n_x, n_b):
            raise ValueError(f"p_obs deve avere shape {(n_x, n_b)}, ricevuta {p_obs.shape}")

    if t is None or w is None:
        t, w = make_bff_quadrature(m_in=m, eps=1e-3)
    else:
        t = np.asarray(t, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)
        m = len(t)

    if len(t) != len(w):
        raise ValueError("t e w devono avere la stessa lunghezza")
    if np.any(t <= 0):
        raise ValueError("i nodi t devono essere strettamente positivi.")

    tau = w / (t * np.log(2.0))
    c_m = float(np.sum(tau))
    m_eff = len(t)

    #print("t =", t)
    #print("w =", w)
    #print("tau =", tau)
    #print("c_m =", c_m)
    #print("m_eff =", m_eff)

    # Se non vengono passati né W_obs né p_obs, prima calcolo il massimo witness compatibile (servirà per il constraint su W_total).
    if W_obs is None and p_obs is None:
        res_W = solve_n_state_discrimination(
            n_x=n_x,
            n_b=n_b,
            n_trunc=n_trunc,
            omega=omega,
            solver=solver,
            verbose=verbose,
            include_extra_words=include_extra_words,
        )
        W_obs = res_W["sdp_upper_bound"]

    rhos, measurements, sigmas = build_operators(n_x, n_b, n_trunc)
    words = build_words(n_x, n_b, n_trunc, include_extra=include_extra_words, bpsk_level=bpsk_level)
    loc_words = build_localizing_words(n_x, n_b, n_trunc)
    photon_lb = 1.0 - omega

    H_total = c_m
    block_values = []
    statuses = []
    num_constraints = []

    # un SDP indipendente per ciascun nodo i=1,...,m-1.
    # range(1,m). Il contributo costante c_m resta fuori.
    for i in range(m_eff):
        constraints = []

        # Classe SDP (di base) comune a tutti i blocchi b
        base = TracialSDP()

        # Un blocco Z_b per ciascun outcome b (nuova classe SDP per le variabili Z)
        blocks = [LiftedTraceBlock(words, base) for _ in range(n_b)]

        G = base.moment_matrix(words)
        constraints.append(G >> 0)

        for r in rhos:
            constraints.append(base.localizing_matrix(loc_words, r) >> 0)
        for r in rhos:
            constraints.append(base.T((r,)) == 1)
        for s in sigmas:
            constraints.append(base.T((s,)) == 1)

        # Completezza POVM: sum_b M_b = I, imposta sui monomi localizing.
        for u in loc_words:
            for v in loc_words:
                lhs = cvx_sum(
                    base.T(tuple(reversed(u)) + (M,) + tuple(v))
                    for M in measurements
                )
                rhs = base.T(tuple(reversed(u)) + tuple(v))
                constraints.append(lhs == rhs)

        # Vincoli fotonici.
        for x, r in enumerate(rhos):
            for n, s in enumerate(sigmas):
                constraints.append(base.T((r, s)) >= photon_lb[x, n])
                constraints.append(base.T((r, s)) <= 1)

        # Dati osservati: distribuzione completa o witness.
        if p_obs is not None:
            for x, r in enumerate(rhos):
                for b, M in enumerate(measurements):
                    constraints.append(base.T((r, M)) == p_obs[x, b])

        if W_obs is not None:
            W_total = cvx_sum(base.T((rhos[x], measurements[x])) for x in range(n_x)) / n_x
            constraints.append(W_total >= W_obs - W_tol)
        else:
            W_total = None

        H_i = 0
        for b, block in enumerate(blocks):
            zG = block.GammaMatrix_from(block.ZT, words)
            hG = block.GammaMatrix_from(block.HT, words)

            constraints.append(zG << psd_tol * np.eye(len(words)))
            constraints.append(cp.bmat([[G, zG], [zG, hG]]) >> -psd_tol * np.eye(2 * len(words)))

            for r in rhos:
                L = base.localizing_matrix(loc_words, r)
                zL = block.localizing_from(block.ZT, loc_words, r)
                hL = block.localizing_from(block.HT, loc_words, r)
                constraints.append(cp.bmat([[L, zL], [zL, hL]]) >> -psd_tol * np.eye(2 * len(loc_words)))

            # Z=zI e Z^2=hI e le normalizzazioni cambiano:
            for r in rhos:
                constraints.append(block.ZT((r,)) == block.z_scalar)
                constraints.append(block.HT((r,)) == block.h_scalar)
            for s in sigmas:
                constraints.append(block.ZT((s,)) == block.z_scalar)
                constraints.append(block.HT((s,)) == block.h_scalar)

            # Completezza delle POVM anche nei layer Z e Z^2.
            for u in loc_words:
                for v in loc_words:
                    lhs_z = cvx_sum(
                        block.ZT(tuple(reversed(u)) + (M,) + tuple(v))
                        for M in measurements
                    )
                    rhs_z = block.ZT(tuple(reversed(u)) + tuple(v))
                    constraints.append(lhs_z == rhs_z)

                    lhs_h = cvx_sum(
                        block.HT(tuple(reversed(u)) + (M,) + tuple(v))
                        for M in measurements
                    )
                    rhs_h = block.HT(tuple(reversed(u)) + tuple(v))
                    constraints.append(lhs_h == rhs_h)

            # Funzione obiettivo
            p_z = block.ZT((rhos[x_star],measurements[b]))
            p_h = block.HT((rhos[x_star],measurements[b]))
            rho_h = block.HT((rhos[x_star],))
            H_i += tau[i] * (2.0 * p_z + (1.0 - t[i]) * p_h + t[i] * rho_h)

        problem = cp.Problem(cp.Minimize(H_i), constraints)

        if solver.upper() == "MOSEK":
            problem.solve(
                solver="MOSEK",
                verbose=verbose,
                mosek_params={
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-7,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-7,
                },
            )
        else:
            problem.solve(solver=solver, verbose=verbose)

        statuses.append(problem.status)
        num_constraints.append(len(constraints))
        if problem.value is None:
            return {
                "H_shannon_bits": None,
                "status": problem.status,
                "failed_node": i,
                "node_values": block_values,
                "statuses": statuses,
            }

        block_values.append(float(problem.value))
        H_total += float(problem.value)

    return {
        "n_x": n_x,
        "n_b": n_b,
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
        "node_values": block_values,
        "statuses": statuses,
        "num_constraints_per_node": num_constraints,
        "num_words": len(words),
        "m_eff": m_eff
    }

# =========================================================
# PREPARAZIONE PROBABILITÀ E BOUND PER PROTOCOLLO BPSK
# =========================================================

def gaussian_bin_prob(alpha, a, b, sign=+1):
    """
    sign=+1 per |+alpha>, sign=-1 per |-alpha>.
    Integra la gaussiana nel bin (a,b).
    """
    mu = sign * sqrt(2.0) * alpha

    erf_b = 1.0 if b == inf else erf(b - mu)
    erf_a = -1.0 if a == -inf else erf(a - mu)

    return 0.5 * (erf_b - erf_a)


def bpsk_bins(n_b, x1=None, x2=None, x3=None):
    if n_b == 2:
        return [(-inf, 0.0), (0.0, inf)]

    if n_b == 4:
        if x1 is None:
            raise ValueError("Per n_b=4 devi fornire x1.")
        return [(-inf, -x1), (-x1, 0.0), (0.0, x1), (x1, inf)]

    if n_b == 8:
        if x1 is None or x2 is None or x3 is None:
            raise ValueError("Per n_b=8 devi fornire x1, x2, x3.")
        return [
            (-inf, -x3),
            (-x3, -x2),
            (-x2, -x1),
            (-x1, 0.0),
            (0.0, x1),
            (x1, x2),
            (x2, x3),
            (x3, inf),
        ]

    raise ValueError("n_b deve essere 2, 4 oppure 8.")


def bpsk_p_obs(alpha, n_b, x1=None, x2=None, x3=None):
    bins = bpsk_bins(n_b, x1=x1, x2=x2, x3=x3)

    p_plus = [gaussian_bin_prob(alpha, a, b, sign=+1) for a, b in bins]
    p_minus = [gaussian_bin_prob(alpha, a, b, sign=-1) for a, b in bins]

    p = np.array([p_plus, p_minus], dtype=float)

    # Correzione numerica piccola
    p[p < 0] = 0.0
    # Normalizzazione
    p = p / p.sum(axis=1, keepdims=True)

    return p

def bpsk_omega(alpha):
    N = alpha**2
    return np.full((2, 1), 1.0 - np.exp(-N))

# =========================
# MAIN
# =========================

N_values = np.linspace(0.005, 0.36, 20)
n_x = 2
n_trunc = 0
#n_b_values = [2]
n_b_values = [2, 4, 8]
mode = "shannon"
#mode = "min"

#crea o entra nel path e salva la configutazione con tutte le info necessarie
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"nx{n_x}_N{N_values[0]:.2f}_{N_values[-1]:.2f}_ntrunc{n_trunc}_{timestamp}"
outdir = Path("results") / "bpsk" / run_name
outdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "bpsk_results.csv"
config_path = outdir / "bpsk_config.txt"

with open(config_path, "w") as f:
    f.write(f"script = shannonGlobal.py\n")
    f.write(f"n_x = {n_x}\n")
    f.write(f"N_values = {N_values}\n")
    f.write(f"n_trunc_values = {n_trunc}\n")
    f.write(f"solver = MOSEK\n")
    f.write(f"mode = {mode}\n")
    f.write(f"include_extra_words = True\n")
    f.write(f"bpsk_level = False\n")


rows = []
#simulazione
results = {}

for n_b in n_b_values:

    H_values = []

    for N in N_values:
        t0 = time.perf_counter()

        alpha = np.sqrt(N)
        omega = bpsk_omega(alpha)

        x1 = (sqrt(2.0) * alpha - erfinv(0.5 * (erf(sqrt(2.0) * alpha) - 1.0)))
        
        if n_b == 2:
            p_obs = bpsk_p_obs(alpha, n_b=2)
        elif n_b == 4:
            p_obs = bpsk_p_obs(alpha, n_b=4, x1=x1)
        elif n_b == 8:
            p_obs = bpsk_p_obs(alpha, n_b=8, x1=x1, x2=2*x1, x3=3*x1)  

        if mode == "shannon":
            res = solve_shannon_entropy_bff_randomness(
                n_x=2,
                n_b=n_b,
                n_trunc=n_trunc,
                omega=omega,
                p_obs=p_obs,
                W_obs=None,
                x_star=0,
                solver="MOSEK",
                include_extra_words=True,
                bpsk_level=False,
                m=4, 
                verbose=False,
            )

            runtime = time.perf_counter() - t0

            rows.append({
                "N": float(N),
                "n_x": n_x,
                "n_b": n_b,
                "n_trunc": n_trunc,

                "p_obs": str(p_obs.tolist()),
                "statuses": str(res["statuses"]),
                "H_shannon_bits": res["H_shannon_bits"],
                
                "c_m": res["c_m"],
                "m_eff": res["m_eff"],

                "runtime_sec": runtime,
                "num_constraints_per_node": str(res["num_constraints_per_node"]),
                "num_words": res["num_words"],

                "node_values": str(res["node_values"]),
            })
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        if mode == "min":
            res = solve_min_entropy_randomness(
                n_x=2,
                n_b=n_b,
                n_trunc=n_trunc,
                omega=omega,
                p_obs=p_obs,
                W_obs=None,
                x_star=0,
                solver="MOSEK",
                include_extra_words=True,
                bpsk_level=False,
                verbose=False,
            )

            runtime = time.perf_counter() - t0

            rows.append({
                "N": float(N),
                "n_x": n_x,
                "n_b": n_b,
                "n_trunc": n_trunc,

                "p_obs": str(p_obs.tolist()),
                "pg": res["guessing_probability_clipped"],
                
                "status": str(res["status"]),
                "H_min_bits": res["H_min_bits"],
                "runtime_sec": runtime,
                "num_constraints": res["num_constraints"],
                "num_blocks": res["num_blocks"],
                "num_words_per_block": res["num_words_per_block"],
            })
            pd.DataFrame(rows).to_csv(csv_path, index=False)


