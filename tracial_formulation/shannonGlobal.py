import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
import chaospy
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
    e la convessità è imposta tramite il vincolo generale
    [[G,zG],[zG,hG]] >= 0.

    Nota: label serve solo per dare nomi diversi alle variabili CVXPY
    quando costruiamo simultaneamente tutti i blocchi (i,b) nello stesso SDP.
    """
    def __init__(self, words, base, label=""):
        self.words = words
        self.base = base
        self.label = label
        self.z_vars = {}
        self.h_vars = {}
        self.z_scalar = cp.Variable(nonpos=True, name=f"z_scalar_{label}")
        self.h_scalar = cp.Variable(nonneg=True, name=f"h_scalar_{label}")

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
            base_name = "ZT_I" if len(key) == 0 else "ZT_" + "_".join(key)
            name = f"{base_name}_{self.label}" if self.label else base_name
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
            base_name = "HT_I" if len(key) == 0 else "HT_" + "_".join(key)
            name = f"{base_name}_{self.label}" if self.label else base_name
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
    W_tol=1e-5,
    psd_tol=1e-8,
):
    """
    SDP per certificare H(B|X=x_star, Lambda) con Shannon entropy
    linearizzata via Brown-Fawzi-Fawzi / Gauss-Radau.

    Convenzioni:
      - scenario n-state discrimination: n_x stati e n_x outcome;
      - una sola measurement y=0, quindi M_b := M_{b|0};
      - omega[x,n] = 1 - P_n(x), quindi Tr(rho_x sigma_n) >= 1 - omega[x,n];
      - se p_obs è dato, viene fissata tutta la distribuzione p(b|x);
        altrimenti si fissa il witness W >= W_obs - W_tol.

    IMPORTANTE:
      Questa versione fa UN SOLO SDP globale:

          H >= c_m + min_Gamma,{Z_{i,b}} sum_i tau_i sum_b [ ... ]

      quindi tutti i nodi di quadratura i condividono la stessa matrice dei
      momenti fisica Gamma, gli stessi stati rho_x, gli stessi operatori M_b e
      gli stessi vincoli osservati.  
    """

    omega = np.asarray(omega, dtype=float)
    if omega.shape != (n_x, n_trunc + 1):
        raise ValueError(f"omega deve avere shape {(n_x, n_trunc + 1)}, ricevuta {omega.shape}")

    if p_obs is not None:
        p_obs = np.asarray(p_obs, dtype=float)
        if p_obs.shape != (n_x, n_x):
            raise ValueError(f"p_obs deve avere shape {(n_x, n_x)}, ricevuta {p_obs.shape}")

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

    # Se non vengono passati né W_obs né p_obs, prima calcolo il massimo witness
    # compatibile coi vincoli fotonici, poi certifico l'entropia a quel valore.
    if W_obs is None and p_obs is None:
        res_W = solve_n_state_discrimination(
            n_x=n_x,
            n_trunc=n_trunc,
            omega=omega,
            solver=solver,
            verbose=verbose,
            include_extra_words=include_extra_words,
        )
        W_obs = res_W["sdp_upper_bound"]

    rhos, measurements, sigmas = build_operators(n_x, n_trunc)
    words = build_words(n_x, n_trunc, include_extra=include_extra_words)
    loc_words = build_localizing_words(n_x, n_trunc)
    photon_lb = 1.0 - omega

    constraints = []

    # Un'unica classe SDP di base: G è comune a tutti i nodi i.
    base = TracialSDP()
    G = base.moment_matrix(words)
    constraints.append(G >> 0)

    # Localizing matrices: rho_x - rho_x^2 >= 0.
    base_localizing = {}
    for r in rhos:
        L = base.localizing_matrix(loc_words, r)
        base_localizing[r] = L
        constraints.append(L >> 0)

    # Normalizzazione degli stati e dei proiettori fotonici.
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

    # Blocchi BFF: uno per ogni coppia (i,b), ma tutti condividono la stessa Gamma G.
    blocks = [[LiftedTraceBlock(words, base, label=f"i{i}_b{b}") for b in range(n_x)] for i in range(m_eff)]

    H_terms_by_node = []
    H_obj = c_m

    for i in range(m_eff):
        H_i = 0
        for b, block in enumerate(blocks[i]):
            zG = block.GammaMatrix_from(block.ZT, words)
            hG = block.GammaMatrix_from(block.HT, words)

            # Z <= 0 e blocco liftato PSD per Z e Z^2.
            constraints.append(zG << psd_tol * np.eye(len(words)))
            constraints.append(cp.bmat([[G, zG], [zG, hG]]) >> -psd_tol * np.eye(2 * len(words)))

            # Versione liftata anche per i localizing constraints rho-rho^2 >= 0.
            for r in rhos:
                L = base_localizing[r]
                zL = block.localizing_from(block.ZT, loc_words, r)
                hL = block.localizing_from(block.HT, loc_words, r)
                constraints.append(cp.bmat([[L, zL], [zL, hL]]) >> -psd_tol * np.eye(2 * len(loc_words)))

            # Z = z I e Z^2 = h I sulle tracce normalizzate rilevanti.
            for r in rhos:
                constraints.append(block.ZT((r,)) == block.z_scalar)
                constraints.append(block.HT((r,)) == block.h_scalar)
            for s in sigmas:
                constraints.append(block.ZT((s,)) == block.z_scalar)
                constraints.append(block.HT((s,)) == block.h_scalar)

            # Completezza POVM anche nei layer Z e Z^2.
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

            # Contributo del noto i e outcome b.
            p_z = block.ZT((rhos[x_star], measurements[b]))
            p_h = block.HT((rhos[x_star], measurements[b]))
            rho_h = block.HT((rhos[x_star],))
            H_i += tau[i] * (2.0 * p_z + (1.0 - t[i]) * p_h + t[i] * rho_h)

        H_terms_by_node.append(H_i)
        H_obj += H_i

    problem = cp.Problem(cp.Minimize(H_obj), constraints)

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

    if problem.value is None:
        return {
            "H_shannon_bits": None,
            "status": problem.status,
            "statuses": [problem.status],
            "node_values": None,
            "num_constraints": len(constraints),
            "num_words": len(words),
        }

    # Se dovesse servire: dopo aver risolto l'SDP globale, salvo i contributi dei singoli nodi.
    node_values = []
    for expr in H_terms_by_node:
        try:
            node_values.append(float(expr.value))
        except Exception:
            node_values.append(None)

    return {
        "n_x": n_x,
        "n_trunc": n_trunc,
        "omega": omega,
        "photon_lb": photon_lb,
        "x_star": x_star,
        "W_obs": W_obs,
        "p_obs": p_obs,
        "H_shannon_bits": max(float(problem.value), 0.0),
        "raw_objective_value": float(problem.value),
        "c_m": c_m,
        "tau": tau,
        "t": t,
        "w": w,
        "node_values": node_values,
        "status": problem.status,
        "statuses": [problem.status],
        "num_constraints": len(constraints),
        "num_words": len(words),
        "num_moment_variables": len(base.vars),
        "num_lifted_blocks": m_eff * n_x,
        "m_eff": m_eff,
    }


# =========================
# MAIN
# =========================

#N_values = [0.005, 0.1, 0.2, 0.5]
N_values = np.linspace(0.01, 1.0, 10)
n_x = 2
n_trunc_values = [0,1,2]
mode = "witness"
#mode = "full_distribution"

#plt.figure(figsize=(7, 5))

#crea o entra nel path e salva la configutazione con tutte le info necessarie
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"nx{n_x}_N{N_values[0]:.2f}_{N_values[-1]:.2f}_ntrunc{'-'.join(map(str, n_trunc_values))}_{timestamp}"
outdir = Path("results") / "shannon2" / run_name
outdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "shannon2_results.csv"
config_path = outdir / "shannon2_config.txt"

with open(config_path, "w") as f:
    f.write(f"script = shannonGlobal.py\n")
    f.write(f"n_x = {n_x}\n")
    f.write(f"N_values = {N_values}\n")
    f.write(f"n_trunc_values = {n_trunc_values}\n")
    f.write(f"solver = MOSEK\n")
    f.write(f"mode = {mode}\n")
    f.write(f"include_extra_words = False\n")


rows = []
#simulazione
for n_trunc in n_trunc_values:
    
    #print(f"\n===== n_trunc = {n_trunc} =====")
    H_values = []
    for N in N_values:
        t0 = time.perf_counter()

        omega = poisson_omega(N, n_x=n_x, n_trunc=n_trunc)

        # Caso 1) calcolo il massimo witness compatibile coi vincoli fotonici
        if mode == "witness":
            res_W = solve_n_state_discrimination(
                n_x=n_x,
                n_trunc=n_trunc,
                omega=omega,
                solver="MOSEK",
                include_extra_words=False,
            )
            W_obs = res_W["sdp_upper_bound"]

            H_result = solve_shannon_entropy_bff_randomness(
                n_x=n_x,
                n_trunc=n_trunc,
                omega=omega,
                W_obs=W_obs,
                p_obs=None,
                x_star=0,
                m=4,
                solver="MOSEK",
                include_extra_words=False,
                W_tol=1e-5,
            )

        
        # Caso 2) distribuzione completa fissata
        elif mode == "full_distribution":

            #(Esempio scemo)
            p_obs = np.eye(n_x)

            print("\nUso distribuzione osservata p_obs =")
            print(p_obs)

            H_result = solve_shannon_entropy_bff_randomness(
                n_x=n_x,
                n_trunc=n_trunc,
                omega=omega,
                W_obs=None,
                p_obs=p_obs,
                x_star=0,
                m=8,
                solver="MOSEK",
                include_extra_words=False,
                W_tol=1e-5,
            )

        else:
            raise ValueError("'mode' deve essere 'witness' oppure 'full_distribution'")

        H_values.append(H_result["H_shannon_bits"])

        runtime = time.perf_counter() - t0
        '''
        print(
            f"N={N:.3f} | "
            #f"W_obs={W_obs:.10f} | "
            #f"p_obs={p_obs} | "
            f"H_total={H_result['H_shannon_bits']:.10f} bits | "
            f"status={H_result['statuses']} | "
        )
        '''
        rows.append({
            "N": float(N),
            "n_x": n_x,
            "n_trunc": n_trunc,
            "mode": mode,

            "W_obs": H_result["W_obs"],
            "status": H_result["status"],
            "H_shannon_bits": H_result["H_shannon_bits"],

            "c_m": H_result["c_m"],
            "m_eff": H_result["m_eff"],
            "num_lifted_blocks": H_result["num_lifted_blocks"],

            "runtime_sec": runtime,
            "num_constraints": H_result["num_constraints"],
            "num_words": H_result["num_words"],

            "node_values": str(H_result["node_values"]),
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    #plt.plot(N_values, H_values, "--", label=fr"$n_{{\mathrm{{trunc}}}}={n_trunc}$")

'''
plt.xlabel(r"$N$")
plt.ylabel(r"$H$ [bits]")
plt.title(fr"Shannon entropy from {n_x}-states discrimination")
plt.grid(True)
plt.legend()
plt.show()
'''
