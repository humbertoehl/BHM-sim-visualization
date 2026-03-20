import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution, minimize
from scipy.special import logsumexp


U_DEFAULT = 1.0
JD_DEFAULT = 1.0
Z_COORDINATION = 4
LOGT_MIN = -2.0
LOGT_MAX = 0.0
PSI_TOL = 1e-3
DRHO_TOL = 1e-3


def local_hilbert_dimension(n_max):
    return n_max + 1


def occupation_arrays(n_max):
    n = np.arange(n_max + 1, dtype=float)
    nn1 = n * (n - 1.0)
    n2 = n * n
    hop = np.sqrt(np.arange(1, n_max + 1, dtype=float))
    return n, nn1, n2, hop


def amplitudes_from_probabilities(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    return np.sqrt(p)


def gutzwiller_moments(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if p.sum() <= 0:
        raise ValueError("Probabilities must have positive norm.")
    p = p / p.sum()

    n_max = len(p) - 1
    n_values, nn1, n2, hop = occupation_arrays(n_max)

    rho = float(np.dot(n_values, p))
    doublon = float(np.dot(nn1, p))
    variance = float(np.dot(n2, p) - rho * rho)

    amps = amplitudes_from_probabilities(p)
    psi = float(np.sum(hop * amps[:-1] * amps[1:]))

    return {
        "probabilities": p,
        "amplitudes": amps,
        "rho": rho,
        "nn1": doublon,
        "variance": variance,
        "psi": psi,
    }


def density_coupling_energy_site(p_odd, p_even, t_over_u, gbar_over_u, num_sites, U=1.0, z=4):
    mO = gutzwiller_moments(p_odd)
    mE = gutzwiller_moments(p_even)

    t = t_over_u * U
    gbar = gbar_over_u * U
    g_local = gbar / max(1, num_sites)

    delta_rho = 0.5 * (mO["rho"] - mE["rho"])

    onsite = 0.25 * U * (mO["nn1"] + mE["nn1"])
    kinetic = -z * t * mO["psi"] * mE["psi"]
    fluctuation = 0.5 * g_local * (mO["variance"] + mE["variance"])
    imbalance = gbar * (delta_rho ** 2)

    energy_site = onsite + kinetic + fluctuation + imbalance

    return {
        "energy_site": float(energy_site),
        "odd": mO,
        "even": mE,
        "delta_rho": float(delta_rho),
        "psi_bar": 0.5 * (abs(mO["psi"]) + abs(mE["psi"])),
        "imbalance_term": float(imbalance),
        "fluctuation_term": float(fluctuation),
        "kinetic_term": float(kinetic),
        "onsite_term": float(onsite),
    }


def initial_density_guess(rho, n_max):
    dim = n_max + 1
    guess = np.zeros(dim, dtype=float)
    low = int(math.floor(rho))
    low = max(0, min(low, n_max))
    high = min(low + 1, n_max)
    frac = rho - low
    if high == low:
        guess[low] = 1.0
    else:
        guess[low] = 1.0 - frac
        guess[high] = frac
    return guess


def refine_probability_candidate(x, density_target, n_values):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    s = x.sum()
    if s <= 0:
        return None
    x = x / s
    if abs(np.dot(n_values, x) - density_target) > 1e-6:
        return None
    return x


def optimize_balanced_branch(rho, t_over_u, gbar_over_u, n_max, num_sites, U=1.0, z=4, seed=1234):
    if rho < 0:
        raise ValueError("The density rho must be nonnegative.")
    if rho > n_max + 1e-12:
        raise ValueError(
            f"Local cutoff n_max={n_max} is too small for rho={rho:.4f}. Increase n_max."
        )

    dim = n_max + 1
    bounds = [(0.0, 1.0)] * dim
    n_values = np.arange(dim, dtype=float)

    def objective(p):
        return density_coupling_energy_site(p, p, t_over_u, gbar_over_u, num_sites, U=U, z=z)["energy_site"]

    linear_constraint = LinearConstraint(
        np.vstack([np.ones(dim), n_values]),
        lb=np.array([1.0, rho]),
        ub=np.array([1.0, rho]),
    )
    constraints = [
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0},
        {"type": "eq", "fun": lambda p: np.dot(n_values, p) - rho},
    ]

    base_guess = initial_density_guess(rho, n_max)
    candidates = [base_guess]
    rng = np.random.default_rng(seed)
    for _ in range(8):
        candidates.append(rng.dirichlet(np.ones(dim)))

    best_x = None
    best_energy = np.inf

    try:
        de = differential_evolution(
            objective,
            bounds=bounds,
            constraints=(linear_constraint,),
            strategy="best1bin",
            maxiter=80,
            popsize=10,
            tol=1e-7,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            seed=seed,
            updating="deferred",
            workers=1,
        )
        if de.success:
            x = refine_probability_candidate(de.x, rho, n_values)
            if x is not None:
                candidates.insert(0, x)
                best_x = x
                best_energy = float(objective(x))
    except Exception:
        pass

    for x0 in candidates:
        try:
            res = minimize(
                objective,
                x0=np.asarray(x0, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1200, "ftol": 1e-12, "disp": False},
            )
        except Exception:
            continue

        if not res.success:
            continue

        x = refine_probability_candidate(res.x, rho, n_values)
        if x is None:
            continue

        energy = float(objective(x))
        if energy < best_energy:
            best_energy = energy
            best_x = x

    if best_x is None:
        raise RuntimeError("Balanced-branch optimization failed.")

    summary = density_coupling_energy_site(best_x, best_x, t_over_u, gbar_over_u, num_sites, U=U, z=z)
    return {
        "branch": "balanced",
        "p_odd": best_x.copy(),
        "p_even": best_x.copy(),
        **summary,
        "success": True,
    }


def build_unbalanced_initial_guess(rho, n_max, delta):
    rho_odd = np.clip(rho + delta, 0.0, float(n_max))
    rho_even = np.clip(2.0 * rho - rho_odd, 0.0, float(n_max))
    if abs(0.5 * (rho_odd + rho_even) - rho) > 1e-10:
        rho_odd = rho
        rho_even = rho
    return initial_density_guess(rho_odd, n_max), initial_density_guess(rho_even, n_max)


def optimize_unbalanced_branch(rho, t_over_u, gbar_over_u, n_max, num_sites, U=1.0, z=4, seed=1234):
    dim = n_max + 1
    total_dim = 2 * dim
    bounds = [(0.0, 1.0)] * total_dim
    n_values = np.arange(dim, dtype=float)

    def split(x):
        x = np.asarray(x, dtype=float)
        return x[:dim], x[dim:]

    def objective(x):
        p_odd, p_even = split(x)
        return density_coupling_energy_site(p_odd, p_even, t_over_u, gbar_over_u, num_sites, U=U, z=z)["energy_site"]

    A = np.zeros((3, total_dim), dtype=float)
    A[0, :dim] = 1.0
    A[1, dim:] = 1.0
    A[2, :dim] = n_values
    A[2, dim:] = n_values
    linear_constraint = LinearConstraint(A, lb=np.array([1.0, 1.0, 2.0 * rho]), ub=np.array([1.0, 1.0, 2.0 * rho]))

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:dim]) - 1.0},
        {"type": "eq", "fun": lambda x: np.sum(x[dim:]) - 1.0},
        {"type": "eq", "fun": lambda x: np.dot(n_values, x[:dim]) + np.dot(n_values, x[dim:]) - 2.0 * rho},
    ]

    candidates = []
    base = initial_density_guess(rho, n_max)
    candidates.append(np.concatenate([base, base]))

    if n_max > 0:
        max_delta = min(rho, n_max - rho)
        for frac in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            delta = frac * max_delta
            p_odd, p_even = build_unbalanced_initial_guess(rho, n_max, delta)
            candidates.append(np.concatenate([p_odd, p_even]))
            candidates.append(np.concatenate([p_even, p_odd]))

    rng = np.random.default_rng(seed)
    for _ in range(10):
        p_odd = rng.dirichlet(np.ones(dim))
        p_even = rng.dirichlet(np.ones(dim))
        rho_sum = np.dot(n_values, p_odd) + np.dot(n_values, p_even)
        target = 2.0 * rho
        shift = target - rho_sum
        if abs(shift) > 1e-8:
            # Try to repair the density by mixing with the nearest integer-density seeds.
            q_odd, q_even = build_unbalanced_initial_guess(rho, n_max, 0.0)
            lam = 0.35
            p_odd = (1.0 - lam) * p_odd + lam * q_odd
            p_even = (1.0 - lam) * p_even + lam * q_even
        candidates.append(np.concatenate([p_odd, p_even]))

    best_x = None
    best_energy = np.inf

    def refine_full_candidate(x):
        p_odd, p_even = split(x)
        p_odd = np.clip(p_odd, 0.0, 1.0)
        p_even = np.clip(p_even, 0.0, 1.0)
        s1 = p_odd.sum()
        s2 = p_even.sum()
        if s1 <= 0 or s2 <= 0:
            return None
        p_odd /= s1
        p_even /= s2
        density_err = abs(np.dot(n_values, p_odd) + np.dot(n_values, p_even) - 2.0 * rho)
        if density_err > 1e-6:
            return None
        return np.concatenate([p_odd, p_even])

    try:
        de = differential_evolution(
            objective,
            bounds=bounds,
            constraints=(linear_constraint,),
            strategy="best1bin",
            maxiter=100,
            popsize=10,
            tol=1e-7,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            seed=seed,
            updating="deferred",
            workers=1,
        )
        if de.success:
            x = refine_full_candidate(de.x)
            if x is not None:
                candidates.insert(0, x)
                best_x = x
                best_energy = float(objective(x))
    except Exception:
        pass

    for x0 in candidates:
        try:
            res = minimize(
                objective,
                x0=np.asarray(x0, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1500, "ftol": 1e-12, "disp": False},
            )
        except Exception:
            continue

        if not res.success:
            continue

        x = refine_full_candidate(res.x)
        if x is None:
            continue

        energy = float(objective(x))
        if energy < best_energy:
            best_energy = energy
            best_x = x

    if best_x is None:
        raise RuntimeError("Unbalanced-branch optimization failed.")

    p_odd, p_even = split(best_x)
    summary = density_coupling_energy_site(p_odd, p_even, t_over_u, gbar_over_u, num_sites, U=U, z=z)
    return {
        "branch": "unbalanced",
        "p_odd": p_odd.copy(),
        "p_even": p_even.copy(),
        **summary,
        "success": True,
    }


def solve_density_coupled_model(lx, ly, n_particles, n_max, t_over_u, gbar_over_u, U=1.0, z=4, seed=1234):
    num_sites = lx * ly
    rho = n_particles / num_sites

    balanced = optimize_balanced_branch(rho, t_over_u, gbar_over_u, n_max, num_sites, U=U, z=z, seed=seed)
    unbalanced = optimize_unbalanced_branch(rho, t_over_u, gbar_over_u, n_max, num_sites, U=U, z=z, seed=seed + 17)

    if unbalanced["energy_site"] < balanced["energy_site"] - 1e-10:
        selected = unbalanced
    else:
        selected = balanced

    selected = dict(selected)
    selected["branch_balanced"] = balanced
    selected["branch_unbalanced"] = unbalanced
    selected["rho_target"] = rho
    selected["num_sites"] = num_sites
    selected["gbar_over_u"] = gbar_over_u
    selected["t_over_u"] = t_over_u
    selected["phase"] = classify_phase(selected)
    return selected


def classify_phase(result):
    psi_bar = float(result["psi_bar"])
    delta_rho = abs(float(result["delta_rho"]))

    if psi_bar < PSI_TOL and delta_rho < DRHO_TOL:
        return "MI"
    if psi_bar < PSI_TOL and delta_rho >= DRHO_TOL:
        return "DW"
    if psi_bar >= PSI_TOL and delta_rho < DRHO_TOL:
        return "SF"
    return "SS"


def build_site_probability_list(lx, ly, p_odd, p_even):
    site_probabilities = []
    for y in range(ly):
        for x in range(lx):
            is_odd = ((x + y) % 2 == 0)
            site_probabilities.append(np.asarray(p_odd if is_odd else p_even, dtype=float))
    return site_probabilities


def build_conditioned_log_partition_sitewise(site_probabilities, total_particles):
    num_sites = len(site_probabilities)
    logZ = np.full((num_sites + 1, total_particles + 1), -np.inf, dtype=float)
    logZ[0, 0] = 0.0

    precomputed_logp = []
    for p in site_probabilities:
        p = np.asarray(p, dtype=float)
        logp = np.full(len(p), -np.inf, dtype=float)
        positive = p > 0.0
        logp[positive] = np.log(p[positive])
        precomputed_logp.append(logp)

    for m in range(1, num_sites + 1):
        logp = precomputed_logp[m - 1]
        n_max = len(logp) - 1
        for s in range(total_particles + 1):
            terms = []
            n_hi = min(n_max, s)
            for n in range(n_hi + 1):
                prev = logZ[m - 1, s - n]
                if np.isfinite(prev) and np.isfinite(logp[n]):
                    terms.append(prev + logp[n])
            if terms:
                logZ[m, s] = logsumexp(terms)

    if not np.isfinite(logZ[num_sites, total_particles]):
        raise RuntimeError(
            "No valid exact-N sample exists for these parameters. Increase n_max or reduce N."
        )

    return logZ


def sample_conditioned_configuration_sitewise(site_probabilities, total_particles, rng, logZ=None):
    if logZ is None:
        logZ = build_conditioned_log_partition_sitewise(site_probabilities, total_particles)

    num_sites = len(site_probabilities)
    state = []
    remaining = total_particles

    precomputed_logp = []
    for p in site_probabilities:
        p = np.asarray(p, dtype=float)
        logp = np.full(len(p), -np.inf, dtype=float)
        positive = p > 0.0
        logp[positive] = np.log(p[positive])
        precomputed_logp.append(logp)

    for m in range(num_sites, 0, -1):
        logp = precomputed_logp[m - 1]
        choices = []
        weights = []
        for n in range(min(len(logp) - 1, remaining) + 1):
            lw = logp[n] + logZ[m - 1, remaining - n]
            if np.isfinite(lw):
                choices.append(n)
                weights.append(lw)
        weights = np.asarray(weights, dtype=float)
        probs = np.exp(weights - logsumexp(weights))
        picked = rng.choice(len(choices), p=probs)
        n_here = choices[picked]
        state.append(n_here)
        remaining -= n_here

    if remaining != 0:
        raise RuntimeError("Internal sampling error in exact-N conditioned draw.")

    state.reverse()
    return tuple(state)


class ExtendedBoseHubbardDensityCouplingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Extended Bose-Hubbard 2D")
        self.root.geometry("1100x750")
        self.root.configure(bg="#ececec")

        self.rng = np.random.default_rng()

        self.cached_key = None
        self.cached_result = None
        self.cached_logZ = None
        self.cached_site_probabilities = None

        self.lx_var = tk.IntVar(value=4)
        self.ly_var = tk.IntVar(value=4)
        self.n_var = tk.IntVar(value=8)
        self.nmax_var = tk.IntVar(value=4)
        self.logt_var = tk.DoubleVar(value=-0.3)
        self.gbar_var = tk.DoubleVar(value=-0.5)

        self.t_string = tk.StringVar()
        self.state_string = tk.StringVar()
        self.info_string = tk.StringVar()

        self._build_ui()
        self._update_t_label()
        self._update_state_label()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Extended Bose-Hubbard 2D",
            font=("Helvetica", 25, "bold"),
            bg="#ececec",
            fg="#202020",
        )
        title.pack(pady=(14, 4))

        subtitle = tk.Label(
            self.root,
            text="2D square lattice with periodic boundary conditions and odd/even sublattice structure",
            font=("Helvetica", 15),
            bg="#ececec",
            fg="#202020",
        )
        subtitle.pack(pady=(0, 10))

        controls = tk.Frame(self.root, bg="#ececec")
        controls.pack(anchor="center", padx=20, pady=6)

        tk.Label(controls, text="Lx =", font=("Helvetica", 15, "bold"), bg="#ececec").grid(row=0, column=0, padx=(0, 6))
        tk.Spinbox(
            controls,
            from_=2,
            to=20,
            textvariable=self.lx_var,
            width=3,
            command=self._update_state_label,
            font=("Helvetica", 15, "bold"),
        ).grid(row=0, column=1, padx=(0, 14))

        tk.Label(controls, text="Ly =", font=("Helvetica", 15, "bold"), bg="#ececec").grid(row=0, column=2, padx=(0, 6))
        tk.Spinbox(
            controls,
            from_=2,
            to=20,
            textvariable=self.ly_var,
            width=3,
            command=self._update_state_label,
            font=("Helvetica", 15, "bold"),
        ).grid(row=0, column=3, padx=(0, 14))

        tk.Label(controls, text="N =", font=("Helvetica", 15, "bold"), bg="#ececec").grid(row=0, column=4, padx=(0, 6))
        tk.Spinbox(
            controls,
            from_=1,
            to=400,
            textvariable=self.n_var,
            width=4,
            command=self._update_state_label,
            font=("Helvetica", 15, "bold"),
        ).grid(row=0, column=5, padx=(0, 14))

        tk.Label(controls, text="n_max =", font=("Helvetica", 15, "bold"), bg="#ececec").grid(row=0, column=6, padx=(0, 6))
        tk.Spinbox(
            controls,
            from_=1,
            to=20,
            textvariable=self.nmax_var,
            width=3,
            command=self._update_state_label,
            font=("Helvetica", 15, "bold"),
        ).grid(row=0, column=7, padx=(0, 18))

        tk.Label(controls, text="log t =", font=("Helvetica", 15, "bold"), bg="#ececec").grid(row=0, column=8, padx=(15, 8))

        slider = tk.Scale(
            controls,
            from_=LOGT_MIN,
            to=LOGT_MAX,
            resolution=0.01,
            orient="horizontal",
            variable=self.logt_var,
            command=lambda _e=None: self._update_t_label(),
            length=150,
            width=20,
            bg="#ececec",
            highlightthickness=0,
        )
        slider.grid(row=0, column=9, padx=(0, 10), pady=(0, 15))

        tk.Label(
            controls,
            textvariable=self.t_string,
            font=("Consolas", 15, "bold"),
            bg="#ececec",
            fg="#1f3b5c",
            width=14,
        ).grid(row=0, column=10, padx=(0, 18))

        tk.Label(controls, text="gN/U =", font=("Helvetica", 15, "bold"), bg="#ececec").grid(row=1, column=8, padx=(15, 8))
        tk.Entry(
            controls,
            textvariable=self.gbar_var,
            width=8,
            font=("Consolas", 15, "bold"),
            justify="center",
        ).grid(row=1, column=9, padx=(0, 10), sticky="w")

        ttk.Button(controls, text="Simulate", command=self.simulate).grid(row=2, column=8, padx=(0, 8))
        ttk.Button(controls, text="New Sample", command=self.new_sample).grid(row=2, column=9, sticky="w")

        state_label = tk.Label(
            self.root,
            textvariable=self.state_string,
            font=("Consolas", 12),
            bg="#ececec",
            fg="#333333",
        )
        state_label.pack(pady=(8, 8))

        self.canvas = tk.Canvas(
            self.root,
            width=800,
            height=320,
            bg="#f6f6f6",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.canvas.pack(padx=20, pady=8)

        info_frame = tk.Frame(self.root, bg="#ececec")
        info_frame.pack(fill="both", expand=False, padx=20, pady=(4, 16))

        scrollbar = tk.Scrollbar(info_frame)
        scrollbar.pack(side="right", fill="y")

        self.info_box = tk.Text(
            info_frame,
            height=5,
            wrap="word",
            font=("Consolas", 14),
            bg="#f8f8f8",
            fg="#202020",
            bd=1,
            relief="solid",
            padx=12,
            pady=10,
            yscrollcommand=scrollbar.set,
        )

        self.info_box.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.info_box.yview)

    def _update_t_label(self):
        t = 10 ** self.logt_var.get()*4
        self.t_string.set(f"(zt/U) = {t:.4}")

    def _update_state_label(self):
        try:
            lx, ly, n_particles, n_max, gbar = self._get_parameters(validate=False)
            num_sites = lx * ly
            rho = n_particles / num_sites
            self.state_string.set(
                rf"Sites = {num_sites:>3}   rho = N/(Lx·Ly) = {rho:.4f}   local dim = n_max+1 = {local_hilbert_dimension(n_max)}   gN/U = {gbar:.4f}"
            )
        except Exception:
            self.state_string.set("Invalid parameters")

    def _get_parameters(self, validate=True):
        lx = int(self.lx_var.get())
        ly = int(self.ly_var.get())
        n_particles = int(self.n_var.get())
        n_max = int(self.nmax_var.get())
        gbar = float(self.gbar_var.get())

        if validate:
            if lx < 2 or ly < 2:
                raise ValueError("Lx and Ly must both be at least 2.")
            if (lx % 2) != 0 or (ly % 2) != 0:
                raise ValueError("For a clean O/E checkerboard with PBC, both Lx and Ly must be even.")
            if n_particles < 1:
                raise ValueError("N must be at least 1.")
            if n_max < 1:
                raise ValueError("n_max must be at least 1.")
            if n_particles > lx * ly * n_max:
                raise ValueError("Need N <= Lx·Ly·n_max so an exact-N sample can exist.")
            rho = n_particles / (lx * ly)
            if rho > n_max + 1e-12:
                raise ValueError(f"Need n_max >= rho. Current rho = {rho:.4f}, n_max = {n_max}.")

        return lx, ly, n_particles, n_max, gbar

    def _ensure_solver_state(self, lx, ly, n_particles, n_max, t_over_u, gbar_over_u):
        key = (lx, ly, n_particles, n_max, round(float(t_over_u), 12), round(float(gbar_over_u), 12))
        if self.cached_key == key:
            return

        result = solve_density_coupled_model(
            lx=lx,
            ly=ly,
            n_particles=n_particles,
            n_max=n_max,
            t_over_u=t_over_u,
            gbar_over_u=gbar_over_u,
            U=U_DEFAULT,
            z=Z_COORDINATION,
            seed=1234,
        )

        site_probabilities = build_site_probability_list(lx, ly, result["p_odd"], result["p_even"])
        logZ = build_conditioned_log_partition_sitewise(site_probabilities, n_particles)

        self.cached_key = key
        self.cached_result = result
        self.cached_site_probabilities = site_probabilities
        self.cached_logZ = logZ

    def simulate(self):
        try:
            lx, ly, n_particles, n_max, gbar = self._get_parameters(validate=True)
            self._update_state_label()
            t_over_u = 10 ** self.logt_var.get()
            self._ensure_solver_state(lx, ly, n_particles, n_max, t_over_u, gbar)
            self._draw_sample()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def new_sample(self):
        if self.cached_result is None:
            self.simulate()
            return
        self._draw_sample()

    def _draw_sample(self):
        lx, ly, n_particles, n_max, gbar = self._get_parameters(validate=True)
        sampled_state = sample_conditioned_configuration_sitewise(
            site_probabilities=self.cached_site_probabilities,
            total_particles=n_particles,
            rng=self.rng,
            logZ=self.cached_logZ,
        )

        self.draw_grid(sampled_state, lx, ly)

        result = self.cached_result
        odd = result["odd"]
        even = result["even"]
        energy_site = result["energy_site"]
        energy_total = energy_site * result["num_sites"]

        top_odd = np.argsort(result["p_odd"])[::-1][: min(4, len(result["p_odd"]))]
        top_even = np.argsort(result["p_even"])[::-1][: min(4, len(result["p_even"]))]
        odd_string = ", ".join([f"n={n}: {result['p_odd'][n]*100:.2f}%" for n in top_odd])
        even_string = ", ".join([f"n={n}: {result['p_even'][n]*100:.2f}%" for n in top_even])

        bal_e = result["branch_balanced"]["energy_site"]
        unbal_e = result["branch_unbalanced"]["energy_site"]

        info = (
            f"zt/U = {result['t_over_u']*4:.4f}\n"
            f"gN/U = {result['gbar_over_u']:.4f}\n"
            f"Selected branch = {result['branch']}   phase = {result['phase']}\n"
            f"E_site = {energy_site:.6f}\n"
            f"E_total = {energy_total:.6f}\n\n"
            f"Balanced branch E_site = {bal_e:.6f}\n"
            f"Unbalanced branch E_site = {unbal_e:.6f}\n"
            f"rho(target) = {result['rho_target']:.6f}\n"
            f"rho_O = {odd['rho']:.6f}   rho_E = {even['rho']:.6f}\n"
            f"Delta rho = (rho_O-rho_E)/2 = {result['delta_rho']:.6f}\n"
            f"psi_O = {odd['psi']:.6f}   psi_E = {even['psi']:.6f}\n"
            f"psi_total = {result['psi_bar']:.6f}\n"
            f"Exact sampled total particles = {sum(sampled_state)}\n"
            f"Measured configuration = {sampled_state}\n"
            f"Odd local probabilities:  {odd_string}\n"
            f"Even local probabilities: {even_string}\n\n"
            f"'Simulate' optimizes Gutzwiller coefficients at fixed density for balanced and unbalanced branches, selecting the lower-energy one.\n"
            f"'New Sample' draws another exact-N configuration from the same product state."
        )
        self.info_box.config(state="normal")
        self.info_box.delete("1.0", tk.END)
        self.info_box.insert(tk.END, info)
        self.info_box.config(state="disabled")

    def draw_grid(self, state, lx, ly):
        self.canvas.delete("all")

        width = int(self.canvas["width"])
        height = int(self.canvas["height"])

        self.canvas.create_text(
            width / 2,
            20,
            text="Sample of an occupation measurement",
            font=("Helvetica", 13, "bold"),
            fill="#202020",
        )

        grid_w = 250
        grid_h = 250
        cell_w = grid_w / lx
        cell_h = grid_h / ly
        cell = min(cell_w, cell_h)

        grid_w_used = cell * lx
        grid_h_used = cell * ly
        x0 = (width - grid_w_used) / 2
        y0 = (height - grid_h_used) / 2 + 18

        max_occ = max(state) if state else 1
        particle_r = int(max(4, min(13, 0.18 * cell, 0.36 * cell / max(1, max_occ))))
        vstep = max(2 * particle_r, int(0.15 * cell))

        odd_bg = "#DE8C3F"
        even_bg = "#3F91DE"

        for y in range(ly):
            for x in range(lx):
                idx = y * lx + x
                n_i = state[idx]
                is_odd = ((x + y) % 2 == 0)

                x1 = x0 + x * cell
                y1 = y0 + y * cell
                x2 = x1 + cell
                y2 = y1 + cell

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=odd_bg if is_odd else even_bg,
                    outline="#d6d6d6",
                    width=1,
                )

                cx = x0 + (x + 0.5) * cell
                cy = y0 + (y + 0.80) * cell

                self.canvas.create_text(
                    x1 + 6,
                    y1 + 6,
                    text="O" if is_odd else "E",
                    anchor="nw",
                    font=("Helvetica", 8, "bold"),
                    fill="#606060",
                )

                self.canvas.create_text(
                    cx,
                    cy + 8,
                    text=f"{idx}",
                    font=("Helvetica", 8, "bold"),
                    fill="#202020",
                )

                if n_i == 0:
                    continue

                shade = int(215 - 195 * (n_i / max_occ)) if max_occ > 0 else 215
                shade = max(15, min(230, shade))
                color = f"#{shade:02x}{shade:02x}{shade:02x}"

                start_y = cy - 15
                for k in range(n_i):
                    py = start_y - k * vstep
                    self.canvas.create_oval(
                        cx - particle_r,
                        py - particle_r,
                        cx + particle_r,
                        py + particle_r,
                        fill=color,
                        outline="#101010",
                        width=1,
                    )


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    app = ExtendedBoseHubbardDensityCouplingApp(root)
    root.mainloop()
