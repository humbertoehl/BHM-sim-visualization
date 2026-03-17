import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from scipy.optimize import differential_evolution, minimize, LinearConstraint
from scipy.special import logsumexp


U_DEFAULT = 1.0
Z_COORDINATION = 4
LOGT_MIN = -2.0
LOGT_MAX = 2.0


def local_hilbert_dimension(n_max):
    return n_max + 1


def gutzwiller_energy_from_probabilities(probabilities, t_over_u, n_max, U=1.0, z=4):
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    n_values = np.arange(n_max + 1, dtype=float)

    onsite = 0.5 * U * np.sum(n_values * (n_values - 1.0) * p)

    amplitudes = np.sqrt(np.clip(p, 0.0, None))
    psi = np.sum(np.sqrt(np.arange(1, n_max + 1, dtype=float)) * amplitudes[:-1] * amplitudes[1:])

    return onsite - z * (t_over_u * U) * (psi ** 2)


def optimize_gutzwiller_probabilities(rho, t_over_u, n_max, U=1.0, z=4, seed=1234):
    if rho < 0:
        raise ValueError("The density rho must be nonnegative.")
    if rho > n_max + 1e-12:
        raise ValueError(
            f"Local cutoff n_max={n_max} is too small for rho={rho:.4f}. "
            f"Increase n_max to >= rho."
        )

    dim = n_max + 1
    bounds = [(0.0, 1.0)] * dim
    n_values = np.arange(dim, dtype=float)

    def objective(p):
        return gutzwiller_energy_from_probabilities(p, t_over_u, n_max, U=U, z=z)

    linear_constraint = LinearConstraint(
        np.vstack([np.ones(dim), n_values]),
        lb=np.array([1.0, rho]),
        ub=np.array([1.0, rho]),
    )

    constraints_slsqp = [
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0},
        {"type": "eq", "fun": lambda p: np.dot(n_values, p) - rho},
    ]

    # Initial guess
    base_guess = np.zeros(dim, dtype=float)
    low = int(math.floor(rho))
    high = min(low + 1, n_max)
    frac = rho - low
    if high == low:
        base_guess[low] = 1.0
    else:
        base_guess[low] = 1.0 - frac
        base_guess[high] = frac

    candidates = [base_guess]
    rng = np.random.default_rng(seed)
    for _ in range(12):
        g = rng.dirichlet(np.ones(dim))
        candidates.append(g)

    best_result = None
    best_energy = np.inf

    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            constraints=(linear_constraint,),
            strategy="best1bin",
            maxiter=220,
            popsize=16,
            tol=1e-7,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            seed=seed,
            updating="deferred",
            workers=1,
        )

        if de_result.success:
            de_x = np.clip(de_result.x, 0.0, 1.0)
            de_x = de_x / de_x.sum()
            candidates.insert(0, de_x)
            best_result = de_result
            best_energy = float(de_result.fun)
    except Exception:
        pass

    # Refine using SLSQP from several starts.
    for x0 in candidates:
        try:
            res = minimize(
                objective,
                x0=np.asarray(x0, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_slsqp,
                options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
            )
        except Exception:
            continue

        if not res.success:
            continue

        x = np.clip(res.x, 0.0, 1.0)
        norm = x.sum()
        if norm <= 0:
            continue
        x = x / norm
        density_error = abs(np.dot(n_values, x) - rho)
        if density_error > 1e-7:
            continue

        energy = objective(x)
        if energy < best_energy:
            best_energy = float(energy)
            best_result = res

    if best_result is None:
        raise RuntimeError("Optimization failed. Try increasing n_max or changing parameters.")

    p = np.clip(np.asarray(best_result.x, dtype=float), 0.0, 1.0)
    p = p / p.sum()

    density = float(np.dot(n_values, p))
    amplitudes = np.sqrt(np.clip(p, 0.0, None))
    psi = float(np.sum(np.sqrt(np.arange(1, dim, dtype=float)) * amplitudes[:-1] * amplitudes[1:]))
    energy_site = float(gutzwiller_energy_from_probabilities(p, t_over_u, n_max, U=U, z=z))

    return {
        "probabilities": p,
        "amplitudes": amplitudes,
        "rho": density,
        "psi": psi,
        "energy_site": energy_site,
        "success": True,
    }


def build_conditioned_log_partition(num_sites, total_particles, probabilities):
    p = np.asarray(probabilities, dtype=float)
    n_max = len(p) - 1

    logp = np.full(n_max + 1, -np.inf, dtype=float)
    positive = p > 0.0
    logp[positive] = np.log(p[positive])

    logZ = np.full((num_sites + 1, total_particles + 1), -np.inf, dtype=float)
    logZ[0, 0] = 0.0

    for m in range(1, num_sites + 1):
        for s in range(total_particles + 1):
            terms = []
            n_hi = min(n_max, s)
            for n in range(n_hi + 1):
                prev = logZ[m - 1, s - n]
                if np.isfinite(logp[n]) and np.isfinite(prev):
                    terms.append(logp[n] + prev)
            if terms:
                logZ[m, s] = logsumexp(terms)

    if not np.isfinite(logZ[num_sites, total_particles]):
        raise RuntimeError(
            "No valid exact-N sample exists for these parameters. "
            "Increase n_max or change N, Lx, Ly."
        )

    return logZ


def sample_conditioned_configuration(num_sites, total_particles, probabilities, rng, logZ=None):
    if logZ is None:
        logZ = build_conditioned_log_partition(num_sites, total_particles, probabilities)

    p = np.asarray(probabilities, dtype=float)
    n_max = len(p) - 1
    logp = np.full(n_max + 1, -np.inf, dtype=float)
    positive = p > 0.0
    logp[positive] = np.log(p[positive])

    state = []
    remaining_particles = total_particles

    for m in range(num_sites, 0, -1):
        choices = []
        weights = []
        for n in range(min(n_max, remaining_particles) + 1):
            lw = logp[n] + logZ[m - 1, remaining_particles - n]
            if np.isfinite(lw):
                choices.append(n)
                weights.append(lw)

        weights = np.asarray(weights, dtype=float)
        probs = np.exp(weights - logsumexp(weights))
        picked = rng.choice(len(choices), p=probs)
        n_here = choices[picked]
        state.append(n_here)
        remaining_particles -= n_here

    if remaining_particles != 0:
        raise RuntimeError("Internal sampling error: leftover particles after conditioned sampling.")

    return tuple(state)


class BoseHubbard2DGutzwillerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bose-Hubbard 2D")
        self.root.geometry("1180x920")
        self.root.configure(bg="#ececec")

        self.rng = np.random.default_rng()

        # Cache
        self.cached_key = None
        self.cached_result = None
        self.cached_logZ = None
        self.cached_t = None

        # GUI
        self.lx_var = tk.IntVar(value=4)
        self.ly_var = tk.IntVar(value=4)
        self.n_var = tk.IntVar(value=4)
        self.nmax_var = tk.IntVar(value=4)
        self.logt_var = tk.DoubleVar(value=0.0)

        self.t_string = tk.StringVar()
        self.state_string = tk.StringVar()
        self.info_string = tk.StringVar()

        self._build_ui()
        self._update_t_label()
        self._update_state_label()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Bose-Hubbard 2D",
            font=("Helvetica", 25, "bold"),
            bg="#ececec",
            fg="#202020",
        )
        title.pack(pady=(14, 4))

        subtitle = tk.Label(
            self.root,
            text=(
                "2D square lattice with periodic boundary conditions"
            ),
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
            to=200,
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

        tk.Label(controls, text="log t =", font=("Helvetica", 15, "bold"), bg="#ececec").grid(row=0, column=8, padx=(35, 8))

        slider = tk.Scale(
            controls,
            from_=LOGT_MIN,
            to=LOGT_MAX,
            resolution=0.01,
            orient="horizontal",
            variable=self.logt_var,
            command=lambda _e=None: self._update_t_label(),
            length=220,
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
        ).grid(row=0, column=10, padx=(0, 10))

        ttk.Button(controls, text="Simulate", command=self.simulate).grid(row=2, column=8)
        ttk.Button(controls, text="New Sample", command=self.new_sample).grid(row=2, column=9)

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

        info_box = tk.Label(
            self.root,
            textvariable=self.info_string,
            justify="left",
            anchor="w",
            font=("Consolas", 10),
            bg="#f8f8f8",
            fg="#202020",
            bd=1,
            relief="solid",
            padx=12,
            pady=10,
        )
        info_box.pack(fill="x", padx=20, pady=(4, 16))

    def _update_t_label(self):
        t = 10 ** self.logt_var.get()
        self.t_string.set(f"(t/U) = {t:.4}")

    def _update_state_label(self):
        try:
            lx, ly, n_particles, n_max = self._get_parameters(validate=False)
            num_sites = lx * ly
            rho = n_particles / num_sites
            self.state_string.set(
                rf"Sites = {num_sites:>3}   rho = N/(Lx·Ly) = {rho:.4f}   local dim = n_max+1 = {local_hilbert_dimension(n_max)}"
            )
        except Exception:
            self.state_string.set("Invalid parameters")

    def _get_parameters(self, validate=True):
        lx = int(self.lx_var.get())
        ly = int(self.ly_var.get())
        n_particles = int(self.n_var.get())
        n_max = int(self.nmax_var.get())

        if validate:
            if lx < 2 or ly < 2:
                raise ValueError("Lx and Ly must both be at least 2.")
            if n_particles < 1:
                raise ValueError("N must be at least 1.")
            if n_max < 1:
                raise ValueError("n_max must be at least 1.")
            rho = n_particles / (lx * ly)
            if rho > n_max + 1e-12:
                raise ValueError(
                    f"Need n_max >= rho. Current rho = {rho:.4f}, n_max = {n_max}."
                )

        return lx, ly, n_particles, n_max

    def _ensure_solver_state(self, lx, ly, n_particles, n_max, t_over_u):
        key = (lx, ly, n_particles, n_max, round(float(t_over_u), 12))
        if self.cached_key == key:
            return

        num_sites = lx * ly
        rho = n_particles / num_sites

        result = optimize_gutzwiller_probabilities(
            rho=rho,
            t_over_u=t_over_u,
            n_max=n_max,
            U=U_DEFAULT,
            z=Z_COORDINATION,
            seed=1234,
        )

        logZ = build_conditioned_log_partition(
            num_sites=num_sites,
            total_particles=n_particles,
            probabilities=result["probabilities"],
        )

        self.cached_key = key
        self.cached_result = result
        self.cached_logZ = logZ
        self.cached_t = t_over_u

    def simulate(self):
        try:
            lx, ly, n_particles, n_max = self._get_parameters(validate=True)
            self._update_state_label()
            t_over_u = 10 ** self.logt_var.get()

            self._ensure_solver_state(lx, ly, n_particles, n_max, t_over_u)
            self._draw_sample()

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def new_sample(self):
        if self.cached_result is None:
            self.simulate()
            return
        self._draw_sample()

    def _draw_sample(self):
        lx, ly, n_particles, n_max = self._get_parameters(validate=True)
        num_sites = lx * ly

        probabilities = self.cached_result["probabilities"]
        sampled_state = sample_conditioned_configuration(
            num_sites=num_sites,
            total_particles=n_particles,
            probabilities=probabilities,
            rng=self.rng,
            logZ=self.cached_logZ,
        )

        self.draw_grid(sampled_state, lx, ly)

        energy_site = self.cached_result["energy_site"]
        energy_total = energy_site * num_sites
        rho = self.cached_result["rho"]
        psi = self.cached_result["psi"]

        top_components = np.argsort(probabilities)[::-1][:min(5, len(probabilities))]
        top_string = ", ".join([f"n={n}: {probabilities[n]*100:.2f}%" for n in top_components])

        info = (
            f"t/U = {self.cached_t:.3f}\n"
            f"E_site = {energy_site:.3f}\n"
            f"E_total = {energy_total:.3f}\n"
            f"rho(target) = N/(Lx·Ly) = {n_particles/num_sites:.3f}\n"
            f"rho(from optimizer) = {rho:.3f}\n"
            f"psi = <b> = {psi:.3f}\n"
            f"Exact sampled total particles = {sum(sampled_state)}\n"
            f"Measured configuration = {sampled_state}\n"
            f"Local probabilities: {top_string}\n\n"
            f"'Simulate' optimizes Gutzwiller coefficients at fixed density.\n"
            f"'New Sample' draws another exact-N configuration from the same product state."
        )
        self.info_string.set(info)

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
        y0 = (height - grid_h_used) / 2 + 15

        max_occ = max(state) if state else 1
        particle_r = int(max(4, min(14, 0.18 * cell, 0.36 * cell / max(1, max_occ))))
        vstep = max(2 * particle_r, int(0.15 * cell))

        # Draw grid
        for ix in range(lx + 1):
            x = x0 + ix * cell
            self.canvas.create_line(x, y0, x, y0 + grid_h_used, fill="#d6d6d6", width=1)
        for iy in range(ly + 1):
            y = y0 + iy * cell
            self.canvas.create_line(x0, y, x0 + grid_w_used, y, fill="#d6d6d6", width=1)

        for y in range(ly):
            for x in range(lx):
                idx = y * lx + x
                n_i = state[idx]

                cx = x0 + (x + 0.5) * cell
                cy = y0 + (y + 0.78) * cell


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

    app = BoseHubbard2DGutzwillerApp(root)
    root.mainloop()
