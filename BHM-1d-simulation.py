import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

# BHM parameters
U_DEFAULT = 1.0
LOGT_MIN = -2.0
LOGT_MAX = 2.0


def hilbert_dimension(num_sites, total_particles):
    return math.comb(total_particles + num_sites - 1, total_particles)


def generate_fock_basis(num_sites, total_particles):
    basis = []
    def rec_build(prefix, sites_left, particles_left):
        if sites_left == 1:
            basis.append(tuple(prefix + [particles_left]))
            return
        for n in range(particles_left + 1):
            rec_build(prefix + [n], sites_left - 1, particles_left - n)

    rec_build([], num_sites, total_particles)
    basis.sort()
    return basis


def build_bose_hubbard_1d_parts(num_sites, total_particles, U=1.0):

    basis = generate_fock_basis(num_sites, total_particles)
    dim = len(basis)
    basis_index = {state: idx for idx, state in enumerate(basis)}

    diagonal = np.zeros(dim, dtype=float)

    rows = []
    cols = []
    data = []

    for col, state in enumerate(basis):
        diagonal[col] = 0.5 * U * sum(n * (n - 1) for n in state)

        for i in range(num_sites):
            j = (i + 1) % num_sites
            n = list(state)
            # hop op.
            if n[j] > 0:
                new_state = n.copy()
                amp = -math.sqrt((new_state[i] + 1) * new_state[j])
                new_state[i] += 1
                new_state[j] -= 1
                row = basis_index[tuple(new_state)]
                rows.append(row)
                cols.append(col)
                data.append(amp)

            # hop op. c.c.
            if n[i] > 0:
                new_state = n.copy()
                amp = -math.sqrt((new_state[j] + 1) * new_state[i])
                new_state[j] += 1
                new_state[i] -= 1
                row = basis_index[tuple(new_state)]
                rows.append(row)
                cols.append(col)
                data.append(amp)

    H_int = diags(diagonal, offsets=0, format="csr")
    H_hop_unit = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()

    return basis, H_int, H_hop_unit


def solve_ground_state_sparse(H):

    dim = H.shape[0]

    if dim == 1:
        return np.array([H[0, 0]], dtype=float), np.array([[1.0]], dtype=float)

    if dim <= 6:
        Hd = H.toarray()
        evals, evecs = np.linalg.eigh(Hd)
        return evals, evecs

    evals, evecs = eigsh(H, k=1, which="SA", tol=1e-10, maxiter=10000)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


def sample_fock_state(basis, probs, rng):
    idx = rng.choice(len(basis), p=probs)
    return idx, basis[idx]


class BoseHubbard1DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bose-Hubbard 1D")
        self.root.geometry("1000x800")
        self.root.configure(bg="#ececec")

        self.rng = np.random.default_rng()

        # current BH model cache
        self.cached_key = None
        self.cached_basis = None
        self.cached_H_int = None
        self.cached_H_hop = None
        self.cached_probs = None
        self.cached_E0 = None
        self.cached_t = None

        # variables for GUI
        self.m_var = tk.IntVar(value=4)
        self.n_var = tk.IntVar(value=4)
        self.logt_var = tk.DoubleVar(value=0.0)

        self.t_string = tk.StringVar()
        self.dim_string = tk.StringVar()
        self.info_string = tk.StringVar()

        self._build_ui()
        self._update_t_label()
        self._update_dimension_label()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Bose–Hubbard 1D",
            font=("Helvetica", 25, "bold"),
            bg="#ececec",
            fg="#202020",
        )
        title.pack(pady=(14, 4))

        subtitle = tk.Label(
            self.root,
            text="1D chain with periodic boundary conditions\n M sites with N particles",
            font=("Helvetica", 15),
            bg="#ececec",
            fg="#202020",
        )
        subtitle.pack(pady=(0, 10))

        controls = tk.Frame(self.root, bg="#ececec")
        controls.pack(anchor='center', padx=20, pady=6)

        tk.Label(
            controls, text="M = ", font=("Helvetica", 15, "bold"), bg="#ececec"
        ).grid(row=0, column=0, padx=(0, 6))

        m_spin = tk.Spinbox(
            controls,
            from_=2,
            to=16,
            textvariable=self.m_var,
            width=2,
            command=self._update_dimension_label,
            font=("Helvetica",15,"bold"),
        )
        m_spin.grid(row=0, column=1, padx=(0, 14))

        tk.Label(
            controls, text="N = ", font=("Helvetica", 15, "bold"), bg="#ececec"
        ).grid(row=0, column=2, padx=(0, 6))

        n_spin = tk.Spinbox(
            controls,
            from_=1,
            to=16,
            textvariable=self.n_var,
            width=2,
            command=self._update_dimension_label,
            font=("Helvetica",15,"bold"),
        )
        n_spin.grid(row=0, column=3, padx=(0, 18))

        tk.Label(
            controls, text="log t = ", font=("Helvetica", 15, "bold"), bg="#ececec"
        ).grid(row=0, column=4, padx=(50, 8))

        slider = tk.Scale(
            controls,
            from_=LOGT_MIN,
            to=LOGT_MAX,
            resolution=0.01,
            orient="horizontal",
            variable=self.logt_var,
            command=lambda _e=None: self._update_t_label(),
            length=200,
            width=20,
            bg="#ececec",
            highlightthickness=0,
        )
        slider.grid(row=0, column=5, padx=(0, 10), pady=(0,15))

        tk.Label(
            controls,
            textvariable=self.t_string,
            font=("Consolas", 15, "bold"),
            bg="#ececec",
            fg="#1f3b5c",
            width=14,
        ).grid(row=0, column=6, padx=(0, 10))

        ttk.Button(controls, text="Simulate", command=self.simulate).grid(
            row=2, column=4
        )
        ttk.Button(controls, text="New Sample", command=self.new_sample).grid(
            row=2, column=5
        )

        dim_label = tk.Label(
            self.root,
            textvariable=self.dim_string,
            font=("Consolas", 12),
            bg="#ececec",
            fg="#333333",
        )
        dim_label.pack(pady=(8, 8))

        self.canvas = tk.Canvas(
            self.root,
            width= 800,
            height=300,
            bg="#f6f6f6",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.canvas.pack(padx=20, pady=12)

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

    def _update_dimension_label(self):
        try:
            M = int(self.m_var.get())
            N = int(self.n_var.get())
            if M < 2 or N < 0:
                self.dim_string.set("Hilbert dimension: invalid parameters")
                return
            dim = hilbert_dimension(M, N)
            self.dim_string.set(rf"dim 𝓗 = Comb(N+M-1, N) = {dim}")
        except Exception:
            self.dim_string.set("Hilbert dimension: invalid parameters")

    def _get_parameters(self):
        M = int(self.m_var.get())
        N = int(self.n_var.get())
        if M < 2:
            raise ValueError("M debe ser al menos 2.")
        if N < 1:
            raise ValueError("N debe ser al menos 1.")
        return M, N

    def _ensure_model(self, M, N):
        key = (M, N, U_DEFAULT)
        if self.cached_key == key:
            return

        dim = hilbert_dimension(M, N)
        if dim > 250000:
            raise ValueError(
                f"Hilbert dimension would be {dim}, too big for iterative exact diagonalization.\n"
                f"Try smaller values for M or N."
            )

        basis, H_int, H_hop = build_bose_hubbard_1d_parts(M, N, U_DEFAULT)

        self.cached_key = key
        self.cached_basis = basis
        self.cached_H_int = H_int
        self.cached_H_hop = H_hop
        self.cached_probs = None
        self.cached_E0 = None
        self.cached_t = None

    def simulate(self):
        try:
            M, N = self._get_parameters()
            self._update_dimension_label()
            self._ensure_model(M, N)

            t = 10 ** self.logt_var.get()
            H = self.cached_H_int + t * self.cached_H_hop

            evals, evecs = solve_ground_state_sparse(H)

            E0 = float(evals[0])
            psi0 = np.asarray(evecs[:, 0]).reshape(-1)
            probs = np.abs(psi0) ** 2
            probs = probs / probs.sum()

            self.cached_probs = probs
            self.cached_E0 = E0
            self.cached_t = t

            self._draw_sample()

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def new_sample(self):
        if self.cached_probs is None:
            self.simulate()
            return
        self._draw_sample()

    def _draw_sample(self):
        idx, sampled_state = sample_fock_state(self.cached_basis, self.cached_probs, self.rng)

        most_prob_idx = int(np.argmax(self.cached_probs))
        most_prob_state = self.cached_basis[most_prob_idx]
        most_prob = float(self.cached_probs[most_prob_idx])
        sampled_prob = float(self.cached_probs[idx])

        self.draw_chain(sampled_state)

        info = (
            f"t = {self.cached_t:.4}\n"
            f"E0 = {self.cached_E0:.4}\n"
            f"Measured configuration= {sampled_state}\n"
            f"Probability of that state = {sampled_prob*100:.4}%\n"
            f"Most probable configuration = {most_prob_state}\n"
            f"Probability of most probable configuration = {most_prob*100:.4}%\n\n"
            f"'Simulate' recalculates the ground state for the actual (t/U).\n"
            f"'New Sample' makes a measurement for the current ground state."
        )
        self.info_string.set(info)

    def draw_chain(self, state):
        self.canvas.delete("all")

        width = int(self.canvas["width"])
        height = int(self.canvas["height"])

        margin_left = 70
        margin_right = 70
        y_chain = 200

        M = len(state)
        usable_width = width - margin_left - margin_right
        spacing = usable_width / max(M - 1, 1)

        xs = [margin_left + i * spacing for i in range(M)]

        self.canvas.create_text(
            width / 2,
            28,
            text="Sample of an occupation measurement",
            font=("Helvetica", 12, "bold"),
            fill="#202020",
        )

        if M > 1:
            self.canvas.create_line(xs[0], y_chain, xs[-1], y_chain, fill="#5a5a5a", width=3)

        max_occ = max(state) if state else 1
        particle_r = max(8, min(16, int(140 / max(M, 6))))
        vstep = 2 * particle_r + 5

        for i, (x, n_i) in enumerate(zip(xs, state)):
            # sitio
            self.canvas.create_oval(
                x - 7, y_chain - 7, x + 7, y_chain + 7,
                fill="#f7f7f7", outline="#222222", width=1.5
            )

            self.canvas.create_text(
                x,
                y_chain + 26,
                text=f"{i}",
                font=("Helvetica", 10, "bold"),
                fill="#202020",
            )

            self.canvas.create_text(
                x,
                y_chain + 46,
                text=f"n={n_i}",
                font=("Helvetica", 10),
                fill="#303030",
            )

            if n_i == 0:
                continue

            # darker color for greater occupation
            if max_occ > 0:
                shade = int(215 - 195 * (n_i / max_occ))
            else:
                shade = 215
            shade = max(15, min(230, shade))
            color = f"#{shade:02x}{shade:02x}{shade:02x}"

            start_y = y_chain - 28
            for k in range(n_i):
                cy = start_y - k * vstep
                self.canvas.create_oval(
                    x - particle_r,
                    cy - particle_r,
                    x + particle_r,
                    cy + particle_r,
                    fill=color,
                    outline="#101010",
                    width=1.2,
                )


if __name__ == "__main__":

    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    app = BoseHubbard1DApp(root)
    root.mainloop()