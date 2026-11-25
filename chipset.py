"""Kähler + Hyperkähler chipset: Qiskit-native pre-layout design model.

Each die is a Qiskit Target (qubit frequencies, native gates rz/sx/x/cz, coupling graph),
circuits are compiled with the Qiskit transpiler and verified with Operator.equiv, the
transmon exchange gate is solved with Qiskit Dynamics (if installed),
a 4-level flux-pulse CZ is optimised and run with Lindblad T1/T2, junction spread is Monte-Carloed, and the Kähler geometry uses ReverseQGT.
All frequencies, couplings, anharmonicities and gate errors are ILLUSTRATIVE placeholders,
to be replaced by the chosen foundry's process data.
Baseline parameters live in PARAMS (value + sweep range).
Run: python chipset.py
"""
import json
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import CZGate, Measure, RZGate, SXGate, XGate, XXPlusYYGate
from qiskit.providers.backend import QubitProperties
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import InstructionProperties, Target
from qiskit_algorithms.eigensolvers import NumPyEigensolver
from scipy.linalg import expm
from scipy.optimize import minimize
try:                                   # optional: needs qiskit<=1.3 (see requirements-dynamics.txt)
    from qiskit_dynamics import Solver
except ImportError:
    Solver = None
import hyperkahler_unit as hk
from kahler_hyperkahler_qiskit import ok, qgt

TWO_PI = 2 * np.pi

# ---------------------------------------------------------------------------------------------
# Generic Al/AlOx/Al-on-silicon baseline: (baseline, (sweep_lo, sweep_hi)). Typical of published
# devices, NOT foundry data and NOT validated by measurement. Foundry-dependent quantities
# (Jc, achieved junction spread, layer stack, measured coherence) stay as swept parameters.
# ---------------------------------------------------------------------------------------------
PARAMS = {
    "EC_MHz":          (220, (180, 260)),      # charging energy; anharmonicity alpha ~ -EC
    "fmax_GHz":        (5.00, (4.5, 5.2)),     # qubit max frequency (sweet spot)
    "g_tri_MHz":       (15, (10, 20)),         # exchange coupling per edge, hyperkahler triangle
    "g_cz_eff_MHz":    (10, (6, 14)),          # effective qubit-qubit g at the CZ point, Kahler die
    "g_direct_MHz":    (7, (4, 10)),           # direct qubit-qubit g, Kahler die
    "g_qc_MHz":        (100, (70, 130)),       # qubit-coupler g (design input, not simulated here)
    "T1_us":           (100, (30, 200)),
    "T2_us":           (80, (20, 150)),
    "t_1q_ns":         (25, (20, 30)),
    "sigma_R_untrimmed": (0.015, (0.010, 0.020)),   # junction resistance spread, as-fabricated
    "sigma_R_trimmed":   (0.002, (0.001, 0.003)),   # after laser-anneal trim
    "sigma_cal_MHz":   (0.5, (0.2, 1.0)),      # residual frequency error after flux calibration
    "su2_tol_MHz":     (1.0, (0.5, 2.0)),      # design target: rms frequency disorder in the triangle
    # --- Stage 0/1 additions ---
    "Jc_uA_per_um2":   (0.5, (0.2, 1.0)),      # junction critical-current density (FOUNDRY-DEPENDENT)
    "coupler_C_fF":    (100, (80, 120)),       # coupler total capacitance
    "coupler_fmax_GHz": (7.0, (6.5, 7.5)),
    "cpw_C_fF_per_um": (0.16, (0.14, 0.18)),   # CPW capacitance per length (geometry-assumed)
    "eps_eff":         (6.45, (6.2, 6.7)),     # effective permittivity, CPW on silicon
    "zz_tol_MHz":      (1.0, (0.5, 2.0)),      # residual ZZ at idle
    "squid_d":         (0.3, (0.2, 0.5)),      # SQUID junction asymmetry (both dies)
    "A_phi_uPhi0":     (2.0, (1.0, 5.0)),      # 1/f flux noise sqrt(A) at 1 Hz, in micro-Phi0
    "g_qr_MHz":        (100, (60, 100)),       # qubit-readout-resonator coupling
    "readout_detuning_GHz": (1.6, (1.2, 2.0)), # resonator minus qubit parking frequency
    "Q_purcell":       (40, (25, 50)),         # Purcell filter quality factor
    "eta":             (0.5, (0.1, 0.7)),      # readout quantum efficiency (needs a paramp for 0.5)
    "t_readout_ns":    (500, (300, 1000)),     # integration window
    "nbar_frac":       (0.1, (0.05, 0.2)),     # readout photons as a fraction of n_crit
}

# Stage 0 decisions (frozen). Changing one of these means re-running every later stage.
DECISIONS = {"hyperkahler_qubits": "flux-tunable", "kahler_qubits": "flux-tunable",
             "kahler_coupler": "flux-tunable", "readout": "per-qubit lambda/4 resonator + Purcell filter, one feedline per die"}
B = lambda k: PARAMS[k][0]
EC, ALPHA = B("EC_MHz"), -B("EC_MHz")


def transmon(f01_GHz, EC_MHz=EC):
    """Derived transmon quantities from f01 = sqrt(8 EJ EC) - EC."""
    ec = EC_MHz * 1e-3
    ej = (f01_GHz + ec) ** 2 / (8 * ec)                              # GHz
    h, e, phi0 = 6.62607015e-34, 1.602176634e-19, 2.067833848e-15
    return {"EJ_GHz": ej, "EJ_over_EC": ej / ec, "C_sigma_fF": e ** 2 / (2 * h * ec * 1e9) * 1e15,
            "Ic_nA": 2 * np.pi * h * ej * 1e9 / phi0 * 1e9}


def coherence_error(t_ns, n_qubits=1, T1=None, T2=None):
    """Coherence-limited average gate error r = n (t/3)(1/T1 + 1/Tphi), 1/Tphi = 1/T2 - 1/(2 T1)."""
    T1, T2 = T1 or B("T1_us"), T2 or B("T2_us")
    rate = 1 / T1 + max(1 / T2 - 1 / (2 * T1), 0)                    # per us
    return n_qubits * (t_ns * 1e-3 / 3) * rate


SPEC = {
    "units": {"freq": "GHz", "alpha": "MHz", "g": "MHz", "error": "per gate", "duration": "s"},
    "kahler_die": {
        "qubits": [{"id": "K0", "tunable": True, "squid_d": B("squid_d"), "f_max": B("fmax_GHz"), "f_park": 4.85, "alpha": ALPHA, "readout_res": round(4.85 + B("readout_detuning_GHz"), 2)},
                   {"id": "K1", "tunable": True, "squid_d": B("squid_d"), "f_max": B("fmax_GHz"), "f_park": 4.45, "alpha": ALPHA, "readout_res": round(4.45 + B("readout_detuning_GHz"), 2)}],
        "couplings": [{"pair": ["K0", "K1"], "g": B("g_direct_MHz")}],
        "coupler": {"f_max": 7.0, "g_qubit_coupler": B("g_qc_MHz")},
    },
    "hyperkahler_die": {
        "qubits": [{"id": f"H{i}", "f_max": 5.50, "f_park": round(4.15 + 0.55 * i, 2), "alpha": ALPHA,
                    "readout_res": round(4.15 + 0.55 * i + B("readout_detuning_GHz"), 2), "tunable": True,
                    "squid_d": B("squid_d")} for i in range(3)],
        "couplings": [{"pair": [f"H{i}", f"H{j}"], "g": B("g_tri_MHz")} for i, j in [(0, 1), (0, 2), (1, 2)]],
        "operating_freq": 5.00,
    },
    "native_gates": {"rz": {"duration": 0, "error": 0},
                     "sx": {"duration": B("t_1q_ns") * 1e-9, "error": coherence_error(B("t_1q_ns"))},
                     "x": {"duration": B("t_1q_ns") * 1e-9, "error": coherence_error(B("t_1q_ns"))},
                     "cz": {"duration": 4e-8, "error": coherence_error(40, 2) + 5e-4}},   # + leakage/phase placeholder
}


def die_target(die):
    """Qiskit Target for a die from SPEC: frequencies, native gates, coupling graph."""
    qs, ng = die["qubits"], SPEC["native_gates"]
    idx = {q["id"]: i for i, q in enumerate(qs)}
    edges = [(idx[a], idx[b]) for a, b in (c["pair"] for c in die["couplings"])]
    edges += [(b, a) for a, b in edges]
    P = lambda k: InstructionProperties(duration=ng[k]["duration"], error=ng[k]["error"])
    t = Target(num_qubits=len(qs), qubit_properties=[QubitProperties(frequency=q["f_park"] * 1e9) for q in qs])
    one = lambda k: {(i,): P(k) for i in range(len(qs))}
    t.add_instruction(RZGate(Parameter("t")), one("rz"))
    t.add_instruction(SXGate(), one("sx")); t.add_instruction(XGate(), one("x"))
    t.add_instruction(CZGate(), {e: P("cz") for e in edges})
    t.add_instruction(Measure(), {(i,): None for i in range(len(qs))})
    return t


def est_fidelity(tc, target):
    f = 1.0
    for inst in tc.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        f *= 1 - (target[name][tuple(tc.find_bit(b).index for b in inst.qubits)].error or 0)
    return f


def compile_check(label, qc, target):
    tc = transpile(qc, target=target, optimization_level=3, seed_transpiler=7)
    eq = Operator.from_circuit(tc).equiv(Operator(qc))
    ops = dict(tc.count_ops())
    f = est_fidelity(tc, target)
    print(f" {label}: native gates {ops} | depth {tc.depth()} | equivalent to source {ok(eq)} | "
          f"est. fidelity {f:.3f}")
    return {"ops": {k: int(v) for k, v in ops.items()}, "depth": int(tc.depth()), "equivalent": bool(eq),
            "est_fidelity": float(f)}


# ---------- 1. compile both dies to native gates ----------
def part_compile():
    print("Compile to each die's native gates (Qiskit Target + transpiler)")
    kt, ht = die_target(SPEC["kahler_die"]), die_target(SPEC["hyperkahler_die"])
    k = QuantumCircuit(2); k.ry(0.9, 0); k.ry(0.5, 1); k.append(XXPlusYYGate(0.8), [0, 1]); k.rz(1.1, 0)
    out = {"kahler": compile_check("Kähler die circuit      ", k, kt)}
    for steps in (1, 5):
        out[f"hyperkahler_{steps}"] = compile_check(f"Hyperkähler unit, {steps} step", hk.circuit(0.8, steps=steps), ht)
    return out


# ---------- 2. Heisenberg pair gate = native XY + ZZ ----------
def check_native_decomposition(theta=0.37):
    ref = QuantumCircuit(2); ref.rxx(2 * theta, 0, 1); ref.ryy(2 * theta, 0, 1); ref.rzz(2 * theta, 0, 1)
    nat = QuantumCircuit(2); nat.append(XXPlusYYGate(4 * theta), [0, 1]); nat.rzz(2 * theta, 0, 1)
    print(f" Heisenberg pair gate = XY(4θ)·ZZ(2θ): {ok(Operator(ref).equiv(Operator(nat)))}")


# ---------- 3. transmon exchange gate with Qiskit Dynamics ----------
def exchange_gate(g_MHz, alpha_MHz, d=3):
    a = Operator(np.diag(np.sqrt(np.arange(1, d)), 1)); I = Operator(np.eye(d))
    ad = a.adjoint(); n = ad.dot(a); nn = n.dot(n) - n
    g, al = TWO_PI * g_MHz * 1e-3, TWO_PI * alpha_MHz * 1e-3      # rad/ns
    H = (al / 2) * (nn.tensor(I) + I.tensor(nn)) + g * (ad.tensor(a) + a.tensor(ad))
    t = np.pi / (2 * g)                                            # full XY swap: g*t = pi/2
    if Solver is not None:
        res = Solver(static_hamiltonian=H.data).solve(t_span=[0, t], y0=np.eye(d * d, dtype=complex),
                                                      method="DOP853", atol=1e-10, rtol=1e-10)
        U = np.array(res.y[-1])
    else:                              # same physics, matrix exponential (time-independent H)
        U = expm(-1j * H.data * t)
    idx = [i * d + j for i in (0, 1) for j in (0, 1)]
    Us = U[np.ix_(idx, idx)]
    leak = 1 - np.mean(np.sum(abs(Us) ** 2, axis=0))
    Uid = Operator(XXPlusYYGate(2 * g * t)).data
    phi = np.angle(Us[3, 3] / Uid[3, 3])
    Ucorr = Us @ np.diag([1, 1, 1, np.exp(-1j * phi)])
    return t, leak, phi, abs(np.trace(Uid.conj().T @ Ucorr)) ** 2 / 16


def part_gate():
    g = B("g_tri_MHz")
    print(f"Transmon-level XY gate ({'Qiskit Dynamics' if Solver else 'matrix exponential; Qiskit Dynamics not installed'}, resonant, g={g} MHz, α={ALPHA} MHz)")
    check_native_decomposition()
    out = {}
    for d in (3, 5):
        t, leak, phi, F = exchange_gate(g, ALPHA, d)
        print(f" {d} levels: gate time {t:.1f} ns | leakage {leak:.2e} | |11> conditional phase {phi:+.3f} rad | "
              f"fidelity after CPhase correction {F:.5f}")
        out[f"levels_{d}"] = {"gate_time_ns": float(t), "leakage": float(leak), "cond_phase_rad": float(phi), "fidelity": float(F)}
    r3, r5 = out["levels_3"], out["levels_5"]
    print(f" leakage < 1e-2 and corrected fidelity > 0.98 (3 levels): {ok(r3['leakage'] < 1e-2 and r3['fidelity'] > 0.98)}")
    print(f" 5-level result agrees with 3-level (fidelity within 1e-3): {ok(abs(r3['fidelity'] - r5['fidelity']) < 1e-3)}")
    return {**r3, "levels_3": r3, "levels_5": r5}


# ---------- 4. SU(2) breaking from frequency disorder ----------
def part_disorder():
    print("Hyperkähler die: SU(2) breaking from qubit-frequency disorder (J=g/2=7.5 MHz)")
    J = 7.5
    H0 = hk.hamiltonian(J)
    rng = np.random.default_rng(1); res = {}
    for rms in (0.05, 0.5, 2.0):
        spreads = []
        for _ in range(20):
            d = rng.normal(0, rms, 3); d -= d.mean()               # uniform part is Zeeman, not a break
            H = H0 + SparsePauliOp.from_list([(hk.pauli_at("Z", [q]), d[q] / 2) for q in range(3)])
            ev = np.sort(NumPyEigensolver(k=8).compute_eigenvalues(H).eigenvalues.real)
            spreads.append(max(np.ptp(ev[:4]), np.ptp(ev[4:])))
        res[rms] = float(np.mean(spreads))
        print(f" δ_rms={rms:>4} MHz -> multiplet splitting {res[rms]:.3f} MHz ({100 * res[rms] / (6 * J):.2f}% of gap)")
    print(f" splitting grows with disorder (tune qubits together): {ok(res[0.05] < res[0.5] < res[2.0])}")
    return {str(k): v for k, v in res.items()}


# ---------- 5. Kähler die geometry ----------
def part_kahler():
    q = ParameterVector("q", 4)
    qc = QuantumCircuit(2)
    qc.ry(q[0], 0); qc.ry(q[1], 1); qc.rxx(q[2], 0, 1); qc.ryy(q[2], 0, 1); qc.rz(q[3], 0)   # rxx·ryy = XY gate
    Q = qgt(qc, [0.9, 0.5, 0.8, 1.1])
    ev = np.linalg.eigvalsh(Q.real)
    print("Kähler die (CP^3) geometry (ReverseQGT)")
    print(f" metric eigenvalues {np.round(ev, 4)} | PSD {ok(ev.min() > -1e-8)} | "
          f"Q Hermitian {ok(np.allclose(Q, Q.conj().T, atol=1e-8))}")
    return {"metric_eigenvalues": ev.tolist()}


# ---------- 6. derived transmon parameters ----------
def part_transmon():
    print("Derived transmon parameters (generic Al/AlOx/Al baseline)")
    res = {}
    for f in (B("fmax_GHz"), *PARAMS["fmax_GHz"][1]):
        res[f"{f:.2f}"] = d = transmon(f)
        print(f" f01={f:.2f} GHz, EC={EC} MHz -> EJ={d['EJ_GHz']:.2f} GHz, EJ/EC={d['EJ_over_EC']:.0f}, "
              f"C_sigma={d['C_sigma_fF']:.0f} fF, Ic={d['Ic_nA']:.1f} nA")
    d = res[f"{B('fmax_GHz'):.2f}"]
    print(f" transmon regime EJ/EC > 50 and C_sigma 70-110 fF: {ok(d['EJ_over_EC'] > 50 and 70 < d['C_sigma_fF'] < 110)}")
    return res


# ---------- 7. flux-pulse CZ via |11> <-> |20>, 4 levels, optional Lindblad T1/T2 ----------
def _ops(d):
    a = np.diag(np.sqrt(np.arange(1, d)), 1); n = a.T @ a
    return a, n, n @ n - n


def _cz_H(g_MHz, delta_MHz, d):
    """Rotating frame with w_b=0, w_a=|alpha|+delta: |11> and |20> are degenerate at delta=0."""
    a, n, nn = _ops(d); I = np.eye(d)
    al, g = -TWO_PI * EC * 1e-3, TWO_PI * g_MHz * 1e-3                # rad/ns
    wa = -al + TWO_PI * delta_MHz * 1e-3
    return (wa * np.kron(n, I) + (al / 2) * (np.kron(nn, I) + np.kron(I, nn))
            + g * (np.kron(a.T, a) + np.kron(a, a.T)))


def _comp(d):
    return [i * d + j for i in (0, 1) for j in (0, 1)]


def _cz_target(Us):
    ph = np.angle(np.diag(Us))                                       # free single-qubit Z phases absorbed
    return np.diag(np.exp(1j * np.array([ph[0], ph[1], ph[2], ph[1] + ph[2] - ph[0] + np.pi])))


def _cz_unitary_fidelity(x, g_MHz, d):
    U = expm(-1j * _cz_H(g_MHz, x[1], d) * x[0])[np.ix_(_comp(d), _comp(d))]
    return abs(np.trace(_cz_target(U).conj().T @ U)) ** 2 / 16, U


def _lindblad_fidelity(H, t, Ut, T1, T2, d):
    a, n, _ = _ops(d); I = np.eye(d)
    g1 = 1 / (T1 * 1e3); gphi = max(1 / (T2 * 1e3) - g1 / 2, 0)      # per ns
    ops = []
    for A, N in ((np.kron(a, I), np.kron(n, I)), (np.kron(I, a), np.kron(I, n))):
        ops += [np.sqrt(g1) * A, np.sqrt(2 * gphi) * N]
    D = d * d; Id = np.eye(D)
    Ls = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    for L in ops:
        LdL = L.conj().T @ L
        Ls = Ls + np.kron(L.conj(), L) - 0.5 * np.kron(Id, LdL) - 0.5 * np.kron(LdL.T, Id)
    S = expm(Ls * t)
    comp = _comp(d); V = np.zeros((D, 4)); V[comp, range(4)] = 1
    Ut_full = V @ Ut
    tot = 0
    for i in range(4):
        for j in range(4):
            rho = np.zeros((D, D), dtype=complex); rho[comp[i], comp[j]] = 1
            out = (S @ rho.flatten(order="F")).reshape(D, D, order="F")
            tot += Ut_full[:, i].conj() @ out @ Ut_full[:, j]
    return tot.real / 16


def part_cz(d=4, g=None, verbose=True):
    g = g or B("g_cz_eff_MHz")
    P = print if verbose else (lambda *a, **k: None)
    t0 = 1e3 / (2 * np.sqrt(2) * g)                                   # full |11>-|20> cycle: 1/(2 sqrt2 g)
    P(f"Flux-pulse CZ via |11>↔|20> ({d} levels, square pulse, g_eff={g} MHz, α={ALPHA} MHz)")
    f0, _ = _cz_unitary_fidelity([t0, 0.0], g, d)
    r = minimize(lambda x: 1 - _cz_unitary_fidelity(x, g, d)[0], [t0, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-12, "maxiter": 2000})
    t, delta = r.x
    F, U = _cz_unitary_fidelity(r.x, g, d)
    ph = np.angle(np.diag(U)); cphase = (ph[3] - ph[2] - ph[1] + ph[0]) % TWO_PI
    leak = 1 - np.mean(np.sum(abs(U) ** 2, axis=0))
    Ut = _cz_target(U)
    H = _cz_H(g, delta, d)
    Fl = _lindblad_fidelity(H, t, Ut, B("T1_us"), B("T2_us"), d)
    err_coh = coherence_error(t, 2)
    P(f" analytic time 1/(2√2 g) = {t0:.1f} ns, unitary error {1 - f0:.2e}")
    P(f" optimised: time {t:.1f} ns, detuning {delta:+.2f} MHz | unitary error {1 - F:.2e} | "
          f"leakage {leak:.2e} | conditional phase {cphase:.4f} rad")
    P(f" with Lindblad T1={B('T1_us')} µs, T2={B('T2_us')} µs: error {1 - Fl:.2e} "
          f"(analytic coherence estimate {err_coh:.2e} + unitary {1 - F:.2e})")
    P(f" conditional phase = π (±0.01) and unitary error < 1e-2: {ok(abs(cphase - np.pi) < 0.01 and 1 - F < 1e-2)}")
    P(f" Lindblad error within 50% of coherence estimate + unitary error: "
          f"{ok(abs((1 - Fl) - (err_coh + 1 - F)) < 0.5 * (err_coh + 1 - F))}")
    return {"levels": d, "g_eff_MHz": g, "time_ns": float(t), "detuning_MHz": float(delta),
            "unitary_error": float(1 - F), "leakage": float(leak), "cond_phase_rad": float(cphase),
            "lindblad_error": float(1 - Fl), "coherence_estimate": float(err_coh),
            "note": "square flux pulse, no ramp/distortion; no 1/f flux noise; no crosstalk"}


# ---------- 8. junction-spread Monte Carlo: yield and SU(2) breaking ----------
def _freqs(f_nom, sigma_R, size, rng):
    """f01 + EC ~ R^(-1/2): sigma_f ~ (f+EC)/2 * sigma_R."""
    return (f_nom + EC * 1e-3) * (1 + sigma_R * rng.standard_normal(size)) ** -0.5 - EC * 1e-3


def part_yield(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    a = abs(ALPHA) * 1e-3
    sk = SPEC["kahler_die"]; p0, p1 = (q["f_park"] for q in sk["qubits"])
    print(f"Junction-spread Monte Carlo ({n} samples per row; f ∝ R^-1/2)")
    sig_f = lambda s: 0.5 * (B("fmax_GHz") + EC * 1e-3) * s * 1e3
    out = {"kahler": {}, "hyperkahler": {}}
    print(" Kähler die (flux-tunable): pass if both qubits reach their parking frequency (+10 MHz margin),")
    print("  K1 can be tuned to K0_park - |α| for the CZ, and idle |Δ| > 50 MHz")
    for name in ("sigma_R_untrimmed", "sigma_R_trimmed"):
        s = B(name)
        f = _freqs(B("fmax_GHz"), s, (n, 2), rng)
        good = (f[:, 0] >= p0 + 0.01) & (f[:, 1] >= p1 + 0.01) & (f[:, 1] >= p0 - a + 0.01) & (abs(p0 - p1) > 0.05)
        out["kahler"][name] = float(good.mean())
        print(f"  sigma_R={100 * s:.1f}% (sigma_f≈{sig_f(s):.0f} MHz): yield {100 * good.mean():.1f}%")
    tol = B("su2_tol_MHz")
    print(f" Hyperkähler die: pass if rms frequency disorder (mean removed) < {tol} MHz")
    rows = {}
    for label, sigma_f_MHz in (("fixed-frequency, untrimmed", None), ("fixed-frequency, trimmed", None),
                               ("flux-tunable, calibrated", B("sigma_cal_MHz"))):
        if sigma_f_MHz is None:
            s = B("sigma_R_untrimmed" if "untrimmed" in label else "sigma_R_trimmed")
            d = (_freqs(5.0, s, (n, 3), rng) - 5.0) * 1e3
        else:
            d = sigma_f_MHz * rng.standard_normal((n, 3))
        d = d - d.mean(axis=1, keepdims=True)
        rms = np.sqrt((d ** 2).mean(axis=1) * 3 / 2)                  # unbiased for 3 qubits, mean removed
        rows[label] = float((rms < tol).mean())
        print(f"  {label:<28}: median rms disorder {np.median(rms):6.2f} MHz, yield {100 * rows[label]:.1f}%")
    out["hyperkahler"] = rows
    print(f" untrimmed fixed-frequency sigma_f is tens of MHz (> g={B('g_tri_MHz')} MHz): "
          f"{ok(30 < sig_f(B('sigma_R_untrimmed')) < 45 and sig_f(B('sigma_R_untrimmed')) > B('g_tri_MHz'))}")
    print(f" trimming raises yield, and calibrated tunable qubits are best: "
          f"{ok(rows['fixed-frequency, untrimmed'] < rows['fixed-frequency, trimmed'] <= rows['flux-tunable, calibrated'])}")
    return out


if __name__ == "__main__":
    out = {"params": {k: {"baseline": v[0], "sweep": list(v[1])} for k, v in PARAMS.items()},
           "spec": SPEC, "transmon": part_transmon(), "compile": part_compile(), "gate": part_gate(),
           "cz": part_cz(), "disorder": part_disorder(), "yield": part_yield(), "kahler": part_kahler()}
    json.dump(out, open("chipset_spec.json", "w"), indent=2)
    print("wrote chipset_spec.json")
