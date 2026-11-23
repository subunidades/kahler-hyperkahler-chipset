"""Kähler + Hyperkähler chipset: Qiskit-native pre-layout design model.

Each die is a Qiskit Target (qubit frequencies, native gates rz/sx/x/cz, coupling graph),
circuits are compiled with the Qiskit transpiler and verified with Operator.equiv, the
transmon exchange gate is solved with Qiskit Dynamics, and the Kähler geometry uses ReverseQGT.
All frequencies, couplings, anharmonicities and gate errors are ILLUSTRATIVE placeholders,
to be replaced by the chosen foundry's process data.
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
try:                                   # optional: needs qiskit<=1.3 (see requirements-dynamics.txt)
    from qiskit_dynamics import Solver
except ImportError:
    Solver = None
import hyperkahler_unit as hk
from kahler_hyperkahler_qiskit import ok, qgt

TWO_PI = 2 * np.pi
SPEC = {
    "units": {"freq": "GHz", "alpha": "MHz", "g": "MHz", "error": "per gate", "duration": "s"},
    "kahler_die": {
        "qubits": [{"id": "K0", "f_max": 5.50, "f_park": 4.90, "alpha": -250, "readout_res": 6.70},
                   {"id": "K1", "f_max": 5.50, "f_park": 5.10, "alpha": -250, "readout_res": 6.80}],
        "couplings": [{"pair": ["K0", "K1"], "g": 15}],
    },
    "hyperkahler_die": {
        "qubits": [{"id": f"H{i}", "f_max": 5.50, "f_park": 4.80 + 0.15 * i, "alpha": -250,
                    "readout_res": 6.60 + 0.1 * i} for i in range(3)],
        "couplings": [{"pair": [f"H{i}", f"H{j}"], "g": 15} for i, j in [(0, 1), (0, 2), (1, 2)]],
        "operating_freq": 5.00,
    },
    "native_gates": {"rz": {"duration": 0, "error": 0}, "sx": {"duration": 2e-8, "error": 1e-4},
                     "x": {"duration": 2e-8, "error": 1e-4}, "cz": {"duration": 4e-8, "error": 5e-3}},
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
    print(f"Transmon-level XY gate ({'Qiskit Dynamics' if Solver else 'matrix exponential; Qiskit Dynamics not installed'}, 3 levels, resonant, g=15 MHz, α=-250 MHz)")
    check_native_decomposition()
    t, leak, phi, F = exchange_gate(15, -250)
    print(f" gate time {t:.1f} ns | leakage {leak:.2e} | |11> conditional phase {phi:+.3f} rad | "
          f"fidelity after CPhase correction {F:.5f}")
    print(f" leakage < 1e-2 and corrected fidelity > 0.98: {ok(leak < 1e-2 and F > 0.98)}")
    return {"gate_time_ns": float(t), "leakage": float(leak), "cond_phase_rad": float(phi), "fidelity": float(F)}


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


if __name__ == "__main__":
    out = {"spec": SPEC, "compile": part_compile(), "gate": part_gate(),
           "disorder": part_disorder(), "kahler": part_kahler()}
    json.dump(out, open("chipset_spec.json", "w"), indent=2)
    print("wrote chipset_spec.json")
