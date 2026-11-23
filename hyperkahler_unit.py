"""Hyperkähler unit: 3 qubits, SU(2)-symmetric Heisenberg coupling, Qiskit-native.
H = J * sum_{i<j} (XX+YY+ZZ). Symmetry via SparsePauliOp algebra, spectrum via NumPyEigensolver,
exact evolution via PauliEvolutionGate, circuit via rxx/ryy/rzz, geometry via ReverseQGT.
Run: python hyperkahler_unit.py
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector, process_fidelity
from qiskit.synthesis import MatrixExponential
from kahler_hyperkahler_qiskit import eigs, evolve, ok, qgt

N = 3
PAIRS = [(0, 1), (0, 2), (1, 2)]


def pauli_at(label, qubits):
    s = ["I"] * N
    for q in qubits:
        s[N - 1 - q] = label                                      # Qiskit is little-endian
    return "".join(s)


def hamiltonian(J=1.0):
    return SparsePauliOp.from_list([(pauli_at(a, p), J) for p in PAIRS for a in "XYZ"])


def spin(a):                                                      # S_a = 1/2 sum sigma_a
    return SparsePauliOp.from_list([(pauli_at(a, [q]), 0.5) for q in range(N)])


def circuit(t, J=1.0, steps=1):
    """exp(-iHt) from native two-qubit rotations; each pair factor is SU(2)-invariant."""
    qc = QuantumCircuit(N)
    dt = t / steps
    for _ in range(steps):
        for i, j in PAIRS:
            qc.rxx(2 * J * dt, i, j); qc.ryy(2 * J * dt, i, j); qc.rzz(2 * J * dt, i, j)
    return qc


def commutes(A, B):
    return np.allclose((A.dot(B) - B.dot(A)).simplify().coeffs, 0, atol=1e-10)


def main():
    H = hamiltonian()
    S = {a: spin(a) for a in "XYZ"}
    S2 = sum((S[a].dot(S[a]) for a in "XYZ")).simplify()
    print("SU(2) symmetry of the hyperkähler unit")
    print(f" [H,S_x]=[H,S_y]=[H,S_z]=0: {ok(all(commutes(H, S[a]) for a in 'XYZ'))}")
    print(f" [H,S²]=0: {ok(commutes(H, S2))}")
    ev, cnt = np.unique(np.round(eigs(H), 8), return_counts=True)
    print(f" spectrum {ev} degeneracies {cnt} (S=3/2 quartet at +3J, two S=1/2 doublets at -3J): "
          f"{ok(np.allclose(ev, [-3, 3]) and list(cnt) == [4, 4])}")

    print("Circuit")
    t = 0.8
    exact = evolve(H, t)
    R = evolve(0.3 * S["X"] - 0.5 * S["Y"] + 0.81 * S["Z"], 0.9)  # collective SU(2) rotation
    errs = []
    for steps in (1, 5, 50):
        U = Operator(circuit(t, steps=steps))
        errs.append(1 - process_fidelity(U, exact))
        print(f" steps={steps}: unitary {ok(U.is_unitary())} | "
              f"commutes with collective SU(2) rotation {ok(U.dot(R) == R.dot(U))} | "
              f"Trotter infidelity {errs[-1]:.2e}")
    print(f" Trotter error shrinks with steps: {ok(errs[0] > errs[1] > errs[2] and errs[2] < errs[1] / 5)}")

    print("Kähler geometry along the symmetry orbit (ReverseQGT)")
    sv0 = Statevector.from_label("000")
    for a, gate in zip("XYZ", ("rx", "ry", "rz")):
        al = Parameter("a")
        qc = QuantumCircuit(N)
        for q in range(N):
            getattr(qc, gate)(al, q)                              # exp(-i a S_a) = product of single-qubit rotations
        g = qgt(qc, [0.3]).real[0, 0]
        var = (sv0.expectation_value(S[a].dot(S[a])) - sv0.expectation_value(S[a]) ** 2).real
        print(f" g_alpha_alpha along S_{a.lower()} = Var(S_{a.lower()}) = {var:.4f}: {ok(np.isclose(g, var, atol=1e-6))}")


if __name__ == "__main__":
    main()
