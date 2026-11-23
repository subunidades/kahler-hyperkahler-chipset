"""Hyperkähler unit: 3 qubits, SU(2)-symmetric Heisenberg coupling, in Qiskit.
H = J * sum_{i<j} (XX+YY+ZZ)_{ij}. Its symmetry is collective SU(2) (the quaternion
group structure of -i*sigma), and the Kähler geometry of the state manifold is
measured on the same circuit. Imports the geometry tools from the main file.
Run: python hyperkahler_unit.py
"""
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from kahler_hyperkahler_qiskit import qgt, ok

N, J = 3, 1.0
PAIRS = [(0, 1), (0, 2), (1, 2)]


def pauli_at(label, qubits):
    s = ["I"] * N
    for q in qubits:
        s[N - 1 - q] = label           # Qiskit is little-endian
    return "".join(s)


def hamiltonian():
    terms = [(pauli_at(a, p), J) for p in PAIRS for a in "XYZ"]
    return SparsePauliOp.from_list(terms)


def spin(a):                            # S_a = 1/2 sum sigma_a
    return SparsePauliOp.from_list([(pauli_at(a, [q]), 0.5) for q in range(N)])


def circuit(t, steps=1):
    """exp(-iHt) as native two-qubit rotations. Each pair factor is SU(2)-invariant,
    so the circuit is exactly symmetric for any step count."""
    qc = QuantumCircuit(N)
    dt = t / steps
    for _ in range(steps):
        for i, j in PAIRS:
            qc.rxx(2 * J * dt, i, j); qc.ryy(2 * J * dt, i, j); qc.rzz(2 * J * dt, i, j)
    return qc


def main():
    H = hamiltonian().to_matrix()
    S = {a: spin(a).to_matrix() for a in "XYZ"}
    S2 = sum(S[a] @ S[a] for a in "XYZ")
    comm = lambda A, B: np.linalg.norm(A @ B - B @ A)

    print("SU(2) symmetry of the hyperkähler unit")
    print(f" [H,S_x]=[H,S_y]=[H,S_z]=0: {ok(all(comm(H, S[a]) < 1e-10 for a in 'XYZ'))}")
    print(f" [H,S²]=0: {ok(comm(H, S2) < 1e-10)}")
    ev, cnt = np.unique(np.round(np.linalg.eigvalsh(H), 8), return_counts=True)
    print(f" spectrum {ev} degeneracies {cnt} (S=3/2 quartet at +3J, two S=1/2 doublets at -3J): "
          f"{ok(np.allclose(ev, [-3 * J, 3 * J]) and list(cnt) == [4, 4])}")

    print("Circuit")
    t = 0.8
    R = expm(-1j * 0.9 * (0.3 * S["X"] - 0.5 * S["Y"] + 0.81 * S["Z"]))   # collective rotation
    exact = expm(-1j * H * t)
    errs = []
    for steps in (1, 5, 50):
        U = Operator(circuit(t, steps)).data
        errs.append(np.linalg.norm(U - exact))
        print(f" steps={steps}: unitary {ok(Operator(U).is_unitary())} | "
              f"commutes with collective SU(2) rotation {ok(np.allclose(U @ R, R @ U, atol=1e-10))} | "
              f"Trotter error {errs[-1]:.2e}")
    print(f" Trotter error shrinks with steps: {ok(errs[0] > errs[1] > errs[2] and errs[2] < errs[1] / 5)}")

    print("Kähler geometry along the symmetry orbit")
    psi0 = Statevector.from_label("000").data
    for a in "XYZ":
        build = lambda q, a=a: _orbit(q[0], S[a], psi0)
        g = qgt(build, [0.3]).real[0, 0]
        var = np.real(psi0.conj() @ S[a] @ S[a] @ psi0 - (psi0.conj() @ S[a] @ psi0) ** 2)
        print(f" g_alpha_alpha along S_{a.lower()} = Var(S_{a.lower()}) = {var:.4f}: {ok(np.isclose(g, var, atol=1e-6))}")


def _orbit(alpha, Sa, psi0):
    qc = QuantumCircuit(N)
    qc.initialize(expm(-1j * alpha * Sa) @ psi0, range(N))
    return qc


if __name__ == "__main__":
    main()
