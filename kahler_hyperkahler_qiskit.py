"""Kähler / Hyperkähler operator lab: Qiskit-native.
PART 1  Operator zoo (unitary, Hermitian, projector, density matrix, resolvent, Pauli, Hadamard)
        with Operator / SparsePauliOp / PauliEvolutionGate / NumPyEigensolver.
PART 2  Kähler: quantum geometric tensor from qiskit_algorithms ReverseQGT on parameterized circuits.
PART 3  Hyperkähler: C^2 = H realified on 2 qubits (ancilla = Re/Im); I, J, K are Pauli-product operators.
Run: python kahler_hyperkahler_qiskit.py
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector
from qiskit.synthesis import MatrixExponential
from qiskit_algorithms.eigensolvers import NumPyEigensolver
from qiskit_algorithms.gradients import DerivativeType, ReverseQGT

ok = lambda b: "PASS" if b else "FAIL"
L = Operator.from_label


def evolve(op, t):
    """exp(-i t op) as a Qiskit Operator (exact matrix exponential synthesis)."""
    return Operator(PauliEvolutionGate(op, time=t, synthesis=MatrixExponential()))


def eigs(op, k=None):
    k = k or Operator(op).dim[0]
    return np.sort(NumPyEigensolver(k=k).compute_eigenvalues(op).eigenvalues.real)


def qgt(circuit, values):
    """Quantum geometric tensor Q = g - (i/2)F via Qiskit's ReverseQGT."""
    return ReverseQGT(StatevectorEstimator(), derivative_type=DerivativeType.COMPLEX).run(
        [circuit], [list(values)]).result().qgts[0]


# ---------------- PART 1 ----------------
def part1():
    print("PART 1: operators")
    I = L("I")
    X, Y, Z, Had = L("X"), L("Y"), L("Z"), L("H")
    ham = SparsePauliOp.from_list([("Z", 0.7), ("X", 0.3)])
    U = evolve(ham, 1.3)                                          # U = e^{-iHt}
    P = L("0")                                                    # |0><0|
    rho_pure = DensityMatrix(Statevector.from_label("0"))
    rho_mix = DensityMatrix(0.5 * I)
    z = 2.0 + 0.5j
    zH = z * I - Operator(ham)
    G = zH.power(-1)                                              # resolvent (zI-H)^-1
    print(f" Paulis & Hadamard self-inverse: {ok(all(A.dot(A) == I for A in (X, Y, Z, Had)))}")
    print(f" U unitary, U^-1 = U†: {ok(U.is_unitary() and U.power(-1) == U.adjoint())}")
    ev = eigs(ham)
    print(f" Hermitian H invertible iff all eigenvalues != 0 (eigs {np.round(ev, 3)}): {ok(np.all(abs(ev) > 1e-10))}")
    print(f" Projector P: P²=P, has zero eigenvalue (not invertible): {ok(P.dot(P) == P and np.min(abs(eigs(P))) < 1e-10)}")
    print(f" Pure rho (purity 1) not invertible; mixed rho=I/2 (purity 1/2) invertible, rho^-1=2I: "
          f"{ok(np.isclose(rho_pure.purity(), 1) and np.min(abs(eigs(Operator(rho_pure)))) < 1e-10 and Operator(rho_mix).power(-1) == 2 * I)}")
    print(f" Resolvent exists for z not in spec(H): {ok(G.dot(zH) == I)}")


# ---------------- PART 2 ----------------
def part2(th=1.0, ph=0.7):
    print("PART 2: Kähler (Fubini-Study / CP^n) via qiskit_algorithms ReverseQGT")
    p = ParameterVector("p", 2)
    qc = QuantumCircuit(1); qc.ry(p[0], 0); qc.rz(p[1], 0)
    Q = qgt(qc, [th, ph]); g = Q.real
    print(f" CP^1 metric = diag(1/4, sin²θ/4): {ok(np.allclose(g, np.diag([0.25, np.sin(th) ** 2 / 4]), atol=1e-8))}")
    s = np.sin(th)
    J = np.array([[0, -s], [1 / s, 0]])                           # J∂θ=(1/sinθ)∂φ, J∂φ=-sinθ∂θ
    print(f" J²=-1: {ok(np.allclose(J @ J, -np.eye(2)))} | g(Ju,Jv)=g(u,v): {ok(np.allclose(J.T @ g @ J, g, atol=1e-8))}")
    print(f" |Kähler form| = |Im Q| = sinθ/4: {ok(np.isclose(abs(Q.imag[0, 1]), s / 4, atol=1e-8))}")
    q = ParameterVector("q", 4)
    c2 = QuantumCircuit(2)
    c2.ry(q[0], 0); c2.ry(q[1], 1); c2.rxx(q[2], 0, 1); c2.ryy(q[2], 0, 1); c2.rz(q[3], 0)
    Q2 = qgt(c2, [0.9, 0.5, 0.3, 1.1])
    print(f" 2-qubit (CP^3) metric PSD: {ok(np.linalg.eigvalsh(Q2.real).min() > -1e-8)} | "
          f"Q Hermitian (F antisymmetric): {ok(np.allclose(Q2, Q2.conj().T, atol=1e-8))}")


# ---------------- PART 3 ----------------
def part3():
    print("PART 3: hyperkähler (C^2 = H, realified on 2 qubits)")
    I, X, Y, Z = L("I"), L("X"), L("Y"), L("Z")
    mi = lambda A: -1j * A
    # realification: v = x + iy -> |0>⊗x + |1>⊗y  (ancilla is the first factor)
    Ii = mi(Y).tensor(I)                                          # multiplication by i
    Jj = I.tensor(mi(Y)).dot(Z.tensor(I))                         # j: v -> -i sigma_y conj(v)
    Kk = Ii.dot(Jj)
    one = L("II")
    ts = (Ii, Jj, Kk)
    print(f" I²=J²=K²=-1: {ok(all(A.dot(A) == -1 * one for A in ts))}")
    print(f" IJ=K, JK=I, KI=J: {ok(Ii.dot(Jj) == Kk and Jj.dot(Kk) == Ii and Kk.dot(Ii) == Jj)}")
    print(f" I,J,K orthogonal (preserve flat metric): {ok(all(A.is_unitary() for A in ts))}")
    print(f" Kähler forms omega_a = g(a·,·) antisymmetric: {ok(all(A == -1 * A.transpose() for A in ts))}")
    n = np.array([0.3, -0.5, 0.81]); n = n / np.linalg.norm(n)
    M = n[0] * Ii + n[1] * Jj + n[2] * Kk
    print(f" every n·(I,J,K) is a complex structure (twistor S²): {ok(M.dot(M) == -1 * one)}")
    Iq, Jq, Kq = mi(X), mi(Y), mi(Z)
    print(f" -iσ triple is quaternion algebra (1 qubit): {ok(Iq.dot(Jq) == Kq and Iq.dot(Iq) == -1 * L('I'))}")


if __name__ == "__main__":
    part1(); part2(); part3()
