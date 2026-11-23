"""Kähler / Hyperkähler operator lab in Qiskit.
Companion to gateway-sbagen-kahler.html (tab 1 operators -> tab 3 geometry).

PART 1  Operator zoo from the invertibility tab: unitary, Hermitian, projector,
        density matrix (mixed/pure), resolvent, Pauli, Hadamard.
PART 2  Kähler: quantum geometric tensor Q = g - (i/2)F on CP^n-1 from any circuit.
        Checks: g symmetric PSD, F antisymmetric, and compatibility
        omega(u,v) = g(Ju,v) for CP^1 with the complex structure J.
PART 3  Hyperkähler: C^2 = H with I = i, J = -i sigma_y∘conj, K = IJ.
        Real 4x4 reps are orthogonal (metric-compatible), I^2=J^2=K^2=-1, IJ=K,
        and each gives a Kähler form omega_a = g(a·,·).
Run: pip install qiskit numpy && python kahler_hyperkahler_qiskit.py
"""
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import (Operator, Pauli, Statevector, DensityMatrix,
                                 SparsePauliOp)

ok = lambda b: "PASS" if b else "FAIL"
I2 = np.eye(2)


# ---------------- PART 1: operator zoo ----------------
def invertible(M, tol=1e-10):
    return abs(np.linalg.det(M)) > tol


def part1():
    print("PART 1: operators")
    X, Y, Z = (Operator(Pauli(s)).data for s in "XYZ")
    H = Operator(np.array([[1, 1], [1, -1]]) / np.sqrt(2)).data
    ham = SparsePauliOp.from_list([("Z", 0.7), ("X", 0.3)]).to_matrix()
    U = expm(-1j * ham * 1.3)                       # U = e^{-iHt}
    P = np.diag([1, 0]).astype(complex)             # |0><0|
    rho_pure = DensityMatrix(Statevector([1, 0])).data
    rho_mix = DensityMatrix(np.eye(2) / 2).data
    z = 2.0 + 0.5j
    G = np.linalg.inv(z * I2 - ham)                 # resolvent (zI-H)^-1
    ev = np.linalg.eigvalsh(ham)
    print(f" Paulis & Hadamard self-inverse, det=-1: "
          f"{ok(all(np.allclose(A @ A, I2) and np.isclose(np.linalg.det(A), -1) for A in (X, Y, Z, H)))}")
    print(f" U unitary, U^-1 = U†: {ok(Operator(U).is_unitary() and np.allclose(np.linalg.inv(U), U.conj().T))}")
    print(f" Hermitian H invertible iff all eigenvalues != 0 (eigs {np.round(ev,3)}): {ok(invertible(ham))}")
    print(f" Projector P: P²=P, not invertible: {ok(np.allclose(P @ P, P) and not invertible(P))}")
    print(f" Pure rho not invertible, mixed rho=I/2 invertible (rho^-1=2I): "
          f"{ok(not invertible(rho_pure) and np.allclose(np.linalg.inv(rho_mix), 2 * I2))}")
    print(f" Resolvent exists for z not in spec(H): {ok(invertible(z * I2 - ham) and np.allclose(G @ (z * I2 - ham), I2))}")


# ---------------- PART 2: Kähler core ----------------
def qgt(build, params, h=1e-5):
    """Q_ij = <di psi|dj psi> - <di psi|psi><psi|dj psi> (gauge invariant)."""
    p = np.array(params, float)
    sv = lambda q: Statevector(build(q)).data
    psi = sv(p)
    d = []
    for i in range(len(p)):
        e = np.zeros(len(p)); e[i] = h
        d.append((sv(p + e) - sv(p - e)) / (2 * h))
    n = len(p)
    return np.array([[np.vdot(d[i], d[j]) - np.vdot(d[i], psi) * np.vdot(psi, d[j])
                      for j in range(n)] for i in range(n)])


def one_qubit(q):
    qc = QuantumCircuit(1); qc.ry(q[0], 0); qc.rz(q[1], 0); return qc


def two_qubit(q):
    qc = QuantumCircuit(2)
    qc.ry(q[0], 0); qc.ry(q[1], 1); qc.cx(0, 1); qc.rz(q[2], 0); qc.ry(q[3], 1)
    return qc


def part2(th=1.0, ph=0.7):
    print("PART 2: Kähler (Fubini-Study / CP^n)")
    Q = qgt(one_qubit, [th, ph]); g, F = Q.real, -2 * Q.imag
    g_exact = np.diag([0.25, np.sin(th) ** 2 / 4])
    print(f" CP^1 metric = diag(1/4, sin²θ/4): {ok(np.allclose(g, g_exact, atol=1e-6))}")
    # complex structure J on (∂θ,∂φ): J∂θ = (1/sinθ)∂φ, J∂φ = -sinθ ∂θ
    s = np.sin(th)
    J = np.array([[0, -s], [1 / s, 0]])             # columns = images of ∂θ, ∂φ
    print(f" J²=-1: {ok(np.allclose(J @ J, -np.eye(2)))} | g(Ju,Jv)=g(u,v): {ok(np.allclose(J.T @ g @ J, g, atol=1e-6))}")
    omega = g @ J                                    # omega(u,v) = g(Ju,v)^T-conv
    print(f" |Kähler form| matches |Im Q| (sinθ/4): {ok(np.isclose(abs(omega[0, 1]), abs(Q.imag[0, 1]), atol=1e-6))}")
    Q2 = qgt(two_qubit, [0.9, 0.5, 0.3, 1.1])
    ev = np.linalg.eigvalsh(Q2.real)
    print(f" 2-qubit (CP^3) metric PSD: {ok(ev.min() > -1e-8)} | F antisymmetric: {ok(np.allclose(Q2.imag, -Q2.imag.T, atol=1e-8))}")


# ---------------- PART 3: hyperkähler core ----------------
def real4(f):
    """Real 4x4 matrix of a map on C^2 in basis (x1,y1,x2,y2)."""
    B = np.eye(4); cols = []
    for e in B:
        v = np.array([e[0] + 1j * e[1], e[2] + 1j * e[3]])
        w = f(v); cols.append([w[0].real, w[0].imag, w[1].real, w[1].imag])
    return np.array(cols).T


def part3():
    print("PART 3: hyperkähler (C^2 = H)")
    Y = Operator(Pauli("Y")).data
    I = real4(lambda v: 1j * v)
    J = real4(lambda v: -1j * Y @ v.conj())          # antilinear quaternionic j
    K = I @ J
    E = np.eye(4)
    print(f" I²=J²=K²=-1: {ok(all(np.allclose(A @ A, -E) for A in (I, J, K)))}")
    print(f" IJ=K, JK=I, KI=J: {ok(np.allclose(I @ J, K) and np.allclose(J @ K, I) and np.allclose(K @ I, J))}")
    print(f" I,J,K orthogonal (preserve flat metric): {ok(all(np.allclose(A.T @ A, E) for A in (I, J, K)))}")
    print(f" Kähler forms omega_a = g(a·,·) antisymmetric: {ok(all(np.allclose(A, -A.T) for A in (I, J, K)))}")
    n = np.array([0.3, -0.5, 0.81]); n = n / np.linalg.norm(n)
    M = n[0] * I + n[1] * J + n[2] * K
    print(f" every n·(I,J,K) is a complex structure (twistor S²): {ok(np.allclose(M @ M, -E))}")
    # same triple from Pauli unitaries on one qubit: -i sigma_a
    X, Yp, Z = (Operator(Pauli(s)).data for s in "XYZ")
    Iq, Jq, Kq = -1j * X, -1j * Yp, -1j * Z
    print(f" -iσ triple is quaternion algebra: {ok(np.allclose(Iq @ Jq, Kq) and np.allclose(Iq @ Iq, -I2))}")


if __name__ == "__main__":
    part1(); part2(); part3()
