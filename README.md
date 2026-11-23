# Kähler / Hyperkähler Quantum Chipset

A pre-layout design model, in Qiskit, for a two-die superconducting chipset built from the
quantum-mechanics operator classes: unitary, Hermitian, projector, density matrix,
resolvent, Pauli and Hadamard.

| Die | Hardware | Geometry |
|---|---|---|
| **Kähler die** | 2 flux-tunable transmons, 1 coupler | Pure-state space CP³ with the Fubini–Study metric and Kähler form |
| **Hyperkähler die** | 3 transmons in a triangle, SU(2)-symmetric Heisenberg coupling | Quaternion triple I, J, K from -iσ; sphere of complex structures |

## Run it from an iPad or iPhone (no folders, no hidden files)

1. **Unzip:** in the Files app, tap the zip. A folder appears with 6 files.
2. **Make a repo:** in Safari go to github.com, sign in, tap **+ → New repository**. Name it, leave "Add a README" **unchecked**, tap **Create repository**.
3. **Upload:** on the empty repo page tap **uploading an existing file**, tap **choose your files**, select all 6 files from the unzipped folder (tap **Select**, then select all), then tap **Commit changes**.
4. **Open a Codespace:** on the repo page tap **Code → Codespaces → Create codespace on main**. If the buttons look cramped, use Safari’s **aA → Request Desktop Website**. Wait a minute for the editor to load.
5. **Open the terminal:** tap the **☰ menu** (top left) **→ Terminal → New Terminal**.
6. **Run everything with one command:**

       bash run_all.sh

   It installs the requirements, then runs all three files. A healthy run ends with 30 `PASS` and 0 `FAIL`, and writes `chipset_spec.json`.

To run one file only: `python chipset.py`, `python hyperkahler_unit.py` or `python kahler_hyperkahler_qiskit.py` (run `pip install -r requirements.txt` once first).

Codespaces stop when idle and your files stay in the repo. Reopen from **Code → Codespaces**.

## Run on a computer

    pip install -r requirements.txt
    bash run_all.sh

## Files

| File | Contents |
|---|---|
| `kahler_hyperkahler_qiskit.py` | **Part 1** operator zoo (invertibility of each class). **Part 2** Kähler: quantum geometric tensor Q = g − (i/2)F from any circuit; checks the CP¹ metric diag(1/4, sin²θ/4), complex structure J² = −1, metric compatibility, and a 2-qubit CP³ case. **Part 3** hyperkähler on C² = ℍ: I = i, J = −iσ_y∘conj, K = IJ, orthogonality, Kähler forms, sphere of complex structures. |
| `hyperkahler_unit.py` | 3-qubit H = J·Σ(XX+YY+ZZ). Checks [H, S_a] = 0 and [H, S²] = 0, spectrum (+3J ×4, −3J ×4), native-gate circuit commuting with collective SU(2) rotations, Trotter-error scaling, and metric = spin variance along the symmetry orbit. |
| `chipset.py` | Native-gate identity (Heisenberg pair gate = XY(4θ)·ZZ(2θ)), 3-level transmon exchange-gate simulation (gate time, leakage, fidelity after CPhase correction), SU(2)-breaking from frequency disorder, Kähler-die geometry from native gates. Writes `chipset_spec.json`. |
| `run_all.sh` | Runs all three in order. |
| `requirements.txt` | qiskit, numpy, scipy. |

## Background

- Single-qubit pure states form CP¹ (the Bloch sphere), a Kähler manifold. Re(Q) is the
  Fubini–Study metric; Im(Q) gives the Kähler form (Berry curvature).
- The Pauli matrices give quaternion units (-iσ_x)(-iσ_y) = -iσ_z, the structure a
  hyperkähler manifold needs (three complex structures I, J, K with IJ = K).
- The hyperkähler die realises the SU(2) symmetry with native gates: XY exchange plus a ZZ phase.

## Status and caveats

- Simulation and design model only. **No layout (GDS) yet** and nothing has been fabricated.
- All frequencies, couplings and anharmonicities in `chipset.py` are **illustrative
  placeholders**. Replace them with the chosen foundry process data.
- Gate numbers (about 16.7 ns, about 0.8% leakage, about 99.2% corrected fidelity) come
  from an idealised 3-level model with no decoherence, flux noise or crosstalk.
- Next steps: choose a foundry, lay out against its design rules, then EM simulation.
