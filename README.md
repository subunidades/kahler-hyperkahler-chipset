# Kähler / Hyperkähler Quantum Chipset

A pre-layout design model, in Qiskit, for a two-die superconducting chipset built from the
quantum-mechanics operator classes: unitary, Hermitian, projector, density matrix,
resolvent, Pauli and Hadamard.

| Die | Hardware | Geometry |
|---|---|---|
| **Kähler die** | 2 flux-tunable transmons, 1 coupler | Pure-state space CP³ with the Fubini–Study metric and Kähler form |
| **Hyperkähler die** | 3 transmons in a triangle, SU(2)-symmetric Heisenberg coupling | Quaternion triple I, J, K from -iσ; sphere of complex structures |

## Run it from an iPad or iPhone (no folders, no hidden files)

1. **Unzip:** in the Files app, tap the zip. A folder appears with 7 files.
2. **Make a repo:** in Safari go to github.com, sign in, tap **+ → New repository**. Name it, leave "Add a README" **unchecked**, tap **Create repository**.
3. **Upload:** on the empty repo page tap **uploading an existing file**, tap **choose your files**, select all 7 files from the unzipped folder (tap **Select**, then select all), then tap **Commit changes**.
4. **Open a Codespace:** on the repo page tap **Code → Codespaces → Create codespace on main**. If the buttons look cramped, use Safari’s **aA → Request Desktop Website**. Wait a minute for the editor to load.
5. **Open the terminal:** tap the **☰ menu** (top left) **→ Terminal → New Terminal**.
6. **Run everything with one command:**

       bash run_all.sh

   It installs the requirements, then runs all three files. A healthy run ends with 33 lines containing `PASS` and none containing `FAIL`, and writes `chipset_spec.json`.

To run one file only: `python chipset.py`, `python hyperkahler_unit.py` or `python kahler_hyperkahler_qiskit.py` (run `pip install -r requirements.txt` once first).

Codespaces stop when idle and your files stay in the repo. Reopen from **Code → Codespaces**.

## Run on a computer

    pip install -r requirements.txt
    bash run_all.sh

## Files

| File | Contents |
|---|---|
| `kahler_hyperkahler_qiskit.py` | **Part 1** operator zoo with `Operator`, `SparsePauliOp`, `PauliEvolutionGate`, `NumPyEigensolver`. **Part 2** Kähler: quantum geometric tensor from `qiskit_algorithms` `ReverseQGT`; CP¹ metric, complex structure, Kähler form, 2-qubit CP³. **Part 3** hyperkähler: C² = ℍ realified on 2 qubits, with I, J, K as Pauli-product `Operator`s. |
| `hyperkahler_unit.py` | 3-qubit H = J·Σ(XX+YY+ZZ). Symmetry via `SparsePauliOp` algebra, spectrum, native-gate circuit vs exact `PauliEvolutionGate` (`process_fidelity`), geometry along the symmetry orbit with `ReverseQGT`. |
| `chipset.py` | A Qiskit `Target` per die (frequencies, native rz/sx/x/cz, coupling graph); `transpile` to native gates verified with `Operator.equiv`; 3-level transmon exchange gate (Qiskit Dynamics if installed, otherwise a matrix exponential, same results); SU(2) disorder tolerance; Kähler-die geometry. Writes `chipset_spec.json`. |
| `run_all.sh` | Runs all three in order (installs requirements first if missing). |
| `requirements.txt` | qiskit >=1.3,<3 (tested on 1.3.0 and 2.6.2), qiskit-algorithms 0.4.0, numpy, scipy. |
| `requirements-dynamics.txt` | Optional, separate environment: Qiskit 1.3.0 + Qiskit Dynamics 0.6.0 for the transmon gate. |

## Background

- Single-qubit pure states form CP¹ (the Bloch sphere), a Kähler manifold. Re(Q) is the
  Fubini–Study metric; Im(Q) gives the Kähler form (Berry curvature).
- The Pauli matrices give quaternion units (-iσ_x)(-iσ_y) = -iσ_z, the structure a
  hyperkähler manifold needs (three complex structures I, J, K with IJ = K).
- The hyperkähler die realises the SU(2) symmetry with native gates: XY exchange plus a ZZ phase.

## Status and caveats

- Simulation and design model only. **No layout (GDS) yet** and nothing has been fabricated;
  nothing runs on IBM hardware.
- **Tested on Qiskit 1.3.0 and 2.6.2** with identical results. Qiskit Dynamics needs qiskit <= 1.3, so it is optional: install `requirements-dynamics.txt` in a separate environment to use it.
- All frequencies, couplings, anharmonicities and gate errors are **illustrative
  placeholders**. Replace them with the chosen foundry process data.
- Gate numbers come from an idealised 3-level model with no decoherence, flux noise or crosstalk.
- Next steps: choose a foundry, lay out against its design rules, then EM simulation.
