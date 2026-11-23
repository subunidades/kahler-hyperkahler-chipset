#!/usr/bin/env bash
set -e
python -c "import qiskit, scipy" 2>/dev/null || pip install -q -r requirements.txt
python kahler_hyperkahler_qiskit.py
python hyperkahler_unit.py
python chipset.py
