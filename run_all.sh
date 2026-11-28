#!/usr/bin/env bash
set -e
python -c "import qiskit_algorithms, gdstk, shapely, matplotlib" 2>/dev/null || pip install -q -r requirements.txt
python -W ignore kahler_hyperkahler_qiskit.py
python -W ignore hyperkahler_unit.py
python -W ignore chipset.py
python -W ignore stage1_budget.py
python -W ignore stage2_quantization.py
python -W ignore stage3_layout.py
python -W ignore stage3b_layout.py
python -u -W ignore stage4_extract.py
