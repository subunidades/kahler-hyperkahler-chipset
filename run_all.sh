#!/usr/bin/env bash
set -e
python kahler_hyperkahler_qiskit.py
python hyperkahler_unit.py
python chipset.py
