"""Minimal OUTCAR reader: Fermi energy, reciprocal lattice and run settings."""

from __future__ import annotations

import numpy as np


def read_outcar(path) -> dict:
    """Return ``efermi``, ``reciprocal_lattice`` (1/Angstrom, incl. 2*pi),
    ``nelect``, ``ispin`` and ``noncollinear`` (last value found wins)."""
    out = {"efermi": None, "reciprocal_lattice": None, "nelect": None,
           "ispin": None, "noncollinear": None}
    with open(path) as f:
        lines = iter(f)
        for line in lines:
            if "E-fermi" in line:
                try:
                    out["efermi"] = float(line.split()[2])
                except (IndexError, ValueError):
                    pass
            elif "reciprocal lattice vectors" in line:
                rows = [next(lines).split() for _ in range(3)]
                try:
                    rec = np.array([[float(x) for x in r[3:6]] for r in rows])
                    out["reciprocal_lattice"] = 2 * np.pi * rec
                except (IndexError, ValueError):
                    pass
            elif "NELECT" in line and "=" in line:
                try:
                    out["nelect"] = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif line.strip().startswith("ISPIN") and "=" in line:
                try:
                    out["ispin"] = int(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif "LNONCOLLINEAR" in line and "=" in line:
                out["noncollinear"] = line.split("=")[1].split()[0].upper().startswith("T")
    return out
