"""Reader for POSCAR/CONTCAR: lattice and per-ion element symbols."""

from __future__ import annotations

import numpy as np


def read_poscar(path) -> dict:
    """Return ``lattice`` (3x3, Angstrom), ``reciprocal_lattice`` (incl. 2*pi)
    and ``ion_symbols`` (one entry per ion; empty if the file has no species line)."""
    with open(path) as f:
        lines = f.readlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:  # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(lattice))) ** (1 / 3)
    lattice *= scale

    tokens = lines[5].split()
    symbols: list[str] = []
    if tokens and not tokens[0].isdigit():
        counts = [int(x) for x in lines[6].split()]
        for sym, n in zip(tokens, counts):
            symbols += [sym.split("/")[0].split("_")[0]] * n
    return {
        "lattice": lattice,
        "reciprocal_lattice": 2 * np.pi * np.linalg.inv(lattice).T,
        "ion_symbols": symbols,
    }
