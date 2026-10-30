"""Reader for the VASP EIGENVAL file."""

from __future__ import annotations

import numpy as np


def read_eigenval(path) -> dict:
    """Return k-points, weights, energies ``(nspin, nk, nb)`` and occupations.

    Handles ISPIN=1/2 and files written with or without occupations.
    """
    with open(path) as f:
        lines = f.readlines()

    ispin = int(lines[0].split()[3])
    nelect, nkpts, nbands = (float(x) for x in lines[5].split()[:3])
    nkpts, nbands = int(nkpts), int(nbands)

    kpoints = np.zeros((nkpts, 3))
    weights = np.zeros(nkpts)
    energies = np.zeros((ispin, nkpts, nbands))
    occupations = np.zeros((ispin, nkpts, nbands))
    has_occ = True

    i = 6
    for k in range(nkpts):
        while not lines[i].strip():
            i += 1
        vals = [float(x) for x in lines[i].split()]
        kpoints[k] = vals[:3]
        weights[k] = vals[3] if len(vals) > 3 else 0.0
        i += 1
        for b in range(nbands):
            cols = lines[i].split()
            i += 1
            if ispin == 1:
                energies[0, k, b] = float(cols[1])
                if len(cols) >= 3:
                    occupations[0, k, b] = float(cols[2])
                else:
                    has_occ = False
            else:
                energies[0, k, b] = float(cols[1])
                energies[1, k, b] = float(cols[2])
                if len(cols) >= 5:
                    occupations[0, k, b] = float(cols[3])
                    occupations[1, k, b] = float(cols[4])
                else:
                    has_occ = False

    return {
        "kpoints": kpoints,
        "weights": weights,
        "energies": energies,
        "occupations": occupations if has_occ else None,
        "nelect": nelect,
        "ispin": ispin,
    }
