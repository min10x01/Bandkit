"""Reader for the VASP PROCAR file (LORBIT = 10, 11, 12)."""

from __future__ import annotations

import re

import numpy as np

_FLOAT = re.compile(r"-?\d+\.\d+(?:[Ee][+-]?\d+)?")
_INT = re.compile(r"\d+")


def read_procar(path) -> dict:
    """Parse a PROCAR file.

    Returns a dict with

    * ``kpoints`` (nk, 3), ``weights`` (nk,)
    * ``energies`` and ``occupations`` (nspin, nk, nb)
    * ``projections`` (nspin, nk, nb, nions, norb)
    * ``spin_texture`` (nk, nb, 3) for non-collinear runs, else None
    * ``orbital_names``, ``noncollinear``

    Phase information (LORBIT=12) is skipped.
    """
    spin_sets = []
    orbital_names: list[str] | None = None
    current = None
    collecting = False
    ion_headers = 0

    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            c = s[0]
            if c.isdigit():
                if collecting:
                    current["rows"].append(s)
            elif s.startswith("tot"):
                continue
            elif s.startswith("band"):
                vals = _FLOAT.findall(s)
                current["energies"].append(float(vals[0]))
                current["occ"].append(float(vals[1]) if len(vals) > 1 else np.nan)
                ion_headers = 0
                collecting = False
            elif s.startswith("ion"):
                ion_headers += 1
                collecting = ion_headers == 1  # a second header starts the phase block
                if orbital_names is None:
                    orbital_names = s.split()[1:-1]
            elif s.startswith("k-point"):
                vals = _FLOAT.findall(s.split(":", 1)[1])
                current["kpoints"].append([float(v) for v in vals[:3]])
                current["weights"].append(float(vals[3]) if len(vals) > 3 else 0.0)
            elif s.startswith("# of k-points"):
                nk, nb, nions = (int(x) for x in _INT.findall(s)[:3])
                current = {"nk": nk, "nb": nb, "nions": nions, "kpoints": [], "weights": [],
                           "energies": [], "occ": [], "rows": []}
                spin_sets.append(current)

    if not spin_sets or orbital_names is None:
        raise ValueError(f"{path}: no projection data found")

    nk, nb, nions = spin_sets[0]["nk"], spin_sets[0]["nb"], spin_sets[0]["nions"]
    norb = len(orbital_names)
    ncols = norb + 2  # ion index, orbitals, tot

    projections, energies, occupations = [], [], []
    spin_texture = None
    noncollinear = False
    for sset in spin_sets:
        data = np.array(" ".join(sset["rows"]).split(), dtype=float)
        nblocks = data.size // (nk * nb * nions * ncols)
        if nblocks * nk * nb * nions * ncols != data.size:
            raise ValueError(f"{path}: unexpected number of projection entries")
        data = data.reshape(nk, nb, nblocks, nions, ncols)[..., 1:-1]
        projections.append(data[:, :, 0])
        if nblocks == 4:
            noncollinear = True
            spin_texture = data[:, :, 1:4].sum(axis=(3, 4))
        energies.append(np.array(sset["energies"]).reshape(nk, nb))
        occupations.append(np.array(sset["occ"]).reshape(nk, nb))

    return {
        "kpoints": np.array(spin_sets[0]["kpoints"]),
        "weights": np.array(spin_sets[0]["weights"]),
        "energies": np.array(energies),
        "occupations": np.array(occupations),
        "projections": np.array(projections),
        "spin_texture": spin_texture,
        "orbital_names": orbital_names,
        "noncollinear": noncollinear,
    }
