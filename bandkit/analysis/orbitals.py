"""Orbital / atomic contributions from PROCAR or vasprun.xml projections."""

from __future__ import annotations

import numpy as np

from ..core import BandStructure

_ALIASES = {"dx2-y2": "x2-y2", "dx2": "x2-y2", "x2": "x2-y2", "z2": "dz2"}


def orbital_group(name: str) -> str:
    """'s', 'p', 'd' or 'f' for a VASP orbital column name (py, dz2, x2-y2, fxyz, ...)."""
    name = name.lower()
    if name == "x2-y2":
        return "d"
    return name[0] if name and name[0] in "spdf" else name


def _require(bs: BandStructure):
    if bs.projections is None:
        raise ValueError("no orbital projections: run VASP with LORBIT=11 (PROCAR / vasprun.xml)")


def resolve_ions(bs: BandStructure, atoms=None) -> list[int]:
    """0-based ion indices from an element symbol, an index, or a list of them."""
    _require(bs)
    if atoms is None:
        return list(range(bs.nions))
    if isinstance(atoms, (str, int, np.integer)):
        atoms = [atoms]
    out: list[int] = []
    for a in atoms:
        if isinstance(a, (int, np.integer)):
            if not 0 <= a < bs.nions:
                raise IndexError(f"ion index {a} out of range (0..{bs.nions - 1})")
            out.append(int(a))
        else:
            hits = [i for i, sym in enumerate(bs.ion_symbols) if sym == a]
            if not hits:
                raise ValueError(f"no ions of element {a!r} (have {sorted(set(bs.ion_symbols))})")
            out += hits
    return sorted(set(out))


def resolve_orbitals(bs: BandStructure, orbitals=None) -> list[int]:
    """Orbital column indices from names ('px', 'dz2') or groups ('s', 'p', 'd', 'f')."""
    _require(bs)
    names = [n.lower() for n in bs.orbital_names]
    if orbitals is None:
        return list(range(len(names)))
    if isinstance(orbitals, str):
        orbitals = [orbitals]
    out: list[int] = []
    for o in orbitals:
        o = _ALIASES.get(o.lower(), o.lower())
        if o in names:
            out.append(names.index(o))
        elif o in ("s", "p", "d", "f"):
            hits = [i for i, n in enumerate(names) if orbital_group(n) == o]
            if not hits:
                raise ValueError(f"no {o} orbitals in projections ({bs.orbital_names})")
            out += hits
        else:
            raise ValueError(f"unknown orbital {o!r} (have {bs.orbital_names})")
    return sorted(set(out))


def parse_projection(spec: str) -> tuple[list | None, list | None]:
    """Parse 'Bi:p', 'Bi', ':d', 'Bi,Te:px,py' or '1-3:s' (1-based ion numbers)
    into (atoms, orbitals) for :func:`projected_weight`."""
    atom_part, _, orb_part = spec.partition(":")
    atoms: list | None = None
    if atom_part.strip():
        atoms = []
        for tok in atom_part.split(","):
            tok = tok.strip()
            if tok.replace("-", "").isdigit():
                a, _, b = tok.partition("-")
                atoms += list(range(int(a) - 1, int(b or a)))
            else:
                atoms.append(tok)
    orbitals = [t.strip() for t in orb_part.split(",") if t.strip()] or None
    return atoms, orbitals


def projected_weight(bs: BandStructure, atoms=None, orbitals=None,
                     normalize: bool = True) -> np.ndarray:
    """Summed projection on the selected ions/orbitals, shape (nspin, nk, nb).

    With ``normalize`` the result is the fraction of the total projected
    weight of each state (0..1); otherwise the raw PROCAR values.
    """
    ions = resolve_ions(bs, atoms)
    orbs = resolve_orbitals(bs, orbitals)
    P = bs.projections
    w = P[:, :, :, ions][..., orbs].sum(axis=(3, 4))
    if normalize:
        total = P.sum(axis=(3, 4))
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 1e-12)
    return w


def band_character(bs: BandStructure, band: int, kindex: int, spin: int = 0,
                   by: str = "element-orbital", threshold: float = 0.0) -> dict[str, float]:
    """Fractional contributions to one state, largest first.

    by
        'element', 'orbital' (s/p/d/f), 'element-orbital' (e.g. 'Bi-p'),
        'element-lm' (e.g. 'Bi-pz') or 'ion' (e.g. 'Bi1').
    """
    _require(bs)
    P = bs.projections[spin, kindex, band]          # (nions, norb)
    total = P.sum()
    if total <= 0:
        return {}
    P = P / total
    out: dict[str, float] = {}
    for i, sym in enumerate(bs.ion_symbols):
        for j, orb in enumerate(bs.orbital_names):
            if by == "element":
                key = sym
            elif by == "orbital":
                key = orbital_group(orb)
            elif by == "element-orbital":
                key = f"{sym}-{orbital_group(orb)}"
            elif by == "element-lm":
                key = f"{sym}-{orb}"
            elif by == "ion":
                key = f"{sym}{i + 1}"
            else:
                raise ValueError(f"unknown grouping {by!r}")
            out[key] = out.get(key, 0.0) + float(P[i, j])
    return {k: v for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v >= threshold and v > 0}


def format_character(character: dict[str, float], n: int = 4) -> str:
    shown = [(k, v) for k, v in character.items() if v >= 0.005][:n]
    return ", ".join(f"{k} {100 * v:.0f}%" for k, v in shown)
