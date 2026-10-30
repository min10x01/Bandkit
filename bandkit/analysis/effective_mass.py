"""Effective masses from parabolic fits along the k-path.

m* / m_e = hbar^2 / (m_e * d^2E/dk^2), with E in eV and k in 1/Angstrom.
Masses of hole-like (downward curving) bands come out negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import HBAR2_OVER_ME, BandStructure
from .edges import find_band_edges


@dataclass
class EffectiveMass:
    band: int
    spin: int
    kindex: int
    direction: str           # e.g. "Γ→X"
    mass: float              # in units of the free-electron mass
    curvature: float         # d^2E/dk^2 in eV*Angstrom^2
    indices: np.ndarray      # k-points used in the fit
    s: np.ndarray            # signed distance of those points from kindex (1/Angstrom)
    coeffs: np.ndarray       # polynomial coefficients E(s), highest power first
    rmse: float              # fit residual (eV)

    def __str__(self):
        return (f"m* = {self.mass:+.4f} m_e  along {self.direction:<10s} "
                f"(band {self.band + 1}, {len(self.indices)} pts, rmse {self.rmse * 1e3:.2f} meV)")


def _fit(bs, band, spin, kindex, direction, idx, s, degree):
    E = bs.energies[spin, idx, band]
    coeffs = np.polyfit(s, E, degree)
    curvature = 2 * coeffs[-3]
    rmse = float(np.sqrt(np.mean((np.polyval(coeffs, s) - E) ** 2)))
    mass = HBAR2_OVER_ME / curvature if curvature != 0 else np.inf
    return EffectiveMass(band, spin, kindex, direction, float(mass), float(curvature),
                         idx, s, coeffs, rmse)


def _window(direction, E0, energies, npoints, energy_window):
    """Number of points (including the origin) to use along one direction."""
    n = min(npoints, len(direction.indices))
    if energy_window is not None:
        dE = np.abs(energies[direction.indices] - E0)
        inside = np.flatnonzero(dE > energy_window)
        n = len(direction.indices) if inside.size == 0 else int(inside[0])
        n = max(n, 3)
        n = min(n, len(direction.indices))
    return n


def effective_mass(bs: BandStructure, band: int, kindex: int, spin: int = 0, npoints: int = 5,
                   energy_window: float | None = None, degree: int = 2) -> list[EffectiveMass]:
    """Effective mass of ``band`` at ``kindex`` along every path direction.

    At a high-symmetry point a one-sided fit is made into each adjacent
    segment. At a point inside a segment (e.g. an off-centre Rashba minimum)
    a symmetric fit over both sides is made.

    npoints
        Points per side, including the extremum (at least 3).
    energy_window
        Alternatively, include points within this energy (eV) of the extremum.
    degree
        Polynomial degree (2 = parabola; 4 can absorb non-parabolicity).
    """
    energies = bs.energies[spin, :, band]
    E0 = energies[kindex]
    dirs = bs.directions_from(kindex)
    results = []
    used_interior = set()
    for d in dirs:
        if d.interior:
            if (d.segment, int(d.indices[0])) in used_interior:
                continue
            partner = next((p for p in dirs if p is not d and p.interior and p.segment == d.segment
                            and p.indices[0] == d.indices[0]), None)
            used_interior.add((d.segment, int(d.indices[0])))
            nr = _window(d, E0, energies, npoints, energy_window)
            nl = _window(partner, E0, energies, npoints, energy_window) if partner else 1
            idx = np.concatenate([partner.indices[1:nl][::-1], d.indices[:nr]]) if partner else d.indices[:nr]
            s = np.concatenate([-partner.dk[1:nl][::-1], d.dk[:nr]]) if partner else d.dk[:nr]
            label = f"{bs.segment_name(d.segment)} @ {bs.label(kindex)}"
        else:
            n = _window(d, E0, energies, npoints, energy_window)
            idx, s, label = d.indices[:n], d.dk[:n], d.label
        if len(idx) < degree + 1:
            continue
        results.append(_fit(bs, band, spin, kindex, label, idx, s, degree))
    return results


def band_edge_masses(bs: BandStructure, npoints: int = 5, energy_window: float | None = None,
                     degree: int = 2) -> dict[str, list[EffectiveMass]]:
    """Electron masses at the CBM and hole masses at the VBM (all directions)."""
    edges = find_band_edges(bs)
    out = {"electron": [], "hole": []}
    if edges.cbm is not None:
        c = edges.cbm
        out["electron"] = effective_mass(bs, c.band, c.kindex, c.spin, npoints, energy_window, degree)
    if edges.vbm is not None:
        v = edges.vbm
        out["hole"] = effective_mass(bs, v.band, v.kindex, v.spin, npoints, energy_window, degree)
    return out
