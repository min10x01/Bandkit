"""Spin / band splitting: Rashba parameters, Zeeman-type and exchange splittings.

Rashba model for a pair of spin-split branches around a degenerate point k_c
(usually a time-reversal invariant momentum, TRIM):

    E_±(k) = E_c + hbar^2 k^2 / 2m* ± alpha_R |k|

The lower (upper, for valence bands) branch has its extremum shifted by the
momentum offset k0 with the Rashba energy E_R, and alpha_R = 2 E_R / k0.
A second estimate, ``alpha_linear``, comes from the small-k slope of the
splitting, Delta E(k) ≈ 2 alpha_R k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import BandStructure
from .edges import find_band_edges


@dataclass
class RashbaSplitting:
    bands: tuple[int, int]      # (lower, upper) branch
    spin: int
    kind: str                   # 'conduction' or 'valence'
    crossing_index: int
    crossing_label: str
    direction: str
    is_trim: bool
    degeneracy: float           # splitting at the crossing point (eV), ~0 for Rashba
    extremum_index: int         # grid point closest to the shifted extremum
    k0: float                   # momentum offset (1/Angstrom)
    E_R: float                  # Rashba energy (eV)
    alpha: float                # 2 E_R / k0 (eV*Angstrom)
    alpha_linear: float         # from the small-k splitting slope (eV*Angstrom)
    x_crossing: float           # positions on the plotted path
    x_extremum: float
    E_crossing: float
    E_extremum: float
    at_segment_end: bool = False   # extremum not reached inside the segment

    def __str__(self):
        if self.k0 == 0:
            return (f"{self.direction:<10s} no momentum offset (alpha_linear = "
                    f"{self.alpha_linear:.3f} eV·Å)")
        flag = "  [extremum at segment end]" if self.at_segment_end else ""
        return (f"{self.direction:<10s} E_R = {self.E_R * 1e3:7.2f} meV  k0 = {self.k0:.4f} 1/Å  "
                f"alpha_R = {self.alpha:.3f} eV·Å  (linear fit: {self.alpha_linear:.3f}){flag}")


@dataclass
class SplittingCharacter:
    bands: tuple[int, int]
    kindex: int
    label: str
    delta_E: float              # eV
    kind: str                   # 'degenerate', 'Rashba-type', 'Zeeman-type', ...
    detail: str = ""

    def __str__(self):
        text = f"{self.label:<8s} ΔE = {self.delta_E * 1e3:8.2f} meV  {self.kind}"
        return text + (f" ({self.detail})" if self.detail else "")


def band_splitting(bs: BandStructure, bands: tuple[int, int], spin: int = 0) -> np.ndarray:
    """Energy difference E_upper - E_lower along the whole path (eV)."""
    lo, hi = bands
    return bs.energies[spin, :, hi] - bs.energies[spin, :, lo]


def splitting_at_high_symmetry(bs: BandStructure, bands: tuple[int, int],
                               spin: int = 0) -> list[tuple[int, str, float]]:
    """Splitting of a band pair at every labelled k-point (each point once)."""
    dE = band_splitting(bs, bands, spin)
    out, seen = [], []
    for k in sorted(bs.labels):
        if any(np.allclose(bs.kpoints[k], bs.kpoints[j], atol=1e-6) for j in seen):
            continue
        seen.append(k)
        out.append((k, bs.label(k), float(dE[k])))
    return out


def exchange_splitting(bs: BandStructure, band: int, kindex: int) -> float:
    """Spin-down minus spin-up energy of a band (collinear ISPIN=2 runs)."""
    if bs.nspin != 2:
        raise ValueError("exchange splitting needs a spin-polarised (ISPIN=2) calculation")
    return float(bs.energies[1, kindex, band] - bs.energies[0, kindex, band])


def _vertex(s, f):
    """Sub-grid minimum of a parabola through three points."""
    a, b, c = np.polyfit(s, f, 2)
    if a <= 0:
        return None
    s0 = -b / (2 * a)
    if not s[0] <= s0 <= s[-1]:
        return None
    return s0, c - b * b / (4 * a)


def rashba_parameters(bs: BandStructure, bands: tuple[int, int], crossing_index: int,
                      kind: str | None = None, spin: int = 0,
                      n_linear: int = 3) -> list[RashbaSplitting]:
    """Rashba energy, momentum offset and alpha_R of a band pair around ``crossing_index``,
    along every path direction leaving that point.

    kind
        'conduction' (look for the minimum of the lower branch) or 'valence'
        (maximum of the upper branch). Guessed from the Fermi level if None.
    """
    lo, hi = bands
    E = bs.energies[spin]
    Ec = 0.5 * (E[crossing_index, lo] + E[crossing_index, hi])
    if kind is None:
        kind = "conduction" if Ec > bs.efermi else "valence"
    branch, sign = (lo, 1.0) if kind == "conduction" else (hi, -1.0)
    dE_all = E[:, hi] - E[:, lo]

    results = []
    for d in bs.directions_from(crossing_index):
        idx, dk = d.indices, d.dk
        f = sign * E[idx, branch]
        j = 0
        while j + 1 < len(idx) and f[j + 1] < f[j] - 1e-7:
            j += 1
        s0, f0 = dk[j], f[j]
        if 0 < j < len(idx) - 1:
            v = _vertex(dk[j - 1:j + 2], f[j - 1:j + 2])
            if v is not None:
                s0, f0 = v
        E_ext = sign * f0
        E_R = sign * (Ec - E_ext) if j > 0 else 0.0
        k0 = float(s0) if j > 0 else 0.0
        alpha = 2 * E_R / k0 if k0 > 0 else np.nan

        m = min(n_linear, len(idx) - 1)
        ds, dd = dk[1:m + 1], dE_all[idx[1:m + 1]] - dE_all[idx[0]]
        alpha_lin = float(ds @ dd / (2 * ds @ ds)) if m > 0 else np.nan

        step = 1 if len(idx) > 1 and idx[1] > idx[0] else -1
        x0 = bs.distances[idx[0]]
        results.append(RashbaSplitting(
            bands=(lo, hi), spin=spin, kind=kind, crossing_index=int(idx[0]),
            crossing_label=bs.label(int(idx[0])), direction=d.label,
            is_trim=bs.is_trim(int(idx[0])), degeneracy=float(dE_all[idx[0]]),
            extremum_index=int(idx[j]), k0=k0, E_R=float(E_R), alpha=float(alpha),
            alpha_linear=alpha_lin, x_crossing=float(x0), x_extremum=float(x0 + step * k0),
            E_crossing=float(Ec), E_extremum=float(E_ext) if j > 0 else float(Ec),
            at_segment_end=j == len(idx) - 1 and j > 0,
        ))
    return results


def find_crossing(bs: BandStructure, bands: tuple[int, int], near: int, spin: int = 0,
                  tol: float = 2e-3) -> int | None:
    """Degenerate point (splitting < tol) of a band pair closest to k-index ``near``,
    preferring time-reversal invariant momenta. None if the pair never meets."""
    dE = np.abs(band_splitting(bs, bands, spin))
    candidates = np.flatnonzero(dE < tol)
    if candidates.size == 0:
        return None
    key = [(abs(bs.distances[c] - bs.distances[near]), not bs.is_trim(int(c))) for c in candidates]
    return int(candidates[min(range(len(key)), key=key.__getitem__)])


def edge_pair(bs: BandStructure, edge: str = "cbm") -> tuple[tuple[int, int], int, int, str] | None:
    """Band pair ``(lower, upper)``, k-index and spin of the CBM or VBM, plus its kind."""
    edges = find_band_edges(bs)
    e = edges.cbm if edge == "cbm" else edges.vbm
    if e is None:
        return None
    if edge == "cbm":
        pair, kind = (e.band, e.band + 1), "conduction"
    else:
        pair, kind = (e.band - 1, e.band), "valence"
    if pair[0] < 0 or pair[1] >= bs.nbands:
        return None
    return pair, e.kindex, e.spin, kind


def find_rashba(bs: BandStructure, edge: str = "cbm", tol: float = 2e-3,
                n_linear: int = 3) -> list[RashbaSplitting]:
    """Rashba analysis of the band pair at the CBM (``edge='cbm'``) or VBM.

    Returns an empty list if the pair is (Kramers) degenerate along most of
    the path (no spin splitting) or never becomes degenerate (no crossing
    point, e.g. Zeeman-type splitting — see :func:`classify_splitting`).
    """
    found = edge_pair(bs, edge)
    if found is None:
        return []
    pair, kindex, spin, kind = found
    dE = np.abs(band_splitting(bs, pair, spin))
    if np.mean(dE < tol) > 0.8:
        return []
    crossing = find_crossing(bs, pair, kindex, spin, tol)
    if crossing is None:
        return []
    return rashba_parameters(bs, pair, crossing, kind, spin, n_linear)


def classify_splitting(bs: BandStructure, bands: tuple[int, int], kindex: int, spin: int = 0,
                       normal=(0.0, 0.0, 1.0), tol: float = 2e-3) -> SplittingCharacter:
    """Heuristic classification of the splitting of a band pair at one k-point.

    With a spin texture (non-collinear PROCAR / vasprun.xml):
      * out-of-plane spin (along ``normal``)            -> Zeeman-type
      * in-plane spin perpendicular to k - k_c          -> Rashba-type
      * in-plane spin parallel to k - k_c               -> Dresselhaus-type
    where k_c is the nearest degenerate TRIM on the same segment.
    Without spin information the decision uses only where the splitting
    vanishes: a k-linear splitting from a degenerate TRIM is Rashba/Dresselhaus
    like, a finite splitting at a TRIM needs broken time-reversal symmetry.
    """
    lo, hi = bands
    dE_all = band_splitting(bs, bands, spin)
    dE = float(dE_all[kindex])
    label = bs.label(kindex)
    if abs(dE) < tol:
        return SplittingCharacter(bands, kindex, label, dE, "degenerate")

    seg = bs.segment_of(kindex)
    s, e = bs.segments[seg]
    ref = None
    for c in range(s, e + 1):
        if abs(dE_all[c]) < tol and bs.is_trim(c):
            if ref is None or abs(c - kindex) < abs(ref - kindex):
                ref = c

    if bs.spin_texture is not None and bs.nspin == 1:
        n = np.asarray(normal, float)
        n /= np.linalg.norm(n)
        S = bs.spin_texture[kindex, lo]
        if np.linalg.norm(S) < 1e-6:
            S = bs.spin_texture[kindex, hi]
        norm = np.linalg.norm(S)
        if norm < 1e-6:
            return SplittingCharacter(bands, kindex, label, dE, "unpolarised", "spin texture ~ 0")
        S = S / norm
        s_out = abs(S @ n)
        if s_out > 0.7:
            return SplittingCharacter(bands, kindex, label, dE, "Zeeman-type",
                                      f"out-of-plane spin, |s·n| = {s_out:.2f}")
        if ref is None or ref == kindex:
            return SplittingCharacter(bands, kindex, label, dE, "in-plane spin splitting",
                                      "no degenerate TRIM on this segment for reference")
        q = bs.kcart[kindex] - bs.kcart[ref]
        q_par = q - (q @ n) * n
        s_par = S - (S @ n) * n
        cos = abs(s_par @ q_par) / (np.linalg.norm(s_par) * np.linalg.norm(q_par))
        detail = f"angle(s, k-{bs.label(ref)}) = {np.degrees(np.arccos(min(cos, 1.0))):.0f}°"
        if cos < 0.35:
            kind = "Rashba-type"
        elif cos > 0.85:
            kind = "Dresselhaus-type"
        else:
            kind = "mixed Rashba/Dresselhaus"
        return SplittingCharacter(bands, kindex, label, dE, kind, detail)

    if bs.is_trim(kindex):
        return SplittingCharacter(bands, kindex, label, dE, "Zeeman/exchange-like",
                                  "finite splitting at a TRIM: time-reversal symmetry broken")
    if ref is not None:
        return SplittingCharacter(bands, kindex, label, dE, "Rashba/Dresselhaus-like",
                                  f"splitting vanishes at {bs.label(ref)}; spin texture needed "
                                  "to tell Rashba from Dresselhaus")
    return SplittingCharacter(bands, kindex, label, dE, "Zeeman-like",
                              "finite splitting away from TRIMs (valley/Zeeman-type)")
