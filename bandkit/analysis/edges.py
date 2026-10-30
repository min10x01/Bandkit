"""Band edges: valence-band maximum, conduction-band minimum and band gap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import BandStructure


@dataclass
class BandEdge:
    energy: float
    band: int
    kindex: int
    spin: int
    label: str

    def __str__(self):
        return f"{self.energy:.4f} eV (band {self.band + 1}, {self.label}, spin {self.spin})"


@dataclass
class BandEdges:
    vbm: BandEdge | None
    cbm: BandEdge | None
    gap: float
    direct: bool
    is_metal: bool

    def __str__(self):
        if self.is_metal:
            return "metallic (bands cross the Fermi level)"
        kind = "direct" if self.direct else "indirect"
        return f"VBM {self.vbm}\nCBM {self.cbm}\ngap {self.gap:.4f} eV ({kind})"


def _occupied_band_count(bs: BandStructure) -> int | None:
    """Number of filled bands from NELECT (only meaningful for nspin == 1)."""
    if bs.nelect is None or bs.nspin != 1:
        return None
    n = bs.nelect if bs.noncollinear else bs.nelect / 2
    if abs(n - round(n)) > 1e-6:
        return None
    return int(round(n))


def find_band_edges(bs: BandStructure, tol: float = 0.01, use_nelect: bool = True) -> BandEdges:
    """Locate the VBM and CBM.

    If NELECT is known (non-spin-polarised or SOC runs) the highest occupied
    band follows from the electron count. Otherwise a band is valence if it
    lies entirely below ``efermi + tol`` and conduction if entirely above
    ``efermi - tol``; any other band crosses the Fermi level (metal).
    """
    E = bs.energies
    nocc = _occupied_band_count(bs) if use_nelect else None

    if nocc is not None and 0 < nocc < bs.nbands:
        vb_mask = np.zeros(bs.nbands, bool)
        vb_mask[:nocc] = True
        valence = np.tile(vb_mask, (bs.nspin, 1))           # (nspin, nb)
        conduction = ~valence
        crossing = np.zeros_like(valence)
    else:
        emax, emin = E.max(axis=1), E.min(axis=1)          # (nspin, nb)
        valence = emax <= bs.efermi + tol
        conduction = (emin >= bs.efermi - tol) & ~valence
        crossing = ~(valence | conduction)

    vbm = cbm = None
    if valence.any():
        Ev = np.where(valence[:, None, :], E, -np.inf)
        s, k, b = np.unravel_index(np.argmax(Ev), E.shape)
        vbm = BandEdge(float(E[s, k, b]), int(b), int(k), int(s), bs.label(int(k)))
    if conduction.any():
        Ec = np.where(conduction[:, None, :], E, np.inf)
        s, k, b = np.unravel_index(np.argmin(Ec), E.shape)
        cbm = BandEdge(float(E[s, k, b]), int(b), int(k), int(s), bs.label(int(k)))

    if vbm is None or cbm is None or crossing.any():
        return BandEdges(vbm, cbm, 0.0, False, True)
    gap = cbm.energy - vbm.energy
    if gap <= 0:
        return BandEdges(vbm, cbm, 0.0, False, True)
    direct = np.allclose(bs.kpoints[vbm.kindex], bs.kpoints[cbm.kindex], atol=1e-6)
    return BandEdges(vbm, cbm, gap, bool(direct), False)
