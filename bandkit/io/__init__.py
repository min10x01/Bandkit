"""Readers for VASP output and a single entry point, :func:`load_band_structure`."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..core import BandStructure
from .eigenval import read_eigenval
from .kpoints import read_kpoints
from .outcar import read_outcar
from .poscar import read_poscar
from .procar import read_procar
from .vasprun import read_vasprun

__all__ = ["load_band_structure", "read_eigenval", "read_kpoints", "read_outcar",
           "read_poscar", "read_procar", "read_vasprun"]


def _find(directory: Path, explicit, *names):
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def load_band_structure(path=".", *, vasprun=None, eigenval=None, procar=None, outcar=None,
                        kpoints=None, poscar=None, efermi=None, projections=True,
                        drop_weighted="auto") -> BandStructure:
    """Read a VASP band-structure calculation.

    Parameters
    ----------
    path
        Calculation directory (or a vasprun.xml file).
    vasprun, eigenval, procar, outcar, kpoints, poscar
        Explicit file paths, overriding the defaults found in ``path``.
    efermi
        Fermi energy in eV. Recommended: the value from the self-consistent run,
        since the non-self-consistent band run can report a shifted value.
    projections
        Read orbital projections (PROCAR or vasprun.xml) if available.
    drop_weighted
        For hybrid-functional band structures the k-point list mixes weighted
        SCF points with zero-weight path points. ``'auto'`` keeps only the
        zero-weight points when both kinds are present.

    Data sources, in order of preference: vasprun.xml (everything), else
    PROCAR/EIGENVAL for energies plus OUTCAR (Fermi energy, reciprocal lattice)
    and POSCAR (element names). KPOINTS supplies the high-symmetry labels.
    """
    path = Path(path)
    directory = path.parent if path.is_file() else path
    if path.is_file() and vasprun is None and path.suffix == ".xml":
        vasprun = path

    f_vasprun = _find(directory, vasprun, "vasprun.xml") if eigenval is None else None
    f_eigenval = _find(directory, eigenval, "EIGENVAL")
    f_procar = _find(directory, procar, "PROCAR") if projections else None
    f_outcar = _find(directory, outcar, "OUTCAR")
    f_kpoints = _find(directory, kpoints, "KPOINTS")
    f_poscar = _find(directory, poscar, "POSCAR", "CONTCAR")

    data: dict = {}
    sources = []
    if f_vasprun is not None:
        data = read_vasprun(f_vasprun)
        sources.append(f_vasprun.name)
        if not projections:
            data["projections"] = data["spin_texture"] = None
    if f_procar is not None and data.get("projections") is None:
        pro = read_procar(f_procar)
        sources.append(f_procar.name)
        if not data:
            data.update(kpoints=pro["kpoints"], weights=pro["weights"], energies=pro["energies"],
                        occupations=pro["occupations"])
        data.update(projections=pro["projections"], spin_texture=pro["spin_texture"],
                    orbital_names=pro["orbital_names"])
        data["noncollinear"] = data.get("noncollinear") or pro["noncollinear"]
    if not data:
        if f_eigenval is None:
            raise FileNotFoundError(f"no vasprun.xml, PROCAR or EIGENVAL found in {directory}")
        data = read_eigenval(f_eigenval)
        sources.append(f_eigenval.name)
    elif data.get("nelect") is None and f_eigenval is not None:
        data["nelect"] = read_eigenval(f_eigenval)["nelect"]

    if f_outcar is not None:
        oc = read_outcar(f_outcar)
        sources.append(f_outcar.name)
        for key in ("efermi", "nelect", "reciprocal_lattice"):
            if data.get(key) is None:
                data[key] = oc[key]
        if oc["noncollinear"]:
            data["noncollinear"] = True
    if f_poscar is not None:
        pc = read_poscar(f_poscar)
        if data.get("reciprocal_lattice") is None:
            data["reciprocal_lattice"] = pc["reciprocal_lattice"]
        if not data.get("ion_symbols"):
            data["ion_symbols"] = pc["ion_symbols"]
    if data.get("reciprocal_lattice") is None:
        raise ValueError("reciprocal lattice not found: provide vasprun.xml, OUTCAR or POSCAR")

    kp = read_kpoints(f_kpoints) if f_kpoints is not None else None

    # hybrid-functional runs: keep only the zero-weight path points
    weights = data.get("weights")
    if drop_weighted and weights is not None:
        zero = np.asarray(weights) == 0
        if zero.any() and not zero.all():
            data = _select_kpoints(data, zero)

    if efermi is not None:
        data["efermi"] = float(efermi)

    segment_size = line_labels = labeled_points = None
    if kp is not None and kp.mode == "line":
        segment_size, line_labels = kp.ndiv, kp.line_labels
        labeled_points = [(pt, lab) for seg, labs in zip(kp.line_points, kp.line_labels)
                          for pt, lab in zip(seg, labs)] if not kp.cartesian else None
    elif kp is not None and kp.mode == "explicit" and not kp.cartesian:
        labeled_points = kp.labeled_points
    if segment_size is None:
        segment_size = data.get("ndiv")

    ion_symbols = list(data.get("ion_symbols") or [])
    if data.get("projections") is not None:
        nions = data["projections"].shape[3]
        if len(ion_symbols) != nions:
            ion_symbols = [f"X{i + 1}" for i in range(nions)]

    bs = BandStructure(
        kpoints=data["kpoints"],
        energies=data["energies"],
        reciprocal_lattice=data["reciprocal_lattice"],
        efermi=data.get("efermi") if data.get("efermi") is not None else np.nan,
        occupations=data.get("occupations"),
        weights=data.get("weights"),
        nelect=data.get("nelect"),
        noncollinear=bool(data.get("noncollinear")),
        projections=data.get("projections"),
        spin_texture=data.get("spin_texture"),
        orbital_names=list(data.get("orbital_names") or []),
        ion_symbols=ion_symbols,
        segment_size=segment_size,
        line_labels=line_labels,
        labeled_points=labeled_points,
        source=" + ".join(sources),
    )
    if np.isnan(bs.efermi):
        from ..analysis.edges import find_band_edges

        edges = find_band_edges(bs)
        if edges.vbm is None:
            raise ValueError("Fermi energy not found: pass efermi=...")
        bs.efermi = edges.vbm.energy
        warnings.warn("Fermi energy not found; using the valence-band maximum instead")
    return bs


def _select_kpoints(data: dict, mask: np.ndarray) -> dict:
    data = dict(data)
    data["kpoints"] = data["kpoints"][mask]
    data["weights"] = data["weights"][mask]
    for key in ("energies", "occupations", "projections"):
        if data.get(key) is not None:
            data[key] = data[key][:, mask]
    if data.get("spin_texture") is not None:
        data["spin_texture"] = data["spin_texture"][mask]
    return data
