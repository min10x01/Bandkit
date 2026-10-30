"""Streaming reader for vasprun.xml.

Only the parts needed for band-structure analysis are kept in memory, so
large files with orbital projections can be read. If the run used
KPOINTS_OPT (VASP >= 6.3) the band structure on that k-point set is returned.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def _rows(texts: list[str]) -> np.ndarray:
    return np.array(" ".join(texts).split(), dtype=float)


def read_vasprun(path) -> dict:
    stack: list[str] = []
    out: dict = {"efermi": None, "nelect": None, "ispin": 1, "noncollinear": False,
                 "nbands": None, "ion_symbols": [], "reciprocal_lattice": None}
    # per k-point set: "main" or "opt" (KPOINTS_OPT)
    sets = {key: {"kpoints": None, "weights": None, "ndiv": None, "eig": [], "neig_spin": 0,
                  "proj": [], "nproj_spin": 0, "fields": []} for key in ("main", "opt")}

    def context() -> dict:
        return sets["opt"] if any(t.endswith("kpoints_opt") for t in stack) else sets["main"]

    def in_projected() -> bool:
        return any(t.startswith("projected") for t in stack)

    for event, elem in ET.iterparse(path, events=("start", "end")):
        tag = elem.tag
        if event == "start":
            stack.append(tag)
            if tag == "eigenvalues" and not in_projected():
                ctx = context()
                ctx["eig"], ctx["neig_spin"] = [], 0
            elif tag == "array" and in_projected() and "eigenvalues" not in stack:
                ctx = context()
                ctx["proj"], ctx["nproj_spin"], ctx["fields"] = [], 0, []
            elif tag == "set" and elem.get("comment", "").startswith("spin"):
                ctx = context()
                if in_projected() and "eigenvalues" not in stack:
                    ctx["nproj_spin"] += 1
                elif "eigenvalues" in stack and not in_projected():
                    ctx["neig_spin"] += 1
            continue

        # ---- end events
        stack.pop()
        name = elem.get("name")
        if tag == "r":
            if "eigenvalues" in stack and not in_projected():
                context()["eig"].append(elem.text)
            elif in_projected() and "eigenvalues" not in stack:
                context()["proj"].append(elem.text)
            elem.clear()
        elif tag == "set" and ("eigenvalues" in stack or in_projected()):
            elem.clear()
        elif tag == "field" and in_projected() and "eigenvalues" not in stack:
            context()["fields"].append(elem.text.strip())
        elif tag == "i":
            text = (elem.text or "").strip()
            if name == "efermi":
                out["efermi"] = float(text)
            elif name == "NELECT":
                out["nelect"] = float(text)
            elif name == "ISPIN":
                out["ispin"] = int(text)
            elif name == "NBANDS":
                out["nbands"] = int(text)
            elif name == "LNONCOLLINEAR":
                out["noncollinear"] = text.upper().startswith("T")
            elif name == "divisions" and "generation" in stack:
                context()["ndiv"] = int(text)
        elif tag == "varray" and name in ("kpointlist", "weights", "rec_basis"):
            vals = np.array([[float(x) for x in v.text.split()] for v in elem])
            if name == "kpointlist":
                context()["kpoints"] = vals
            elif name == "weights":
                context()["weights"] = vals.ravel()
            elif name == "rec_basis":
                out["reciprocal_lattice"] = 2 * np.pi * vals
        elif tag == "array" and name == "atoms" and "atominfo" in stack:
            out["ion_symbols"] = [rc[0].text.strip() for rc in elem.iter("rc")]

    ctx = sets["opt"] if sets["opt"]["eig"] else sets["main"]
    if ctx["kpoints"] is None or not ctx["eig"]:
        raise ValueError(f"{path}: no eigenvalues found")
    nk = len(ctx["kpoints"])
    nspin = max(ctx["neig_spin"], 1)
    eig = _rows(ctx["eig"])
    ncol = 2
    nb = eig.size // (nspin * nk * ncol)
    eig = eig.reshape(nspin, nk, nb, ncol)
    out.update(kpoints=ctx["kpoints"], weights=ctx["weights"], ndiv=ctx["ndiv"],
               energies=eig[..., 0], occupations=eig[..., 1],
               projections=None, spin_texture=None, orbital_names=[])

    if ctx["proj"]:
        fields = ctx["fields"]
        proj = _rows(ctx["proj"])
        nblocks = max(ctx["nproj_spin"], 1)
        nions = proj.size // (nblocks * nk * nb * len(fields))
        proj = proj.reshape(nblocks, nk, nb, nions, len(fields))
        out["orbital_names"] = fields
        if out["noncollinear"] or nblocks == 4 and out["ispin"] == 1:
            out["noncollinear"] = True
            out["projections"] = proj[:1]
            out["spin_texture"] = proj[1:4].sum(axis=(3, 4)).transpose(1, 2, 0)
        else:
            out["projections"] = proj
    return out
