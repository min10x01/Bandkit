"""Synthetic VASP output for tests: analytic band models written in VASP formats.

The Rashba model uses a square lattice (a = 4 Å) and the path X–Γ–M with
two spin-split pairs whose parameters are known exactly:

    valence:    E = -hbar²k²/2m_v ± alpha_v k     (degenerate at Γ, E = 0)
    conduction: E = E_c + hbar²k²/2m_c ± alpha_c k
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HBAR2_ME = 7.619964231070681
ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]

# model parameters
A = 4.0
M_V, ALPHA_V = 0.5, 1.0
M_C, ALPHA_C, E_C = 0.2, 2.0, 1.5
NDIV = 101


@dataclass
class Model:
    lattice: np.ndarray
    kpoints: np.ndarray
    energies: np.ndarray                 # (nspin, nk, nb)
    projections: np.ndarray              # (nspin, nk, nb, nions, norb)
    spin_texture: np.ndarray | None      # (nk, nb, 3)
    efermi: float
    nelect: float
    symbols: list[str]
    ndiv: int
    path: list[tuple[tuple, tuple, str, str]]
    weights: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def noncollinear(self):
        return self.spin_texture is not None

    @property
    def rec(self):
        return 2 * np.pi * np.linalg.inv(self.lattice).T


def _path_kpoints(path, ndiv):
    return np.concatenate([np.linspace(a, b, ndiv) for a, b, _, _ in path])


def _proj(weights_by_band, nk):
    """weights_by_band: list of (nions, norb) arrays -> (1, nk, nb, nions, norb)."""
    P = np.stack(weights_by_band)                    # (nb, nions, norb)
    return np.broadcast_to(P, (nk,) + P.shape)[None].copy()


def rashba_model() -> Model:
    lattice = np.diag([A, A, 20.0])
    path = [((0.5, 0, 0), (0, 0, 0), "X", "G"), ((0, 0, 0), (0.5, 0.5, 0), "G", "M")]
    kfrac = _path_kpoints(path, NDIV)
    rec = 2 * np.pi * np.linalg.inv(lattice).T
    kc = kfrac @ rec
    k = np.linalg.norm(kc, axis=1)

    deep = -1.5 - HBAR2_ME / 2 / 1.0 * k**2
    v_par = -HBAR2_ME / (2 * M_V) * k**2
    c_par = E_C + HBAR2_ME / (2 * M_C) * k**2
    bands = [deep, deep, v_par - ALPHA_V * k, v_par + ALPHA_V * k,
             c_par - ALPHA_C * k, c_par + ALPHA_C * k]
    E = np.stack(bands, axis=1)[None]                # (1, nk, 6)

    # Rashba spin texture: in-plane, perpendicular to k, opposite in the two branches
    khat = np.divide(kc, k[:, None], out=np.zeros_like(kc), where=k[:, None] > 0)
    tangential = np.column_stack([khat[:, 1], -khat[:, 0], np.zeros(len(k))])
    S = np.zeros((len(k), 6, 3))
    for b, sign in ((2, 1), (3, -1), (4, -1), (5, 1)):
        S[:, b] = sign * tangential

    def w(**kw):
        a = np.zeros((2, len(ORBITALS)))
        for key, val in kw.items():
            ion, orb = key.split("_")
            a[{"Bi": 0, "Te": 1}[ion], ORBITALS.index(orb)] = val
        return a

    proj = _proj([w(Te_s=0.9), w(Te_s=0.9),
                  w(Te_pz=0.63, Bi_s=0.27), w(Te_pz=0.63, Bi_s=0.27),
                  w(Bi_px=0.36, Bi_py=0.36, Te_s=0.18), w(Bi_px=0.36, Bi_py=0.36, Te_s=0.18)],
                 len(k))
    return Model(lattice, kfrac, E, proj, S, efermi=0.2, nelect=4, symbols=["Bi", "Te"],
                 ndiv=NDIV, path=path)


def zeeman_model() -> Model:
    """Valley at the non-TRIM point (1/4, 1/4, 0) with a constant spin-z splitting."""
    lattice = np.diag([A, A, 20.0])
    path = [((0, 0, 0), (0.5, 0.5, 0), "G", "M")]
    kfrac = _path_kpoints(path, NDIV)
    rec = 2 * np.pi * np.linalg.inv(lattice).T
    q = np.linalg.norm((kfrac - [0.25, 0.25, 0]) @ rec, axis=1)
    delta = 0.15
    v = -1.0 - HBAR2_ME / 2 * q**2
    c = 1.0 + HBAR2_ME / 2 * q**2
    E = np.stack([v - delta / 2, v + delta / 2, c - 0.01, c + 0.01], axis=1)[None]
    S = np.zeros((len(q), 4, 3))
    S[:, 0, 2], S[:, 1, 2], S[:, 2, 2], S[:, 3, 2] = 1, -1, 1, -1
    ones = np.zeros((2, len(ORBITALS)))
    ones[0, 0] = ones[1, 2] = 0.5
    proj = _proj([ones] * 4, len(q))
    return Model(lattice, kfrac, E, proj, S, efermi=0.0, nelect=2, symbols=["Mo", "S"],
                 ndiv=NDIV, path=path, extra={"delta": delta})


def spin_polarized_model() -> Model:
    """Collinear ISPIN=2: rigid exchange splitting of 0.3 eV, gap 1 eV."""
    lattice = np.diag([A, A, A])
    path = [((0, 0, 0), (0.5, 0, 0), "G", "X")]
    kfrac = _path_kpoints(path, 21)
    k = np.linalg.norm(kfrac @ (2 * np.pi * np.linalg.inv(lattice).T), axis=1)
    v, c = -k**2, 1.3 + k**2
    up = np.stack([v, c], axis=1)
    E = np.stack([up, up + 0.3])
    proj = np.ones((2, len(k), 2, 1, 1)) * 0.5
    return Model(lattice, kfrac, E, proj, None, efermi=0.5, nelect=2, symbols=["Fe"],
                 ndiv=21, path=path)


# ---------------------------------------------------------------------------- writers
def write_kpoints(m: Model, path: Path):
    lines = ["k-path", str(m.ndiv), "Line-mode", "Reciprocal"]
    for a, b, la, lb in m.path:
        lines += [f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} ! {la}",
                  f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} ! {lb}", ""]
    path.write_text("\n".join(lines) + "\n")


def write_poscar(m: Model, path: Path):
    lines = ["synthetic", "1.0"] + [" ".join(f"{x:.8f}" for x in row) for row in m.lattice]
    lines += [" ".join(m.symbols), " ".join("1" for _ in m.symbols), "Direct"]
    lines += [f"{0.0:.6f} {0.0:.6f} {0.1 * i:.6f}" for i in range(len(m.symbols))]
    path.write_text("\n".join(lines) + "\n")


def _occ(m, s):
    if m.energies.shape[0] == 2:
        return (m.energies[s] < m.efermi).astype(float)
    occ = np.zeros(m.energies.shape[1:])
    occ[:, :int(m.nelect if m.noncollinear else m.nelect / 2)] = 1.0
    return occ


def write_eigenval(m: Model, path: Path, with_occ=True):
    nspin, nk, nb = m.energies.shape
    out = [f"    2    2    1    {nspin}", "  0.6400000E+02  0.4E-09  0.4E-09  0.2E-08  0.5E-15",
           "  1.000000000000000E-004", "  CAR", " synthetic", f"    {int(m.nelect)}  {nk}  {nb}"]
    for ik in range(nk):
        kx, ky, kz = m.kpoints[ik]
        wt = 0.0 if m.weights is None else m.weights[ik]
        out += ["", f"  {kx:.7E}  {ky:.7E}  {kz:.7E}  {wt:.7E}"]
        for b in range(nb):
            es = [m.energies[s, ik, b] for s in range(nspin)]
            os_ = [_occ(m, s)[ik, b] for s in range(nspin)] if with_occ else []
            out.append(f"{b + 1:5d}" + "".join(f"  {e:14.6f}" for e in es) + "".join(f"  {o:9.6f}" for o in os_))
    path.write_text("\n".join(out) + "\n")


def write_outcar(m: Model, path: Path):
    rec = m.rec / (2 * np.pi)
    out = [f"   NELECT =      {m.nelect:.4f}    total number of electrons",
           f"   ISPIN  =      {m.energies.shape[0]}    spin polarized calculation?",
           f"   LNONCOLLINEAR =      {'T' if m.noncollinear else 'F'}    non collinear calculations",
           "      direct lattice vectors                 reciprocal lattice vectors"]
    for a, b in zip(m.lattice, rec):
        out.append("    " + " ".join(f"{x:12.9f}" for x in a) + "    " + " ".join(f"{x:12.9f}" for x in b))
    out += ["", f" E-fermi :   {m.efermi:.4f}     XC(G=0): -10.0000     alpha+bet : -10.0000", ""]
    path.write_text("\n".join(out) + "\n")


def _fmt_row(label, vals):
    return f"{label:<5s}" + "".join(f"{v:7.3f}" for v in vals) + f"{sum(vals):7.3f}"


def write_procar(m: Model, path: Path, fuse_negative=False):
    nspin, nk, nb = m.energies.shape
    nions = len(m.symbols)
    out = ["PROCAR lm decomposed"]
    for s in range(nspin):
        out.append(f"# of k-points:  {nk}         # of bands:  {nb}         # of ions:  {nions}")
        occ = _occ(m, s)
        for ik in range(nk):
            kx, ky, kz = m.kpoints[ik]
            coords = f"{kx:11.8f}{ky:11.8f}{kz:11.8f}" if fuse_negative else f"{kx:11.8f} {ky:11.8f} {kz:11.8f}"
            out += ["", f" k-point {ik + 1:5d} :   {coords}     weight = 0.00000000", ""]
            for b in range(nb):
                out += [f"band {b + 1:5d} # energy {m.energies[s, ik, b]:14.8f} # occ. {occ[ik, b]:11.8f}", ""]
                out.append("ion " + "".join(f"{o:>7s}" for o in ORBITALS[:m.projections.shape[-1]]) + "    tot")
                P = m.projections[s, ik, b]
                blocks = [P]
                if m.noncollinear:
                    blocks += [P * m.spin_texture[ik, b, c] for c in range(3)]
                for blk in blocks:
                    for i in range(nions):
                        out.append(_fmt_row(f"{i + 1:4d}", blk[i]))
                    out.append(_fmt_row("tot", blk.sum(axis=0)))
                out.append("")
    path.write_text("\n".join(out) + "\n")


def write_vasprun(m: Model, path: Path, with_projections=True):
    nspin, nk, nb = m.energies.shape
    nions = len(m.symbols)
    norb = m.projections.shape[-1]
    x = ['<?xml version="1.0" encoding="ISO-8859-1"?>', "<modeling>",
         ' <incar><i type="string" name="SYSTEM">synthetic</i></incar>', " <kpoints>",
         '  <generation param="listgenerated">', f'   <i name="divisions" type="int"> {m.ndiv} </i>']
    for a, b, _, _ in m.path:
        x += [f"   <v> {a[0]} {a[1]} {a[2]} </v>", f"   <v> {b[0]} {b[1]} {b[2]} </v>"]
    x += ["  </generation>", '  <varray name="kpointlist">']
    x += [f"   <v> {k[0]:.8f} {k[1]:.8f} {k[2]:.8f} </v>" for k in m.kpoints]
    x += ["  </varray>", '  <varray name="weights">'] + ["   <v> 0.0 </v>"] * nk + ["  </varray>", " </kpoints>"]
    x += [" <parameters>", '  <separator name="electronic">',
          f'   <i name="NELECT"> {m.nelect:.4f} </i>', f'   <i type="int" name="NBANDS"> {nb} </i>',
          '   <separator name="electronic spin">', f'    <i type="int" name="ISPIN"> {nspin} </i>',
          f'    <i type="logical" name="LNONCOLLINEAR"> {"T" if m.noncollinear else "F"} </i>',
          '    <v type="logical" name="LSORBIT"> T </v>',
          "   </separator>", "  </separator>", " </parameters>"]
    x += [" <atominfo>", f"  <atoms> {nions} </atoms>", '  <array name="atoms">',
          '   <dimension dim="1">ion</dimension>', '   <field type="string">element</field>',
          '   <field type="int">atomtype</field>', "   <set>"]
    x += [f"    <rc><c>{s:2s}</c><c>   {i + 1}</c></rc>" for i, s in enumerate(m.symbols)]
    x += ["   </set>", "  </array>", '  <array name="atomtypes">', "   <set>"]
    x += [f"    <rc><c>   1</c><c>{s:2s}</c><c> 1.0 </c></rc>" for s in m.symbols]
    x += ["   </set>", "  </array>", " </atominfo>"]

    def structure(name):
        s = [f' <structure name="{name}">', "  <crystal>", '   <varray name="basis">']
        s += ["    <v> " + " ".join(f"{v:.8f}" for v in row) + " </v>" for row in m.lattice]
        s += ["   </varray>", '   <varray name="rec_basis">']
        s += ["    <v> " + " ".join(f"{v:.8f}" for v in row) + " </v>" for row in m.rec / (2 * np.pi)]
        s += ["   </varray>", "  </crystal>", '  <varray name="selective">',
              "   <v type=\"logical\"> T T T </v>", "  </varray>", " </structure>"]
        return s

    x += structure("initialpos")
    x += [" <calculation>", "  <eigenvalues>", "   <array>", "    <field>eigene</field>",
          "    <field>occ</field>", "    <set>"]
    for s in range(nspin):
        x.append(f'     <set comment="spin {s + 1}">')
        occ = _occ(m, s)
        for ik in range(nk):
            x.append(f'      <set comment="kpoint {ik + 1}">')
            x += [f"       <r> {m.energies[s, ik, b]:12.6f} {occ[ik, b]:8.4f} </r>" for b in range(nb)]
            x.append("      </set>")
        x.append("     </set>")
    x += ["    </set>", "   </array>", "  </eigenvalues>",
          f'  <dos><i name="efermi"> {m.efermi:12.8f} </i></dos>']
    if with_projections:
        x += ["  <projected>", "   <array>", '    <dimension dim="1">ion</dimension>']
        x += [f"    <field>{o}</field>" for o in ORBITALS[:norb]]
        x.append("    <set>")
        blocks = []
        for s in range(nspin):
            blocks.append(m.projections[s])
        if m.noncollinear:
            P = m.projections[0]
            blocks += [P * m.spin_texture[:, :, c, None, None] for c in range(3)]
        for n, blk in enumerate(blocks):
            x.append(f'     <set comment="spin{n + 1}">')
            for ik in range(nk):
                x.append(f'      <set comment="kpoint {ik + 1}">')
                for b in range(nb):
                    x.append(f'       <set comment="band {b + 1}">')
                    x += ["        <r> " + " ".join(f"{v:.4f}" for v in blk[ik, b, i]) + " </r>"
                          for i in range(nions)]
                    x.append("       </set>")
                x.append("      </set>")
            x.append("     </set>")
        x += ["    </set>", "   </array>", "  </projected>"]
    x += [" </calculation>"] + structure("finalpos") + ["</modeling>"]
    path.write_text("\n".join(x) + "\n")


def write_all(m: Model, directory: Path, files=("KPOINTS", "POSCAR", "EIGENVAL", "OUTCAR", "PROCAR",
                                                "vasprun.xml")):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    writers = {"KPOINTS": write_kpoints, "POSCAR": write_poscar, "EIGENVAL": write_eigenval,
               "OUTCAR": write_outcar, "PROCAR": write_procar, "vasprun.xml": write_vasprun}
    for name in files:
        writers[name](m, directory / name)
    return directory
