# bandkit

Parse, analyse and plot VASP band structures:

- **Read** energies, k-points, high-symmetry labels and orbital projections from
  `vasprun.xml`, `PROCAR`, `EIGENVAL`, `OUTCAR`, `KPOINTS` and `POSCAR`.
- **Find** the band edges (VBM, CBM, gap, direct or indirect).
- **Compute** effective masses along every path direction.
- **Analyse** band splitting: Rashba parameters (E_R, k0, α_R), Zeeman-type, Dresselhaus-type and
  exchange splitting.
- **Report** the orbital and atomic character of any state.
- **Plot** with switchable overlays: fat bands, projection gradients, the dominant orbital, spin
  texture, effective-mass fits, Rashba markers and splitting arrows.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
pip install -e ".[test]" && pytest
```

## Package layout

```
bandkit/
├── core.py               BandStructure container, k-path (segments, distances, labels)
├── io/                   file readers + load_band_structure()
│   ├── vasprun.py        streaming vasprun.xml reader (incl. projections, KPOINTS_OPT)
│   ├── procar.py         PROCAR (LORBIT 10/11/12, ISPIN 1/2, non-collinear spin blocks)
│   ├── eigenval.py       EIGENVAL
│   ├── outcar.py         E-fermi, reciprocal lattice, NELECT, ISPIN, LNONCOLLINEAR
│   ├── kpoints.py        line-mode and explicit KPOINTS (labels after "!")
│   └── poscar.py         lattice and element names
├── analysis/
│   ├── edges.py          VBM / CBM / band gap
│   ├── effective_mass.py parabolic (or higher-order) fits along each direction
│   ├── splitting.py      Rashba, Zeeman, Dresselhaus, exchange splitting
│   └── orbitals.py       projected weights and band character
├── plotting.py           plot_bands(), plot_splitting()
└── cli.py                `bandkit` command
```

## Python usage

```python
import bandkit as bk
from bandkit.plotting import plot_bands

bs = bk.load_band_structure("path/to/band_run", efermi=-1.234)  # Fermi level from the SCF run

print(bk.find_band_edges(bs))

# effective masses (negative for holes), one result per direction
for m in bk.effective_mass(bs, band=27, kindex=100, npoints=5):
    print(m)
masses = bk.band_edge_masses(bs)          # {"electron": [...], "hole": [...]}

# Rashba analysis of the band pair at the CBM / VBM
rashba = bk.find_rashba(bs, "cbm") + bk.find_rashba(bs, "vbm")
for r in rashba:
    print(r)                              # E_R, k0, alpha_R (and alpha from the linear slope)

# any band pair: splitting along the path and a classification at a given k-point
dE = bk.band_splitting(bs, (26, 27))
print(bk.classify_splitting(bs, (26, 27), kindex=150))

# orbital character
print(bk.band_character(bs, band=27, kindex=100, by="element-orbital"))   # {'Bi-p': 0.62, ...}
w = bk.projected_weight(bs, atoms="Bi", orbitals=["px", "py"])          # (nspin, nk, nb)

# plot only what you want to show
plot_bands(bs, projections=["Bi:p", "I:p"], mode="fat", rashba=rashba,
           effective_masses=masses["electron"], edges=True, energy_range=(-2, 2),
           filename="bands.png")
plot_bands(bs, spin_texture="z", filename="spin_z.png")
```

`examples/analyze.py` runs the full analysis on a directory.

## Command line

```bash
bandkit info     path/                      # summary, gap, band-edge character
bandkit emass    path/ --npoints 5          # masses at VBM/CBM (or --band B --kpoint K)
bandkit split    path/                      # Rashba/Zeeman analysis at the VBM and CBM
bandkit split    path/ --bands 27 28        # a specific band pair
bandkit orbitals path/ --by element-lm      # orbital character of the VBM/CBM
bandkit plot     path/ --proj Bi:p --proj I:p --rashba --emass -o bands.png
bandkit plot     path/ --spin-texture z --erange -1 1 -o spin.png
bandkit plot     path/ --proj Bi --proj I --mode dominant --splitting --splitting-panel cbm -o b.png
```

Projection specs: `Bi:p`, `Bi,I:s,p`, `:d` (all atoms), `1-4:px,py` (ions 1–4).
All commands accept `--efermi`.

## Conventions

| | |
|---|---|
| Energies | eV |
| k distances | 1/Å, including 2π |
| Effective mass | units of m_e; m* = ħ²/(d²E/dk²), negative for hole-like bands |
| α_R | eV·Å |
| Indices | **0-based in Python** (bands, k-points, ions, spins); **1-based on the command line** (as in VASP) |
| Spin channels | `energies` has shape `(nspin, nk, nb)`; SOC runs have `nspin = 1` |

## Method

**Band edges.** If NELECT is known (non-spin-polarised or SOC runs), the highest occupied band
comes from the electron count. Otherwise bands are sorted by their position relative to E_F.

**Effective mass.** At a high-symmetry point the code fits a one-sided polynomial into each
adjacent path segment. At a point inside a segment, such as an off-centre Rashba minimum, it fits
both sides together. Choose the fit range with `npoints` or `energy_window`.

**Rashba.** The two branches meet at a crossing point k_c, normally a time-reversal-invariant
momentum (TRIM) where |ΔE| < `tol`. Along each direction leaving k_c, the code follows the lower
branch to its minimum (conduction) or the upper branch to its maximum (valence). That extremum
gives k0 and E_R, refined below the grid spacing with a parabola, and α_R = 2E_R/k0. A second
estimate, `alpha_linear`, comes from the small-k slope ΔE ≈ 2α_R k.

**Splitting classification** (`classify_splitting`) is heuristic:
- With the non-collinear spin texture:
  - out-of-plane spin → *Zeeman-type*
  - in-plane spin ⟂ (k − k_c) → *Rashba-type*
  - in-plane spin ∥ (k − k_c) → *Dresselhaus-type*
  - anything in between → *mixed*
- Without spin data, the code uses only where the splitting vanishes. It needs LORBIT = 11 and
  LSORBIT = .TRUE. to tell Rashba from Dresselhaus. The surface normal defaults to z (`normal=`).

**Exchange splitting** (ISPIN = 2): `exchange_splitting(bs, band, k)` = E↓ − E↑.

## Practical notes

- Use the Fermi energy from the self-consistent run (`efermi=` / `--efermi`). The
  non-self-consistent band run can report a shifted value.
- Orbital character and spin texture need `LORBIT = 11` (PROCAR or vasprun.xml).
- Hybrid-functional band structures with weighted SCF k-points plus zero-weight path points
  are handled automatically: only the zero-weight points are kept.
- Rashba fits need enough k-points between k_c and k0. Use a dense line-mode path near the
  crossing point.

## Tests

`tests/synthetic.py` writes VASP-format files (KPOINTS, POSCAR, EIGENVAL, OUTCAR, PROCAR,
vasprun.xml) for analytic models with known answers: a Rashba model, a Zeeman valley model and
a collinear spin-polarised model. The tests check that the parsers agree and that the analysis
recovers the input parameters (α_R, m*, E_R, k0, orbital weights).
