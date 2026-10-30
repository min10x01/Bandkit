"""Example: full analysis of a VASP band-structure run.

Usage:  python examples/analyze.py /path/to/band/calculation
"""

import sys

import matplotlib.pyplot as plt

import bandkit as bk
from bandkit.plotting import plot_bands, plot_splitting

path = sys.argv[1] if len(sys.argv) > 1 else "."
bs = bk.load_band_structure(path)          # pass efermi=... to use the SCF Fermi level
print(bs.summary())

# --- band edges and gap
edges = bk.find_band_edges(bs)
print(edges)

# --- effective masses at the band edges (every path direction)
masses = bk.band_edge_masses(bs, npoints=5)
for carrier, results in masses.items():
    for m in results:
        print(carrier, m)

# --- spin splitting at the band edges
rashba = []
for edge in ("vbm", "cbm"):
    found = bk.edge_pair(bs, edge)
    if found is None:
        continue
    pair, k, spin, kind = found
    print(f"\n{edge.upper()} pair {pair[0] + 1}/{pair[1] + 1}:", bk.classify_splitting(bs, pair, k, spin))
    for r in bk.find_rashba(bs, edge):
        print("  ", r)
        rashba.append(r)

# --- orbital character of the band edges
if bs.has_projections and not edges.is_metal:
    for name, e in (("VBM", edges.vbm), ("CBM", edges.cbm)):
        print(name, bk.format_character(bk.band_character(bs, e.band, e.kindex, e.spin)))

# --- plots: pick what to show
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_bands(bs, ax=axes[0], edges=True, rashba=rashba,
           effective_masses=[m for ms in masses.values() for m in ms], title="Rashba / m*")
if bs.has_projections:
    species = list(dict.fromkeys(bs.ion_symbols))
    plot_bands(bs, ax=axes[1], projections=[f"{s}:p" for s in species] + [f"{s}:s" for s in species],
               mode="fat", title="Orbital character")
fig.tight_layout()
fig.savefig("bands_analysis.png", dpi=200)

found = bk.edge_pair(bs, "cbm")
if found:
    plot_splitting(bs, found[0], filename="splitting_cbm.png")
print("\nsaved bands_analysis.png")
