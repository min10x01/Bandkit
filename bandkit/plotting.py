"""Band-structure plots with optional orbital, spin and splitting overlays."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .analysis.edges import find_band_edges
from .analysis.orbitals import parse_projection, projected_weight
from .analysis.splitting import band_splitting
from .core import BandStructure

_YLABEL = {"fermi": r"$E - E_\mathrm{F}$ (eV)", "vbm": r"$E - E_\mathrm{VBM}$ (eV)", None: "Energy (eV)"}


def _normalize_projections(projections):
    out = []
    for n, p in enumerate(projections or []):
        if isinstance(p, str):
            atoms, orbitals = parse_projection(p)
            p = {"atoms": atoms, "orbitals": orbitals, "label": p}
        p = dict(p)
        p.setdefault("label", f"{p.get('atoms') or 'all'}:{p.get('orbitals') or 'all'}")
        p.setdefault("color", f"C{n}")
        out.append(p)
    return out


def _colored_lines(ax, x, E, values, segments, bands, **kw):
    """LineCollection of the selected bands, each piece coloured by ``values`` (nk, nb)."""
    pieces, colors = [], []
    for s, e in segments:
        if e <= s:
            continue
        for b in bands:
            pts = np.column_stack([x[s:e + 1], E[s:e + 1, b]])
            pieces.append(np.stack([pts[:-1], pts[1:]], axis=1))
            v = values[s:e + 1, b]
            colors.append(0.5 * (v[:-1] + v[1:]))
    lc = LineCollection(np.concatenate(pieces), **kw)
    lc.set_array(np.concatenate(colors))
    ax.add_collection(lc)
    return lc


def plot_bands(bs: BandStructure, ax=None, *, reference="fermi", energy_range=(-3.0, 3.0),
               spin=None, color="k", linewidth=1.0,
               projections=None, mode="fat", fat_scale=40.0, cmap="viridis",
               spin_texture=None, edges=False, effective_masses=None, rashba=None,
               splittings=None, title=None, legend=True, filename=None, dpi=300,
               figsize=(6, 5)):
    """Plot the band structure.

    reference
        'fermi', 'vbm', a number (eV) or None for absolute energies.
    energy_range
        (Emin, Emax) relative to the reference; None for automatic limits.
    spin
        Spin channel to show (ISPIN=2); both by default (spin down dashed).
    projections
        Orbital/atom selections to show, e.g. ``["Bi:p", "I:p"]`` or dicts
        ``{"atoms": "Bi", "orbitals": ["px", "py"], "label": ..., "color": ...}``.
    mode
        How projections are drawn: 'fat' (marker size ∝ weight),
        'gradient' (line colour = weight of the first projection) or
        'dominant' (line colour = projection with the largest weight).
    spin_texture
        'x', 'y' or 'z': colour bands by the normalised spin component
        (non-collinear runs).
    edges
        Mark the VBM and CBM.
    effective_masses
        List of :class:`EffectiveMass` results: draws the fitted parabolas.
    rashba
        List of :class:`RashbaSplitting` results: marks crossing point,
        shifted extremum, k0 / E_R and alpha_R.
    splittings
        List of :class:`SplittingCharacter` results: arrow with ΔE.
    filename
        Save the figure there.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    E0 = bs.reference_energy(reference)
    x = bs.distances
    spins = [spin] if spin is not None else list(range(bs.nspin))
    projections = _normalize_projections(projections)
    window = energy_range if energy_range is not None else (-np.inf, np.inf)
    handles = []

    for s in spins:
        E = bs.energies[s] - E0
        bands = [b for b in range(bs.nbands)
                 if E[:, b].max() >= window[0] - 1 and E[:, b].min() <= window[1] + 1]
        if not bands:
            continue
        ls = "--" if s == 1 else "-"
        coloured = spin_texture is not None or (projections and mode in ("gradient", "dominant"))
        base_color = "0.6" if projections and mode == "fat" else (color if s == 0 else "tab:red")
        if not coloured:
            for a, b in bs.segments:
                ax.plot(x[a:b + 1], E[a:b + 1][:, bands], color=base_color, lw=linewidth, ls=ls)
            if bs.nspin == 2 and spin is None and not projections:
                handles.append(Line2D([], [], color=base_color, ls=ls, label="spin ↑" if s == 0 else "spin ↓"))

        if spin_texture is not None:
            if bs.spin_texture is None:
                raise ValueError("no spin texture: needs a non-collinear run with LORBIT=11")
            comp = "xyz".index(spin_texture)
            S = bs.spin_texture
            norm = np.linalg.norm(S, axis=-1)
            val = np.divide(S[..., comp], norm, out=np.zeros_like(norm), where=norm > 1e-8)
            lc = _colored_lines(ax, x, E, val, bs.segments, bands, cmap="coolwarm", lw=linewidth * 1.5)
            lc.set_clim(-1, 1)
            plt.colorbar(lc, ax=ax, label=rf"$\langle\sigma_{spin_texture}\rangle$", pad=0.02)
        elif projections and mode == "fat":
            for p in projections:
                w = projected_weight(bs, p.get("atoms"), p.get("orbitals"))[s]
                for b in bands:
                    ax.scatter(x, E[:, b], s=fat_scale * w[:, b], color=p["color"], alpha=0.5,
                               linewidths=0, zorder=3)
                if s == spins[0]:
                    handles.append(Line2D([], [], marker="o", ls="", color=p["color"], alpha=0.6,
                                          label=p["label"]))
        elif projections and mode == "gradient":
            p = projections[0]
            w = projected_weight(bs, p.get("atoms"), p.get("orbitals"))[s]
            lc = _colored_lines(ax, x, E, w, bs.segments, bands, cmap=cmap, lw=linewidth * 1.8,
                                linestyle=ls)
            lc.set_clim(0, 1)
            if s == spins[0]:
                plt.colorbar(lc, ax=ax, label=f"{p['label']} weight", pad=0.02)
        elif projections and mode == "dominant":
            W = np.stack([projected_weight(bs, p.get("atoms"), p.get("orbitals"))[s] for p in projections])
            dom = np.argmax(W, axis=0).astype(float)
            from matplotlib.colors import BoundaryNorm, ListedColormap

            cm = ListedColormap([p["color"] for p in projections])
            norm = BoundaryNorm(np.arange(len(projections) + 1) - 0.5, len(projections))
            _colored_lines(ax, x, E, dom, bs.segments, bands, cmap=cm, norm=norm,
                           lw=linewidth * 1.8, linestyle=ls)
            if s == spins[0]:
                handles += [Line2D([], [], color=p["color"], lw=2, label=p["label"]) for p in projections]
        elif projections:
            raise ValueError(f"unknown projection mode {mode!r}")

    # ---- overlays
    if edges:
        be = find_band_edges(bs)
        for e, c, name in ((be.vbm, "tab:blue", "VBM"), (be.cbm, "tab:red", "CBM")):
            if e is None or be.is_metal:
                continue
            xs = [x[i] for i in bs.equivalent_indices(e.kindex)]
            ax.scatter(xs, [e.energy - E0] * len(xs), s=60, facecolor="none", edgecolor=c,
                       linewidths=1.5, zorder=5)
            handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=c, label=name))

    for m in effective_masses or []:
        xp = x[m.indices]
        slope, icpt = np.polyfit(m.s, xp, 1) if np.ptp(m.s) > 0 else (0.0, xp[0])
        sg = np.linspace(m.s.min(), m.s.max(), 50)
        ax.plot(slope * sg + icpt, np.polyval(m.coeffs, sg) - E0, color="tab:orange", lw=2,
                ls=":", zorder=4)
        end = sg[-1] if abs(sg[-1]) >= abs(sg[0]) else sg[0]
        xe = slope * end + icpt
        right = xe >= x[m.kindex]
        ax.annotate(f"m*={m.mass:.2f}", (xe, np.polyval(m.coeffs, end) - E0), fontsize=7,
                    color="tab:orange", xytext=(3 if right else -3, -10 if m.mass < 0 else 3),
                    textcoords="offset points", ha="left" if right else "right")

    for r in rashba or []:
        if r.k0 <= 0:
            continue
        yc, ye = r.E_crossing - E0, r.E_extremum - E0
        ax.plot([r.x_crossing, r.x_extremum], [ye, ye], color="tab:green", lw=1, ls="--", zorder=4)
        ax.plot([r.x_crossing, r.x_crossing], [yc, ye], color="tab:green", lw=1, ls="--", zorder=4)
        ax.scatter([r.x_crossing, r.x_extremum], [yc, ye], color="tab:green", s=20, zorder=5)
        right = r.x_extremum >= r.x_crossing
        ax.annotate(rf"$\alpha_R$={r.alpha:.2f}", (r.x_extremum, ye), fontsize=7, color="tab:green",
                    xytext=(4 if right else -4, -11 if r.kind == "conduction" else 5),
                    textcoords="offset points", ha="left" if right else "right")

    for sp in splittings or []:
        xk = x[sp.kindex]
        s = 0
        lo, hi = bs.energies[s, sp.kindex, sp.bands[0]] - E0, bs.energies[s, sp.kindex, sp.bands[1]] - E0
        ax.annotate("", (xk, hi), (xk, lo), arrowprops=dict(arrowstyle="<->", color="tab:purple", lw=1))
        ax.annotate(f"{sp.delta_E * 1e3:.0f} meV", (xk, 0.5 * (lo + hi)), fontsize=7,
                    color="tab:purple", xytext=(3, 0), textcoords="offset points", va="center")

    # ---- axes
    ticks = bs.ticks()
    for xt, _ in ticks:
        ax.axvline(xt, color="k", lw=0.6)
    ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks])
    ax.set_xlim(x[0], x[-1])
    if energy_range is not None:
        ax.set_ylim(*energy_range)
    ax.axhline(bs.efermi - E0, color="0.4", lw=0.7, ls="--", zorder=0)
    ax.set_ylabel(_YLABEL.get(reference if isinstance(reference, str) or reference is None else None,
                              "Energy (eV)"))
    if title:
        ax.set_title(title)
    if legend and handles:
        ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    if filename:
        ax.figure.tight_layout()
        ax.figure.savefig(filename, dpi=dpi)
    return ax


def plot_splitting(bs: BandStructure, bands: tuple[int, int], ax=None, spin: int = 0,
                   color="tab:purple", filename=None, dpi=300):
    """Splitting ΔE(k) = E_upper - E_lower of a band pair along the path (meV)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2.5))
    dE = band_splitting(bs, bands, spin) * 1e3
    x = bs.distances
    for a, b in bs.segments:
        ax.plot(x[a:b + 1], dE[a:b + 1], color=color)
    ticks = bs.ticks()
    for xt, _ in ticks:
        ax.axvline(xt, color="k", lw=0.6)
    ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(f"ΔE bands {bands[0] + 1}/{bands[1] + 1} (meV)")
    if filename:
        ax.figure.tight_layout()
        ax.figure.savefig(filename, dpi=dpi)
    return ax
