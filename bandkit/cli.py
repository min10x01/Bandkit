"""Command-line interface: ``bandkit <command> [directory] [options]``.

Band, k-point and ion numbers on the command line are 1-based, as in VASP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import analysis as an
from .io import load_band_structure


def _load(args, projections=True):
    return load_band_structure(args.path, efermi=args.efermi,
                               projections=projections and not getattr(args, "no_proj", False))


def _header(title):
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def _edge_states(bs, args):
    """[(name, band, kindex, spin)] from --band/--kpoint or the band edges."""
    if args.band is not None:
        k = args.kpoint - 1 if args.kpoint is not None else None
        if k is None:
            raise SystemExit("--band needs --kpoint")
        return [(f"band {args.band} @ {bs.label(k)}", args.band - 1, k, args.spin)]
    edges = an.find_band_edges(bs)
    out = []
    if edges.vbm is not None:
        out.append(("VBM", edges.vbm.band, edges.vbm.kindex, edges.vbm.spin))
    if edges.cbm is not None:
        out.append(("CBM", edges.cbm.band, edges.cbm.kindex, edges.cbm.spin))
    return out


def cmd_info(args):
    bs = _load(args)
    _header("Band structure")
    print(bs.summary())
    _header("Band edges")
    edges = an.find_band_edges(bs)
    print(edges)
    if bs.has_projections and not edges.is_metal:
        for name, e in (("VBM", edges.vbm), ("CBM", edges.cbm)):
            ch = an.band_character(bs, e.band, e.kindex, e.spin)
            print(f"{name} character: {an.format_character(ch)}")


def cmd_emass(args):
    bs = _load(args, projections=False)
    for name, band, k, spin in _edge_states(bs, args):
        _header(f"Effective mass: {name} (band {band + 1}, {bs.label(k)})")
        for m in an.effective_mass(bs, band, k, spin, npoints=args.npoints,
                                   energy_window=args.window, degree=args.degree):
            print(m)


def _print_splitting_pair(bs, pair, spin, kind, near, tol):
    lo, hi = pair
    print(f"bands {lo + 1}/{hi + 1}")
    print("splitting at high-symmetry points:")
    for k, lab, dE in an.splitting_at_high_symmetry(bs, pair, spin):
        print("  " + str(an.classify_splitting(bs, pair, k, spin, tol=tol)))
    dE = np.abs(an.band_splitting(bs, pair, spin))
    if np.mean(dE < tol) > 0.8:
        print("bands are degenerate along the path: no spin splitting (inversion symmetry?)")
        return []
    crossing = an.find_crossing(bs, pair, near, spin, tol)
    print(f"splitting at {bs.label(near)}: {an.classify_splitting(bs, pair, near, spin, tol=tol)}")
    if crossing is None:
        print("no degenerate crossing point: splitting is not Rashba-like")
        return []
    print(f"Rashba analysis around {bs.label(crossing)}:")
    results = an.rashba_parameters(bs, pair, crossing, kind, spin)
    for r in results:
        print("  " + str(r))
    return results


def cmd_split(args):
    bs = _load(args)
    if bs.nspin == 2:
        _header("Exchange splitting (spin ↓ − spin ↑)")
        for name, band, k, _ in _edge_states(bs, args):
            print(f"{name}: {an.exchange_splitting(bs, band, k) * 1e3:.2f} meV")
    if args.bands:
        pair = (args.bands[0] - 1, args.bands[1] - 1)
        near = args.kpoint - 1 if args.kpoint is not None else int(np.argmin(np.abs(an.band_splitting(bs, pair, args.spin))))
        _header("Band splitting")
        _print_splitting_pair(bs, pair, args.spin, None, near, args.tol)
        return
    for edge in ("vbm", "cbm"):
        found = an.edge_pair(bs, edge)
        if found is None:
            continue
        pair, k, spin, kind = found
        _header(f"Splitting at the {edge.upper()}")
        _print_splitting_pair(bs, pair, spin, kind, k, args.tol)


def cmd_orbitals(args):
    bs = _load(args)
    if not bs.has_projections:
        raise SystemExit("no projections found (need PROCAR or vasprun.xml with LORBIT=11)")
    for name, band, k, spin in _edge_states(bs, args):
        _header(f"Orbital character: {name} (band {band + 1}, {bs.label(k)})")
        for key, val in an.band_character(bs, band, k, spin, by=args.by, threshold=args.threshold).items():
            print(f"  {key:<12s} {100 * val:6.1f} %")


def cmd_plot(args):
    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    from .plotting import plot_bands, plot_splitting

    bs = _load(args, projections=bool(args.proj or args.spin_texture))
    emass = rashba = splits = None
    if args.emass:
        emass = [m for ms in an.band_edge_masses(bs, npoints=args.npoints).values() for m in ms]
    if args.rashba:
        rashba = an.find_rashba(bs, "vbm") + an.find_rashba(bs, "cbm")
    if args.splitting:
        splits = []
        for edge in ("vbm", "cbm"):
            found = an.edge_pair(bs, edge)
            if found:
                pair, k, spin, _ = found
                splits.append(an.classify_splitting(bs, pair, k, spin))
    reference = args.reference if args.reference in ("fermi", "vbm") else float(args.reference)
    ax = plot_bands(bs, reference=reference, energy_range=tuple(args.erange), spin=args.plot_spin,
                    projections=args.proj, mode=args.mode, fat_scale=args.fat_scale,
                    spin_texture=args.spin_texture, edges=args.edges or args.emass,
                    effective_masses=emass, rashba=rashba, splittings=splits, title=args.title,
                    filename=args.output)
    if args.splitting_panel:
        found = an.edge_pair(bs, args.splitting_panel)
        if found:
            out = None
            if args.output:
                o = Path(args.output)
                out = str(o.with_name(f"{o.stem}_splitting_{args.splitting_panel}{o.suffix}"))
            plot_splitting(bs, found[0], spin=found[2], filename=out)
            if out:
                print(f"saved {out}")
    if args.output:
        print(f"saved {args.output}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
    return ax


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bandkit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("path", nargs="?", default=".", help="calculation directory or vasprun.xml")
    common.add_argument("--efermi", type=float, help="Fermi energy (eV), e.g. from the SCF run")

    state = argparse.ArgumentParser(add_help=False)
    state.add_argument("--band", type=int, help="band number (1-based); default: band edges")
    state.add_argument("--kpoint", type=int, help="k-point number (1-based)")
    state.add_argument("--spin", type=int, default=0, help="spin channel (0 or 1)")

    s = sub.add_parser("info", parents=[common], help="summary, band gap and band-edge character")
    s.add_argument("--no-proj", action="store_true", help="do not read projections")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("emass", parents=[common, state], help="effective masses")
    s.add_argument("--npoints", type=int, default=5, help="points per side in the fit")
    s.add_argument("--window", type=float, help="energy window (eV) instead of --npoints")
    s.add_argument("--degree", type=int, default=2, help="polynomial degree")
    s.set_defaults(func=cmd_emass)

    s = sub.add_parser("split", parents=[common, state], help="Rashba / Zeeman / exchange splitting")
    s.add_argument("--bands", type=int, nargs=2, metavar=("LOWER", "UPPER"),
                   help="band pair (1-based); default: pairs at the VBM and CBM")
    s.add_argument("--tol", type=float, default=2e-3, help="degeneracy tolerance (eV)")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("orbitals", parents=[common, state], help="orbital contributions")
    s.add_argument("--by", default="element-orbital",
                   choices=["element", "orbital", "element-orbital", "element-lm", "ion"])
    s.add_argument("--threshold", type=float, default=0.01, help="hide contributions below this")
    s.set_defaults(func=cmd_orbitals)

    s = sub.add_parser("plot", parents=[common], help="plot the band structure")
    s.add_argument("-o", "--output", help="save to this file instead of showing")
    s.add_argument("--erange", type=float, nargs=2, default=[-3, 3], metavar=("EMIN", "EMAX"))
    s.add_argument("--reference", default="fermi", help="'fermi', 'vbm' or an energy in eV")
    s.add_argument("--plot-spin", type=int, choices=[0, 1], help="only this spin channel (ISPIN=2)")
    s.add_argument("--proj", action="append",
                   help="projection, e.g. 'Bi:p', 'I:s,p', '1-4:d' (repeatable)")
    s.add_argument("--mode", default="fat", choices=["fat", "gradient", "dominant"])
    s.add_argument("--fat-scale", type=float, default=40.0)
    s.add_argument("--spin-texture", choices=["x", "y", "z"], help="colour by spin component (SOC)")
    s.add_argument("--edges", action="store_true", help="mark VBM and CBM")
    s.add_argument("--emass", action="store_true", help="show effective-mass fits at the edges")
    s.add_argument("--npoints", type=int, default=5)
    s.add_argument("--rashba", action="store_true", help="mark Rashba parameters at the edges")
    s.add_argument("--splitting", action="store_true", help="mark band splitting at the edges")
    s.add_argument("--splitting-panel", choices=["vbm", "cbm"],
                   help="extra plot of ΔE(k) of the pair at the VBM or CBM")
    s.add_argument("--title")
    s.set_defaults(func=cmd_plot)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
