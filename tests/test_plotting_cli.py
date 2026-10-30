import matplotlib.pyplot as plt
import pytest

from bandkit import band_edge_masses, classify_splitting, edge_pair, find_rashba
from bandkit.cli import main
from bandkit.plotting import plot_bands, plot_splitting


@pytest.mark.parametrize("kwargs", [
    {},
    {"projections": ["Bi:p", "Te:p"], "mode": "fat"},
    {"projections": ["Bi:p"], "mode": "gradient"},
    {"projections": ["Bi", {"atoms": "Te", "orbitals": ["s"], "color": "k"}], "mode": "dominant"},
    {"spin_texture": "y", "reference": "vbm"},
    {"reference": None, "energy_range": None, "edges": True},
])
def test_plot_modes(bs, tmp_path, kwargs):
    out = tmp_path / "bands.png"
    ax = plot_bands(bs, filename=out, **kwargs)
    assert out.stat().st_size > 0
    assert [t.get_text() for t in ax.get_xticklabels()] == ["X", "Γ", "M"]
    plt.close("all")


def test_plot_overlays(bs, tmp_path):
    masses = [m for ms in band_edge_masses(bs).values() for m in ms]
    rashba = find_rashba(bs, "cbm") + find_rashba(bs, "vbm")
    pair, k, spin, _ = edge_pair(bs, "cbm")
    splits = [classify_splitting(bs, pair, k, spin)]
    plot_bands(bs, effective_masses=masses, rashba=rashba, splittings=splits, edges=True,
               filename=tmp_path / "overlay.png")
    plot_splitting(bs, pair, filename=tmp_path / "split.png")
    plt.close("all")


def test_cli_commands(rashba_dir, tmp_path, capsys):
    d = str(rashba_dir)
    assert main(["info", d]) == 0
    assert "gap 1.4148 eV (indirect)" in capsys.readouterr().out
    assert main(["emass", d]) == 0
    assert "+0.2000 m_e" in capsys.readouterr().out
    assert main(["split", d]) == 0
    out = capsys.readouterr().out
    assert "alpha_R = 2.000" in out and "alpha_R = 1.000" in out
    assert main(["split", d, "--bands", "5", "6"]) == 0
    assert main(["orbitals", d, "--band", "5", "--kpoint", "101", "--by", "element"]) == 0
    assert "Bi" in capsys.readouterr().out
    out_png = tmp_path / "cli.png"
    assert main(["plot", d, "-o", str(out_png), "--proj", "Bi:p", "--rashba", "--emass",
                 "--splitting", "--splitting-panel", "cbm"]) == 0
    assert out_png.exists() and (tmp_path / "cli_splitting_cbm.png").exists()
    plt.close("all")


def test_cli_error(tmp_path, capsys):
    assert main(["info", str(tmp_path)]) == 1
    assert "error" in capsys.readouterr().err
