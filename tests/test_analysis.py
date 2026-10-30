import numpy as np
import pytest
import synthetic
from conftest import to_band_structure

from bandkit import (BandStructure, band_character, band_edge_masses, classify_splitting,
                     effective_mass, exchange_splitting, find_band_edges, find_rashba,
                     parse_projection, projected_weight, rashba_parameters,
                     splitting_at_high_symmetry)

HB = synthetic.HBAR2_ME


# ----------------------------------------------------------------- band edges
def test_band_edges(bs):
    e = find_band_edges(bs)
    assert not e.is_metal and not e.direct
    assert (e.vbm.band, e.cbm.band) == (3, 4)
    assert e.vbm.energy == pytest.approx(synthetic.ALPHA_V**2 * synthetic.M_V / (2 * HB), abs=2e-4)
    assert e.cbm.energy == pytest.approx(synthetic.E_C - synthetic.ALPHA_C**2 * synthetic.M_C / (2 * HB),
                                         abs=2e-4)


def test_band_edges_without_nelect(model):
    bs = to_band_structure(model)
    bs.nelect = None
    e = find_band_edges(bs)
    assert (e.vbm.band, e.cbm.band) == (3, 4)


def test_metal_detection(model):
    bs = to_band_structure(model)
    bs.nelect, bs.efermi = None, 2.0          # Fermi level inside the conduction band
    assert find_band_edges(bs).is_metal


def test_spin_polarized_edges_and_exchange():
    bs = to_band_structure(synthetic.spin_polarized_model())
    e = find_band_edges(bs)
    assert e.vbm.spin == 1 and e.cbm.spin == 0
    assert e.gap == pytest.approx(1.0)
    assert exchange_splitting(bs, 0, 0) == pytest.approx(0.3)


# ------------------------------------------------------------- effective mass
def test_edge_masses(bs):
    m = band_edge_masses(bs)
    assert [r.mass for r in m["electron"]] == pytest.approx([synthetic.M_C], rel=1e-3)
    assert [r.mass for r in m["hole"]] == pytest.approx([-synthetic.M_V], rel=1e-3)


def test_mass_at_high_symmetry_point_all_directions(bs):
    gamma = 100
    res = effective_mass(bs, 0, gamma, npoints=4)
    assert sorted(r.direction for r in res) == ["Γ→M", "Γ→X"]
    for r in res:
        assert r.mass == pytest.approx(-1.0, rel=1e-3)


def test_mass_energy_window(bs):
    e = find_band_edges(bs).cbm
    res = effective_mass(bs, e.band, e.kindex, energy_window=0.02)
    assert res[0].mass == pytest.approx(synthetic.M_C, rel=1e-3)
    assert len(res[0].indices) > 3


# ------------------------------------------------------------------ splitting
@pytest.mark.parametrize("edge, alpha, m", [("cbm", synthetic.ALPHA_C, synthetic.M_C),
                                            ("vbm", synthetic.ALPHA_V, synthetic.M_V)])
def test_rashba_parameters(bs, edge, alpha, m):
    res = find_rashba(bs, edge)
    assert sorted(r.direction for r in res) == ["Γ→M", "Γ→X"]
    k0 = alpha * m / HB
    E_R = alpha**2 * m / (2 * HB)
    for r in res:
        assert r.crossing_label == "Γ" and r.is_trim
        assert r.degeneracy == pytest.approx(0, abs=1e-6)
        assert r.k0 == pytest.approx(k0, rel=1e-3)
        assert r.E_R == pytest.approx(E_R, rel=1e-3)
        assert r.alpha == pytest.approx(alpha, rel=1e-3)
        assert r.alpha_linear == pytest.approx(alpha, rel=1e-3)
        assert r.kind == ("conduction" if edge == "cbm" else "valence")


def test_rashba_explicit_pair(bs):
    res = rashba_parameters(bs, (4, 5), 100)
    assert res[0].kind == "conduction"
    assert res[0].alpha == pytest.approx(synthetic.ALPHA_C, rel=1e-3)


def test_no_rashba_when_kramers_degenerate(model):
    model.energies[0, :, 4] = model.energies[0, :, 5] = synthetic.E_C  # flat, degenerate
    bs = to_band_structure(model)
    assert find_rashba(bs, "cbm") == []


def test_classify_rashba(bs):
    for k, lab, dE in splitting_at_high_symmetry(bs, (4, 5)):
        c = classify_splitting(bs, (4, 5), k)
        assert c.kind == ("degenerate" if lab == "Γ" else "Rashba-type")
    assert classify_splitting(bs, (4, 5), 90).kind == "Rashba-type"


def test_classify_dresselhaus(model):
    kc = model.kpoints @ model.rec
    k = np.linalg.norm(kc, axis=1, keepdims=True)
    khat = np.divide(kc, k, out=np.zeros_like(kc), where=k > 0)
    model.spin_texture[:, 4] = khat
    model.spin_texture[:, 5] = -khat
    bs = to_band_structure(model)
    assert classify_splitting(bs, (4, 5), 90).kind == "Dresselhaus-type"


def test_classify_without_spin_texture(model):
    bs = to_band_structure(model, spin_texture=False)
    assert classify_splitting(bs, (4, 5), 90).kind == "Rashba/Dresselhaus-like"


def test_zeeman_classification():
    m = synthetic.zeeman_model()
    bs = to_band_structure(m)
    valley = 50                                           # (1/4, 1/4, 0)
    c = classify_splitting(bs, (0, 1), valley)
    assert c.kind == "Zeeman-type"
    assert c.delta_E == pytest.approx(m.extra["delta"])
    assert find_rashba(bs, "vbm") == []                   # never degenerate: not Rashba
    bs_nospin = to_band_structure(m, spin_texture=False)
    assert classify_splitting(bs_nospin, (0, 1), valley).kind == "Zeeman-like"
    assert classify_splitting(bs_nospin, (0, 1), 0).kind == "Zeeman/exchange-like"   # Γ is a TRIM


# ------------------------------------------------------------------- orbitals
def test_band_character(bs):
    e = find_band_edges(bs)
    ch = band_character(bs, e.vbm.band, e.vbm.kindex)
    assert list(ch)[:2] == ["Te-p", "Bi-s"]
    assert ch["Te-p"] == pytest.approx(0.7, abs=1e-3)
    assert band_character(bs, e.cbm.band, e.cbm.kindex, by="element")["Bi"] == pytest.approx(0.8, abs=1e-3)
    lm = band_character(bs, e.cbm.band, e.cbm.kindex, by="element-lm", threshold=0.01)
    assert set(lm) == {"Bi-px", "Bi-py", "Te-s"}
    assert band_character(bs, 0, 0, by="ion") == pytest.approx({"Te2": 1.0}, abs=1e-3)


def test_projected_weight(bs):
    w = projected_weight(bs, "Bi", "p")
    assert w.shape == (1, bs.nkpts, bs.nbands)
    np.testing.assert_allclose(w[0, :, 4], 0.8, atol=2e-3)
    np.testing.assert_allclose(w[0, :, 2], 0.0, atol=2e-3)
    total = projected_weight(bs)
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    assert projected_weight(bs, [1], ["pz"])[0, 0, 2] == pytest.approx(0.7, abs=2e-3)


def test_parse_projection():
    assert parse_projection("Bi:p") == (["Bi"], ["p"])
    assert parse_projection("Bi,Te") == (["Bi", "Te"], None)
    assert parse_projection(":d") == (None, ["d"])
    assert parse_projection("1-3:px,py") == ([0, 1, 2], ["px", "py"])


def test_projection_errors(bs):
    with pytest.raises(ValueError):
        projected_weight(bs, "Xe")
    with pytest.raises(ValueError):
        projected_weight(bs, None, "q")
    plain = BandStructure(kpoints=bs.kpoints, energies=bs.energies,
                          reciprocal_lattice=bs.reciprocal_lattice, efermi=0.0)
    with pytest.raises(ValueError, match="LORBIT"):
        band_character(plain, 0, 0)
