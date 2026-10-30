import re
import shutil

import numpy as np
import pytest
import synthetic
from conftest import to_band_structure

from bandkit import BandStructure, load_band_structure
from bandkit.io import read_eigenval, read_kpoints, read_outcar, read_poscar, read_procar, read_vasprun


def test_kpoints_line_mode(rashba_dir):
    kp = read_kpoints(rashba_dir / "KPOINTS")
    assert kp.mode == "line" and kp.ndiv == synthetic.NDIV
    assert kp.line_labels == [("X", "G"), ("G", "M")]
    np.testing.assert_allclose(kp.line_points[1][1], [0.5, 0.5, 0])


def test_outcar_and_poscar(rashba_dir, model):
    oc = read_outcar(rashba_dir / "OUTCAR")
    assert oc["efermi"] == pytest.approx(model.efermi)
    assert oc["nelect"] == 4 and oc["noncollinear"] is True
    np.testing.assert_allclose(oc["reciprocal_lattice"], model.rec, atol=1e-6)
    pc = read_poscar(rashba_dir / "POSCAR")
    assert pc["ion_symbols"] == ["Bi", "Te"]
    np.testing.assert_allclose(pc["reciprocal_lattice"], model.rec)


def test_all_readers_agree(rashba_dir, model):
    vr = read_vasprun(rashba_dir / "vasprun.xml")
    pr = read_procar(rashba_dir / "PROCAR")
    ev = read_eigenval(rashba_dir / "EIGENVAL")
    for data in (vr, pr, ev):
        np.testing.assert_allclose(data["kpoints"], model.kpoints, atol=1e-6)
        np.testing.assert_allclose(data["energies"], model.energies, atol=1e-5)
    assert ev["nelect"] == 4
    for data in (vr, pr):
        assert data["noncollinear"]
        assert data["projections"].shape == (1, 202, 6, 2, 9)
        np.testing.assert_allclose(data["projections"], model.projections, atol=1e-3)
        np.testing.assert_allclose(data["spin_texture"], model.spin_texture * 0.9, atol=1e-2)
        assert data["orbital_names"] == synthetic.ORBITALS
    assert vr["ion_symbols"] == ["Bi", "Te"]
    assert vr["ndiv"] == synthetic.NDIV
    np.testing.assert_allclose(vr["reciprocal_lattice"], model.rec, atol=1e-6)


def test_procar_fused_negative_coordinates(tmp_path):
    m = synthetic.rashba_model()
    m.kpoints = -m.kpoints
    synthetic.write_procar(m, tmp_path / "PROCAR", fuse_negative=True)
    assert re.search(r"\d-\d", (tmp_path / "PROCAR").read_text())   # numbers run together
    np.testing.assert_allclose(read_procar(tmp_path / "PROCAR")["kpoints"], m.kpoints, atol=1e-7)


def test_load_paths_equivalent(rashba_dir, tmp_path):
    full = load_band_structure(rashba_dir)
    for files in (["KPOINTS", "POSCAR", "OUTCAR", "PROCAR"], ["KPOINTS", "POSCAR", "OUTCAR", "EIGENVAL"]):
        d = tmp_path / files[-1]
        d.mkdir()
        for f in files:
            shutil.copy(rashba_dir / f, d / f)
        other = load_band_structure(d)
        np.testing.assert_allclose(other.energies, full.energies, atol=1e-5)
        np.testing.assert_allclose(other.distances, full.distances, atol=1e-6)
        assert other.efermi == pytest.approx(full.efermi)
        assert other.ion_symbols == ["Bi", "Te"]
        assert other.noncollinear
        assert [lab for _, lab in other.ticks()] == ["X", "Γ", "M"]
    assert full.has_projections and full.spin_texture is not None


def test_efermi_override_and_no_projections(rashba_dir):
    bs = load_band_structure(rashba_dir, efermi=0.5, projections=False)
    assert bs.efermi == 0.5 and not bs.has_projections


def test_kpath_line_mode(bs):
    assert bs.segments == [(0, 100), (101, 201)]
    assert bs.labels[0] == "X" and bs.labels[100] == "Γ" and bs.labels[201] == "M"
    gx = np.pi / synthetic.A                     # |Γ-X| = 0.5 * 2π/a
    assert bs.distances[100] == pytest.approx(gx)
    assert bs.distances[101] == pytest.approx(gx)
    assert bs.distances[-1] == pytest.approx(gx * (1 + np.sqrt(2)))
    assert bs.is_trim(100) and bs.is_trim(0) and not bs.is_trim(50)


def test_kpath_heuristic_segments(model):
    m = to_band_structure(model)
    bs = BandStructure(kpoints=m.kpoints, energies=m.energies, reciprocal_lattice=m.reciprocal_lattice,
                       efermi=0.0)
    assert bs.segments == [(0, 100), (101, 201)]


def test_discontinuous_path_labels():
    rec = 2 * np.pi * np.eye(3) / 4
    k1 = np.linspace([0.5, 0, 0], [0, 0, 0], 11)
    k2 = np.linspace([0.5, 0.5, 0], [0, 0, 0], 11)
    k = np.vstack([k1, k2])
    E = np.linalg.norm(k, axis=1)[None, :, None]
    bs = BandStructure(kpoints=k, energies=E, reciprocal_lattice=rec, efermi=0.0, segment_size=11,
                       line_labels=[("X", "G"), ("M", "G")])
    assert [lab for _, lab in bs.ticks()] == ["X", "Γ|M", "Γ"]
    assert bs.distances[10] == bs.distances[11]
    # without line-mode information the jump is still detected
    bs2 = BandStructure(kpoints=k, energies=E, reciprocal_lattice=rec, efermi=0.0)
    assert bs2.segments == [(0, 10), (11, 21)]


def test_hybrid_weighted_kpoints_dropped(tmp_path):
    m = synthetic.rashba_model()
    scf = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.25, 0.25, 0]])
    m.kpoints = np.vstack([scf, m.kpoints])
    m.energies = np.concatenate([m.energies[:, :3] - 5, m.energies], axis=1)
    m.weights = np.r_[np.ones(3) / 3, np.zeros(202)]
    synthetic.write_eigenval(m, tmp_path / "EIGENVAL")
    synthetic.write_outcar(m, tmp_path / "OUTCAR")
    lines = ["explicit", str(len(m.kpoints)), "Reciprocal"]
    for i, (k, w) in enumerate(zip(m.kpoints, m.weights)):
        lab = {3: "X", 103: "G", 104: "G", 204: "M"}.get(i, "")
        lines.append(f"{k[0]:.8f} {k[1]:.8f} {k[2]:.8f} {w:.6f}" + (f" ! {lab}" if lab else ""))
    (tmp_path / "KPOINTS").write_text("\n".join(lines) + "\n")
    bs = load_band_structure(tmp_path)
    assert bs.nkpts == 202
    assert [lab for _, lab in bs.ticks()] == ["X", "Γ", "M"]
    np.testing.assert_allclose(bs.energies, synthetic.rashba_model().energies, atol=1e-5)


def test_spin_polarized_eigenval(tmp_path):
    m = synthetic.spin_polarized_model()
    synthetic.write_all(m, tmp_path, files=("KPOINTS", "POSCAR", "EIGENVAL", "OUTCAR"))
    bs = load_band_structure(tmp_path)
    assert bs.nspin == 2 and not bs.noncollinear
    np.testing.assert_allclose(bs.energies, m.energies, atol=1e-5)


def test_eigenval_without_occupations(tmp_path, model):
    synthetic.write_eigenval(model, tmp_path / "EIGENVAL", with_occ=False)
    data = read_eigenval(tmp_path / "EIGENVAL")
    assert data["occupations"] is None
    np.testing.assert_allclose(data["energies"], model.energies, atol=1e-5)


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_band_structure(tmp_path)
