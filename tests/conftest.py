import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent))

import synthetic  # noqa: E402

from bandkit import BandStructure, load_band_structure  # noqa: E402


def to_band_structure(m: synthetic.Model, spin_texture=True) -> BandStructure:
    return BandStructure(
        kpoints=m.kpoints, energies=m.energies, reciprocal_lattice=m.rec, efermi=m.efermi,
        nelect=m.nelect, noncollinear=m.noncollinear, projections=m.projections,
        spin_texture=m.spin_texture if spin_texture else None,
        orbital_names=synthetic.ORBITALS[:m.projections.shape[-1]], ion_symbols=m.symbols,
        segment_size=m.ndiv, line_labels=[(a, b) for _, _, a, b in m.path],
    )


@pytest.fixture(scope="session")
def rashba_dir(tmp_path_factory):
    return synthetic.write_all(synthetic.rashba_model(), tmp_path_factory.mktemp("rashba"))


@pytest.fixture(scope="session")
def bs(rashba_dir):
    return load_band_structure(rashba_dir)


@pytest.fixture
def model():
    return synthetic.rashba_model()
