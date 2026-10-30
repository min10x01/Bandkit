"""Core data container for an electronic band structure along a k-path.

Conventions
-----------
* Energies are in eV, reciprocal-space distances in 1/Angstrom (2*pi included).
* ``energies`` always has shape ``(nspin, nkpts, nbands)``. ``nspin`` is 2 for
  collinear spin-polarised runs (ISPIN=2) and 1 otherwise (including SOC /
  non-collinear runs, where every band holds one electron).
* All indices in the Python API (spin, k-point, band, ion) are 0-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# hbar^2 / m_e in eV * Angstrom^2
HBAR2_OVER_ME = 7.619964231070681

_LABEL_ALIASES = {
    "G": "Γ", "GAMMA": "Γ", "\\GAMMA": "Γ", "$\\GAMMA$": "Γ", "Γ": "Γ",
    "SIGMA": "Σ", "\\SIGMA": "Σ", "DELTA": "Δ", "\\DELTA": "Δ",
    "LAMBDA": "Λ", "\\LAMBDA": "Λ",
}


def pretty_label(label: str) -> str:
    """Normalise common spellings of high-symmetry labels (G, GAMMA, \\Gamma -> Γ)."""
    label = label.strip()
    key = label.upper().replace("$", "")
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    # keep subscripts like M_1 readable
    return label.replace("$", "")


@dataclass
class PathDirection:
    """A walk along one straight k-path segment, starting at a given k-point.

    ``indices[0]`` is the starting k-point and ``dk[i]`` is the Cartesian
    distance (1/Angstrom) between ``indices[i]`` and the start.
    """

    indices: np.ndarray
    dk: np.ndarray
    segment: int
    label: str
    interior: bool = False  # start point lies strictly inside the segment


@dataclass
class BandStructure:
    kpoints: np.ndarray                      # (nk, 3) fractional coordinates
    energies: np.ndarray                     # (nspin, nk, nb) eV
    reciprocal_lattice: np.ndarray           # (3, 3) rows b1, b2, b3 in 1/Angstrom, incl. 2*pi
    efermi: float
    occupations: np.ndarray | None = None    # (nspin, nk, nb)
    weights: np.ndarray | None = None        # (nk,)
    nelect: float | None = None
    noncollinear: bool = False
    # orbital projections (from PROCAR / vasprun.xml)
    projections: np.ndarray | None = None    # (nspin, nk, nb, nions, norb)
    spin_texture: np.ndarray | None = None   # (nk, nb, 3) <sigma_x,y,z>, non-collinear only
    orbital_names: list[str] = field(default_factory=list)
    ion_symbols: list[str] = field(default_factory=list)
    # k-path description
    segment_size: int | None = None          # points per segment in line-mode KPOINTS
    line_labels: list[tuple[str, str]] | None = None    # (start, end) label of every segment
    labeled_points: list[tuple[np.ndarray, str]] | None = None  # explicit (frac coord, label)
    source: str = ""

    def __post_init__(self):
        self.kpoints = np.asarray(self.kpoints, dtype=float).reshape(-1, 3)
        self.energies = np.asarray(self.energies, dtype=float)
        if self.energies.ndim == 2:
            self.energies = self.energies[None]
        self.reciprocal_lattice = np.asarray(self.reciprocal_lattice, dtype=float)
        if self.energies.shape[1] != len(self.kpoints):
            raise ValueError(
                f"energies have {self.energies.shape[1]} k-points but {len(self.kpoints)} "
                "k-point coordinates were given"
            )
        self.kcart = self.kpoints @ self.reciprocal_lattice
        self.segments = _detect_segments(self.kcart, self.segment_size)
        self.distances = _path_distances(self.kcart, self.segments)
        self.labels = self._assign_labels()

    # ------------------------------------------------------------------ sizes
    @property
    def nspin(self) -> int:
        return self.energies.shape[0]

    @property
    def nkpts(self) -> int:
        return self.energies.shape[1]

    @property
    def nbands(self) -> int:
        return self.energies.shape[2]

    @property
    def nions(self) -> int:
        return 0 if self.projections is None else self.projections.shape[3]

    @property
    def has_projections(self) -> bool:
        return self.projections is not None

    # ------------------------------------------------------------- k-path API
    def _assign_labels(self) -> dict[int, str]:
        labels: dict[int, str] = {}
        if self.line_labels and len(self.line_labels) == len(self.segments):
            for (s, e), (ls, le) in zip(self.segments, self.line_labels):
                labels[s] = pretty_label(ls)
                labels[e] = pretty_label(le)
        if self.labeled_points:
            endpoints = {i for seg in self.segments for i in seg}
            for coord, lab in self.labeled_points:
                for i in endpoints:
                    if i not in labels and np.allclose(self.kpoints[i], coord, atol=1e-5):
                        labels[i] = pretty_label(lab)
        return labels

    def label(self, kindex: int) -> str:
        """Label of a k-point, or its fractional coordinates if it has none."""
        if self.labels.get(kindex):
            return self.labels[kindex]
        return "(" + ",".join(f"{c:.3f}" for c in self.kpoints[kindex]) + ")"

    def ticks(self) -> list[tuple[float, str]]:
        """(x position, label) of every segment boundary, merged for plotting."""
        ticks: list[tuple[float, str]] = []
        for n, (s, e) in enumerate(self.segments):
            if n == 0:
                ticks.append((self.distances[s], self.labels.get(s, "")))
            else:
                prev_end = self.segments[n - 1][1]
                a, b = self.labels.get(prev_end, ""), self.labels.get(s, "")
                if prev_end == s or a == b or not a or not b:
                    lab = a or b
                elif np.allclose(self.kpoints[prev_end], self.kpoints[s], atol=1e-6):
                    lab = a
                else:
                    lab = f"{a}|{b}"
                ticks[-1] = (ticks[-1][0], lab)
            ticks.append((self.distances[e], self.labels.get(e, "")))
        return ticks

    def segment_of(self, kindex: int) -> int:
        for n, (s, e) in enumerate(self.segments):
            if s <= kindex <= e:
                return n
        raise IndexError(kindex)

    def segment_name(self, n: int) -> str:
        s, e = self.segments[n]
        return f"{self.label(s)}–{self.label(e)}"

    def equivalent_indices(self, kindex: int, tol: float = 1e-6) -> list[int]:
        """All path indices with the same fractional coordinates as ``kindex``."""
        k0 = self.kpoints[kindex]
        return [int(i) for i in np.flatnonzero(np.all(np.abs(self.kpoints - k0) < tol, axis=1))]

    def is_trim(self, kindex: int, tol: float = 1e-4) -> bool:
        """True if the k-point is a time-reversal invariant momentum (2k = G)."""
        twice = 2 * self.kpoints[kindex]
        return bool(np.all(np.abs(twice - np.round(twice)) < tol))

    def directions_from(self, kindex: int) -> list[PathDirection]:
        """Every straight walk along the path that starts at this k-point.

        A high-symmetry point that ends one segment and starts the next gives
        two directions; a point inside a segment gives two opposite ones.
        Repeated visits of the same k-point anywhere on the path are included.
        """
        origins = self.equivalent_indices(kindex)
        dirs: list[PathDirection] = []
        for n, (s, e) in enumerate(self.segments):
            if e == s:
                continue
            for o in origins:
                if o == s:
                    walks = [(np.arange(s, e + 1), False)]
                elif o == e:
                    walks = [(np.arange(e, s - 1, -1), False)]
                elif s < o < e:
                    walks = [(np.arange(o, s - 1, -1), True), (np.arange(o, e + 1), True)]
                else:
                    continue
                for idx, interior in walks:
                    dk = np.linalg.norm(self.kcart[idx] - self.kcart[idx[0]], axis=1)
                    label = f"{self.label(int(idx[0]))}→{self.label(int(idx[-1]))}"
                    dirs.append(PathDirection(idx, dk, n, label, interior))
        return dirs

    # ---------------------------------------------------------------- helpers
    def shifted_energies(self, reference: str | float | None = "fermi") -> np.ndarray:
        """Energies relative to ``'fermi'``, ``'vbm'``, a number, or absolute (None)."""
        return self.energies - self.reference_energy(reference)

    def reference_energy(self, reference: str | float | None = "fermi") -> float:
        if reference is None:
            return 0.0
        if isinstance(reference, (int, float)):
            return float(reference)
        if reference == "fermi":
            return self.efermi
        if reference == "vbm":
            from .analysis.edges import find_band_edges

            edges = find_band_edges(self)
            return edges.vbm.energy if edges.vbm is not None else self.efermi
        raise ValueError(f"unknown energy reference {reference!r}")

    def summary(self) -> str:
        spin = "non-collinear (SOC)" if self.noncollinear else ("spin-polarised" if self.nspin == 2 else "non-spin-polarised")
        lines = [
            f"source        : {self.source}",
            f"k-points      : {self.nkpts} in {len(self.segments)} segments "
            f"({' '.join(lab for _, lab in self.ticks())})",
            f"bands         : {self.nbands} ({spin})",
            f"E-fermi       : {self.efermi:.4f} eV",
        ]
        if self.has_projections:
            species = sorted(set(self.ion_symbols), key=self.ion_symbols.index)
            lines.append(f"projections   : {self.nions} ions ({', '.join(species)}), "
                         f"orbitals {' '.join(self.orbital_names)}")
            if self.spin_texture is not None:
                lines.append("spin texture  : available (sx, sy, sz)")
        return "\n".join(lines)


def _detect_segments(kcart: np.ndarray, segment_size: int | None = None) -> list[tuple[int, int]]:
    """Split the k-point list into straight segments ``(start, stop)`` (inclusive).

    With a line-mode KPOINTS the segment length is known. Otherwise segments
    are broken at repeated points, jumps and changes of direction.
    """
    nk = len(kcart)
    if nk == 0:
        return []
    if segment_size and segment_size > 1 and nk % segment_size == 0:
        return [(s, s + segment_size - 1) for s in range(0, nk, segment_size)]
    if nk == 1:
        return [(0, 0)]

    steps = np.diff(kcart, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    tiny = 1e-8
    # classify every step j (from point j to j+1): repeated point, jump or regular
    hard = norms < tiny
    for j in np.flatnonzero(~hard):
        neighbours = [norms[n] for n in (j - 1, j + 1) if 0 <= n < len(norms) and norms[n] > tiny]
        if neighbours and norms[j] > 5 * max(neighbours):
            hard[j] = True

    segments = []
    start = 0
    for i in range(1, nk):
        if hard[i - 1]:                                   # break between i-1 and i
            segments.append((start, i - 1))
            start = i
        elif i + 1 < nk and not hard[i]:
            cosang = steps[i - 1] @ steps[i] / (norms[i - 1] * norms[i])
            if cosang < 0.9999:                           # kink: shared endpoint
                segments.append((start, i))
                start = i
    segments.append((start, nk - 1))
    return segments


def _path_distances(kcart: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    x = np.zeros(len(kcart))
    for n, (s, e) in enumerate(segments):
        if n > 0 and s != segments[n - 1][1]:
            x[s] = x[segments[n - 1][1]]
        if e > s:
            steps = np.linalg.norm(np.diff(kcart[s:e + 1], axis=0), axis=1)
            x[s + 1:e + 1] = x[s] + np.cumsum(steps)
    return x
