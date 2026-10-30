"""Reader for VASP KPOINTS files (line mode and explicit lists)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KpointsFile:
    mode: str                                   # 'line', 'explicit' or 'automatic'
    cartesian: bool = False
    ndiv: int | None = None                     # points per segment (line mode)
    # line mode: (start, end) of every segment
    line_points: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    line_labels: list[tuple[str, str]] = field(default_factory=list)
    # explicit mode
    kpoints: np.ndarray | None = None
    weights: np.ndarray | None = None
    labeled_points: list[tuple[np.ndarray, str]] = field(default_factory=list)


def _split_label(line: str) -> tuple[list[str], str]:
    """Return (numeric tokens, label). Labels are given after '!' or as an extra column."""
    if "!" in line:
        data, comment = line.split("!", 1)
        tokens = data.split()
        label = comment.strip().split()[0] if comment.strip() else ""
    else:
        tokens = line.split()
        label = ""
    return tokens, label


def read_kpoints(path) -> KpointsFile:
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) < 3:
        raise ValueError(f"{path}: too short to be a KPOINTS file")

    try:
        nk_line = int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: cannot read line 2 of KPOINTS") from exc
    mode_char = lines[2].strip()[:1].lower()

    if mode_char == "l":
        cartesian = lines[3].strip()[:1].lower() in ("c", "k")
        points: list[tuple[np.ndarray, str]] = []
        for line in lines[4:]:
            if not line.strip():
                continue
            tokens, label = _split_label(line)
            if len(tokens) < 3:
                continue
            try:
                coord = np.array([float(t) for t in tokens[:3]])
            except ValueError:
                continue
            if not label and len(tokens) >= 4:
                label = tokens[3]
            points.append((coord, label))
        if len(points) % 2:
            points = points[:-1]
        line_points = [(points[i][0], points[i + 1][0]) for i in range(0, len(points), 2)]
        line_labels = [(points[i][1], points[i + 1][1]) for i in range(0, len(points), 2)]
        return KpointsFile("line", cartesian, nk_line, line_points, line_labels)

    if nk_line == 0:
        return KpointsFile("automatic")

    # explicit list: line 3 is the coordinate system
    cartesian = mode_char in ("c", "k")
    coords, weights, labeled = [], [], []
    for line in lines[3:3 + nk_line]:
        tokens, label = _split_label(line)
        if len(tokens) < 3:
            continue
        coord = np.array([float(t) for t in tokens[:3]])
        weight = float(tokens[3]) if len(tokens) > 3 else 1.0
        if not label and len(tokens) >= 5:
            label = tokens[4]
        coords.append(coord)
        weights.append(weight)
        if label:
            labeled.append((coord, label))
    return KpointsFile("explicit", cartesian, kpoints=np.array(coords), weights=np.array(weights),
                       labeled_points=labeled)
