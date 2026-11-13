from pathlib import Path

import numpy as np

from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
)

PROFILES = [
    ("straight", HeadingRateProfile(0.0, 0.0)),
    ("arc", HeadingRateProfile(90.0, 90.0, interpolation="linear")),
    (
        "s-curve",
        HeadingRateProfile(
            0.0,
            0.0,
            changes=(HeadingRateChange(0.30, 240.0), HeadingRateChange(0.70, -240.0)),
        ),
    ),
    (
        "loop-like",
        HeadingRateProfile(
            80.0,
            20.0,
            changes=(HeadingRateChange(0.25, 480.0), HeadingRateChange(0.65, 220.0)),
        ),
    ),
]

rows = []
for index, (name, profile) in enumerate(PROFILES):
    sample = Trajectory2D(
        length=140.0,
        start_heading_deg=0.0,
        heading_rate=profile,
    ).evaluate(count=513)
    xy = sample.xy.copy()
    xy[:, 1] *= -1
    xy += np.array([40.0, 70.0 + index * 120.0])
    points = " ".join(f"{x:.3f},{y:.3f}" for x, y in xy)
    rows.append(f'<text x="15" y="{25 + index * 120}" font-size="14">{name}</text>')
    rows.append(f'<polyline fill="none" stroke="black" stroke-width="2" points="{points}"/>')

svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="520" viewBox="0 0 520 520">'
    + "".join(rows)
    + "</svg>"
)
Path("curvature_gallery.svg").write_text(svg, encoding="utf-8")
print("wrote curvature_gallery.svg")
