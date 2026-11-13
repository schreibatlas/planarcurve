import numpy as np

from planarcurve.polyline import polyline_length, resample_polyline_by_arclength
from planarcurve.svg import analyze_path, sample_path, transform_path
from planarcurve.transforms import Affine2D


def main() -> None:
    path = "M 0 0 C 20 0 30 30 50 30 Q 70 30 80 10"
    info = analyze_path(path, samples_per_curve=64)
    assert info.move_count == 1 and not info.closed
    print(info)

    transform = Affine2D.from_scale_shear_translation(
        scale_x=1.5, scale_y=0.75, shear_x=0.2, dx=10, dy=-5
    )
    transformed = transform_path(path, transform)
    transformed_info = analyze_path(transformed)
    samples = sample_path(transformed, samples_per_curve=64)
    assert np.allclose(transformed_info.end, transform.apply(info.end))
    assert np.allclose(samples[-1], transform.apply(info.end))
    print(transformed)

    uniform = resample_polyline_by_arclength(samples, 12)
    assert uniform.shape == (12, 2)
    assert np.allclose(uniform[[0, -1]], samples[[0, -1]])
    print("sampled length:", polyline_length(samples))
    print("uniform points:\n", np.array2string(uniform, precision=3))


if __name__ == "__main__":
    main()
