# Transforms

`Affine2D` represents a 2D linear map followed by a translation. Its matrix convention is

```text
[x']   [m00 m01] [x]   [dx]
[y'] = [m10 m11] [y] + [dy]
```

The constructors cover identity, translation, uniform or nonuniform scale, rotation in degrees, and a basis transform:

```python
from planarcurve.transforms import Affine2D, Similarity2D

identity = Affine2D.identity()
move = Affine2D.translation_only(4.0, -2.0)
scale = Affine2D.scale(2.0, 0.5)
rotate = Affine2D.rotation(30.0)
basis = Affine2D.from_basis(origin=(1, 2), x_axis=(2, 0), y_axis=(0, 3))
custom = Affine2D.from_scale_shear_translation(
    scale_x=2.0, scale_y=0.5, shear_x=0.25, dx=4.0, dy=-2.0
)
```

The `scale_x`/`shear_x` constructor uses `x' = scale_x * (x + shear_x*y) + dx` and `y' = scale_y*y + dy`.

## Apply, inspect, and invert

```python
points = [(0.0, 0.0), (1.0, 0.0)]
placed = custom.apply(points)
vectors = custom.apply_vectors(points)  # no translation
area_scale = custom.determinant
restored = custom.inverse().apply(placed)
```

`determinant` is the determinant of the linear component. `inverse()` raises `ValueError` for singular transforms. Both point and vector methods accept one point or arrays with final dimension 2.

## Composition order

`a.compose(b)` means `a ∘ b`: first apply `b`, then apply `a`.

```python
placement = Affine2D.translation_only(10.0, 0.0).compose(Affine2D.rotation(90.0))
result = placement.apply([(1.0, 0.0)])  # rotate to (0, 1), then translate to (10, 1)
```

`Similarity2D` packages a positive uniform scale, rotation in degrees, and translation. `as_affine()` returns the equivalent `Affine2D`, and `apply()` transforms points in the same order: uniform scale and rotation, followed by translation. Similarities are useful when occurrences may change overall size and placement but should preserve shape.
