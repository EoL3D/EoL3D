# The scanner's point clouds

`$DATA/reconstructions/scanner/D<NN>/S<nn>.ply`: one point cloud per annotated scanner step, named like the picture it
belongs to (`annotations/scanner/D<NN>/images/S<nn>.png`). Binary little-endian PLY with one vertex element:

| property | type | meaning |
|---|---|---|
| `x`, `y`, `z` | float | millimetres, in the scanner camera frame (z away from the camera) |
| `red`, `green`, `blue` | uchar | colour |
| `instance` | ushort | the instance id of the part the point lies on (`labels/S<nn>.png` at the point's pixel); 0 = none |
| `category` | uchar | the COCO `category_id` of that instance; 0 = none |

- `eol3d_ply.py`: `read_ply(path)` returns the vertices as a numpy structured array; `pixels(v)` gives each point's
  column and row in the 1600 x 1600 picture.
- `show_step.py`: lists the parts of a step with their points, and writes one part as its own PLY.
- `parts3d.py` and `HOW_TO_EXTRACT_3D_PARTS.md`: for parts that are never fully in view, the route through the removal
  order (the points that are there before the part leaves and gone after), with a plan per part
  (`annotations/scanner/parts_3d_plan.csv`).

```python
from eol3d_ply import read_ply
v = read_ply("D13/S01.ply")
fan = v[v["instance"] == 19]            # names: annotations/scanner/D13/instances.json -> instances["19"]["key"]
coolers = v[v["category"] == 7]         # every point on a CPU cooler
```
