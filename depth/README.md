# Depth from a single image

Two scripts, numpy only.

```bash
python depth/make_reference.py --data "$DATA" --out work/depth_ref
python depth/eval_depth.py --ref work/depth_ref --pred work/my_model [--json work/my_model.json]
```

`make_reference.py` turns the scanner's point clouds (`$DATA/reconstructions/scanner/D<NN>/S<nn>.ply`) into reference
depth maps: 1600 x 1600, metres along the scanner's optical axis, NaN where nothing was measured (about half of each
picture is measured). By default it writes the first step (`S01`) of D03, D04, D10, D11 and D13; `--desktops` and
`--step` choose others.

`eval_depth.py` reads one prediction per reference map under the same file name (`D03_S01.npy`, ...), in metres. A
prediction at another resolution is resized to the reference grid. For every image it computes

| | measures |
|---|---|
| as predicted | MAE (m), RMSE (m), AbsRel, share of pixels with max(p / r, r / p) < 1.25, and the scale: the median of predicted over measured depth |
| after scale alignment | the prediction divided by that scale: MAE (m), RMSE (m), AbsRel, share of pixels within 1.05 |

and prints the mean over the images, each image counting the same.

The picture a model sees is `$DATA/annotations/scanner/D<NN>/images/S<nn>.png`.
