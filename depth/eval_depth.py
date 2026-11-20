"""Score depth predictions against the reference maps of make_reference.py.

    python depth/eval_depth.py --ref ref --pred preds/my_model [--json scores.json]

<pred> holds one array per reference map under the same file name (D03_S01.npy, ...): the predicted depth of every
pixel of that scanner picture, in metres (another unit only changes the unscaled columns). A prediction at another
resolution is resized to the reference grid (bilinear). Pixels count where both maps hold a positive, finite depth.

Per image, then averaged with equal weight per image:
  as predicted            MAE (m), RMSE (m), AbsRel, share of pixels with max(p/r, r/p) < 1.25, and the scale: the
                          median of predicted over reference depth (1 is exact)
  after scale alignment   the prediction divided by that median: MAE (m), RMSE (m), AbsRel, share with ratio < 1.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resize(depth: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of a 2-D map to h x w (pixel centres aligned, edges clamped)."""
    depth = np.asarray(depth, np.float32)
    if depth.shape == (h, w):
        return depth
    H, W = depth.shape
    y = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    x = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    fy, fx = (y - y0)[:, None].astype(np.float32), (x - x0)[None, :].astype(np.float32)
    top = depth[y0][:, x0] * (1 - fx) + depth[y0][:, x1] * fx
    bottom = depth[y1][:, x0] * (1 - fx) + depth[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


def errors(x: np.ndarray, r: np.ndarray) -> dict:
    th = np.maximum(x / r, r / x)
    return {"mae": float(np.abs(x - r).mean()), "rmse": float(np.sqrt(((x - r) ** 2).mean())),
            "absrel": float((np.abs(x - r) / r).mean()), "d125": float((th < 1.25).mean() * 100), "d105": float((th < 1.05).mean() * 100)}


def metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """The measures of one image (see the module text)."""
    pred = resize(pred, *ref.shape)
    m = np.isfinite(ref) & np.isfinite(pred) & (pred > 0) & (ref > 0)
    if not m.any():
        raise ValueError("no pixel holds both a predicted and a reference depth")
    p, r = pred[m].astype(np.float64), ref[m].astype(np.float64)
    scale = float(np.median(p / r))
    raw, ali = errors(p, r), errors(p / scale, r)
    return {"pixels": int(m.sum()), "scale": scale, "mae": raw["mae"], "rmse": raw["rmse"], "absrel": raw["absrel"], "d125": raw["d125"],
            "mae_aligned": ali["mae"], "rmse_aligned": ali["rmse"], "absrel_aligned": ali["absrel"], "d105": ali["d105"]}


def evaluate(ref_dir: Path, pred_dir: Path) -> dict:
    per = {}
    for f in sorted(ref_dir.glob("*.npy")):
        p = pred_dir / f.name
        if not p.exists():
            raise FileNotFoundError(f"no prediction for {f.name} in {pred_dir}")
        per[f.stem] = metrics(np.load(p), np.load(f))
    if not per:
        raise FileNotFoundError(f"no reference maps in {ref_dir}")
    keys = [k for k in next(iter(per.values())) if k != "pixels"]
    return {"images": len(per), "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in keys}, "per_image": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="also write the scores to this file")
    a = ap.parse_args()
    out = evaluate(a.ref, a.pred)
    m = out["mean"]
    print(f"{out['images']} images")
    print(f"as predicted          MAE {m['mae']:.3f} m   RMSE {m['rmse']:.3f} m   AbsRel {m['absrel']:.3f}   d<1.25 {m['d125']:.2f} %   scale {m['scale']:.3f}")
    print(f"after scale alignment MAE {m['mae_aligned']:.3f} m   RMSE {m['rmse_aligned']:.3f} m   AbsRel {m['absrel_aligned']:.3f}   d<1.05 {m['d105']:.2f} %")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
