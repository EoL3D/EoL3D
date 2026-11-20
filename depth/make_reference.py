"""Reference depth maps from the scanner's 3D reconstructions.

    python depth/make_reference.py --data $DATA --out ref [--desktops D03 D04 D10 D11 D13] [--step S01]

Reads $DATA/reconstructions/scanner/D<NN>/S<nn>.ply and writes <out>/D<NN>_S<nn>.npy: a 1600 x 1600 float32 map of the
measured depth in metres along the scanner's optical axis, NaN where the scanner measured nothing. Each point goes to
its pixel of the scanner picture (the calibration is in reconstructions/eol3d_ply.py); where two points share a pixel the nearer one
is kept. The default desktops and step are the ones the dataset's depth results were computed on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reconstructions"))
from eol3d_ply import SIZE, pixels, read_ply  # noqa: E402

DESKTOPS = ["D03", "D04", "D10", "D11", "D13"]


def reference(ply: Path) -> np.ndarray:
    v = read_ply(ply)
    u, w = pixels(v)
    ok = (u >= 0) & (u < SIZE) & (w >= 0) & (w < SIZE) & (v["z"] > 0)
    z = v["z"][ok].astype(np.float32)
    flat = w[ok] * SIZE + u[ok]
    order = np.lexsort((z, flat))                            # by pixel, nearest point first
    pix, first = np.unique(flat[order], return_index=True)   # the nearest point of every pixel that has one
    out = np.full(SIZE * SIZE, np.nan, np.float32)
    out[pix] = z[order][first] / 1000.0                      # millimetres -> metres
    return out.reshape(SIZE, SIZE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="the folder the dataset was downloaded into")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--desktops", nargs="+", default=DESKTOPS)
    ap.add_argument("--step", default="S01")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.desktops:
        ref = reference(a.data / "reconstructions" / "scanner" / d / f"{a.step}.ply")
        np.save(a.out / f"{d}_{a.step}.npy", ref)
        seen = np.isfinite(ref)
        print(f"{d} {a.step}: depth at {seen.mean():.1%} of the pixels, {np.nanmin(ref):.3f} to {np.nanmax(ref):.3f} m")


if __name__ == "__main__":
    main()
