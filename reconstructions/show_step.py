"""The parts of one step's point cloud, and optionally one part saved on its own.

    python reconstructions/show_step.py --data $DATA --desktop D13 --step S01
    python reconstructions/show_step.py --data $DATA --desktop D13 --step S01 --part cpu_cooler.fan.01 --out fan.ply

Reads $DATA/reconstructions/scanner/D<NN>/S<nn>.ply (every point carries the instance it lies on) and the part names
from $DATA/annotations/scanner/D<NN>/instances.json, and prints one line per part with its points. With --part it
writes that part's points (x y z red green blue) as a PLY that any viewer opens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eol3d_ply import read_ply


def write_ply(path: Path, v: np.ndarray) -> None:
    out = np.empty(len(v), np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
    for name in out.dtype.names:
        out[name] = v[name]
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(out)}", "property float x", "property float y",
            "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(out.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="the folder the dataset was downloaded into")
    ap.add_argument("--desktop", required=True, help="e.g. D13")
    ap.add_argument("--step", required=True, help="e.g. S01")
    ap.add_argument("--part", help="a part key from instances.json, e.g. cpu_cooler.fan.01")
    ap.add_argument("--out", type=Path, help="with --part: the PLY to write")
    a = ap.parse_args()

    v = read_ply(a.data / "reconstructions" / "scanner" / a.desktop / f"{a.step}.ply")
    inst = json.loads((a.data / "annotations" / "scanner" / a.desktop / "instances.json").read_text(encoding="utf-8"))["instances"]
    ids, counts = np.unique(v["instance"], return_counts=True)
    print(f"{a.desktop} {a.step}: {len(v):,} points, {int((v['instance'] > 0).sum()):,} on {int((ids > 0).sum())} parts")
    for i, n in sorted(zip(ids.tolist(), counts.tolist()), key=lambda t: -t[1]):
        if i:
            print(f"  {i:3d}  {inst[str(i)]['key']:34s} {inst[str(i)]['class']:18s} {n:9,d}")
    if a.part:
        match = [int(i) for i, rec in inst.items() if rec["key"] == a.part]
        if not match:
            raise SystemExit(f"no part named {a.part!r} on {a.desktop}")
        pts = v[v["instance"] == match[0]]
        if not len(pts):
            raise SystemExit(f"{a.part} has no measured points on {a.step} (hidden or already removed)")
        if not a.out:
            raise SystemExit("--part needs --out")
        write_ply(a.out, pts)
        print(f"wrote {len(pts):,} points of {a.part} to {a.out}")


if __name__ == "__main__":
    main()
