"""Reading the dataset's point clouds (binary little-endian PLY, one vertex element) with numpy alone."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# scanner calibration: the pixel of a point is u = FX x / z + CX, v = FY y / z + CY (pictures are 1600 x 1600)
FX, FY, CX, CY = 2939.196, 2938.742, 824.546, 830.927
SIZE = 1600
_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "ushort": "<u2", "uint16": "<u2",
          "short": "<i2", "int": "<i4", "uint": "<u4"}


def read_ply(path: str | Path) -> np.ndarray:
    """The vertices of a PLY as a structured array; its fields are the file's own properties (x, y, z, red, green,
    blue and, in reconstructions/scanner/, instance and category)."""
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            head += line
        text = head.decode("ascii")
        if "format binary_little_endian 1.0" not in text:
            raise ValueError(f"{path}: not a binary little-endian PLY")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        dtype = np.dtype([(name, _TYPES[kind]) for kind, name in re.findall(r"property (\w+) (\w+)", text)])
        return np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)


def pixels(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column and row of each point in its scanner picture (rounded to the pixel)."""
    z = v["z"].astype(np.float64)
    u = np.rint(FX * v["x"] / z + CX).astype(np.int64)
    w = np.rint(FY * v["y"] / z + CY).astype(np.int64)
    return u, w
