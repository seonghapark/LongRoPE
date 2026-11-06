#!/usr/bin/env python3
"""Compare two safetensors model dirs without torch. Handles BF16 by hand.

numpy has no bfloat16, so we read the raw little-endian uint16 payload and
widen it to float32 by shifting into the high half of the word -- that is
exactly what a bf16->fp32 cast is.
"""
import json, struct, sys, os, glob
import numpy as np


def index(d):
    """tensor name -> (file, dtype, shape, byte_start, byte_end)"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            out[k] = (f, v["dtype"], tuple(v["shape"]), base + s, base + e)
    return out


def load(entry):
    f, dt, shape, s, e = entry
    with open(f, "rb") as fh:
        fh.seek(s)
        buf = fh.read(e - s)
    if dt == "BF16":
        u16 = np.frombuffer(buf, dtype="<u2")
        u32 = u16.astype(np.uint32) << 16
        a = u32.view(np.float32)
    elif dt == "F32":
        a = np.frombuffer(buf, dtype="<f4")
    elif dt == "F16":
        a = np.frombuffer(buf, dtype="<f2").astype(np.float32)
    else:
        raise RuntimeError("unhandled dtype " + dt)
    return a.reshape(shape), dt


A, B = sys.argv[1], sys.argv[2]
ia, ib = index(A), index(B)
print("in keys: %d  out keys: %d" % (len(ia), len(ib)))
print("only in IN :", sorted(set(ia) - set(ib))[:10])
print("only in OUT:", sorted(set(ib) - set(ia))[:10])

common = sorted(set(ia) & set(ib))
shape_bad = [k for k in common if ia[k][2] != ib[k][2]]
print("shape mismatches:", shape_bad[:10], "(%d)" % len(shape_bad))

# NaN/Inf sweep over every OUT tensor, plus delta on a sample.
nan_bad = []
for k in sorted(ib):
    a, dt = load(ib[k])
    if not np.isfinite(a).all():
        nan_bad.append((k, int(np.isnan(a).sum()), int(np.isinf(a).sum())))
print("NaN/Inf tensors:", nan_bad if nan_bad else "none")

sample = [k for k in common if any(t in k for t in (
    "embed_tokens", "layers.0.", "layers.13.", "layers.25.", "norm.weight", "lm_head"))]
print("\n%-58s %-7s %12s %12s" % ("tensor", "dtype", "max|delta|", "rel_rms"))
nchanged = 0
for k in sample[:24]:
    x, dt = load(ia[k])
    y, _ = load(ib[k])
    if x.shape != y.shape:
        print("%-58s SHAPE %s vs %s" % (k, x.shape, y.shape))
        continue
    d = np.abs(x - y)
    md = float(d.max())
    rms = float(np.sqrt((d.astype(np.float64) ** 2).mean()) /
                (np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-20))
    if md > 0:
        nchanged += 1
    print("%-58s %-7s %12.6g %12.6g" % (k, dt, md, rms))
print("\nsampled %d tensors, %d changed from the input checkpoint" % (len(sample[:24]), nchanged))
