#!/usr/bin/env python3
"""Compare two directories of per-layer qtensors produced by a conversion.

Reads safetensors directly - no CUDA, no exllamav3, no torch - and compares the
raw tensor bytes, so what it reports is bit equality, not float closeness.

  usage: compare_qtensors.py A B [--layer N ...]

Exits 0 when every compared tensor matches, 1 otherwise.
"""
import json
import os
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
        data_start = 8 + n
    hdr.pop("__metadata__", None)
    return hdr, data_start


def read_raw(path, meta, data_start):
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        return f.read(end - start)


def differing_bytes(a, b):
    n = 0
    for i in range(0, min(len(a), len(b)), 65536):
        ca, cb = a[i:i + 65536], b[i:i + 65536]
        if ca != cb:
            n += sum(1 for x, y in zip(ca, cb) if x != y)
    n += abs(len(a) - len(b))
    return n


def compare_file(a, b, limit = 6):
    ha, da = read_header(a)
    hb, db = read_header(b)
    only_a = sorted(set(ha) - set(hb))
    only_b = sorted(set(hb) - set(ha))
    bad = []
    checked = 0
    for key in sorted(set(ha) & set(hb)):
        ma, mb = ha[key], hb[key]
        checked += 1
        if ma["shape"] != mb["shape"] or ma["dtype"] != mb["dtype"]:
            bad.append((key, "shape/dtype %s %s vs %s %s" %
                        (ma["shape"], ma["dtype"], mb["shape"], mb["dtype"])))
            continue
        ra = read_raw(a, ma, da)
        rb = read_raw(b, mb, db)
        if ra != rb:
            bad.append((key, "%d of %d bytes differ" % (differing_bytes(ra, rb), max(len(ra), len(rb)))))
    return checked, only_a, only_b, bad


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    layers = []
    rest = sys.argv[3:]
    i = 0
    while i < len(rest):
        if rest[i] == "--layer" and i + 1 < len(rest):
            layers.append(rest[i + 1])
            i += 2
        else:
            i += 1

    if not os.path.isdir(a_dir) or not os.path.isdir(b_dir):
        print("missing directory: %s / %s" % (a_dir, b_dir))
        return 2

    files = sorted(f for f in os.listdir(a_dir) if f.endswith(".safetensors"))
    if layers:
        want = {"model.language_model.layers.%s.safetensors" % l for l in layers}
        files = [f for f in files if f in want]
    if not files:
        print("no matching files in %s" % a_dir)
        return 2

    failed = 0
    for f in files:
        pa, pb = os.path.join(a_dir, f), os.path.join(b_dir, f)
        if not os.path.exists(pb):
            print("%-52s MISSING in B" % f)
            failed += 1
            continue
        checked, only_a, only_b, bad = compare_file(pa, pb)
        if not bad and not only_a and not only_b:
            print("%-52s identical (%d tensors)" % (f, checked))
            continue
        failed += 1
        print("%-52s DIFFERS" % f)
        if only_a:
            print("   only in A: %s" % ", ".join(only_a[:6]))
        if only_b:
            print("   only in B: %s" % ", ".join(only_b[:6]))
        for key, why in bad[:6]:
            print("   %s: %s" % (key, why))
        if len(bad) > 6:
            print("   ... and %d more" % (len(bad) - 6))

    print()
    print("RESULT:", "IDENTICAL" if not failed else "%d file(s) differ" % failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
