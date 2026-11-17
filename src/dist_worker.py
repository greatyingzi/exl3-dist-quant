#!/usr/bin/env python3
"""Distributed quantization peer for exllamav3 convert: quantizes a caller-chosen
subset of a module's linears with the caller's Hessians, returns the qtensors."""
import argparse, io, json, os, socket, struct, sys, time, traceback
import torch


def send_blob(f, b):
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def recv_blob(f):
    hdr = f.read(8)
    if not hdr:
        return None
    n = struct.unpack("<Q", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("blob truncated")
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required = True)
    ap.add_argument("--wd", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--port", type = int, default = 9311)
    o = ap.parse_args()
    os.makedirs(o.wd, exist_ok = True)
    os.makedirs(o.out, exist_ok = True)

    import exllamav3.conversion.convert_model as cm

    argv = ["-i", o.src, "-o", o.out, "-w", o.wd, "-b", "3.0", "-cb", "mcg",
            "-hb", "16", "-eb", "16", "-mb", "16", "-vb", "16"]
    args, job_state, ok, err = cm.prepare(cm.parser.parse_args(argv))
    assert ok, f"prepare failed: {err}"
    args["max_module"] = 0
    config, model, mtp_model, vision_model, tokenizer, use_ref = cm.get_base_model(args)
    device = torch.device(0)
    torch.set_grad_enabled(False)
    print("[peer] model ready", flush = True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", o.port))
    srv.listen(1)
    print(f"[peer] listening on {o.port}", flush = True)

    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = conn.makefile("rwb", buffering = 1 << 20)
        cached = {"idx": None}
        print("[peer] client connected", flush = True)
        while True:
            line = f.readline()
            if not line:
                break
            req = json.loads(line.decode("utf-8"))
            op = req.get("op")
            t0 = time.time()
            resp = {"ok": True}
            payload = None
            try:
                if op == "ping":
                    resp["pong"] = True
                elif op == "init":
                    resp["modules"] = len(model.modules)
                    resp["trident"] = "1"
                    resp["modules"] = len(model.modules)
                elif op == "load":
                    idx = int(req["idx"])
                    if cached["idx"] is not None and cached["idx"] != idx:
                        model.modules[cached["idx"]].unload()
                        cm.free_mem()
                        cached["idx"] = None
                    mod = model.modules[idx]
                    if cached["idx"] != idx:
                        mod.load(device, keep_source_weights = True)
                        mod.config.stc.close()
                        cached["idx"] = idx
                    resp["key"] = mod.key
                elif op == "quant":
                    blob = recv_blob(f)
                    H = torch.load(io.BytesIO(blob), map_location = "cpu", weights_only = False)
                    del blob
                    idx = int(req["idx"])
                    keys = set(req["keys"])
                    strategy = {k: int(v) for k, v in req["strategy"].items()}
                    mod = model.modules[idx]
                    linears = [m for m in mod if isinstance(m, cm.Linear) and m.qmap and m.device is not None and m.key in keys]
                    assert len(linears) == len(keys), f"found {len(linears)} of {len(keys)} linears"
                    if not os.environ.get("X3_NO_SWAP"):
                        for l in linears:
                            l.inner.swap_cpu()
                    if all(strategy[l.key] <= 8 for l in linears):
                        cm.quantize_linears_parallel(args, linears, config, strategy, idx, [0], None, H, True, None)
                    else:
                        cm.quantize_linears_single(args, linears, config, strategy, idx, [0], None, H, True, None)
                    tens = {}
                    for m in mod:
                        if isinstance(m, cm.Linear) and m.key in keys:
                            tens.update(m.get_tensors())
                    buf = io.BytesIO()
                    torch.save(tens, buf)
                    payload = buf.getvalue()
                    resp["n_tensors"] = len(tens)
                    del tens, buf
                    torch.cuda.empty_cache()
                elif op == "unload":
                    if cached["idx"] is not None:
                        model.modules[cached["idx"]].unload()
                        cached["idx"] = None
                    cm.free_mem()
                elif op == "quit":
                    resp["bye"] = True
                    f.write((json.dumps(resp) + "\n").encode("utf-8")); f.flush()
                    sys.exit(0)
                else:
                    resp = {"ok": False, "err": f"unknown op {op}"}
            except Exception as e:
                resp = {"ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
            resp["sec"] = round(time.time() - t0, 2)
            resp["has_payload"] = payload is not None
            f.write((json.dumps(resp) + "\n").encode("utf-8")); f.flush()
            if payload is not None:
                send_blob(f, payload)
                del payload
        f.close(); conn.close()


if __name__ == "__main__":
    main()
