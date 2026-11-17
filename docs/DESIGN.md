# Design

## Where the time goes

A routed-expert module dominates a conversion. Measured on Sage-S2.1-Flash at
3.0 bpw on one DGX Spark, before any of this work:

| phase | what it does | time |
| --- | --- | --- |
| load | read the module, dequantize FP8 -> BF16 | ~8 s |
| capture | forward all 250 calibration rows with every expert active, accumulate Hessians | ~325 s |
| prep | move every tensor to host memory | 50-180 s |
| **quantize** | LDLQ + trellis for ~870 tensors | **~1030 s** |
| advance | forward the 250 rows through the quantized module | ~150 s |

Two of those are easy to remove without touching the algorithm: `prep` is pure
waste on a unified-memory machine, and `quantize` is the only phase that
parallelizes cleanly across machines. Together they are most of the module.

## Why layers cannot be split across machines

The calibration state is advanced with the module that was **just quantized**:

```python
config.stc.set_new_tensors(q_tensors)
module.load(..., source = q_tensors)     # the quantized module
...
state[i] = module.forward(state[i], params)   # new state for the next layer
```

So layer `N` is calibrated on activations that already carry the quantization
error of layers `0..N-1`. This is what makes the conversion good: each layer
sees the errors it will actually see at serving time.

Consequences:

* A layer-range split ("A does 0-22, B does 23-45") gives B the wrong input
  activations. Its output would be a different model, not a faster copy of the
  same one.
* Running both sides through the whole chain and discarding one side's output
  buys nothing.
* Only parallelism *inside* one module keeps the chain intact.

## Why group splitting is exact

`quantize_linears_parallel()` builds groups with `group_quant_linears()`:

* tensors that share a Hessian (same qmap, same bitrate, same input width) are
  concatenated along `out_features` and quantized as one batch;
* tensors of the same shape with individual Hessians are stacked up to
  `max_stack`.

Each group is then quantized by one worker and its result written into the shared
`q_tensors` dict. Nothing in the quantization of a group reads another group's
weights, scales or errors, and the per-tensor global scale search operates within
the tensor.

Therefore:

* partitioning the **groups** of a module and running the partitions on different
  workers produces the same tensors as running them all in one worker;
* partitioning must happen on group boundaries, and the order within each
  partition must be preserved, so that re-grouping a sublist yields the same
  chunks. Splitting an individual group would not be safe.

Both properties are checked empirically, per tensor and per module, in
[MEASUREMENTS.md](MEASUREMENTS.md).

## Concurrency, not sequencing

The first version of this kit sent the peer its work and waited for the reply
before starting the local half. The quantization phase then took
`peer + local` instead of `max(peer, local)`, which cancelled the benefit of the
second machine almost exactly.

The peer request now runs in a worker thread started by `dist_begin()`, the
local half is quantized on the main thread, and `dist_finish()` joins and merges.
The peer reports its own duration, which makes the overlap visible:

```
-- dist peer: 436 tensors on peer, 433 local, peer alone 511.66s, wall 517.7s, xfer 1.29 GiB
```

If `wall` were much larger than `peer alone`, the halves would be serialized
again.

## Transport

Control messages are one JSON object per line. Bulk payloads (the Hessians going
out, the qtensors coming back) are `torch.save` blobs preceded by an 8-byte
little-endian length. The worker tolerates a dropped connection and keeps
listening: a frame that fails to parse only closes that connection, because a
protocol desync must never take the worker process down in the middle of a
multi-hour job.

## What the local model still does

Capture and the state advance stay on the orchestrator. They are row-parallel
and could in principle be split too (the code already contains
`calibration_row_shards()` for multi-GPU calibration), but each row carries
`quant_preserve` state across modules, so splitting them means keeping two
copies of that state in sync. The gain is smaller than the quantization phase,
so it is left alone here.
