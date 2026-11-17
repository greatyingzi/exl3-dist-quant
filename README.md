# exl3-dist-quant

Split an [exllamav3](https://github.com/turboderp-org/exllamav3) EXL3 conversion
across two machines so that a long quantization finishes in about half the wall
time — **without changing a single weight**.

Measured on Sage-S2.1-Flash (321B MoE, 45 routed-expert layers), 3.0 bpw mcg,
2x DGX Spark:

| | per expert layer | 45 layers |
| --- | --- | --- |
| single machine | ~1860 s | ~23 h |
| this kit | **~1085 s** | **~13.5 h** |

Outputs are bit-identical to a local run; see [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

## The problem it solves

`convert_model.py` advances a calibration state through every module it has just
quantized:

```python
module.load(source = q_tensors)        # reload the quantized module
state[i] = module.forward(state[i])    # advance the calibration rows
```

The state entering module `N` is the output of the **quantized** modules
`0..N-1`. The layers form a strict chain: "node A does layers 0-22, node B does
23-45" changes the calibration input of everything on the second side, so it is
not a scheduling problem — it is what the algorithm does.

Split **inside** a module instead. `quantize_linears_parallel()` already
partitions a module into *groups* of tensors that share a Hessian and quantizes
each group independently before merging the results. Which worker computes a
group has no effect on its output, and the calibration capture and state advance
both stay on one machine, so the chain is never cut.

```
node 01 (orchestrator)                        node 00 (worker)
-----------------------                       ----------------
load module (FP8 -> BF16)
capture 250 rows -> H
split groups ~50/50 by weight count
  send H subset + key list  --------------->  dist_worker receives
  quantize local half       <-- concurrent --> quantize peer half
  receive qtensors          <--------------  returns qtensors
merge into one q_tensors dict
reload quantized module, advance state
save, checkpoint
```

## Requirements

* exllamav3 at **v1.6.7** (`8c593e4e`), which the patch is generated against
* two machines, each able to hold one module of the model plus quantization
  buffers in GPU memory
* the **same source model on both machines** — the worker loads modules from its
  own local copy
* a network path with room for about 1.3 GiB per module in each direction

## Install

On both machines, in the exllamav3 checkout:

```bash
./scripts/install.sh /path/to/exllamav3
```

That applies `patch/0001-distributed-quantization.patch` to
`exllamav3/conversion/convert_model.py` and copies `src/dist_worker.py` next to
it. The patch is opt-in: without the new flags, behaviour is unchanged.

## Run

On the worker machine:

```bash
SRC=/models/Sage-S2.1-Flash-FP8 WD=/tmp/peer-wd OUT=/tmp/peer-out PORT=9311 \
  X3_NO_SWAP=1 ./scripts/quantize_peer.sh
```

Wait for `[peer] listening on 9311`. Then on the orchestrator:

```bash
SRC=/models/Sage-S2.1-Flash-FP8 OUT=/models/Sage-3bpw WD=/tmp/orch-wd \
  PEER=192.168.1.20:9311 X3_NO_SWAP=1 ./scripts/quantize_orchestrator.sh
```

Add `EXTRA_ARGS=-r` to resume from a checkpoint. The worker stays up across
orchestrator restarts.

## What the patch adds

| Change | Default behaviour |
| --- | --- |
| `--dist-peer host:port` splits a module's tensor groups with a remote worker | everything stays local |
| `[phases]` line per module: load / capture / quantize / advance | unchanged, logging only |

The patch is purely additive against v1.6.7 (153 added lines, none removed).

The numbers in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) were taken on an
earlier checkout (`151539c`) with an equivalent patch. Upstream `256287f3`
later added a capture-only fast path for block-sparse layers with every expert
active, which targets the same `capture` phase the `[phases]` line reports and
claims a further 1.6x on that phase; a conversion on v1.6.7 should therefore be
faster than those numbers. The CPU-swap workaround that older checkouts needed
is upstream as `--no_swap` (auto-enabled on integrated GPUs) since
`b658e643`, so it is not part of this patch.

## Limitations

* If the worker dies the orchestrator aborts; there is no silent fallback to
  local quantization (that would double the time without telling you).
* The worker keeps one module resident; GPU memory must allow that on top of the
  quantization working set.
* Transport is `torch.save` over a length-prefixed stream. It assumes a trusted
  network; do not expose the port.

## License

MIT for this repository's own code. The patch modifies exllamav3, which is MIT,
Copyright (c) 2025 Turboderp — see [NOTICE](NOTICE).
