# Measurements

Model: Sage-S2.1-Flash (321B MoE, 45 routed-expert layers + MTP head)
Recipe: 3.0 bpw, mcg codebook, 250 x 2048 calibration rows
Hardware: 2x DGX Spark (GB10), one GPU per node, 200 GbE point-to-point link

Each row is the `[phases]` line the patched converter prints for one
routed-expert layer, in seconds.

## Running the peer request concurrently

The first version waited for the peer to finish before quantizing locally, so
the two halves were serialized and the second machine bought nothing.

| quantize | load | capture | prep | quantize | save | advance | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| serialized | 7.9 | 340.1 | 59.0 | 1029.7 | 1.3 | 139.1 | 1577.2 |
| serialized | 8.0 | 320.0 | 50.2 | 1010.7 | 0.9 | 88.5 | 1478.4 |
| concurrent | 4.3 | 326.4 | 43.0 | **519.4** | 0.8 | 171.9 | **1066.0** |
| concurrent | 8.0 | 318.3 | 50.0 | **519.5** | 0.9 | 186.8 | **1083.8** |
| concurrent | 9.4 | 323.3 | 42.9 | **518.8** | 0.9 | 190.8 | **1086.3** |

The quantization phase halves; the rest is unchanged. It was about 65% of the
module, so the module total drops by roughly a third.

## Skipping the per-tensor CPU swap

| layer kind | with swap | `X3_NO_SWAP=1` |
| --- | --- | --- |
| routed-expert (~870 tensors) | 50-180 s in `prep` | ~0 s |
| dense (~5 tensors) | 15.7 s | 8.5 s |

## Correctness: the remote half produces identical tensors

The same layer was converted locally and split, then compared on both the
per-tensor proxy error / global scale and the whole-module error metrics.

| check | local | split (remote worker) |
| --- | --- | --- |
| layers.0 qkv_proj proxy_err / g_sc | 0.002347 / 0.932887 | 0.002347 / 0.932887 |
| layers.0 o_proj proxy_err / g_sc | 0.007529 / 0.933273 | 0.007529 / 0.933273 |
| layers.0 down_proj proxy_err / g_sc | 0.000913 / 0.934380 | 0.000913 / 0.934380 |
| layers.0 whole module rfn / cos / sqnr | 0.096260 / 0.004501 / 23.846789 | 0.096260 / 0.004501 / 23.846789 |

## Approaches that do not work

* **Layer-range split across nodes** — see [DESIGN.md](DESIGN.md). The
  calibration state is advanced through the quantized layers.
* **Two quantization threads on one GPU** (`--devices 0,0`). Measured 6.4 s per
  expert tensor against 4.1 s single threaded: the two streams contend for the
  same memory bandwidth, so this is slower, not faster.
* **Per-tensor bitrate recipes as the split mechanism.** In the exllamav3 version
  used here, a recipe that assigns 3 bits to the `mcg` codebook does not
  quantize the routed experts at all — no trellis tensors are emitted and the
  layer reports 16 bpw. A recipe with a value that *does* quantize them was
  silently quantizing to a different bitrate than intended.
