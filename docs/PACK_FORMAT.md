# What a serving pack has to look like

The quantization recipe and the *serving* pack format are two different things. `convert.py`
writes a pack that reflects how it quantized; an inference engine that expects "EXL3 routed
experts + everything else at source precision" will reject it. This page is the list of
mismatches we hit when turning a fresh conversion into a loadable pack, so you can check them
before spending days on a run.

None of these are visible to a structural check of the safetensors headers. **Only an actual
load finds them.**

## 1. `-b` is not "experts only"

`create_q_strategy()` in `exllamav3/conversion/allocation.py` collects every quantizable Linear
whose `qbits_key` is `"bits"` into one budget pool: they all start at `rate_floor(bpw)` and the
remaining budget is handed out by priority. Attention projections, the shared-expert MLP and the
dense MLP are in that same pool. `-hb` / `-eb` / `-mb` / `-vb` pin the head, embedding, MTP layer
and vision tower, but **there is no flag that exempts attention**.

Two ways to get "routed experts only":

* `-rcp recipe.yaml` with an explicit per-tensor map (`16` means store unquantized), or
* quantize everything, then graft the non-expert tensors back from the source.

The recipe approach is the one that also makes the calibration correct: the per-layer
calibration state is advanced through the *quantized* module, so changing attention's precision
changes every downstream Hessian. Grafting afterwards is cheaper but the expert quantization was
fitted to a state that never existed in the final model.

## 2. Fused tensors are the wrong shape for the fp16 path

`convert.py` fuses `q_proj`/`k_proj`/`v_proj` into one `qkv_proj` Linear (and the three
`conv1d` taps into one `conv1d`). The *quantized* path reads the fused name; the *float* path
looks the parts up individually and stops with `MissingTensor: ...self_attn.q_proj.weight`.

Do not split them by hand if you can avoid it — take the tensors from the source checkpoint,
where they are already separate.

## 3. Unquantized linears must be bf16

`convert.py`'s `strategy == 16` path writes **float16**, and it also upcasts fp8 sources to
float16. A loader that reads unquantized linears as bf16 reports:

```
...declared NVFP4, stored f16 { 8192, 4096 } (FormatMismatch); this loader reads it as { .bf16 }
...stored f8_e4m3 { 12288, 4096 } (FormatMismatch); this loader reads it as { .bf16 }
```

So: cast the plain `.weight` tensors to bf16, and dequantize fp8 block-scaled tensors
(`<base>.weight` + `<base>.weight_scale_inv` on a 128x128 grid) to bf16, dropping the scale
tensor. Leave the EXL3 pieces (`trellis`, `suh`, `svh`, `mcg`) exactly as they are.

Whether the source's fp8 tensors may stay fp8 depends entirely on the engine; declaring
`non_routed_dtype_policy` does not make a loader accept a dtype it does not implement.

## 4. A leftover `quantization_config.json` is worse than none

`convert.py` emits a `quantization_config.json` holding a per-module `tensor_storage` table with
tens of thousands of entries. Once you graft, that table is stale. An engine that reads it will
validate the pack against a description that no longer matches and fail with a terse
`UnexpectedTensor` that names no tensor.

Either regenerate the table or ship the pack without the file. Check what a known-good pack in
your deployment contains before deciding.

## 5. Custom code files are easy to lose

Tokenizer/processor `auto_map` entries point at Python files that live next to the weights
(`tokenization_*.py`, `processing_*.py`). Rebuilding a pack from an index-only pass silently
drops them, and then `AutoTokenizer` fails for every consumer even though the engine loads fine.

```bash
grep -l auto_map config.json tokenizer_config.json processor_config.json
# any non-null entry means the referenced module has to be shipped as well
```

## 6. Read the deployed pack first

If a model of the same family is already in production, its `config.json` says how it was built:

```json
{"quant_method": "exl3", "bits": 3, "codebook": "mcg", "head_bits": 16,
 "scope": "...routed_experts_only", "non_routed_dtype_policy": "official_source_native",
 "grafted_by": "<path to the tool that made it>"}
```

`scope` and `non_routed_dtype_policy` are not decoration — a pack without them failed to load
here, and adding them was the difference between `UnexpectedTensor` and a working server. When
the producing tool is named, use it instead of writing your own; the semantics (which tensors are
byte-copied, which shard layout, which config fields) are already encoded there.
