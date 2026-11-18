# ONNX export contract

An exported model is a contract with whoever runs it on the device. The four constraints
below each cost real debugging time when they were ignored, and none of them fails
loudly. They are written down here so the handoff does not depend on a conversation.

## The interface

| item | value | note |
|---|---|---|
| `input_ids` | `(1, L)` int64 | `L` fixed at export time; padded with `0` |
| `attention_mask` | `(1, L)` int64 | `1` for real tokens, `0` for padding |
| `logits` | `(1, C)` float32 | one score per class, **order fixed below** |
| post-processing | `p = softmax(logits)`, `score = p[1] + p[2]` | score for the positive region, i.e. the two non-base tiers |
| decision | `score >= threshold` → positive | threshold is a product decision, never chosen on the verification set |

## 1. Static, fixed shapes are a hard requirement

Edge accelerators compile a graph for a specific shape. A dynamic dimension either fails
compilation or silently falls back to a slower path, and in both cases the failure shows
up as "it works on my laptop".

- Pad / truncate during tokenization so the tensor is exactly `L`.
- Keep batch size 1 on device; batch on the host if you need it.
- Ship reference tensors padded to the same `L`, otherwise the desktop comparison is
  comparing two different shapes.

## 2. The class order must be frozen and documented *outside* the model

The output is a vector. It has no field names, no schema, and no metadata that survives
most conversion toolchains. Reversing `[tier-0, tier-1, tier-2]` into
`[tier-2, tier-1, tier-0]` raises nothing: every number is still in range, inference
still runs, the pipeline still reports a class. It just reports the wrong one - at
whatever error rate the permutation happens to produce, which is invisible without a
per-item comparison.

Practices that make this survivable:

- Write the order in the delivery README, in the verification script (`--labels`), and in
  the post-processing formula. Three places, one meaning.
- Prove it with a per-item argmax comparison against the reference implementation: any
  permutation collapses the agreement rate.
- Never infer the order from the model file, and never "fix" a mismatch by permuting the
  labels at runtime - that hides the bug for the next model.

## 3. The tokenizer has to ship with the model

Weights-only is the most common handoff mistake, and it fails silently: raw strings get
mapped to the wrong ids, the model computes something, nothing errors, and the output is
confidently wrong. Ship the tokenizer configuration **with** the weights:

- vocabulary / merges, `do_lower_case`, and the special-token ids (`[PAD]=0`, `[CLS]`,
  `[SEP]`, `[UNK]`), plus the padding convention (`0`) and truncation direction.
- Ship pre-tokenized tensors alongside the reference dump: feeding those ids lets you
  skip tokenization entirely and separate "tokenizer wrong" from "model wrong" (§5).

Truncation direction is its own trap. BERT-style tokenizers truncate from the tail, so if
the span of interest sits at the end of the input, it is cut off and the model is
effectively trained on inputs that do not contain the thing being classified. Measured
in this project: with the target span placed last, **36.9% of samples** had it truncated
away. Fix it by input ordering (put the span first), not by raising the max length.

## 4. Quantisation calibration: class balance decides minority recall

Post-training INT8 usually holds up, but only if the calibration set looks like the
deployment distribution. A calibration set that is 90% one class tunes the activation
ranges to that class and compresses the sensitivity of the rare one - the failure mode is
"the rare class recall quietly drops while overall accuracy looks fine".

- Expect a 1–3 point drop; treat a drop beyond ~5 points as "check the calibration set
  first, before touching the model".
- Keep the rare classes represented in the calibration set, and record the calibration
  set composition next to the quantised metrics.
- Validate the fp32 conversion before the INT8 one: if fp32 already disagrees, the
  problem is the conversion, not quantisation.

## 5. The four-layer triage

When on-device results differ from desktop results, isolate instead of guessing. Each step
changes exactly one thing, so the first step that disagrees localises the fault:

| layer | check | if it matches |
|---|---|---|
| ① tokenizer | feed the shipped pre-tokenized ids, skip tokenization | the bug is in tokenization |
| ② export | run `python -m src.verify_onnx --onnx … --data …` on the desktop | the bug is downstream of the export |
| ③ conversion | same ids through the converted model, compare with the ONNX output | the bug is in the conversion |
| ④ quantisation | check calibration coverage; compare fp32 conversion vs INT8 | the bug is in calibration |

## 6. What "verified" means here

`src/verify_onnx.py` passes when **both** hold:

- `max |Δ logits| <= 1e-3` (measured 1.5e-05 for the model exported in this project), and
- per-item argmax agreement is **100%** (measured 707/707).

Reading the deviation magnitude tells you which layer to open:

| max abs deviation | reading |
|---|---|
| `<= 1e-5` | float accumulation order only - the export is lossless |
| `<= 1e-3` | above pure float noise - suspect operator substitution |
| `<= 1e-1` | wrong file, permuted class order, or a different tokenizer |
| `> 1e-1` | the two sides are not the same model |

And the boundary: this check answers "is this the same model", nothing else. It is not an
evaluation set - no threshold may be chosen on it - and passing it says nothing about
whether the model is good.
