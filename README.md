# ondevice-pipeline-verification

Verification tooling for a fully local, multi-stage inference pipeline: prove it stays
offline, prove the exported model is the same model, and prove the deterministic
post-processing layer never invents text.

## What this is

A three-stage pipeline runs entirely on the device - a small classifier judges each turn,
a router sends it either to a code-owned text module or to a local quantised language
model, and a deterministic post-processing layer trims what the generator produced.
This repo is the tooling built around that pipeline: an in-process network isolation test,
an export-consistency check, a backend cold-start comparison, and the post-processing
layer itself with its regression suite.

Nothing here is a demo of a model. It is a demo of **how the pipeline's claims are
measured**, including the numbers that had to be retracted.

## Key engineering results

- **Process-level network isolation**: 27 replay rounds complete with **0 outbound
  connection attempts**, plus a positive control that proves the block is actually active.
  The full pipeline it was built for ran 9/9 turns with 0 attempts and 0 generator calls
  on protected turns.
- **Regression suites**: four suites green in the source project - rule guard **40/40**,
  protected path **40/0**, cue detector **22/0**, isolation **9/9**. The two domain-free
  ones ship here; the guard suite is re-expressed as **32 pytest cases, 32 passed**.
- **Export consistency**: ONNX vs framework `max |Δ logits|` = **1.5e-05** with per-item
  argmax agreement **707/707**, at **11.5 ms/item** on CPU.
- **Backend cold start**: **18.3 s vs 2.9 s** total startup for the same weights on the
  same machine (first request 13.5 s vs 0.3 s - the slow one compiles kernels at first
  call). Generation speed differed inside noise, and the ranking reversed between runs.

Every number above is listed with the command and artifact that produced it in
[`docs/results.md`](docs/results.md) - including the ones that were withdrawn.

## Architecture

```
                       user turn
                           │
                           ▼
             ┌───────────────────────────┐   protected tier
             │ stage 1  judge            │──────────────┐
             │ small classifier          │              │
             └─────────────┬─────────────┘              ▼
                           │ open tiers      ┌──────────────────────────┐
                           ▼                 │ verified-text module     │
             ┌───────────────────────────┐   │ code-owned strings,      │
             │ stage 2  generator        │   │ never touched by a model │
             │ local quantised LLM       │   └───────────┬──────────────┘
             └─────────────┬─────────────┘               │
                           ▼                             │
             ┌───────────────────────────┐               │
             │ stage 3  guard            │               │
             │ deterministic, only deletes│              │
             └─────────────┬─────────────┘               │
                           └──────────────┬──────────────┘
                                          ▼
                                     reply to user

verification layer (what this repo ships)
  src/offline_test.py     no outbound traffic; protected turns never reach the generator
  src/verify_onnx.py      exported model == framework model, item by item
  src/bench_backends.py   startup / throughput / whether output quality moved
  tests/test_guard.py     one negative case for every guard rule
```

## Design decisions

**Constraints that must hold are written in code, not in the prompt.** A prompt clause
asking for a conditional behaviour moved the violation count 9 → 8 - inside noise, i.e.
no effect. The same condition checked in code moved it 8 → 4. Small models cannot execute
conditional rules reliably; code can, when it knows the context. The paired replay on one
frozen output dump: 11 turns affected, 7 actually changed, **0 became empty**, 105
unaffected turns untouched.

**The safety-critical path never goes through the generator.** Numbers and fixed wording
come from a code-owned module, and the test proves it with a call counter instead of
reading the output: protected turns called the generator 0 times. The generator path
separately strips any sentence carrying a number shape - verified by a deliberately
fault-injected round where the stub emits one, and it never reaches the user.

**Post-processing only deletes, and never returns empty.** Every rule removes content;
none rewrites it. The worst case is a terse reply, never a fabricated one, and if every
sentence gets filtered the original is kept. That invariant is what makes it acceptable to
run this layer unconditionally, on every reply.

**Protection is sentence-level, not passage-level.** A word allowlist that short-circuited
the whole module looked safe and silently disabled all four rules at once - question-cap
violations reappeared in production output. Now the caller supplies a predicate, only the
matching sentences are exempt, and everything else is trimmed as usual.

**Export = fixed shape + frozen class order + tokenizer shipped alongside.** The output is
a bare vector with no field names, so a permuted class order reports the wrong class
without raising anything; weights-only handoff produces ids that are wrong and output that
is confidently incorrect with no error. Both are documented as contract clauses in
[`docs/onnx-export-contract.md`](docs/onnx-export-contract.md).

## Quick start

```bash
python -m pip install -r requirements.txt

python -m pytest -q                      # 32 rule-level cases
python -m src.offline_test --rounds 1    # 9 rounds, 0 outbound attempts, positive control
python -m src.verify_onnx --demo         # export check, synthetic (reporting path only)
python -m src.bench_backends --stub      # backend comparison harness, synthetic
```

With your own artifacts:

```bash
python -m src.verify_onnx --onnx model.onnx --data reference.npz \
       --labels tier-0,tier-1,tier-2 --json out.json
python -m src.bench_backends --name cuda --api-base http://127.0.0.1:8080 \
       --launch-cmd "path/to/server -m model.gguf -ngl 99" --repeat 2
```

## What I learned

- **Compare rules on one frozen output dump.** Re-sampling per rule produced a 50%
  "improvement" between identical configurations; replayed paired, the honest number was
  7 changed turns out of 11 candidates and nothing broken.
- **When seed noise beats the effect, change the metric.** Single-point F1 moved 8.7
  points across seeds while the effect being measured was 1.8; area under the PR curve
  moved 2.7 points and gave a stable answer.
- **A broken ruler is not a finding.** A test set 100% contained in training voided the
  headline metric, and an inter-rater kappa of 0.11 made a whole evaluation unusable. Both
  are written up as methodology results rather than quietly replaced.

## Scope

The pipeline itself, its weights, its data, and two of the four regression suites are
bound to private content and are **not** part of this repository - only the domain-free
verification code, its tests, and the measurements. See
[`docs/engineering-note.md`](docs/engineering-note.md) for how the claims were verified
and [`docs/results.md`](docs/results.md) for what is reproducible here versus measured in
the source project.

## License

MIT - see [`LICENSE`](LICENSE).
