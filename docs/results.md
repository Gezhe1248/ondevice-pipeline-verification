# Results and their provenance

Every number in the README appears here with what produced it. Two groups, kept apart on
purpose:

- **A. Reproducible in this repo** - run the command, get the number.
- **B. Measured in the source project** - the code lives here, the private data and model
  files it ran against do not. The command and the generated artifact are recorded so the
  number is checkable rather than merely asserted.

Nothing below is rounded up, and withdrawn numbers are listed as withdrawn.

## A. Reproducible in this repo

| Claim | Value | Command |
|---|---|---|
| Rule-level regression suite | **32 passed** | `python -m pytest -q` |
| Isolation replay | **27 rounds, 0 outbound attempts**, 18 generator calls (= 27 − 9 protected) | `python -m src.offline_test --rounds 3` |
| Isolation positive control | outbound probe rejected while the block is active | same command |
| Export check reporting path | runs end-to-end on synthetic arrays (`--demo`) | `python -m src.verify_onnx --demo` |
| Backend harness reporting path | runs end-to-end on synthetic arrays (`--stub`) | `python -m src.bench_backends --stub` |

The last two are **synthetic**: they exercise the code and the report format and produce
no claim about any model. Their output is labelled as such.

## B. Measured in the source project

| Claim | Value | Produced by | Artifact |
|---|---|---|---|
| Process-level isolation | **9/9 turns** completed, outbound attempts **0**, protected turns called the generator **0** times | offline harness | availability report (script-generated) |
| Regression suites, four of them | rule guard **40/40** · protected-path **40/0** · cue detector **22/0** · isolation **9/9** | four suites | suite output |
| ONNX vs framework | `max |Δ logits|` **1.5e-05**, argmax agreement **707/707**, **11.5 ms/item** on CPU | export packager + verify script | delivery README (written by the packager) |
| Backend comparison (n=12) | total startup **18.3 s vs 2.9 s**; first request **13.5 s vs 0.3 s**; generation 117.7 vs 183.3 tok/s | backend harness | backend comparison report |
| Leakage audit | test set **707/707** contained in training set; protocol sweep F1 **79.5–80.3%** | audit script | audit report |
| Multi-seed validation | single-point F1 span **8.7 pt**; AP deltas **+1.6 / +0.4 / +3.5** (mean **+1.8**, sd **1.3**) | seed-repeat script, 3 seeds × 2 variants | multi-seed report |
| Two-rater agreement | Cohen's kappa **0.11**, exact agreement **39.5%**, 39 items | blind rating script | blind rating report |

Two of the four suites (protected-path, cue detector) are bound to domain content and are
**not distributed** in this repo; only their counts are reported. The two domain-free ones
are the rule guard (re-expressed here as `tests/test_guard.py`) and the isolation harness
(`src/offline_test.py`).

## Withdrawn numbers - do not quote

| Number | Why it is void |
|---|---|
| F1 **82.8%**, recall **89.6%**, **7** misses | computed on the 707-item test set, which was later found to be 707/707 contained in the training set. Retracted; re-measured on four disjoint splits instead. |
| Sentence-cap change "38.9% → 50.0%" | produced by re-sampling each rule separately, i.e. by noise. Retracted; the paired replay gives 11 turns affected, 7 changed, 0 empty, 105 untouched. |
| Generation speed ranking between backends | the two backends swapped places between consecutive runs on the same machine (165 vs 184 tok/s in both orders), so the ranking is not a finding. Only the startup difference survived repeated runs. |

Keeping this table is the point: a number that has been retracted is still a fact about
the work, and deleting it would make the remaining numbers unreadable.

## Not measured

Written as `not measured` rather than guessed:

- INT8 quantisation accuracy drop on device, and on-device latency: produced by a
  different workstream and not included here.
- Memory headroom of the full pipeline on the target board.
- Agreement rate of a self-implemented tokenizer against the reference tokenizer
  (the isolation was done with pre-tokenized tensors, which is cheaper and unambiguous).
- Any metric on data outside the four splits named in the audit.
