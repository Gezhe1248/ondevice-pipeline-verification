# Engineering note: how I verify a system that is not reliable by construction

The pipeline has four stages that can each be wrong: a small classifier decides which
route a turn takes, a router picks either a fixed-text module or a generator, the
generator is a 1.7–4B quantised model, and a deterministic post-processing layer trims
what the generator produced. None of these stages is trustworthy on its own. The
small models are probabilistic, the router is code written by someone in a hurry, and
the post-processing rules are regular expressions that can quietly stop matching.

So the unit of work was never "make the model better". It was "make it possible to tell
that something went wrong, and to tell *which* layer did it" — before a user notices.
The five practices below each came out of a concrete failure, and each one has a number
attached.

## 1. Build the evaluation before you touch the model

Three layers, in increasing cost: a fixed input set, deterministic counters, and human
ratings.

- **Fixed inputs.** 12 scenarios / 40 turns, taken from real logs where possible;
  invented scenarios only fill in failure modes the logs did not contain. Every later
  change is replayed on the same inputs. Changing the topic between runs makes a
  comparison meaningless.
- **Deterministic counters.** Fragment sentences (unbalanced quotes), sentence-level
  repetition against the last three replies, sentence and question caps, forbidden
  imperative patterns, and - on the generator path only - any number shape at all.
  These are cheap, exact, and anybody can recompute them.
- **Human ratings.** The script emits a rating sheet (5 dimensions, 1–5, plus a
  hard-defect checkbox); two people fill it in independently; the script reports means,
  defect counts and inter-rater agreement.

The reason to insist on all three: style metrics were "all green" on a day when the
output was useless. Every style check passed while the assistant answered the wrong
question. Worse, one of those checks was itself broken for weeks - the sentence
splitter treated a question mark as a separator, so "at most one question per reply"
never actually held. An all-green dashboard built on a broken metric is the most
expensive kind of false confidence: **the metric is part of the thing under test.**

## 2. Paired experiments: one frozen output dump, two rules

When I compared two post-processing rules by running the model twice, I measured a
50% improvement that did not exist. The same unmodified configuration produced
30.6 / 41.7 / 55.6 / 61.1 / 69.4 / 72.2% on the same metric (n=36) across runs - the
spread between identical configurations is larger than any effect I was trying to
detect.

The rule that stuck: a post-processing change is replayed on **one frozen dump of raw
generator output**, once through rule A and once through rule B, and the two results
are compared item by item. Re-sampling per rule is not a comparison, it is noise.

Concrete case: raising the sentence cap from 3 to 4. With re-sampling this looked like
a 38.9% → 50.0% improvement, which I withdrew. Replayed on the same dump, the honest
statement is: 11 turns affected, 7 actually changed, **0 became empty**, and the 105
turns that were already fine were not touched at all.

## 3. Multi-seed validation: when noise wins, change the metric, not the seed

The test set had 67 positive items. Across three training seeds, a single-point F1 for
the same configuration ranged 72.5–81.2 - an **8.7 point** span. The effect I was
trying to measure (soft-label distillation) was about **1.8 points**. The ruler was
five times coarser than the thing being measured, so F1 could not rank these models at
all, in either direction.

Two fixes, applied together:

1. Repeat over 3 seeds × 2 variants and report the per-seed deltas, not the mean of one
   run: +1.6 / +0.4 / +3.5 points (mean +1.8, sd 1.3), all positive.
2. Switch the primary metric to one that is not tied to a working point - area under
   the precision-recall curve, plus precision at matched recall. AP varied only 2.7
   points across the same three seeds.

The distilled model's single-point F1 was *lower* in one seed and *higher* in another.
Reporting either one alone would have been a true statement and a false conclusion.

## 4. Leakage self-check: audit which decisions touched the test set

Written as a script, not as a promise, and run over four places where a test set
usually leaks in: training/early stopping, variant selection, decision-threshold
selection, and hand-written regression cases.

It found two leaks. Selecting the threshold on the test set inflated F1 by 0.6–0.8
points; selecting the variant (context / no context) on the test set had no effect.
Recomputing with the correct protocol and with no tuning at all gave F1 across
79.5–80.3%, a range of 0.8 points - so the conclusion did not depend on the leak, which
is exactly the kind of sentence you can only write if you measured it.

Then the audit got worse news: the 707-item test set turned out to be 707/707 contained
in the training set. Every metric computed on it - F1 82.8%, recall 89.6%, 7 misses -
was void. I retracted them and re-measured on four mutually disjoint splits (one
real-conversation set, three held-out sets), with the split rule "whole conversations
are assigned to one split, never individual turns", because turns of one conversation
are not independent.

The general lesson: **the protocol is a deliverable.** "Which decisions used which
split" has to be published next to the number, otherwise the number cannot be read.

## 5. Agreement gate: an unusable evaluation is not a weak result

Two raters, 39 items, five dimensions: overall kappa **0.11** (39.5% exact agreement),
and the two raters pointing in opposite directions (one preferred model A on 46% of
items, the other on 55%).

The protocol said agreement below 0.4 means the criteria themselves are unclear. So the
result was not averaged and not reported as "no difference": the evaluation was declared
unusable, the cause was written up (single turns shown without context, two different
private notions of "good reply", an under-specified task), and the redesign starts with
a 10-item pilot where disagreements are discussed *before* the real run, with a hard
gate: kappa ≥ 0.4 or the run does not count.

Low agreement is a defect of the ruler, not a finding about the model. Averaging it
away converts a broken measurement into a confident number.

## What every claim in this repo has to survive

1. **A positive control.** "Zero outbound connection attempts" only means something if
   the block is demonstrably active - so `src/offline_test.py` also probes an
   unroutable address and requires that probe to fail. A check that cannot fail is not
   a check.
2. **A stated boundary.** The offline test blocks the socket calls of *this* process;
   a separately launched local server is not covered by it and is covered by a
   different argument (it only listens on loopback and only reads local weight files).
   The test set shares topics with the training set, so the metric describes
   generalisation to new conversations, not to new topics.
3. **`not measured` where it was not measured.** Unknown is written as unknown. A
   fabricated number is unrecoverable once it is in a report; an empty cell is not.
