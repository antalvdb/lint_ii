# LiNT-II self-diagnosis eval harness

Repeatable quality measurement for the suggestion engine: labelled corpora of
paragraph-length Dutch inputs with ground truth (`should_suggest`, `phenomena`,
`must_not`), a runner that posts them to the live box, and an LLM-as-judge pass
over the captured suggestions.

## Workflow

```
python3 build_corpusN.py                                   # regenerate corpusN.json
python3 run_eval.py --corpus scripts/eval/corpus5.json \
                    --results scripts/eval/results5.json --fresh
```

- The runner is sequential, resumable (`--fresh` ignores prior results), and
  cache-busts every item with a per-run nonce. It prints presence/absence
  precision/recall; per-suggestion quality judging (wrong / debatable / right)
  is done by Claude from the results file afterwards.
- `results*.json` are **gitignored** (they exist on the machine that ran the
  eval — historically Antal's Mac). Re-run to regenerate.
- Each suggestion in the results carries `model` (`None` = Hunspell spelling
  pass, a model name = LLM) and `error_category` — attribute spelling failures
  to the right pass before fixing anything (eval 4's spelling regression was
  chased into the wrong pass for lack of this).

## Corpus inventory

Five independent 100-item sets, same label scheme, disjoint texts/domains.
Set 1 is the DEV set (thresholds were tuned on it); 2–5 are honest held-out
sets. When authoring a new set: validate every deterministic design assumption
against the local pipeline first (wordfreq/compound positives must trigger,
family/shortlist/clean negatives must NOT — see the validation snippets in the
corpus5 commit), and keep rare-looking words out of negatives unless a guard
provably suppresses them.

- `corpus.json` — dev set.
- `corpus2.json`, `corpus3.json` — held-out; drove the connective relation
  guard, abstract min-count, passive agent-gate, max_sdl length-gate.
- `corpus4.json` — first Qwen-era set; `family-*` guard group (rare surface
  forms of common word families must not fire word_frequency).
- `corpus5.json` — consolidation set for the 2026-07-30/31 fixes. First
  `spelling-*` DETECTION positives (planted typos + dt-errors); enum mixes
  nominalized-infinitive lists (surface route) with two plain NP-list
  sentinels (known conj-route gap, expected misses); family-* covers
  lemma/comparative/diminutive/particle mechanisms; conn-9/10 are temporal
  consequences tracking the model-side GEEN residual (conn-9 fires as of the
  set-5 run; conn-10 is the surviving inferential case — see backlog item 1).

## Cross-set results (presence/absence, precision/recall)

| Set | Mistral@0.7 | Qwen@0.3 | Qwen + July-30/31 fixes | + `8397cae` connective |
|-----|-------------|----------|--------------------------|------------------------|
| 1 (dev) | 0.84 / 1.00 | 0.93 / 0.98 | — | — |
| 2 | 0.88 / 1.00 | 0.86 / 1.00 | — | — |
| 3 | 0.88 / 0.98 | 0.86 / 0.95 | — | — |
| 4 | — | — | 0.90 / 0.92 | **0.91 / 0.95** (re-run 2026-07-31) |
| 5 | — | — | **0.94 / 0.95** | — |

Set-4 notes: family guard 5/5, zero same-family swaps; recall dip was
connective (6/10, since fixed to 8/10) plus one wordfreq FN; the spelling
"regression" (3/3 hallucinated) traced to the HUNSPELL pass on out-of-
dictionary compounds, both passes now gated by `_correction_plausible`.

Set-4 re-run notes (2026-07-31, after `8397cae` + `c7fdb54`): 0.91 / 0.95, up
from 0.90 / 0.92 — but **do not attribute the recall gain to the connective
work**. Connective recall is 8/10 in BOTH runs; what changed is *which* two
fail. conn-8 (the backlog case) is fixed, and conn-6 took its place — and
conn-6 is wobble, not a regression: a 6-rep probe scores it 5/6 under the old
and new prompt alike, so the eval simply caught its 1-in-6 miss. Across all
eight connective positives that were outside the tuning set, old and new
prompt tie exactly at 42/48. The recall gain therefore comes from `c7fdb54`
and run variance, not from this change.

The 6 FPs are all `word_frequency` / `abstract_nouns` (pompstation, "vereniging
van moestuiniers", storing→"Het spijt ons", reparatie→werkzaamheden) — the
known Qwen semantic-swap class, backlog item 2. **Zero connective FPs**, which
is the precision half of the connective claim. Remaining FNs: conn-6 (wobble),
conn-10 (inferential, item 1), compound-7. The fragment-1 422 is expected
(non-prose input).

A caution this run earned: a 10-item phenomenon group cannot resolve a
one-item recall change. The connective claim rests on the 6-rep probe
(conn-8 0/5 → 6/6, no regression across 48 case-reps); the eval's job here was
regression detection on the other passes, and it passed.

Set-5 notes (best held-out result to date): family guards 5/5 across all
four mechanisms; connective 8/10 incl. the FIRST 'gevolg' fire on a
temporal consequence (conn-9); enum exactly as designed (5/5 surface route,
2 NP sentinels fell back to prose rewrites); **spelling detection 5/6** —
all five catches were the Hunspell pass, the one miss ("Ik wordt") needs
LLM dt-detection, confirming the wobble. FPs (4): one abstract_nouns
meaning-shift on clean text (volksuniversiteit→avondschool, the known Qwen
semantic class), one authoring bait (vakantieweken in conj-3 — also exposed
a no-op word_frequency suggestion slipping the filters), one defensible
14-word max_sdl split (URL preserved intact), and "terugzwemmen → te
rugzwemmen" — fixed in `c7fdb54` (split corrections now require every part
to be a common word). `Suggestion.model` serialization also fixed there, so
spelling-pass attribution works from the next deploy.

## Prompt iteration against Qwen (method)

Deploy round-trips are far too slow to tune a prompt, and a 100-item eval
resolves a one-item recall change no better than noise. `connective_probe.py`
rebuilds the exact `{paragraph}`/`{boundaries}` the connective pass sends,
calls the Hetzner endpoint with production settings (temp 0.3,
`enable_thinking:false`), and applies the real `parse_block_response` +
relation whitelist — a variant sweep in ~90s.

```
python3 connective_probe.py --reps 6                       # score the live prompt
python3 connective_probe.py --variant mine --compare base  # A/B a candidate
python3 connective_probe.py --audit                        # example/corpus word collisions
```

Cases are read from the corpora (all 20 `conn-*` positives, plus every
multi-sentence negative that actually reaches the pass) so they cannot drift.
Add a candidate to `VARIANTS` via `variant()`, which refuses a substitution
whose anchor is missing — otherwise an edit to `prompts.py` silently turns
your candidate back into base and you A/B a prompt against itself. Two rules
it earned:

- **Run every case 5–6× before believing a delta.** At 3 reps one variant
  looked like 8/12; at 5 it was 9/20. Single-run comparisons of connective
  recall are worthless — the same variant swung 3/3 → 3/5 on one item.
- **Validate the probe against a real eval run first.** It independently
  reproduced the set-5 conn-9 result, which is what made the rest trustworthy.

What it settled, against the hypothesis behind the reverted `0041a47`:
narrowing the "simpele opeenvolging" guard clause **alone changes nothing**
(scored identically to base). Qwen moves on *worked examples*, not abstract
guidance — the same lesson as the deterministic-post-filter rule in CLAUDE.md,
one level down. And prompt examples leak lexically: an example opening "De zaal
was tot de laatste stoel gevuld" knocked corpus5 conn-4 ("De zaal was … toch
uitverkocht") from 5/5 to 1/5. Keep example vocabulary clear of corpus text,
and keep negatives that share words with new examples in the probe set.

## Current residuals / backlog (priority order)

1. **Connective GEEN on inferential consequences** (corpus4 conn-10, corpus5
   conn-10) — narrowed from the original item by `8397cae`. The *temporal*
   half is fixed: corpus4 conn-8 went 0/5 → 6/6 by teaching the delay/measure
   shape with worked examples. What is left is the shape where the causal link
   runs through an unstated inference ("kreeg een tweede ster" → demand →
   "reserveren kan maanden vooruit"). Both stayed 0/6 under every variant
   tried. Treat this as a **defensible GEEN, not a bug**: that shape is not
   safely separable from the thematic pairs the clean-* negatives protect, and
   loosening for it is what drove false positives in the discarded variants.
   Note corpus5 conn-9 is NOT part of this residual — it has fired since the
   set-5 run (base 5/5 on a direct probe); the old item text was stale.
2. **Qwen semantic swaps on positives** (monumentale→groot,
   "verloren gewaande"→vermeende, koeling→koelkast class): low volume,
   meaning-changing. Likely prompt work; same box-side loop.
3. **Enumeration conj-route gap**: plain 4-item NP lists parse as pair-chains,
   only nominalized-infinitive lists are surface-detected. corpus5 enum-6/7
   are sentinels (expected misses) — a surface route for NP lists must not
   break the shortlist guards.
4. **Spelling detection wobble**: Qwen flags planted dt-errors in ~1 of 3 runs;
   corpus5's spelling-* group measures it for the first time.
