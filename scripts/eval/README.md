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
  consequences tracking the model-side GEEN residual.

## Cross-set results (presence/absence, precision/recall)

| Set | Mistral@0.7 | Qwen@0.3 | Qwen + July-30/31 fixes |
|-----|-------------|----------|--------------------------|
| 1 (dev) | 0.84 / 1.00 | 0.93 / 0.98 | — |
| 2 | 0.88 / 1.00 | 0.86 / 1.00 | — |
| 3 | 0.88 / 0.98 | 0.86 / 0.95 | — |
| 4 | — | — | 0.90 / 0.92 (run of 2026-07-31) |
| 5 | — | — | pending (run in flight 2026-07-31) |

Set-4 notes: family guard 5/5, zero same-family swaps; recall dip was
connective (6/10, since fixed to 8/10) plus one wordfreq FN; the spelling
"regression" (3/3 hallucinated) traced to the HUNSPELL pass on out-of-
dictionary compounds, both passes now gated by `_correction_plausible`.

## Current residuals / backlog (priority order)

1. **Connective GEEN on temporally-expressed consequences** (corpus4 conn-8/10,
   corpus5 conn-9/10): model-side — journal shows no discard lines, the merges
   are never proposed. Needs prompt iteration directly against Qwen
   (`HETZNER_API_KEY`, box side). Do NOT add rhetorical questions to the
   prompt (collapses block formatting); the current prompt (05e8b3d state) is
   journal-verified at 8/10 with clean precision.
2. **Qwen semantic swaps on positives** (monumentale→groot,
   "verloren gewaande"→vermeende, koeling→koelkast class): low volume,
   meaning-changing. Likely prompt work; same box-side loop.
3. **Enumeration conj-route gap**: plain 4-item NP lists parse as pair-chains,
   only nominalized-infinitive lists are surface-detected. corpus5 enum-6/7
   are sentinels (expected misses) — a surface route for NP lists must not
   break the shortlist guards.
4. **Spelling detection wobble**: Qwen flags planted dt-errors in ~1 of 3 runs;
   corpus5's spelling-* group measures it for the first time.
