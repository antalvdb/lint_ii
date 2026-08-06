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

**Check provider health before AND after any eval run.** A run on 2026-08-06
scored 0.71 recall and looked like a catastrophic regression; the cause was
295 HTTP 500s from `inference.hetzner.com` during the run. Every pass is
fail-open, so a provider outage silently becomes "no suggestions" and reads as
a recall collapse. The tell is items losing suggestion types the change under
test cannot touch. Gate on it:

```
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  https://inference.hetzner.com/api/v1/chat/completions \
  -H "Authorization: Bearer $HETZNER_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"ok"}],
       "max_tokens":5,"chat_template_kwargs":{"enable_thinking":false}}'
grep -c "500 Internal Server Error" /var/log/lint-ii/app.log   # before vs after
```

A handful of 500s is survivable (81 in the next run cost exactly one item);
hundreds voids the run.

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
  nominalized-infinitive lists with two plain NP lists (enum-6/7, authored as
  conj-route sentinels, detecting since `fe0ff4c` — note they never actually
  measured that gap at the headline level, see the third-run notes);
  family-* covers
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
| 5 | — | — | 0.94 / 0.95 | **0.95 / 0.94** (5th run 2026-08-06, `bb8783d`) |

Set 5 has been run four times. The sequence matters more than any single
number, and it is the strongest argument in this file for judging phenomenon
fixes by phenomenon counts rather than by precision/recall:

| run | commit | headline | what actually changed |
|-----|--------|----------|------------------------|
| 1 | `8397cae`+`c7fdb54` | 0.95 / 0.95 | spelling attribution corrected |
| 2 | `a4246f3` | 0.95 / 0.97 | conj-3 no-op FP fixed; recall gain was conn-6 wobble |
| 3 | `fe0ff4c` | 0.95 / 0.95 | enum 5/7 → **7/7** — invisible to the metric |
| 4 | `48c8fa8` | 0.97 / 0.94 | dt fix — invisible; family-4 FP cleared |
| — | `bebbecd` | ~~0.95 / 0.71~~ | **VOID** — 295 provider 500s, not a regression |
| 5 | `bb8783d` | 0.95 / 0.94 | swap judge: 3 bad swaps removed, 1 good one lost |

Twice the headline moved OPPOSITE to a fix that demonstrably worked, and once
it measured a provider outage. Each run is described below; read them together.

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
~~all five catches were the Hunspell pass~~ **(WRONG — see the re-run note
below; `model` was `None` for everything because of the serialization bug,
and `None` was read as "Hunspell". The pass was unknown, not Hunspell.)** —
the one miss ("Ik wordt") needs LLM dt-detection, confirming the wobble.
FPs (4): one abstract_nouns
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

python3 wordfreq_probe.py --reps 12                        # item 2: semantic swaps
python3 wordfreq_probe.py --group control --reps 4         # ...must still simplify
python3 swap_judge_probe.py --judge calibrated             # can Qwen judge its own swaps?
```

**Reps: 6 for connective, 12 for word-frequency.** The word-frequency
aggregate is far noisier — two runs of an IDENTICAL prompt scored 55% and 42%
(a 13-point swing, larger than the effect being measured), which briefly
produced a confident wrong conclusion. 5 reps is enough for a per-case
verdict, not for a total.

**Always run the control group.** `wordfreq_probe.py --group control` scores
words that MUST still be simplified, where refusing is the failure. A variant
can otherwise look excellent purely by declining to do its job — and the
winning variant does add an ONGEWIJZIGD escape hatch, so this is a live risk
rather than a theoretical one.

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

Set-5 re-run notes (2026-07-31, after `8397cae` + `c7fdb54`): **0.95 / 0.95**,
up from 0.94 / 0.95. Three findings, only one of which is the headline:

- **The set-5 spelling attribution above was wrong, and this is the important
  correction.** With `Suggestion.model` serializing for the first time, all 7
  spelling suggestions carry the Qwen model name and the Hunspell pass
  contributed **zero suggestions in the whole 100-item run** (0 of 107 have
  `model=None`; `hunspell_spelling.py` hardcodes `model=None`, the LLM pass
  sets `provider.model_name`, so the count is unambiguous). The catches are
  the LLM pass, not Hunspell. This is exactly the mis-attribution this README
  warns about, sprung by the very bug the `model` field was added to prevent —
  when attribution is broken, "unknown" reads as whichever pass you assumed.
  Detection was 6/6 this run, including the previously-missed dt-error
  "Ik wordt" → "Ik word", which is backlog item 4's wobble, not a fix.
- Connective 7/10, down from 8/10, and **not a regression**: the misses are
  conn-5, conn-6 and conn-10. A 6-rep probe scores conn-5 at 1/6 and conn-6 at
  0-1/6 under the OLD and new prompt alike — they are ~1-in-6 firers either
  way, and the 8/10 run caught them on a lucky draw. conn-10 is the documented
  inferential residual (item 1). conn-9 fired again.
- FPs down 4 → 3: `family-4` ("terugzwemmen" → "te rugzwemmen") is gone,
  confirming `c7fdb54`. Survivors are clean-13 (volksuniversiteit → "openbare
  school voor volwassenen", item 2), url-1 (defensible max_sdl split, URL
  intact), and conj-3 — where the `word_frequency` suggested text is
  BYTE-IDENTICAL to the original. That no-op suggestion is still slipping the
  filters and is worth a deterministic guard.

Also seen: on spelling-4 and spelling-6 the connective pass merged sentences
that still contain the planted typo ("...kwam onmiddelijk in actie, dus..."),
since it does not spell-check. Harmless to scoring, but a tester would see a
suggestion containing a visible misspelling.

Set-5 SECOND re-run (2026-07-31, after `a4246f3`): **0.95 / 0.97**. A good
illustration of how little a single 100-item run resolves — one real fix,
otherwise noise:

- **conj-3 is gone from the FPs.** That is the one change attributable to the
  commit, and the live probe agrees (the input now yields no suggestions).
- Precision did NOT move, because an unrelated FP replaced it: `family-4`
  drew a `max_sdl` rewrite ("Wie durft, mag na de les een stukje
  terugzwemmen..." → "Na de les mag wie durft..."). `a4246f3` only removes
  suggestions from the bundled word-frequency path, and family-4 produced
  nothing at all in the previous run, so this is the max_sdl pass wobbling.
  Note family-4's `must_not` is word_frequency — the family guard HELD.
- Recall 0.95 → 0.97 is the conn-5/conn-6 wobble landing favourably: conn-6
  fired, taking connective to 8/10. Both are ~1-in-6 firers under the old and
  new prompt alike, so anything in 7-9/10 means "unchanged". Spelling 6/6.

Emerging pattern worth a decision: 2 of the 3 surviving FPs (url-1, family-4)
are now `max_sdl` firing on borderline-length sentences and producing
DEFENSIBLE rewrites rather than errors. That is a scoring-convention question
(should a sound rewrite of a 13-word sentence count against precision?) more
than a quality defect — worth settling before chasing max_sdl precision.

Set-5 THIRD run (2026-08-04, after `fe0ff4c`): 0.95 / 0.95. **The headline is
the wrong place to look for this fix, and the reason is a flaw in the harness
worth understanding before designing another sentinel.**

- The fix worked: enum 5/7 → **7/7**, enum-6 and enum-7 both detect, zero new
  FPs (the route fires on 0 of 185 non-enum items offline).
- Yet recall went 0.97 → 0.95. Two independent things moved: enum gained 2,
  and conn-6 — a ~1-in-6 firer under the old and new connective prompt alike —
  drew its miss again. The connective loss is sampling; the enum gain is a
  deterministic detector change.
- They do not cancel arithmetically because **enum-6/7 were already counted as
  TPs**. Both attract `word_frequency` suggestions, so the item scored as
  "suggested something" while the enumeration itself was missing.

The lesson for corpus design: presence/absence is an ITEM-level metric, so it
only moves when an item goes from zero suggestions to some. A sentinel for a
missing PHENOMENON is invisible to it unless the item is otherwise clean —
enum-6/7 never measured the gap they were authored to track. Judge a
phenomenon fix by its phenomenon count (here 5/7 → 7/7) plus offline
validation, not by precision/recall.

Set-5 FOURTH run (2026-08-04, after `48c8fa8`): **0.97 / 0.94**. The dt fix is
again invisible to the headline — spelling was already 6/6 in the three
previous runs, so there was no room to gain — and the run's real job was
regression detection, which it passed:

- **Zero spelling FPs on any negative.** That was the risk worth checking: a
  dt-focused instruction could have made the pass over-flag correct verb
  forms. It did not, on all 35 negatives with production filters applied.
- Precision 0.95 → 0.97: `family-4`'s max_sdl rewrite did not recur (the same
  borderline-length wobble noted two runs earlier). Not attributable.
- Recall 0.95 → 0.94: one new FN, `compound-6`, and it is NOT the spelling
  change — that commit only touches the spelling prompt. Probing the word
  directly 8x: the model never returns ONGEWIJZIGD, but 4 of 8 times it
  answers with a shorter yet still-rare compound ("fietsparkeersysteem",
  "fietsenstalling") instead of splitting, and the frequency band check
  correctly rejects those. Pre-existing compound wobble in word_frequency.
- connective 7/10, spelling 6/6, enum 7/7 — all unchanged.

Set-5 FIFTH run (2026-08-06, after `bb8783d`, swap judge live): **0.95 / 0.94**.
Valid despite 81 provider 500s — only one item lost everything, and the judge
explains that one, not the outage. (The run before it, at `bebbecd`, scored
0.71 recall and is VOID: 295 500s, and 15 items shed suggestion types the
judge cannot touch.)

The judge made 5 rejections across 100 items — the direct evidence, better
than the headline:

| rejected | verdict |
|----------|---------|
| verharding → vastberadenheid | correct |
| verzakelijking → zakelijkheid | correct (also produced ungrammatical Dutch) |
| insinuaties → suggesties | correct |
| aggresief → agressief | harmless — the SPELLING pass still supplies that fix |
| ambivalent → verdeeld | **false alarm, and it cost a whole item** |

The denominalization fix held: `verlaging → minder` and `afname → minder` are
absent from the list and the `abstract-*` group is intact (all seven keep
their suggestions bar abstract-6, where the judge correctly removed
verharding).

The one costly error is instructive. `verdeeld` is a GOOD replacement for
ambivalent, and it was that item's only suggestion, so rejecting it dropped
wordfreq-4 to zero and made it the run's new FN. The judge has no stable view
of that word: it rejects both the bad swap (`→ twijfelachtig`) and the good one
(`→ verdeeld`). **A false alarm on a single-suggestion item costs the whole
item**, which is a sharper failure than on an item with several — worth
considering if the judge is ever tightened.

Net for the run: 3 meaning-changing suggestions removed, 1 legitimate one
destroyed. The other FP change (family-4) is the max_sdl borderline wobble,
unrelated.

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
2. **Qwen semantic swaps on positives** — PARTLY addressed by `3992b09`;
   what remains is characterised, not guessed. Measure with
   `wordfreq_probe.py` (11 reproducible bad swaps + 10 must-still-simplify
   controls). The item said "likely prompt work"; that is only half right.
   - **Shipped:** meaning-preservation examples in the bundle prompt, BAD
     67% → 52% at 12 reps, controls 40/40 with 0 refusals (no recall cost).
   - **The class splits in two, and this is the key finding.** Every swap
     failing 12/12 at base still fails 12/12 after the fix
     (monumentale→groot, gewaande→vermeende, insinuaties→suggesties,
     notoire→bekende — 48 of the 68 remaining failures). The entire gain came
     from partially-failing cases (koeling 8→3, verharding 10→5, reder 4→0).
     **Prompt examples move wobbly cases and cannot touch confident ones.**
     Do not spend more prompt effort on the confident four.
   - **Vector similarity is ruled out** as a deterministic guard: spaCy
     `nl_core_news_lg` cosine gives BAD mean 0.440 vs GOOD 0.529, heavily
     overlapping, because the worst swaps are topically CLOSE
     (koeling→koelkast 0.547, insinuaties→suggesties 0.561 both score above
     the legitimate beoogt→wil 0.288). Relatedness is not substitutability.
   - **Verification pass — SHIPPED** in `bebbecd` + `bb8783d`
     (`_verify_word_swaps`, prompt `swap_judge`, `LINT_II_SWAP_JUDGE=0`
     disables; defaults ON). Qwen does NOT share its own generator error — it
     rejects monumentale→groot when asked directly — so a second call reaches
     what prompt work cannot.
     Cost is a non-issue, which was the open question: ~8 completion tokens
     and ~0.22s per call, ~40 per 100-item set, ~0.6% of the per-minute output
     budget. FAIL-OPEN throughout (exception, unparseable answer or missing
     verdict all KEEP the suggestion).
     **Calibration is the entire design, and every direction has been measured
     to fail in a different way:**
     | judge wording | detection | false alarms |
     |---------------|-----------|--------------|
     | "precies hetzelfde?" | 100% | **60%** — unusable |
     | "zet dit de lezer op het verkeerde been?" | 47% | 0% on probes, but rejected `verlaging→minder` live |
     | + blanket "simplifying noun constructions is good" | 25% | 0% |
     | + NARROW change-nominalization rule (shipped) | 37% | 0% |
     The false-alarm side governs: a false alarm deletes a legitimate
     simplification, which is the product. On a live run the shipped judge
     removed 3 meaning-changing swaps and destroyed 1 good one.
     **Known weakness:** it has no stable view of some words — it rejects both
     `ambivalent→twijfelachtig` (bad) and `ambivalent→verdeeld` (good). And a
     false alarm on a single-suggestion item costs the WHOLE item.
     Still untouched by either layer: monumentale→grote, conservator→bewaarder,
     notoire→bekende all pass the judge as readily as they pass the generator.
   - **Scope limit of the shipped fix:** it patches `word_frequency_bundle`
     only. A trigger folded into a consolidated sentence_rewrite uses a
     different prompt carrying none of this guidance. Observed benign once
     (monumentale preserved), but unguarded and unmeasured.
3. ~~**Enumeration conj-route gap**~~ — FIXED in `fe0ff4c`. corpus5 enum-6/7
   detect; the enum group is 7/7. A second surface route counts plain comma
   lists ("A, B, C en D") with phrase-level items.
   The diagnosis was worse than "parses as pair-chains": on enum-6 spaCy
   chains 2 of 4 items, and on enum-7 it chains "binnenstad" — a noun from
   INSIDE the third item — to the wrong head. The chain is wrong, not just
   short, so no tuning of the conj route could reach these.
   The warning about the shortlist guards was aimed at the wrong gate:
   shortlist-* was never held back by item count (it hits
   `enumeration_min_items` exactly, chained correctly) but by SPAN, 4-6
   against a threshold of 12. The new route reuses that span gate and the
   margin stays wide (negatives 4-6, positives 15-24).
4. ~~**Spelling detection wobble**~~ — FIXED in `48c8fa8`, and the diagnosis in
   the old item text was wrong twice over.
   - It was never stochastic across the group and never a DETECTION failure.
     At 10 reps: non-word typos 30/30, dt-errors 24/30, and all six failures
     are ONE case. On "Ik wordt volgende maand geopereerd" the model flags
     "wordt" 10/10 and 6 of those returns `CORRECTIE: wordt` — the word
     unchanged. The pipeline correctly drops that (`suggested_text ==
     sent_text`), so a **correction-formation** failure surfaces as a missing
     suggestion and reads like flaky detection. That is also why three
     consecutive runs all scored 6/6 while the item predicted ~1 in 3: at item
     granularity you sample a 40%-failure case once per run.
   - Fixed with conjugation guidance + the rule that CORRECTIE must differ from
     WOORD. dt 24/30 → 30/30 (the one case 4/10 → 10/10), non-word unchanged,
     and clean controls widened to 16 items IMPROVED 72/80 → 76/80 — the dt
     focus did not cause over-flagging. Measure with `spelling_probe.py`.
   - **The Hunspell pass is correct, not dead.** It produced nothing across 100
     items because it skips words the LLM already flagged, and the LLM catches
     all six typos first. Given uncovered input it still fires
     (acomodatie → accommodatie). It STRUCTURALLY cannot catch dt-errors —
     word/vind/loop are valid dictionary entries — and misses "onmiddelijk"
     because spylls' `suggest()` returns empty for it. Non-word typos are
     Hunspell's; dt-errors are the LLM's alone, which is why this item was
     always an LLM-prompt problem.
5. ~~**No-op `word_frequency` suggestions slip the filters**~~ — FIXED for the
   suggestion layer. The BUNDLED word-frequency path lacked the
   `_is_noop_rewrite` check the per-trigger path already had, so a rewrite
   identical to the original reached the user (corpus5 conj-3, two consecutive
   runs). Added there.
   **The underlying trigger is still live and is an analyzer issue, not an LLM
   one:** spaCy's Dutch tokenizer keeps `vol.` as ONE token (a known
   abbreviation — *vol.* = volume), unlike `mogelijk.` which splits. Such a
   token is absent from SUBTLEX, so it scores as rare and fires a
   word_frequency trigger whose only possible "fix" is the same word without
   the period. The band check cannot catch this (rare → common always passes).
   Note this also means `word_frequency` — a LiNT scoring feature — treats
   these tokens as rare, so the effect is not confined to suggestions.
   Fixing it properly means normalising abbreviation-final tokens before the
   frequency lookup, which CHANGES LiNT SCORES and must be validated against
   the LiNT reference first. Left deliberately untouched.
