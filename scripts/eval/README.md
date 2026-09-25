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

**The runner now does this itself** (2026-09-24): it reads the service log
around every item, counts provider-call failures (429 AND 5xx), retries a
contaminated item up to `--retries` times (default 2), and ends with a
`VALIDITY:` line — `CLEAN`, `CONTAMINATED` (naming the items), or `NOT CHECKED`
when the log is unreadable, which is the case on the Mac. Resuming with the same
`--results` re-runs contaminated items and skips clean ones.

**Why it had to be automatic.** The manual gate below counted only 500s. On
2026-09-11 set 3 ran through 19 **429s** with a clean 500 count and was read as
a healthy run. Six of its seven "misses" — including all three connective ones —
produced suggestions when re-tested, so the reported 0.89 recall, and the
conclusion drawn from it that recall/connective was the engine's weak axis, were
a rate-limit artifact. 429s rose ~40x in September 2026; they are now the
common failure mode, not the rare one.

Manual equivalent, if you ever need it:

```
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  https://inference.hetzner.com/api/v1/chat/completions \
  -H "Authorization: Bearer $HETZNER_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"ok"}],
       "max_tokens":5,"chat_template_kwargs":{"enable_thinking":false}}'
grep -cE '"HTTP/1.1 (429|5[0-9][0-9])' /var/log/lint-ii/app.log   # before vs after
```

Count **429s as well as 5xx** — a clean 500 count proves nothing about 429s. A
handful of errors can be survivable; with per-item retries the runner now
decides that per item rather than per run.

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

### Scoring convention: `must_not` vs `should_suggest` (SETTLED 2026-09-20)

`must_not` was recorded from the start but **never scored** — every guard it
names was checked by hand. `run_eval.py` now checks it, and reports three
numbers instead of one:

1. **Presence/absence precision/recall** — unchanged, so the cross-set table
   stays comparable. Read it knowing what it conflates (below).
2. **Guard violations** — the `must_not` rules actually breached. This is the
   metric that asserts something; it should be **0**.
3. **Negatives that fired, split by kind** — `must_not=[]` (should have been
   silent: a genuine precision concern) versus guarded items where a
   *different* suggestion type fired.

The convention, stated once: **`should_suggest=False` with a non-empty
`must_not` asserts only that the named behaviour must not occur.** A different,
legitimate suggestion on that item is not a defect. `should_suggest=False` with
an EMPTY `must_not` (clean-*, good-*) does mean "nothing should fire".

Why this matters, measured over sets 2-5: **14 of 25 false positives (56%) were
guarded items whose guard held.** Treating those as defects moves precision
from 0.96 to 0.91 — so the legacy number understates the engine by ~5 points,
and conj-4's `invalidenplaatsen → parkeerplaatsen voor gehandicapten`, a good
suggestion, was being counted against us.

| set | legacy precision | guard-aware | FP split (silent / guarded) |
|-----|------------------|-------------|------------------------------|
| 2 | 0.86 | 0.93 | 5 / 5 |
| 3 | 0.92 | 0.98 | 1 / 4 |
| 4 | 0.90 | 0.94 | 4 / 3 |
| 5 | 0.95 | 0.98 | 1 / 2 |
| all | 0.91 | **0.96** | 11 / 14 |

**One thing the split does NOT mean.** A guarded item firing something else is
not automatically *good* — set 3's shortlist-4 kept its enumeration guard and
still produced "De gereedschap bevat ..." (item 7). Those cases leave the
automatic FP count and enter the LLM-as-judge queue; they are not waved through.

Validation: run over the four stored result sets, the checker reports **0
violations and 0 unchecked rules**, matching the by-hand conclusion from each
of those runs. All 8 distinct `must_not` values across the five corpora have
predicates; a new, unrecognised rule is reported as `UNCHECKED` rather than
silently passing.

## Deterministic-guard unit tests (`tests/`)

The pure guard/filter methods on `SuggestionEngine` — the deterministic
backstops that reject bad LLM output — have a pytest suite (`tests/`,
`python -m pytest`, ~4s, no LLM calls, no API keys). Until 2026-09 they were
validated only by 40-minute eval runs. `test_text_guards.py` covers the
text-level guards (`_is_noop_rewrite`, `_breaks_clause_conjunction`,
`_alters_url`, band/family checks, `_connective_adds_content`,
`_correction_plausible`, enumeration parsing/anchoring);
`test_parse_guards.py` covers the spaCy/Hunspell-dependent ones
(`_np_coordination_list`, `_nominalized_infinitive_list`,
`_dehet_disagreement`, `_introduces_misspelling`). Every guard is tested in
both directions — fires on a breach, silent on legitimate input (method
lesson 4). Known defects are encoded as strict `xfail` so a fix flips them
loudly: backlog item 6 (`banenzwemmen → banenzwemmer` passes
`_correction_plausible`) is there now. Run the suite before an eval run when
touching a guard: it catches shape regressions for free; the eval's job is
what the units cannot see (LLM behaviour, pass interactions).

Frontend logic that is pure enough to run outside a browser has Node tests in
`tests/js/` (`node --test tests/js/`, built-in runner, no npm install); the
first covers the variant labels in `suggestion-popup.js`.

### max_sdl on short sentences (DECIDED 2026-09-24: leave as is)

**Decision: no change.** `max_sdl` keeps its gate (SDL > 5 in a sentence of
≥ `max_sdl_min_words` = 12 words), including on 12–15-word sentences.

The question was first posed as scoring: *should a sound max_sdl rewrite of a
~13-word sentence count against precision?* The `must_not` convention above
answered that before anyone had to. Across all five sets on the current engine,
max_sdl fires on a negative item exactly **5 times, and every one is a guarded
item** — `url-1` ×3, `shortlist-1`, `family-4` — so none is a false positive
under the settled scoring. It fires on **zero** silent-required (`must_not=[]`)
items. There was no scoring decision left to make.

What remained was behavioural: should max_sdl fire on sentences that short at
all? The 5 firings all sit at the gate's floor (12–15 words), and the rewrites
are defensible — `url-1` splits cleanly after the URL, `shortlist-1` simplifies
"In het pakket zitten…" to "Het pakket bevat…". The one borderline case is
`family-4` ("Wie durft, mag na de les…" → "Na de les mag wie durft…"), a
reordering that shortens nothing.

**Rejected alternative: raise `max_sdl_min_words` 12 → 16.** It would suppress
those 5 firings, but on POSITIVE items it would also remove 8 max_sdl
suggestions on ≤15-word sentences — and on 3 of those max_sdl is the item's
ONLY suggestion, so ~3 items of recall would be lost. That trades real recall
to suppress firings that are neither false positives nor bad. Measured:

*(Corrected 2026-09-24: this paragraph originally also called recall "the axis
the five-set sweep showed is weakest (set 3: 0.89)". That premise was a
rate-limit artifact — see the provider-error gate. The decision does not depend
on it: losing ~3 items of recall to suppress non-defects is a bad trade whichever
axis is weaker.)*

| sentence length | max_sdl on positives | item's only suggestion |
|-----------------|----------------------|------------------------|
| ≤ 15 words | 8 | 3 |
| 16–20 words | 2 | 1 |
| > 20 words | 5 | 4 |

**Caveat, recorded so the evidence is not overstated.** `family-4` counts as
guarded somewhat incidentally: its `must_not` is about word_frequency, not
max_sdl, so it is excluded because the *family* guard held, not because anyone
judged max_sdl fine there. The convention as written treats that correctly,
but the real evidence is four clear cases and one borderline one.

**Revisit if:** a max_sdl firing appears on a `must_not=[]` item (that would be
a genuine false positive the convention does not excuse), or reorderings like
family-4's — moving a clause without shortening anything — recur often enough
to be a pattern rather than one case.

### Intermediate (two-sentence) rewrite variant — shipped 2026-09-25

`476cb6b` merges the Mac's TUSSENVORM feature (`0d6ba5e`) plus a box-side fix
(`35099a5`). Measure with `rewrite_variant_probe.py`; presence/absence cannot
see this feature at all, since adding a variant changes what is inside a
suggestion, not whether one appears.

**The probe caught a regression the branch was waiting to find.** Asking for a
third variant pushed BEHOUDEND to split on long sentences — one-sentence
compliance fell from 46/72 to 30-37/72, one-directionally (4 sentences worse,
0 better), and on c5-long-7 BEHOUDEND came back as essentially VOLLEDIG
reworded. The one-sentence option had not moved to the middle; it had vanished.
A worked example (BEHOUDEND explicitly one sentence "ook al is die lang"),
vetted against all five corpora, restored it to 47/72 and raised TUSSENVORM
usability to 43/72, with VOLLEDIG stable and no content loss.

**Regression gate, set 3 (`results3d.json`)**: `VALIDITY: CLEAN`, 0.93 / 0.97 —
identical to the pre-feature run on every count (same two misses, same silent
FP, 0 guard violations). 17 of 28 consolidated suggestions offer all three
variants; 10 collapse to conservative+full (a sentence with one natural split
point, as designed). Cost: 286k tokens vs 275k, **+4% per run** — much less than
the +26% per consolidated call, because consolidated calls are a minority. The
gate's per-item retry fired for the first time in real use (one 429, recovered).

**Watch item — BEHOUDEND sometimes returns the sentence unchanged.** The example's
BEHOUDEND changes one word, and occasionally the model reads "minimal" as "none".
The no-op guard then drops it and the popup falls back to intermediate (1 of 28
in the gate; long-3). It fails gracefully, and on long-3 it replaced a worse
failure — before the feature that sentence's "one-sentence" option was actually
two sentences. The probe's 12 sentences showed zero no-ops, so this is exactly
the kind of thing a larger or different sample finds. Revisit if it becomes
common.

**Pre-existing, and a decision rather than a bug:** even the original prompt leaves
BEHOUDEND split on roughly a third of long sentences, and some (c3-long-1,
c5-long-2) split under every prompt in every run — so the "one-sentence" option
is sometimes mislabelled. A deterministic gate (`"conservative": 1` beside
`"intermediate": 2` in `_VARIANT_SENTENCE_COUNT`) would stop the mislabelling but
DROP BEHOUDEND on those sentences, changing what the frontend shows by default.

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

All figures are LEGACY presence/absence (any suggestion on a negative item is a
false positive), kept for comparability across the whole history. The
`guard-aware` column applies the settled scoring convention — see the
scoring-convention section — and is the number that reflects the engine.

| Set | Mistral@0.7 | Qwen@0.3 | Qwen + July-30/31 fixes | current engine (legacy) | guard-aware |
|-----|-------------|----------|--------------------------|-------------------------|-------------|
| 1 (dev) | 0.84 / 1.00 | 0.93 / 0.98 | — | 0.97 / 0.97 (2026-09-25, `a9952be`, **VALIDITY: CLEAN**) | **1.00** / 0.97 |
| 2 | 0.88 / 1.00 | 0.86 / 1.00 | — | 0.89 / 1.00 (2026-09-25, `a9952be`, **VALIDITY: CLEAN**) | **0.94** / 1.00 |
| 3 | 0.88 / 0.98 | 0.86 / 0.95 | — | 0.93 / 0.97 (2026-09-24, `a9952be`, **VALIDITY: CLEAN**) | **0.98** / 0.97 |
| 4 | — | — | 0.90 / 0.92 | 0.90 / 0.94 (2026-08-11, `c60953d`) | **0.94** / 0.94 |
| 5 | — | — | 0.94 / 0.95 | 0.95 / 0.94 (2026-08-06, `bb8783d`) | **0.98** / 0.94 |

Recall is identical in both columns: the convention only changes how negatives
are counted. **Guard violations are 0 on all five sets** — every `must_not` the
corpora assert has held on the current engine.

**Discount the set-1 row.** It is the DEV set: the thresholds were fitted to it,
which is why it has the table's highest precision and 9/10 connective against
set 3's 7/10. A strong number there confirms little; a weak one would have been
the informative outcome. Judge the engine on sets 2-5.

**Set 3 was re-run clean on 2026-09-24** (`results3c.json`, first run under the
automatic provider-error gate): `VALIDITY: CLEAN`, zero retries needed, 275k
tokens. Recall **0.89 → 0.97**, connective **7/10 → 10/10**, guard-aware
precision 0.98. The 11-September figures (0.92 / 0.89) are superseded and should
not be cited. Its two remaining misses are both word_frequency: `wordfreq-4`
("inconsistent", missed in both runs — genuine) and `wordfreq-6`
("ongefundeerd", which fired on 11 September, so ordinary word-frequency wobble).

**CORRECTED 2026-09-24 — recall is NOT the weak axis.** This paragraph used to
say it was, "floored by set 3's 0.89 (mostly connective)". Set 3's run hit 19
provider 429s, which the fail-open passes turned into silent misses; re-tested,
6 of its 7 misses produce suggestions. With that artifact removed, neither axis
stands out — guard-aware precision and recall both sit in the mid-to-high 0.90s.
Sets 1 and 2, also run in September with 5 provider 429s each, were **re-run
clean on 2026-09-25** (`results1c.json`, `results2c.json`; `VALIDITY: CLEAN`, zero
retries, 553k tokens for both). Recall rose in both — set 1 0.94 → 0.97, set 2
0.97 → 1.00 — and **only by recoveries: no item got worse in either set**
(recovered: set 1 passive-5/passive-6, set 2 multi-5/wordfreq-5). That
one-directional pattern is what removing contamination looks like; wobble would
lose items as well as gain them. The old logs cannot tie individual 429s to
items, so no single recovery is claimed as a 429. Set 1's remaining misses
(abstract-5, compound-6) missed in both runs and are genuine.

Two false positives also disappeared, and those ARE attributable — to this
week's fixes, not to the re-run: set 2's clean-8 was the `banenzwemmen →
banenzwemmer` Hunspell bug (item 6, `a9952be`), and set 1's clean-1 was
`pas. → kaart` (item 5's trigger half, `23fbe2a`). The first full-eval
confirmation of both fixes.

Sets 1-3 are now gate-verified clean. Sets 4 and 5 ran in August, before the
rate-limit rise and before the gate existed, so they are clean by circumstance
rather than by measurement. What survives from the old
paragraph is the scoring point: under the legacy metric alone, precision looks
like the problem, which is why the convention needed settling.

Sets 2-5's guard-aware numbers are a RE-SCORE of the same runs, not fresh
measurements — the runner is resumable, so pointing it at a stored
`results*.json` re-reports with no new API calls, which isolates the scoring
change from run variance. Set 1 was measured fresh, with the new scoring already
in the runner.

Do not read guard-aware precision as "those items are fine". A guarded item can
hold its guard and still produce something bad — set 3's shortlist-4 kept its
enumeration guard while emitting "De gereedschap bevat ..." (item 7). Such cases
leave the automatic FP count and enter the judging queue.

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

Set-4 runs 2 and 3 (2026-08-07 `bb8783d`, 2026-08-11 `c60953d`) exist to test
the **swap judge on a set it was not tuned on**, and that is what they were
worth. Both valid (0 new provider 500s).

Run 2 (0.92 / 0.91) found the judge's real defect. It rejected
`vermindering → minder` and `verkorting → minder` — the same denominalization
shape that `bb8783d` had supposedly fixed on set 5. The clause had patched the
two measured instances, not the cause. **A second corpus is what exposed it;
set 5 alone looked fixed.** The cause was that the judge saw only the ORIGINAL
sentence plus the bare pair, so it judged a substitution the pipeline never
makes ("een minder van de overlegdruk") rather than the real proposal
("... en minder overlegdruk"). Fixed in `c60953d` by passing suggested_text.

Run 3 (0.90 / 0.94) confirms the fix and shows its limits:
- The denominalization rejections are gone. Prediction confirmed.
- Seeing the rewrite adds a capability the word pair cannot give: it catches
  BROKEN rewrites. `leidingwerk → leidingen` produced "het vervangen van het
  volledige leidingen" (ungrammatical) and `instrueren → uitleggen` produced
  "het uitleggen van de vrijwilligers" (explaining *the volunteers*). Both
  correctly rejected; neither is visible from the pair alone.
- But the errors MOVED rather than reduced. Two new false alarms —
  `monumentenvergunningstraject → procedure voor een monumentenvergunning` (a
  good compound split, and it cost compound-6) and `onvoorspelbaarder →
  moeilijker te voorspellen`. And by no longer rejecting `moestuinvereniging`
  it stopped incidentally fixing the clean-9 FP, so precision fell.

**Do not read 0.92/0.91 vs 0.90/0.94 as an A/B of the judge**: the two runs
generated different swaps, so generator wobble is mixed in.

The pattern across three runs on two corpora: **this judge sits at roughly 2
false alarms per ~40 swaps whatever the wording, and each fix relocates the
errors** — denominalizations, then compound splits. Further prompt tuning has
low expected value. Its best property, catching ungrammatical rewrites, is
probably better served by a deterministic grammar check.

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
**DECIDED 2026-09-24: leave max_sdl as is** — see "max_sdl on short sentences"
in the scoring-convention section.

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

Set-2 re-run (2026-09-11, `c60953d`): **0.86 / 0.97** against Qwen@0.3's
0.86 / 1.00. The first time ANY of the July/August work met a set it had not
been tuned on, and the headline is the least informative part of it.

**The engine held.** Recall cost is two items (wordfreq-5 "indicatief",
multi-5 passive) across five engine changes, every one of which only ever
REMOVES suggestions. No guard misfired: in all five `must_not` items the
forbidden behaviour did not occur.

**Half the FPs are a scoring artefact, not an engine fault.** In five of the
ten, the `must_not` guard held and a DIFFERENT suggestion type fired:

| item | forbids | fired | guard |
|------|---------|-------|-------|
| shortlist-1 | enumeration | max_sdl | held |
| conj-1, conj-4 | split ", maar " | abstract_nouns / word_frequency, `maar` intact | held |
| url-1, url-3 | alter URL | max_sdl / abstract_nouns, URL intact | held |

conj-4's `invalidenplaatsen → parkeerplaatsen voor gehandicapten` is a good
suggestion counted against us. A binary `should_suggest` cannot express "no
enumeration here, but a word-frequency suggestion is fine" — the same
scoring-convention question raised by the set-5 max_sdl FPs, now confirmed on
independent data. **Worth settling before anyone tunes a pass for precision.**

The rest: two defensible connectives (clean-13, good-3, both real causal
links); two mild quality issues (good-1 drops a sentence, clean-14 is a lateral
swap); one real bug (backlog item 6); and `clean-13`'s `omleiding → omweg` —
the semantic-swap class, and **the swap judge rejected exactly that pair on
set 4**. It is off, so it got through. First evidence the judge would help on
unseen data.

Set-3 re-run (2026-09-11, `3e85b3f`): **0.92 / 0.89** against Qwen@0.3's
0.86 / 0.95 — precision up six points, recall down six. With this, every
held-out set has been measured against the current engine.

~~**The recall drop is mostly connective** … **the connective pass is weaker on
unseen data than sets 4 and 5 imply.**~~ **WRONG — retracted 2026-09-24.** This
run saw 19 provider 429s, and every pass is fail-open, so a rate-limited call
became a silent "miss". Re-tested through the live pipeline, 6 of the 7 FNs
produce suggestions: conn-2 → "…afgesloten, **want** er wordt…", conn-3 →
"…niet, **dus** de lessen…", conn-4 → "…recordwinst, **maar** de werknemers…"
(all three also 6/6 in the connective probe), plus wordfreq-5, long-1 and
multi-7. Only wordfreq-4 ("inconsistent") is a genuine miss. The honest
held-out connective picture is ~36/40 on sets 2-5, every remaining miss already
documented (conn-10's inferential consequences on sets 4 and 5; set 5's
conn-5/conn-6, ~1-in-6 firers under both prompts). The 500 count for this run
was clean, and that is exactly why nobody noticed: the gate did not count 429s.
The runner now does — see the provider-error gate at the top of this file.

**The scoring artefact recurs for a third set.** conj-4 (`, maar ` intact),
url-1 and url-2 (URLs intact) all had their `must_not` guard hold while a
different, legitimate suggestion type fired. Sets 2, 3 and 5 now agree, so this
is systematic: **precision on these corpora understates the engine**, and
tuning any pass for precision optimises against the measurement rather than the
product. **SETTLED 2026-09-20** — see the scoring-convention section above;
`run_eval.py` now scores `must_not` and splits the negatives.

Also seen: `bezorging → levering` fired here (conj-4) and on set 2 (clean-14) —
the same lateral swap twice, neither simpler nor wrong, the kind the swap judge
is aimed at. And shortlist-4 produced ungrammatical Dutch: backlog item 7.

**Measured cost of a 100-item run** (first run with token logging live):
**255,210 tokens over ~240 calls**, ~1,080 tokens/call. An earlier estimate of
"1-2M per run" was wrong — it extrapolated from one atypical text with many
triggers. The per-month figures in this file derive from CALL counts, so they
are unaffected. A full five-corpus sweep is ~1.3M tokens, which makes
validating a provider change cheap.

## Comparing models / providers (and a worked example)

The probes take `LINT_II_LLM_MODEL`, so swapping the model is a config change,
not a code change:

```
LINT_II_LLM_MODEL=<other-model> python3 connective_probe.py --reps 6 --workers 2
LINT_II_LLM_MODEL=<other-model> python3 wordfreq_probe.py   --reps 12 --workers 2
LINT_II_LLM_MODEL=<other-model> python3 wordfreq_probe.py   --group control --reps 4
```

**Run all three.** The third is not optional: the hard group can be "improved"
by a model that simply refuses to simplify, and only the control group
distinguishes that from a real gain.

### Worked example: Qwen3.8-27B vs Qwen3.6-35B-A3B-FP8 (2026-09-20)

Hetzner began serving a second model (`Qwen3.8-27B`) alongside ours. Measured
rather than guessed:

| dimension | Qwen3.6 (current) | Qwen3.8-27B |
|-----------|-------------------|-------------|
| semantic swaps, BAD rate | 52% | **34%** |
| refusals on must-simplify | 0/40 | 0/40 |
| connective probe | 90% | 89% (no real difference) |
| latency per call | 0.41s | **1.51s** (~3.7x) |

Compatibility is fine: it parses the block format, speaks correct Dutch, and
honours the same `enable_thinking:false` kwarg (thinking still defaults ON, so
the kwarg stays load-bearing). `HetznerProvider` would drive it unchanged.

**The finding that matters is about backlog item 2, not about switching.**
`gewaande → vermeende` failed **12/12 under every prompt variant tried** and is
**0/12** on the newer model. So part of that residual is
**model-capability-bound, not engineering-bound** — which vindicates stopping
prompt work on those cases, and means the swap judge compensates for something
a model upgrade partly fixes. Note `notoire → bekende` is 12/12 on BOTH models:
that one is a stable property of Qwen's Dutch, not a capability gap.

**But a newer model is not a free upgrade.** `koeling → koelkast` regressed
3/12 → 11/12 — the "action becomes an appliance" error, much worse. And the
apparent fixes on verharding/structureel/verwaarloosde are the ONGEWIJZIGD
escape hatch, i.e. declining to simplify; the control group is what showed this
was discriminating (0/40 refusals on words that DO have simpler synonyms)
rather than general laziness. Without that control the 34% would have been
over-read.

**Recommendation as of 2026-09-20: do not switch.** A gain on one pass, a
regression on another, nothing on connective, at 3.7x latency — and with 429s
at 6.6% (see below) slower calls make throughput worse, not better.

### Hetzner rate limits tightened (2026-09)

429 responses as a share of calls: **0.1% (Jul), 0.2% (Aug), 6.6% (Sep)** — a
~40x rise, consistent with the experiment being commercialised. Nothing is
broken (auth, model, structured output and thinking control all verified
2026-09-20) but the headroom is gone: CLAUDE.md's "parallel-3 can brush the
cap" is now "parallel-3 hits it routinely". Probe runs lose calls to 429s, and
because failed calls are EXCLUDED from a probe's denominator, two runs can
report different observation counts — check those before comparing rates.

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
     (`_verify_word_swaps`, prompt `swap_judge`). **OFF by default; set
     `LINT_II_SWAP_JUDGE=1` to enable**, same pattern as `LINT_II_CONNECTIVES`
     — it only ever REMOVES suggestions and its false-alarm rate is measured on
     one corpus, not on tester text. Qwen does NOT share its own generator error — it
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
     **Known weaknesses, measured on two corpora — read before tuning it:**
     - It sits at roughly **2 false alarms per ~40 swaps whatever the wording**,
       and each fix RELOCATES the errors rather than removing them
       (denominalizations → compound splits). Two successive "fixes" turned out
       to be patches on the instances measured; the second corpus caught both.
       Further prompt tuning has low expected value.
     - It has no stable view of some words: it rejects both
       `ambivalent→twijfelachtig` (bad) and `ambivalent→verdeeld` (good).
     - A false alarm on a **single-suggestion item costs the WHOLE item**
       (wordfreq-4 on set 5, compound-6 on set 4).
     - Still untouched by either layer: monumentale→grote,
       conservator→bewaarder, notoire→bekende pass the judge as readily as they
       pass the generator.
     - **Part of this residual is MODEL-capability-bound, not
       engineering-bound** (measured 2026-09-20, see the model-comparison
       section): `gewaande → vermeende` failed 12/12 under every prompt variant
       and is 0/12 on Qwen3.8-27B. So stopping prompt work on the confident
       cases was right, and a future model change may retire part of this item
       for free. `notoire → bekende` is 12/12 on both models, so that one is
       not capability-bound — do not expect a model upgrade to fix it.
     **Net:** clearly positive on set 5 (3 bad removed, 1 good lost), roughly
     break-even on set 4. It is opt-in for that reason. Its most valuable
     behaviour — rejecting rewrites that are ungrammatical ("het volledige
     leidingen") or semantically broken ("het uitleggen van de vrijwilligers")
     — is arguably better served by a deterministic grammar check than by an
     LLM judge.
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
5. ~~**No-op `word_frequency` suggestions slip the filters**~~ / ~~abbreviation
   tokens fire spurious triggers~~ — the TRIGGER half is now FIXED too; the
   LiNT-score half remains open. FIXED for the
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
   **UPGRADED 2026-09-21 — it is worse than "a no-op the guard catches".** Set 1
   clean-1 produced `pas.` → `kaart` ("U kunt boeken lenen met uw kaart"), a
   REAL suggestion on a word that was never difficult, and the no-op guard
   cannot touch it because the text genuinely changed. So the spurious trigger
   reaches the user whenever the model answers with a different word rather
   than echoing the original.
   Confirmed on two sets with two different words, so it is not a `vol.`-specific
   quirk: spaCy keeps both `pas.` and `vol.` as single tokens, and SUBTLEX has
   them at Zipf 2.26 / 2.36 against 5.51 / 5.32 for the bare forms — a ~3-point
   drop that is purely the attached period. Any sentence-final word that spaCy
   treats as an abbreviation is a candidate.
   This also raises the priority of the LiNT-score half: the same ~3-point error
   feeds the frequency metric on every such token.
   **TRIGGER HALF FIXED 2026-09-21**: `_check_word_frequency` now skips a token
   whose period-stripped form is frequent enough, alongside the existing
   word-family guard. Validated over all 500 corpus items: exactly 5 tokens
   suppressed (`pas.` 2.26/5.51, `hand.` 2.77/5.30, `vol.` 2.36/5.32 ×3 — a
   third word, `hand.`, surfaced that no eval run had flagged), 0 legitimate
   abbreviation triggers lost, and neither affected positive loses its other
   suggestion types. Genuinely rare words still trigger, sentence-final ones
   included (spaCy splits the period for those, so the token is already bare).
   **The LiNT-score half is still open and is the harder one**: `WordFeatures.
   word_frequency` continues to score these tokens ~3 Zipf too low, so the
   published metric is affected. Fixing that changes LiNT scores and needs
   validating against the LiNT reference first — deliberately not touched.
6. ~~**Hunspell mangles valid compounds when BOTH forms are unknown to SUBTLEX**~~ — FIXED
   (set 2 clean-8, found 2026-09-11). The Hunspell pass "corrected"
   `banenzwemmen` → `banenzwemmer`, yielding ungrammatical Dutch ("In de
   ochtend is er banenzwemmer") and turning a gerund into a person.
   Mechanism, so it need not be re-derived: `_correction_plausible` has a
   same-stem branch (`common >= max(3, min(len) - 2)`) meant for short
   inflection fixes like word/wordt and loop/loopt. It has **no length ceiling
   and no requirement that either form be a known word**, so two 12-letter
   compounds differing in the final letter match it and return True before the
   frequency check is ever reached. Both forms are absent from SUBTLEX, so the
   frequency rule would have had no signal either.
   This is the failure CLAUDE.md warns about ("not in SUBTLEX/Hunspell does NOT
   mean not a word"), reaching the user through the one pass whose corrections
   are not LLM-generated. Likely fix: cap the same-stem exemption by length, or
   require at least one of the two forms to be in SUBTLEX — but re-verify the
   gate's existing cases (word/wordt, loop/loopt, aparte/apart, te veel,
   terugzwemmen) before changing it, since it is load-bearing for two passes.
   **FIXED 2026-09-21.** Neither suggested fix was the right one; the sharp
   distinction is that a real inflection fix ADDS OR REMOVES A SUFFIX, so one
   form is a PREFIX of the other (word/wordt, loop/loopt, apart/aparte).
   Sharing a long prefix and then DIVERGING is a different word — and
   banenzwemmen → banenzwemmer is a final-letter substitution turning a gerund
   into a person. The branch now requires the prefix relation plus a ≤3-char
   length delta (Dutch inflectional suffixes are short: -t, -e, -en, -de, -te,
   -s). A length ceiling alone would have been fragile, and a known-word
   requirement would have rejected genuine typo fixes on productive compounds
   that SUBTLEX lacks — the trap CLAUDE.md warns about.
   Regression-checked against every spelling correction the engine has ever
   produced across all stored result files: 9 distinct pairs, 7 kept, 2
   rejected — the banenzwemmen bug, and `clandestiene → clandestine`, a SECOND
   hallucination the old rule admitted. That one is worth noting: `clandestiene`
   is valid Dutch (Hunspell, Zipf 2.70) and `clandestine` is not a Dutch word
   at all (absent from Hunspell, Zipf 1.66), so the LLM pass was corrupting
   correct text and shipping it. The strict xfail in `tests/` is now a normal
   assertion, with parametrised cases for both sides of the distinction.
7. **`de`/`het` article disagreement reaches the user** (set 3 shortlist-4,
   found 2026-09-11). The word-frequency pass swapped `gereedschapsset →
   gereedschap`, yielding "**De** gereedschap bevat een hamer..." — it is *het*
   gereedschap. Two independent gaps, both needed for a fix:
   - `_dehet_disagreement` checks only ADJECTIVE inflection (`buitenlandse
     bezit` → `buitenlands bezit`). Article/noun agreement is a different
     phenomenon and was never in scope, so it returns None here.
   - The bundled word-frequency path (`_build_wordfreq_suggestion`) does not
     call that guard AT ALL. It applies the no-op, frequency-band,
     conjunction, URL and misspelling checks; de/het is applied only in the
     connective path, the rewrite backstop and the per-trigger path.
   So extending the guard without also wiring it into the bundled path would
   fix nothing. Note a swap changing a noun's gender is exactly what a
   word-frequency replacement does, which makes that path the one most likely
   to produce this error — and the guard's docstring warns the reverse
   direction (uninflected `houten`, `gouden`) must stay allowed, so widening
   it needs the same care as item 6.
   **Attempted and REVERTED on 2026-09-11 — read this before trying again.**
   The obvious fix is to compare the article against spaCy's gender tag on the
   noun. That cannot work: **spaCy's in-context gender tag is derived from the
   article itself.** `De gereedschap` tags gereedschap `zijd`; `Het
   gereedschap` tags the same word `onz`. The tag always agrees with the
   article, so it carries no independent signal.
   The next idea — probe the noun in a determiner-free frame ("Wij zagen X
   daar.") — looked sound on a 36-word de/het battery (0 errors) and then
   produced **12 false positives on the 500 corpus items**, flagging correct
   Dutch like "Het wijkcentrum", "Het orgel" and "De hoofdprijs". Without a
   determiner the tagger simply guesses, defaulting to `zijd` for anything it
   does not know well — which is most compounds, i.e. exactly this corpus's
   vocabulary. The battery was unrepresentative: common short nouns are the
   case spaCy gets right.
   **Dutch gender is lexical and cannot be derived from the parser.** A real
   fix needs a de/het GENDER LEXICON (Hunspell is no help — its Y-flags encode
   diminutive formation, not gender; NOUN_DATA has no gender field). Treat
   that data dependency, and its licensing, as the actual task. Until then the
   500-item sweep is the acceptance test: a candidate guard must fire on "De
   gereedschap bevat..." and hit 0 of the 500.
