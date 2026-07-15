# Design: connective-words (coherence) pass

Status: proposal · Author: LiNT-II dev · Context: Henk Pander Maat email (~2026-07-13); earlier zin 4

## Motivation

LiNT-II never suggests inserting a **connective** (verbindingswoord —
*doordat, want, daarom, hierdoor, immers, namelijk, toch, bovendien*) where one
is missing, because every stage works inside a single sentence and so never sees
the *relation between* sentences. Henk flagged this as the current lacuna, and
noted it is the opposite of the "lekentheorie" risk: coherence marking is one of
the better **evidence-based** comprehensibility interventions (Kleijn / LiNT
research), not folk wisdom. Example (test zin 4):

> De productie moest enkele uren worden onderbroken. Een van de machines was
> kapotgegaan.

clearer as:

> De productie moest enkele uren worden onderbroken, **doordat** een van de
> machines kapot was gegaan.   *(merge)*

or, keeping two sentences:

> De productie moest enkele uren worden onderbroken. **Een machine was namelijk
> kapotgegaan.**   *(sentence-initial)*

## Key insight — this is cheaper than it looks

The relation is cross-sentence, but the *edit* need not be. Split into two cases:

- **Case A — sentence-initial connective, sentences stay separate**
  ("Daarom …", "Hierdoor …", "… was namelijk …"). This edits only the SECOND
  sentence, so it is representable as an ordinary single-sentence rewrite of
  sentence N+1 — **the existing accept/merge/score path already handles it.**
- **Case B — merge two sentences with a connective** ("…, want …"). This turns
  two sentences into one: a cross-sentence structural change the per-sentence
  accept model can't represent (same block-surgery problem as
  [enumeration→bullets](design-enumeration-to-bullets.md)).

**Scope this feature to Case A.** It covers a large, useful class and reuses the
whole frontend. Case B is a later extension that shares the `_blockOverrides`
work from the enumeration design.

So the novelty is concentrated in **detection** (pair/paragraph-aware) and
**prompting** (needs the previous sentence as context) — not in accept UI.

## Architectural novelty — the first LLM-*detected* trigger

Every existing trigger is detected deterministically (frequency, SDL, passive…)
and only the *rewrite* is LLM-driven. A missing connective cannot be detected
from surface features — the relation is semantic — so **detection itself is
LLM-driven** here. That is the real new thing, and it drives the cost model
(§Cost) and the conservative-bias requirement (§Validation).

## 1. Detection (backend, `suggestions.py`)

Add `SuggestionType.CONNECTIVE`. Work **per paragraph**, not per sentence — the
preprocessor already groups sentences into blocks; a "paragraph" = a maximal run
of consecutive `sentence` blocks (never cross a heading / blank / list / quote).

Cheap deterministic pre-filter to bound LLM work — only consider a boundary
(sentence N → N+1) as a *candidate* when:
- both are declarative main clauses (not a question/imperative; check the parse);
- sentence N+1 does **not** already start with a discourse marker (match a
  connective lexicon, case-insensitive, first token/first two tokens);
- neither sentence is very short/formulaic (headings mis-parsed as sentences,
  greetings) — length ≥ ~4 content words.

Then a **single LLM call per paragraph** decides which candidate boundaries
genuinely benefit and with which connective (see §2). No per-pair calls.

## 2. Generation (backend, `prompts.py`)

New `connective` template, given the whole paragraph and the candidate
boundaries. It must be conservative: only mark a boundary when the relation is
**already implied by the existing content**, prefer to abstain when unsure, and
never invent facts. Structured output, one block per accepted boundary:

```
Hieronder staat een alinea, met genummerde zinnen. Geef alleen aan tussen welke
opeenvolgende zinnen een expliciet verbindingswoord de samenhang duidelijker
maakt — uitsluitend als die relatie al besloten ligt in de tekst. Verzin geen
relaties en voeg geen inhoud toe. Twijfel je, laat de overgang dan ongemoeid.

Alinea:
{numbered_sentences}

Per verbetering één blok:
---
NA_ZIN: [nummer van de eerste zin]
RELATIE: [oorzaak | gevolg | tegenstelling | toelichting | opsomming]
HERSCHRIJVING: [de TWEEDE zin, herschreven met het verbindingswoord vooraan en
               correcte woordvolgorde; verder ongewijzigd]
UITLEG: [hoogstens tien woorden]
---
Geen verbeteringen? Antwoord met GEEN.
```

Parse with the existing `parse_block_response` machinery. Each block yields one
suggestion attached to sentence **N+1** (the rewritten one).

## 3. Data model

Reuse `Suggestion` as-is:
- `type = CONNECTIVE`, `sentence_index = N+1`,
- `original_text` = sentence N+1, `suggested_text` = rewritten N+1,
- `explanation` = UITLEG, `new_sentence_metrics` from the rewrite.

Optional: a `context_sentence_index = N` field (serialized) so the popup can show
the preceding sentence — the user needs it to judge the relation. Small add to
`as_dict`, mirroring `component_types`.

## 4. Validation backstops (deterministic, cheap, fail-open)

Same philosophy as `_alters_url` / `_introduces_misspelling` / `_dehet_disagreement`:
- **Connective lexicon guard**: the rewrite must add exactly a connective from an
  allowed list at/near the front; reject if the model prepended arbitrary text.
- **Containment guard**: every content word of the rewritten N+1 (minus the added
  connective) must appear in the original N+1 — no invented content, no smuggled
  facts from elsewhere.
- **Reuse** `_introduces_misspelling` and `_dehet_disagreement` on the rewrite.
- Known gap: none of the current backstops check **V2 word order** (a
  sentence-initial adverbial connective forces inversion: "Daarom lag er…", not
  "Daarom er lag…"). Either trust the model + spot-check, or add a light
  finite-verb-position check via spaCy (see §Risks).

## 5. Frontend

Almost entirely reuses the existing path, because the edit is a single-sentence
rewrite of N+1:
- Highlight sentence N+1 (its own cluster). Optionally emphasise the
  sentence-initial insertion point.
- **Popup shows the preceding sentence as read-only context** ("na: …<zin N>")
  above the usual Origineel/Suggestie/Uitleg, so the user can judge the relation.
- A **"beoordeel zelf" note** (reuse the passive-note pattern) — relation
  inference is interpretive, so nudge rather than assert.
- **Exclusivity**: add `'connective'` to `SENTENCE_SCOPED_TYPES` in `editor.js`,
  so it is mutually exclusive with any other edit on sentence N+1.
- Label: `_typeLabel` → `'connective': 'Samenhang'` (or 'Verbindingswoord').
- Score/export: nothing new — it is a normal sentence rewrite.

## 6. Cost

One LLM call per paragraph (not per boundary, not per sentence), gated by the
deterministic pre-filter, and only for paragraphs with ≥1 candidate boundary. On
the box (Mistral) this is affordable; it runs alongside the existing per-sentence
jobs in the same batch/round-robin. Log candidates considered vs. accepted to
tune the pre-filter.

## 7. Risks

- **Inventing a relation** (asserting causality that isn't there) — the central
  risk. Mitigate with the conservative prompt (abstain-on-doubt), the containment
  guard, and the "beoordeel zelf" framing. Consider defaulting to weaker
  connectives (*ook, verder*) over strong causal ones when the model is unsure.
- **V2 inversion** correctness (see §4).
- **Over-marking**: a paragraph peppered with connectives reads worse. Cap
  suggestions per paragraph (e.g. ≤ ~⅓ of boundaries) and prefer the strongest
  single case.
- **Pronoun/reference shifts** when a connective changes the framing — the
  containment guard catches added content but not subtle meaning drift.

## 8. Phasing

1. **Backend detect + generate + serialize** behind a flag; run on real
   paragraphs in `py311` (deterministic pre-filter is testable offline; the LLM
   step needs the box). Tune the pre-filter and measure false-relation rate on a
   sample before exposing.
2. **Frontend**: context-sentence in the popup + "beoordeel zelf" note +
   exclusivity + label. Small, since accept/score/export are unchanged.
3. Later / optional: **Case B** (merge with *want/omdat/doordat*), which needs the
   `_blockOverrides` block-surgery from the enumeration design (two sentences → one).

## 9. Open decisions

- Case A only (recommended) vs. also Case B merges from day one?
- Connective lexicon + which relations to attempt (start narrow: gevolg
  *daarom/hierdoor*, toelichting *namelijk/immers*; add contrast/cause later).
- Abstain threshold / max suggestions per paragraph.
- Add a V2 word-order backstop now, or trust the model and spot-check on the box?
- Priority vs. the other backlog items (long-compound prompt, enumeration build).
