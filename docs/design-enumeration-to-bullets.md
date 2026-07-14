# Design: enumeration → bullet-list suggestion

Status: proposal · Author: LiNT-II dev · Context: Henk Pander Maat test (zin 2, 3)

## Motivation

Henk suggested that long in-sentence enumerations — several coordinated items
crammed into one running sentence — would be clearer presented as a bulleted
list. Example (zin 3):

> De nieuwe maatregelen hebben betrekking op de veiligheid van door werknemers
> of aannemers uitgevoerde werkzaamheden, de veiligheid van producten, het
> voorkomen van lucht-, water- en bodemverontreiniging en het zo veel mogelijk
> voorkomen van schade als gevolg van onvoorziene gebeurtenissen.

Desired result:

> De nieuwe maatregelen hebben betrekking op:
> - de veiligheid van door werknemers of aannemers uitgevoerde werkzaamheden;
> - de veiligheid van producten;
> - het voorkomen van lucht-, water- en bodemverontreiniging;
> - het zo veel mogelijk voorkomen van schade door onvoorziene gebeurtenissen.

This is fundamentally different from every existing suggestion: the accepted
output is **structural** (one sentence → a lead-in + a multi-item list block),
not an inline word-level rewrite. That is why it is a feature, not a tweak — the
whole accept/merge/render/export path currently assumes single-sentence prose.

## Scope of the change

| Layer | Change | Size |
|---|---|---|
| Detection | new `ENUMERATION` trigger | M |
| Generation | new `enumeration` prompt + structured parse | M |
| Data model | list payload on `Suggestion` + serialize | S |
| Frontend render | highlight + list preview in popup | M |
| Frontend accept | replace sentence block with lead-in + list block | **L (the hard part)** |
| Score / export | metrics per resulting line; emit list markdown | M |
| Exclusivity | add to `SENTENCE_SCOPED_TYPES` | S |

## 1. Detection (backend, `suggestions.py`)

Add `SuggestionType.ENUMERATION` and `_check_enumeration(sent_analysis, ...)`.

Heuristic on the spaCy parse — require ALL of, to keep precision high:
- A coordination with **≥ 3 conjuncts** sharing one head (walk `conj` chains;
  reuse the `heads`/`conjuncts` logic already in `word_features.py`), **or** a
  run of **≥ 3 comma-separated constituents** closed by a final `en`/`of` item.
- The enumeration spans a meaningful share of the sentence (e.g. sentence length
  ≥ ~20 words, or the coordinated span ≥ ~12 words) — so we skip short, natural
  lists like "koffie, thee en water".
- The items are phrase-level (NPs/PPs), not clause-level with finite verbs
  (splitting full clauses is a different problem, already covered by max_sdl /
  subordinate_clause).

Emit one trigger carrying the detected item spans (char offsets) as context, so
the prompt can be anchored and the parse doesn't have to be redone by the model.

Tunable thresholds live in `DEFAULT_THRESHOLDS` (`enumeration_min_items`,
`enumeration_min_span_words`). Expect to iterate these against real texts.

## 2. Generation (backend, `prompts.py`)

New `enumeration` template. It must NOT invent content — same conservative
guardrails as the rest — only re-present the existing items. Ask for structured
output so we don't have to re-parse prose:

```
Herschrijf de opsomming in deze zin als een puntsgewijze lijst. Verzin geen
nieuwe items en verander de betekenis niet.

Zin: "{sentence}"

Geef je antwoord in dit formaat:
INLEIDING: [de aanloopzin tot en met de dubbele punt]
ITEM: [eerste item]
ITEM: [tweede item]
ITEM: [...]
UITLEG: [hoogstens tien woorden]
```

Parse with the existing `parse_block_response` style (repeated `ITEM:` lines).
Validation backstops: item count ≥ 3, every item's content words must appear in
the original (a bag-of-words containment check, mirroring `_alters_url` /
`_introduces_misspelling` philosophy), else discard.

## 3. Data model (backend `Suggestion` + `as_dict`)

The suggestion's output is a list, so a single `suggested_text` string is not
enough. Add:

```python
list_intro: str | None = None       # lead-in, ends with ":"
list_items: list[str] = field(default_factory=list)
```

Serialize both when present. Keep `suggested_text` populated with a plain-text
rendering ("intro\n- a\n- b\n- c") as a fallback for the copy/export path and
for any code that only knows `suggested_text`.

`new_sentence_metrics`: run the analysis on the flattened rendering so the score
recompute reflects the shorter resulting "sentences"; the editor already handles
one original sentence expanding to several results (`getEffectiveSentenceLevels`
returns one entry per resulting unit).

## 4. Frontend rendering & accept (the hard part, `editor.js` + visualizer)

The inline word-diff model cannot represent "one sentence → a block of bullets".
Handle enumeration suggestions on a **separate, block-level path**:

- **Highlight**: mark the whole sentence as the clickable span (not per-word);
  `_buildClusters` gives it its own cluster keyed on the sentence.
- **Popup**: render the proposed list (intro + `<ul>`) instead of an inline
  diff, with the usual accept/ignore buttons.
- **Accept → mutate the block model.** Today `getEditedText()` walks `blocks`
  (`sentence`, `list_item`, `heading`, `quote`, `blank`). On accepting an
  enumeration suggestion, replace the single `{type:"sentence", sentence_index}`
  block with:
  - a `sentence` block for the intro, and
  - N `list_item` blocks (`ordered:false`) for the items.
  Render that region as a real indented `<ul>` in the sentences view, and mark it
  accepted (green). `getEditedText()` then emits `- item` lines for free, since
  the list_item branch already exists.
- **Reset/ignore** restores the original single sentence block.

This needs a small amount of new state (an override list of blocks for accepted
enumerations) because `blocks` is currently derived once from the backend. The
cleanest approach: keep an `_blockOverrides` map (sentence_index → replacement
blocks) that `getEditedText()` and the renderer consult, so the original
`blocks`/`sentences` stay immutable and reset is trivial.

## 5. Exclusivity

An enumeration rewrite reformulates the whole sentence, so it must be mutually
exclusive with any other suggestion on that sentence. Add `'enumeration'` to
`SENTENCE_SCOPED_TYPES` in `editor.js` (the rule shipped for the multi-accept
fix already generalises to this).

## 6. Popup label

Reuse the `_typeLabel` map: `'enumeration': 'Opsomming als lijst'`. If an
enumeration is ever consolidated with other issues, it appears in
`component_types` like the others.

## 7. Edge cases / risks

- **Items containing commas** ("lucht-, water- en bodemverontreiniging") — rely
  on the model's segmentation (structured `ITEM:` output) rather than splitting
  on commas ourselves.
- **Trailing clause after the enumeration** — the intro/last item must absorb it;
  the containment check guards against dropped content.
- **False positives** — coordinated short lists; mitigated by the span/length
  gate. Log what was flagged so thresholds can be tuned.
- **Mobile rendering** of the list block; the export markdown round-tripping back
  through `/convert`.
- **Nested/again-analysed** — after accept, the score uses the flattened metrics;
  we do not re-run LiNT on the literal bullet markup.

## 8. Suggested phasing

1. **Backend detect + generate + serialize** behind a flag; verify on real texts
   in `py311`, tune thresholds, confirm 0 content-invention via the containment
   backstop. (No UI yet — inspect the JSON.)
2. **Frontend block-level accept** with the `_blockOverrides` model + list
   rendering + export. This is the bulk of the work and where to spend review.
3. Polish: popup preview styling, exclusivity, mobile, score animation.

## 9. Decisions needed before building

- Item terminator style: semicolons + final period, or plain items? (affects the
  prompt and export.)
- Ordered vs unordered — always bullets, or numbered when the source implies
  sequence? (default: unordered.)
- Minimum item count / span thresholds — start at 3 items and ~20-word sentence,
  tune on a corpus.
- Do we want this at all given the added UI surface, or is it lower priority than
  other backlog items? (It is the largest Tier-3 item by far.)
