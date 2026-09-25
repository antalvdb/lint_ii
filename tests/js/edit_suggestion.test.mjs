// Editing a suggestion before accepting it (EditorController).
// Run with: node --test tests/js/
// The fixture is real backend output (ReadabilityAnalysis.as_dict() plus
// suggestions scored by _analyze_suggested_text) for two sentences: r0 is a
// two-variant rewrite of sentence 0, r1 a single rewrite of sentence 1, w1 a
// word-level swap, c1 a connective merging the two sentences.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { EditorController } from '../../src/visualizer/core/editor.js'

const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/two_sentences.json', import.meta.url)))
const fresh = () => new EditorController(structuredClone(FIXTURE))
const EDIT = 'De fiets ging vorige week kapot, want de ketting was defect.'
const variantKeys = s => s.variants.map(v => v.key)

test('only whole-sentence prose rewrites are editable', () => {
    const ed = fresh()
    assert.equal(EditorController.isEditable(ed.getSuggestion('r0')), true)
    assert.equal(EditorController.isEditable(ed.getSuggestion('r1')), true)
    assert.equal(EditorController.isEditable(ed.getSuggestion('w1')), false)
    assert.equal(EditorController.isEditable(ed.getSuggestion('c1')), false)
    assert.equal(ed.setEditedText('w1', EDIT), false)
})

test('an edit becomes the chosen "edited" variant, after the model variants', () => {
    const ed = fresh()
    assert.equal(ed.setEditedText('r0', EDIT), true)
    const s = ed.getSuggestion('r0')
    assert.deepEqual(variantKeys(s), ['conservative', 'full', 'edited'])
    assert.equal(ed.getChosenVariantKey('r0'), 'edited')
    assert.equal(s.suggested_text, EDIT)
    assert.equal(s.new_sentence_metrics, null)
    assert.equal(s.edited, true)
})

test('a single rewrite gains its model text as a variant to choose between', () => {
    const ed = fresh()
    const modelText = ed.getSuggestion('r1').suggested_text
    const modelMetrics = ed.getSuggestion('r1').new_sentence_metrics
    ed.setEditedText('r1', 'Hij was daardoor veel te laat op zijn werk.')
    const s = ed.getSuggestion('r1')
    assert.equal(EditorController.hasVariants(s), true)
    assert.deepEqual(variantKeys(s), ['suggestion', 'edited'])
    assert.equal(s.variants[0].suggested_text, modelText)
    assert.deepEqual(s.variants[0].new_sentence_metrics, modelMetrics)
})

test('empty text and text equal to the current sentence change nothing', () => {
    const ed = fresh()
    const s = ed.getSuggestion('r0')
    const before = JSON.stringify(s.variants)
    assert.equal(ed.setEditedText('r0', '   '), false)
    assert.equal(ed.setEditedText('r0', `  ${s.original_text}  `), false)
    assert.equal(JSON.stringify(s.variants), before)
    assert.equal(ed.getChosenVariantKey('r0'), 'conservative')
})

test('editing again replaces the edit; whitespace is normalised', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    ed.setEditedText('r0', '  De fiets   ging kapot.\n')
    const s = ed.getSuggestion('r0')
    assert.equal(s.variants.filter(v => v.key === 'edited').length, 1)
    assert.equal(s.suggested_text, 'De fiets ging kapot.')
})

test('an unscored edit is requested once, then scored and re-announced', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    assert.deepEqual(ed.textsNeedingMetrics(), [EDIT])

    const announced = []
    ed.addEventListener('editor-change', e => announced.push(e.detail.suggestionId))
    const metrics = FIXTURE.suggestions.suggestions[0].variants[1].new_sentence_metrics
    ed.setTextMetrics(EDIT, metrics)

    assert.deepEqual(ed.textsNeedingMetrics(), [])
    assert.deepEqual(ed.getSuggestion('r0').new_sentence_metrics, metrics)
    assert.deepEqual(announced, ['r0'])
})

test('a failed text is not re-requested until the user edits again', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    ed.markMetricsFailed(EDIT)
    assert.deepEqual(ed.textsNeedingMetrics(), [])
    assert.equal(ed.editedMetricsFailed('r0'), true)
    ed.setEditedText('r0', EDIT)
    assert.deepEqual(ed.textsNeedingMetrics(), [EDIT])
    assert.equal(ed.editedMetricsFailed('r0'), false)
})

test('an accepted edit scores as the original until its metrics arrive', () => {
    const ed = fresh()
    const baseline = ed.computeUpdatedScore().score
    ed.setEditedText('r0', EDIT)
    ed.accept('r0')
    assert.equal(ed.computeUpdatedScore().score, baseline)

    ed.setTextMetrics(EDIT, FIXTURE.suggestions.suggestions[0].variants[1].new_sentence_metrics)
    assert.notEqual(ed.computeUpdatedScore().score, baseline)
})

test('undo keeps the edit as a choosable variant', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    ed.accept('r0')
    ed.reset('r0')
    assert.equal(ed.getState('r0'), 'pending')
    assert.deepEqual(variantKeys(ed.getSuggestion('r0')), ['conservative', 'full', 'edited'])
    ed.chooseVariant('r0', 'full')
    assert.equal(ed.getSuggestion('r0').suggested_text, FIXTURE.suggestions.suggestions[0].suggested_text)
})

test('the exported text carries an accepted edit', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    ed.accept('r0')
    assert.ok(ed.getEditedText().includes(EDIT))
})

// Composed merges: the backend's composed_metrics were computed from the
// rewrite's PRIMARY text (the full variant), so they hold only while the
// rewrite shows that text.

test('merge with the primary rewrite uses the precomputed composed metrics', () => {
    const ed = fresh()
    ed.chooseVariant('r0', 'full')
    ed.accept('r0')
    ed.accept('c1')
    assert.deepEqual(ed.textsNeedingMetrics(), [])
    assert.deepEqual(ed._getEffectiveMetrics(0), FIXTURE.suggestions.suggestions[3].composed_metrics.r0)
})

test('merge with another variant fetches metrics for the actual composed text', () => {
    const ed = fresh()
    ed.accept('r0')  // default variant: conservative, not the primary
    ed.accept('c1')
    const composed = ed._composedMergeText(ed.getSuggestion('c1'))
    assert.ok(composed.startsWith('De fiets ging vorige week kapot door een defect aan de ketting'))
    assert.deepEqual(ed.textsNeedingMetrics(), [composed])
    // Until then: the merge's own metrics, never the stale full-variant ones.
    assert.deepEqual(ed._getEffectiveMetrics(0), ed.getSuggestion('c1').new_sentence_metrics)

    const exact = { ...ed.getSuggestion('c1').new_sentence_metrics, word_freq_count: 99 }
    ed.setTextMetrics(composed, exact)
    assert.deepEqual(ed._getEffectiveMetrics(0), exact)
})

test('merge with an edited rewrite composes and scores the edit', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    ed.accept('r0')
    ed.accept('c1')
    const composed = ed._composedMergeText(ed.getSuggestion('c1'))
    assert.ok(composed.startsWith(EDIT.replace(/\.$/, '')))
    assert.ok(ed.textsNeedingMetrics().includes(composed))
})
