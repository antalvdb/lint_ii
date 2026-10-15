// Two accepted word-level edits in one sentence co-apply, but each one's
// precomputed metrics describe the sentence with only that edit. The editor
// asks /sentence-metrics for the composed sentence and scores that.
// Run with: node --test tests/js/
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { EditorController } from '../../src/visualizer/core/editor.js'

const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/two_sentences.json', import.meta.url)))
const BASE = FIXTURE.suggestions.suggestions[0].new_sentence_metrics
const metrics = freqSum => ({
    ...BASE, n_sentences: 1, sdl_values: [BASE.sdl_values[0]],
    cwpc_values: [BASE.cwpc_values[0]], word_freq_sum: freqSum,
})
const ONE = metrics(BASE.word_freq_sum + 1)
const BOTH = metrics(BASE.word_freq_sum + 2)
const COMPOSED = 'De fiets ging vorige week plots kapot door een fout aan de ketting.'

function twoSwaps() {
    const data = structuredClone(FIXTURE)
    data.suggestions.suggestions = [
        { id: 'a', type: 'word_frequency', sentence_index: 0, word: 'onverwacht', word_index: 5,
          replacement_word: 'plots', original_text: 'onverwacht',
          suggested_text: 'De fiets ging vorige week plots kapot door een defect aan de ketting.',
          explanation: '', new_sentence_metrics: ONE },
        { id: 'b', type: 'word_frequency', sentence_index: 0, word: 'defect', word_index: 9,
          replacement_word: 'fout', original_text: 'defect',
          suggested_text: 'De fiets ging vorige week onverwacht kapot door een fout aan de ketting.',
          explanation: '', new_sentence_metrics: ONE },
    ]
    return new EditorController(data)
}

test('one accepted swap scores from its own metrics, without a request', () => {
    const ed = twoSwaps()
    ed.accept('a')
    assert.deepEqual(ed.textsNeedingMetrics(), [])
    assert.equal(ed.getSentenceScoreChange(0).scoring, false)
})

test('a second swap in the sentence asks for the composed text', () => {
    const ed = twoSwaps()
    ed.accept('a')
    ed.accept('b')
    assert.equal(ed.getState('a'), 'accepted')
    assert.equal(ed.getState('b'), 'accepted')
    assert.deepEqual(ed.textsNeedingMetrics(), [COMPOSED])
    assert.equal(ed.getSentenceScoreChange(0).scoring, true)
})

test('the composed metrics move the score past the single-swap score', () => {
    const ed = twoSwaps()
    ed.accept('a')
    const single = ed.computeUpdatedScore().score
    ed.accept('b')
    const announced = []
    ed.addEventListener('editor-change', e => announced.push(e.detail.suggestionId))
    ed.setTextMetrics(COMPOSED, BOTH)
    assert.ok(ed.computeUpdatedScore().score < single)
    assert.equal(ed.getSentenceScoreChange(0).scoring, false)
    assert.deepEqual(ed.textsNeedingMetrics(), [])
    assert.deepEqual(announced.sort(), ['a', 'b'])
})

test('a failed request is reported and not retried', () => {
    const ed = twoSwaps()
    ed.accept('a')
    ed.accept('b')
    ed.markMetricsFailed(COMPOSED, 'time-out')
    const change = ed.getSentenceScoreChange(0)
    assert.equal(change.failed, true)
    assert.equal(change.reason, 'time-out')
    assert.deepEqual(ed.textsNeedingMetrics(), [])
})

test('undoing one swap returns to its own precomputed metrics', () => {
    const ed = twoSwaps()
    ed.accept('a')
    const single = ed.computeUpdatedScore().score
    ed.accept('b')
    ed.setTextMetrics(COMPOSED, BOTH)
    ed.reset('b')
    assert.equal(ed.computeUpdatedScore().score, single)
    assert.deepEqual(ed.textsNeedingMetrics(), [])
})
