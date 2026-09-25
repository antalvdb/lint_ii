// The popup's sentence-score line: editor.getSentenceScoreChange and the
// sentenceScoreText formatter. Run with: node --test tests/js/
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { EditorController } from '../../src/visualizer/core/editor.js'
import { sentenceScoreText } from '../../src/visualizer/core/suggestion-popup.js'

const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/two_sentences.json', import.meta.url)))
const fresh = () => new EditorController(structuredClone(FIXTURE))
const EDIT = 'De fiets ging vorige week kapot, want de ketting was stuk.'
const FULL_METRICS = FIXTURE.suggestions.suggestions[0].variants[1].new_sentence_metrics

test('formatter: an easier sentence, with a level change', () => {
    const line = sentenceScoreText({
        before: { score: 47.25, level: 3 }, after: { score: 38.4, level: 2 }, scoring: false,
    })
    assert.equal(line.text, 'Deze zin: 47,3 → 38,4 (makkelijker) · niveau 3 → 2')
    assert.equal(line.trend, 'easier')
})

test('formatter: a harder sentence at the same level', () => {
    const line = sentenceScoreText({
        before: { score: 20, level: 1 }, after: { score: 22.5, level: 1 }, scoring: false,
    })
    assert.equal(line.text, 'Deze zin: 20,0 → 22,5 (moeilijker)')
    assert.equal(line.trend, 'harder')
})

test('formatter: below 0.05 reads as no noticeable change', () => {
    const line = sentenceScoreText({
        before: { score: 30.01, level: 1 }, after: { score: 30.04, level: 1 }, scoring: false,
    })
    assert.equal(line.text, 'Deze zin: 30,0, geen merkbare verandering')
    assert.equal(line.trend, 'same')
})

test('formatter: scoring in progress, and no score at all', () => {
    assert.equal(sentenceScoreText({ before: null, after: null, scoring: true }).trend, 'scoring')
    assert.equal(sentenceScoreText({ before: { score: null }, after: { score: 3 }, scoring: false }), null)
})

test('editor: nothing accepted means before equals after', () => {
    const { before, after, scoring } = fresh().getSentenceScoreChange(0)
    assert.equal(scoring, false)
    assert.equal(before.score, after.score)
})

test('editor: an edit is "scoring" until its metrics arrive, then it moves', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    ed.accept('r0')
    let change = ed.getSentenceScoreChange(0)
    assert.equal(change.scoring, true)
    assert.equal(change.after.score, change.before.score)

    ed.setTextMetrics(EDIT, FULL_METRICS)
    change = ed.getSentenceScoreChange(0)
    assert.equal(change.scoring, false)
    assert.notEqual(change.after.score, change.before.score)
})

test('editor: a failed edit stops "scoring" and keeps the original score', () => {
    const ed = fresh()
    ed.setEditedText('r0', EDIT)
    ed.accept('r0')
    ed.markMetricsFailed(EDIT)
    const change = ed.getSentenceScoreChange(0)
    assert.equal(change.scoring, false)
    assert.equal(change.after.score, change.before.score)
})

test('editor: an accepted model option is never "scoring"', () => {
    const ed = fresh()
    ed.accept('r0')
    const change = ed.getSentenceScoreChange(0)
    assert.equal(change.scoring, false)
    assert.notEqual(change.after.score, change.before.score)
})
