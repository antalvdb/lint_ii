// Identical-swap groups (group_ids from the backend): one action accepts or
// ignores every still-open occurrence; each stays individually overridable.
// Run with: node --test tests/js/
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { EditorController } from '../../src/visualizer/core/editor.js'

const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/two_sentences.json', import.meta.url)))

function withGroup() {
    const data = structuredClone(FIXTURE)
    const ids = ['w0', 'w1']
    data.suggestions.suggestions = [
        { id: 'w0', type: 'word_frequency', sentence_index: 0, word: 'fiets', word_index: 1,
          replacement_word: 'rijwiel', original_text: 'fiets', suggested_text: 'rijwiel',
          explanation: '', group_ids: ids },
        { id: 'w1', type: 'word_frequency', sentence_index: 1, word: 'werk', word_index: 8,
          replacement_word: 'baan', original_text: 'werk', suggested_text: 'baan',
          explanation: '', group_ids: ids },
    ]
    return new EditorController(data)
}

test('acceptGroup accepts every pending occurrence', () => {
    const ed = withGroup()
    assert.deepEqual(ed.groupPending('w0'), ['w0', 'w1'])
    ed.acceptGroup('w1')
    assert.equal(ed.getState('w0'), 'accepted')
    assert.equal(ed.getState('w1'), 'accepted')
    assert.deepEqual(ed.groupPending('w0'), [])
})

test('an occurrence ignored singly is not overridden by acceptGroup', () => {
    const ed = withGroup()
    ed.ignore('w1')
    ed.acceptGroup('w0')
    assert.equal(ed.getState('w0'), 'accepted')
    assert.equal(ed.getState('w1'), 'ignored')
})

test('ignoreGroup, then one occurrence reset and accepted on its own', () => {
    const ed = withGroup()
    ed.ignoreGroup('w0')
    assert.equal(ed.getState('w0'), 'ignored')
    assert.equal(ed.getState('w1'), 'ignored')
    ed.reset('w1')
    ed.accept('w1')
    assert.equal(ed.getState('w0'), 'ignored')
    assert.equal(ed.getState('w1'), 'accepted')
})

test('an ungrouped suggestion has no group', () => {
    const ed = new EditorController(structuredClone(FIXTURE))
    assert.deepEqual(ed.groupPending('r0'), [])
})
