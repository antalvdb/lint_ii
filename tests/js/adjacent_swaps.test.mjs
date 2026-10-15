// Adjacent accepted edits must compose in reading order. Study text E,
// sentence 3, swapped beoogde→bedoelde and toepassing→gebruik and came out
// as "De door u gebruik bedoelde ...". Run with: node --test tests/js/
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { EditorController } from '../../src/visualizer/core/editor.js'
import { rightToLeft } from '../../src/visualizer/core/word-diff.js'

const WORDS = ['De', 'door', 'u', 'beoogde', 'toepassing', 'van', 'gerecycled', 'polypropyleen', 'is', 'toegestaan', '.']
const ORIGINAL = 'De door u beoogde toepassing van gerecycled polypropyleen is toegestaan.'

function editor(suggestions) {
    const word_features = WORDS.map(text => text === '.'
        ? { text, pos: 'PUNCT' }
        : { text, pos: 'NOUN', ...(text === 'toegestaan' ? { punctuation: { trailing: '.' } } : {}) })
    const data = {
        sentences: [{ text: ORIGINAL, word_features }],
        suggestions: { suggestions },
    }
    return new EditorController(data)
}
const swap = (id, wordIndex, word, replacement) => ({
    id, type: 'word_frequency', sentence_index: 0, word, word_index: wordIndex,
    replacement_word: replacement, original_text: word, explanation: '',
    suggested_text: ORIGINAL.replace(word, replacement),
})

test('adjacent swaps compose in reading order, whichever is accepted first', () => {
    for (const order of [['a', 'b'], ['b', 'a']]) {
        const ed = editor([swap('a', 3, 'beoogde', 'bedoelde'), swap('b', 4, 'toepassing', 'gebruik')])
        for (const id of order) ed.accept(id)
        assert.equal(ed.getEditedText().trim(),
            'De door u bedoelde gebruik van gerecycled polypropyleen is toegestaan.')
    }
})

test('a replacement goes in front of an insertion right after it', () => {
    const ed = editor([
        swap('a', 3, 'beoogde', 'bedoelde'),
        { id: 'c', type: 'spelling', sentence_index: 0, word: 'toepassing', word_index: 4,
          original_text: 'toepassing', explanation: '',
          suggested_text: 'De door u beoogde nieuwe toepassing van gerecycled polypropyleen is toegestaan.' },
    ])
    ed.accept('c')
    ed.accept('a')
    assert.equal(ed.getEditedText().trim(),
        'De door u bedoelde nieuwe toepassing van gerecycled polypropyleen is toegestaan.')
})

test('rightToLeft: an insertion sorts before a replacement at the same point', () => {
    const replace = { insertBeforeIdx: 4, origIndices: [3] }
    const insert = { insertBeforeIdx: 4, origIndices: [] }
    assert.deepEqual([replace, insert].sort(rightToLeft), [insert, replace])
})
