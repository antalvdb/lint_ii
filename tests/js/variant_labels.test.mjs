// Run with: node --test tests/js/
// Zero dependencies: node:test and node:assert are built in.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { variantLabels } from '../../src/visualizer/core/suggestion-popup.js'

const v = (key, n) => ({
    key,
    suggested_text: key,
    new_sentence_metrics: n === undefined ? {} : { n_sentences: n },
})

const cases = [
    ['full variant named by its count',
        [v('conservative', 1), v('intermediate', 2), v('full', 3)],
        ['Eén zin, niet gesplitst', 'Twee zinnen', 'Drie zinnen']],
    ['intermediate collapsed into a two-sentence full',
        [v('conservative', 1), v('full', 2)],
        ['Eén zin, niet gesplitst', 'Twee zinnen']],
    ['conservative dropped',
        [v('intermediate', 2), v('full', 3)],
        ['Twee zinnen', 'Drie zinnen']],
    ['same count as the intermediate is marked as a further rewrite',
        [v('intermediate', 2), v('full', 2)],
        ['Twee zinnen', 'Twee zinnen, verder herschreven']],
    ['one-sentence full beside the conservative is marked too',
        [v('conservative', 1), v('full', 1)],
        ['Eén zin, niet gesplitst', 'Eén zin, verder herschreven']],
    ['results without n_sentences keep the fixed labels',
        [v('conservative'), v('full')],
        ['Eén zin, niet gesplitst', 'Opgesplitst']],
    ['counts beyond the word list fall back to digits',
        [v('conservative', 1), v('full', 12)],
        ['Eén zin, niet gesplitst', '12 zinnen']],
]

for (const [name, variants, expected] of cases) {
    test(name, () => assert.deepEqual(variantLabels(variants), expected))
}
