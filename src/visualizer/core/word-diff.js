/**
 * LCS-based word diff shared by the display path (lint_ii_visualizer.js,
 * DOM spans) and the export path (editor.js, plain strings), so the text a
 * user copies always matches what the editor showed.
 */

/** Normalization used to align original and suggested tokens. */
export const stripToken = t => t.replace(/[,;:()"'“”]/g, '').toLowerCase()

/**
 * Word diff that returns independent change regions.
 * Each region: { origIndices: [...], newTexts: [...], insertBeforeIdx }
 *
 * Alignment is on the normalized (stripToken) forms, so edits that live only
 * in the stripped-away layer — added punctuation, letter case, quote style —
 * align as "keep" and would otherwise be invisible, re-emitting the original
 * surface. Pass `origTokens` (the raw original surfaces) to catch these: a kept
 * pair whose RAW surfaces differ becomes a 1:1 replacement so the suggested
 * surface reaches the output (Henk Pander Maat: dropped comma zin 17, kept
 * original casing zin 8). Omit `origTokens` for the legacy surface-blind
 * behaviour.
 */
export function computeWordDiff(origBare, sugBare, sugTokens, origTokens = null) {
    const m = origBare.length
    const n = sugBare.length

    // LCS dynamic programming
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            dp[i][j] = origBare[i - 1] === sugBare[j - 1]
                ? dp[i - 1][j - 1] + 1
                : Math.max(dp[i - 1][j], dp[i][j - 1])
        }
    }

    // Backtrack to produce alignment operations
    const ops = []
    let i = m, j = n
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && origBare[i - 1] === sugBare[j - 1]) {
            ops.push({ type: 'keep', origIdx: i - 1, sugIdx: j - 1 })
            i--; j--
        } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
            ops.push({ type: 'insert', sugIdx: j - 1 })
            j--
        } else {
            ops.push({ type: 'delete', origIdx: i - 1 })
            i--
        }
    }
    ops.reverse()

    // A "keep" is only a true region boundary when the raw surfaces also match.
    // A surface-differing keep is a hidden edit and joins the change region.
    const isBoundary = (op) =>
        op.type === 'keep' &&
        (origTokens === null || origTokens[op.origIdx] === sugTokens[op.sugIdx])

    // Group contiguous change operations (including surface-differing keeps)
    // into change regions
    const regions = []
    let idx = 0
    while (idx < ops.length) {
        if (isBoundary(ops[idx])) { idx++; continue }

        const origIndices = []
        const newTexts = []
        while (idx < ops.length && !isBoundary(ops[idx])) {
            const op = ops[idx]
            if (op.type === 'delete') {
                origIndices.push(op.origIdx)
            } else if (op.type === 'insert') {
                newTexts.push(sugTokens[op.sugIdx])
            } else {
                // Surface-differing keep → replace the original word in place.
                origIndices.push(op.origIdx)
                newTexts.push(sugTokens[op.sugIdx])
            }
            idx++
        }

        // Determine insertion point
        const insertBeforeIdx = origIndices.length > 0
            ? origIndices[origIndices.length - 1] + 1
            : (idx < ops.length ? ops[idx].origIdx : origBare.length)

        regions.push({ origIndices, newTexts, insertBeforeIdx })
    }

    return regions
}

/**
 * Tokenize a suggested_text the same way in both paths: strip wrapping
 * quotes, split on whitespace, and capitalize the first token so sentence
 * capitalization never produces a spurious diff region.
 */
export function suggestionTokens(suggestedText) {
    const text = suggestedText.replace(/^[""“]+|[""”]+$/g, '').trim()
    const tokens = text.split(/\s+/).filter(Boolean)
    if (tokens.length > 0) {
        tokens[0] = capitalizeToken(tokens[0])
    }
    return tokens
}

/** Uppercase the first letter of a token, skipping leading quotes/brackets. */
export function capitalizeToken(token) {
    const lead = token.match(/^[("'“]*/)
    const off = lead ? lead[0].length : 0
    if (off < token.length) {
        return token.slice(0, off) + token.charAt(off).toUpperCase() + token.slice(off + 1)
    }
    return token
}
