import { computeWordDiff, stripToken, suggestionTokens, capitalizeToken } from './word-diff.js?v=2'

// Suggestion types that reformulate a whole sentence. Any two of these on the
// same sentence — or one of these plus any other edit — cannot be safely merged
// by the diff-splice compose path, so accepting one makes the rest of the
// sentence's suggestions mutually exclusive. Only word-level edits
// (word_frequency, spelling) are localized enough to co-apply. See accept().
const SENTENCE_SCOPED_TYPES = new Set([
    'sentence_rewrite',
    'max_sdl',
    'content_words_per_clause',
    'abstract_nouns',
    'passive',
    'subordinate_clause',
    'sentence_length',
])

/**
 * EditorController manages suggestion state and text editing.
 *
 * Tracks accepted/ignored suggestions, clusters overlapping suggestions,
 * and computes edited text by applying accepted changes.
 */
export class EditorController {
    constructor(data) {
        this._data = data
        this._suggestionStates = new Map() // id -> 'pending' | 'accepted' | 'ignored'
        this._eventTarget = new EventTarget()
        this._clusters = new Map()            // clusterId -> { suggestionIds, sentenceIdx, wordIndices }
        this._suggestionToCluster = new Map() // suggestionId -> clusterId
        this._wordToCluster = new Map()       // "sentIdx:wordIdx" -> clusterId

        // Initialize all suggestions as pending
        if (data.suggestions?.suggestions) {
            for (const suggestion of data.suggestions.suggestions) {
                this._suggestionStates.set(suggestion.id, 'pending')
            }
            this._buildClusters()
        }

        // Precompute per-sentence metrics from token data for score recomputation
        this._originalSentenceMetrics = this._computeOriginalMetrics()
        this._originalDocumentScore = data.document_lint_score
        this._originalDocumentLevel = data.document_difficulty_level
    }

    /**
     * Build clusters of suggestions that overlap in affected words.
     *
     * Affected words are determined by diffing each suggestion's rewritten
     * text against the original sentence — only the words that actually
     * change are part of the span. This keeps clusters as small as possible.
     * Suggestions that share any affected word in the same sentence are
     * merged into a single cluster using union-find.
     */
    _buildClusters() {
        const suggestions = this.suggestions
        if (!suggestions.length) return

        // Step 1: compute affected word indices via diff for each suggestion
        const affectedWords = new Map() // suggestionId -> { sentenceIdx, wordIndices: Set }

        for (const s of suggestions) {
            const sentence = this._data.sentences[s.sentence_index]
            if (!sentence) continue

            const wordIndices = this._computeAffectedIndices(s, sentence)

            // Fallback: if diff found nothing but we have a word_index, use it
            if (wordIndices.size === 0 && s.word_index != null) {
                const filteredIdx = this._toFilteredIndex(sentence, s.word_index, s.word)
                if (filteredIdx !== null) wordIndices.add(filteredIdx)
            }

            affectedWords.set(s.id, { sentenceIdx: s.sentence_index, wordIndices })
        }

        // Step 2: union-find to cluster overlapping suggestions
        const parent = new Map()
        for (const s of suggestions) parent.set(s.id, s.id)

        const find = (x) => {
            while (parent.get(x) !== x) {
                parent.set(x, parent.get(parent.get(x))) // path compression
                x = parent.get(x)
            }
            return x
        }
        const union = (a, b) => {
            const ra = find(a), rb = find(b)
            if (ra !== rb) parent.set(ra, rb)
        }

        for (let i = 0; i < suggestions.length; i++) {
            for (let j = i + 1; j < suggestions.length; j++) {
                const a = affectedWords.get(suggestions[i].id)
                const b = affectedWords.get(suggestions[j].id)
                if (!a || !b || a.sentenceIdx !== b.sentenceIdx) continue
                for (const idx of a.wordIndices) {
                    if (b.wordIndices.has(idx)) {
                        union(suggestions[i].id, suggestions[j].id)
                        break
                    }
                }
            }
        }

        // Step 3: build cluster map
        for (const s of suggestions) {
            const clusterId = find(s.id)
            this._suggestionToCluster.set(s.id, clusterId)

            if (!this._clusters.has(clusterId)) {
                this._clusters.set(clusterId, {
                    suggestionIds: new Set(),
                    sentenceIdx: affectedWords.get(s.id).sentenceIdx,
                    wordIndices: new Set()
                })
            }
            const cluster = this._clusters.get(clusterId)
            cluster.suggestionIds.add(s.id)
            for (const idx of affectedWords.get(s.id).wordIndices) {
                cluster.wordIndices.add(idx)
            }
        }

        // Step 4: build word-to-cluster reverse map
        for (const [clusterId, cluster] of this._clusters) {
            for (const wordIdx of cluster.wordIndices) {
                this._wordToCluster.set(`${cluster.sentenceIdx}:${wordIdx}`, clusterId)
            }
        }
    }

    /**
     * Convert a backend word_index (over all word_features) to the
     * filtered index used in rendering (excludes bare PUNCT tokens).
     * Falls back to text matching if the index doesn't align.
     */
    _toFilteredIndex(sentence, backendIndex, wordText) {
        const tokens = sentence.word_features
        let filteredIdx = 0
        for (let i = 0; i < tokens.length; i++) {
            const keep = tokens[i].pos !== 'PUNCT' || 'punctuation' in tokens[i]
            if (keep) {
                if (i === backendIndex) return filteredIdx
                filteredIdx++
            }
        }
        // Fallback: match by text
        filteredIdx = 0
        for (let i = 0; i < tokens.length; i++) {
            const keep = tokens[i].pos !== 'PUNCT' || 'punctuation' in tokens[i]
            if (keep) {
                if (tokens[i].text === wordText) return filteredIdx
                filteredIdx++
            }
        }
        return null
    }

    /**
     * Compute which word indices actually change for a suggestion by
     * diffing its suggested_text against the original sentence tokens.
     */
    _computeAffectedIndices(suggestion, sentence) {
        const strip = t => t.replace(/[.,;:!?()"'\u201c\u201d]/g, '').toLowerCase()
        const filteredTokens = sentence.word_features.filter(
            t => t.pos !== 'PUNCT' || 'punctuation' in t
        )
        const origBare = filteredTokens.map(t => strip(t.text))

        const sugText = suggestion.suggested_text
            .replace(/^[""\u201c]+|[""\u201d]+$/g, '').trim()
        const sugTokens = sugText.split(/\s+/).filter(Boolean)
        const sugBare = sugTokens.map(strip)

        const regions = this._computeWordDiff(origBare, sugBare, sugTokens)

        // Bare forms the suggestion newly inserts (not aligned to any original
        // word). A deleted original word whose bare form reappears among these
        // was only MOVED, not changed, so it must not be marked as affected.
        // Without this, reordered-but-identical words were highlighted as if
        // changed even though they stay put in the revision (Henk Pander Maat,
        // feedback H7).
        const insertedAvail = new Map()
        for (const region of regions) {
            for (const t of region.newTexts) {
                const b = strip(t)
                insertedAvail.set(b, (insertedAvail.get(b) || 0) + 1)
            }
        }

        const raw = new Set()
        const indices = new Set()
        for (const region of regions) {
            for (const idx of region.origIndices) {
                raw.add(idx)
                const bare = origBare[idx]
                const avail = insertedAvail.get(bare) || 0
                if (avail > 0) {
                    insertedAvail.set(bare, avail - 1) // moved word — not a change
                } else {
                    indices.add(idx)
                }
            }
        }

        // Safety net: never suppress every word (e.g. a pure reorder), or the
        // suggestion would have no span to click. Fall back to the raw set.
        return indices.size > 0 ? indices : raw
    }

    /**
     * LCS-based word diff returning independent change regions.
     * Each region: { origIndices: [...], newTexts: [...], insertBeforeIdx }
     */
    _computeWordDiff(origBare, sugBare, sugTokens) {
        const m = origBare.length
        const n = sugBare.length

        const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                dp[i][j] = origBare[i - 1] === sugBare[j - 1]
                    ? dp[i - 1][j - 1] + 1
                    : Math.max(dp[i - 1][j], dp[i][j - 1])
            }
        }

        const ops = []
        let i = m, j = n
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && origBare[i - 1] === sugBare[j - 1]) {
                ops.push({ type: 'keep', origIdx: i - 1 })
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

        const regions = []
        let idx = 0
        while (idx < ops.length) {
            if (ops[idx].type === 'keep') { idx++; continue }
            const origIndices = []
            const newTexts = []
            while (idx < ops.length && ops[idx].type !== 'keep') {
                if (ops[idx].type === 'delete') {
                    origIndices.push(ops[idx].origIdx)
                } else {
                    newTexts.push(sugTokens[ops[idx].sugIdx])
                }
                idx++
            }
            const insertBeforeIdx = origIndices.length > 0
                ? origIndices[origIndices.length - 1] + 1
                : (idx < ops.length ? ops[idx].origIdx : origBare.length)
            regions.push({ origIndices, newTexts, insertBeforeIdx })
        }
        return regions
    }

    /**
     * Get cluster info for a suggestion.
     */
    getClusterForSuggestion(suggestionId) {
        const clusterId = this._suggestionToCluster.get(suggestionId)
        if (!clusterId) return null
        return { id: clusterId, ...this._clusters.get(clusterId) }
    }

    /**
     * Get cluster info for a word at a given position.
     */
    getClusterForWord(sentenceIdx, wordIdx) {
        const clusterId = this._wordToCluster.get(`${sentenceIdx}:${wordIdx}`)
        if (!clusterId) return null
        return { id: clusterId, ...this._clusters.get(clusterId) }
    }

    /**
     * Aggregate cluster state: 'accepted' if any suggestion accepted,
     * 'ignored' if all ignored, else 'pending'.
     */
    getClusterState(clusterId) {
        const cluster = this._clusters.get(clusterId)
        if (!cluster) return 'pending'

        let hasAccepted = false
        let allIgnored = true
        for (const sid of cluster.suggestionIds) {
            const state = this._suggestionStates.get(sid)
            if (state === 'accepted') hasAccepted = true
            if (state !== 'ignored') allIgnored = false
        }
        if (hasAccepted) return 'accepted'
        if (allIgnored) return 'ignored'
        return 'pending'
    }

    /**
     * Get all suggestion objects in a cluster.
     */
    getClusterSuggestions(clusterId) {
        const cluster = this._clusters.get(clusterId)
        if (!cluster) return []
        return [...cluster.suggestionIds].map(id => this.getSuggestion(id)).filter(Boolean)
    }

    // ── LiNT Score Recomputation ──────────────────────────────

    static COEFFICIENTS = {
        constant: -4.20782,
        freq_log: 17.283729,
        max_sdl: -1.624415,
        content_words_per_clause: -2.536780,
        proportion_concrete: 16.001231,
    }

    /**
     * Extract per-sentence metrics from the original token data.
     */
    _computeOriginalMetrics() {
        return this._data.sentences.map(sentence => {
            let wordFreqSum = 0, wordFreqCount = 0
            let nConcrete = 0, nAbstract = 0, nUndefined = 0

            for (const wf of sentence.word_features) {
                if (wf.word_frequency != null) {
                    wordFreqSum += wf.word_frequency
                    wordFreqCount++
                }
                if (wf.super_sem_type === 'concrete') nConcrete++
                else if (wf.super_sem_type === 'abstract') nAbstract++
                else if (wf.super_sem_type === 'undefined') nUndefined++
            }

            return {
                word_freq_sum: wordFreqSum,
                word_freq_count: wordFreqCount,
                sdl_values: sentence.max_sdl != null ? [sentence.max_sdl] : [],
                cwpc_values: sentence.content_words_per_clause != null
                    ? [sentence.content_words_per_clause] : [],
                n_concrete: nConcrete,
                n_abstract: nAbstract,
                n_undefined: nUndefined,
            }
        })
    }

    /**
     * Compute the document LiNT score and level using current suggestion states.
     * For sentences with accepted suggestions, their precomputed new_sentence_metrics
     * replace the original metrics.
     */
    computeUpdatedScore() {
        let totalFreqSum = 0, totalFreqCount = 0
        const allSdls = []
        const allCwpcs = []
        let totalConcrete = 0, totalAbstract = 0, totalUndefined = 0

        for (let i = 0; i < this._originalSentenceMetrics.length; i++) {
            const metrics = this._getEffectiveMetrics(i)
            totalFreqSum += metrics.word_freq_sum
            totalFreqCount += metrics.word_freq_count
            allSdls.push(...metrics.sdl_values)
            allCwpcs.push(...metrics.cwpc_values)
            totalConcrete += metrics.n_concrete
            totalAbstract += metrics.n_abstract
            totalUndefined += metrics.n_undefined
        }

        const meanFreq = totalFreqCount > 0 ? totalFreqSum / totalFreqCount : null
        const meanSdl = allSdls.length > 0
            ? allSdls.reduce((a, b) => a + b, 0) / allSdls.length : null
        const meanCwpc = allCwpcs.length > 0
            ? allCwpcs.reduce((a, b) => a + b, 0) / allCwpcs.length : null
        const totalNouns = totalConcrete + totalAbstract + totalUndefined
        const propConcrete = totalNouns > 0 ? totalConcrete / totalNouns : null

        return this._lintScore(meanFreq, meanSdl, meanCwpc, propConcrete)
    }

    /**
     * Compute the LiNT score and difficulty level for a single sentence using
     * its effective metrics (accepted suggestion metrics when available).
     */
    getEffectiveSentenceLevel(sentenceIndex) {
        const metrics = this._getEffectiveMetrics(sentenceIndex)
        const meanFreq = metrics.word_freq_count > 0
            ? metrics.word_freq_sum / metrics.word_freq_count : null
        const meanSdl = metrics.sdl_values.length > 0
            ? metrics.sdl_values.reduce((a, b) => a + b, 0) / metrics.sdl_values.length : null
        const meanCwpc = metrics.cwpc_values.length > 0
            ? metrics.cwpc_values.reduce((a, b) => a + b, 0) / metrics.cwpc_values.length : null
        const totalNouns = metrics.n_concrete + metrics.n_abstract + metrics.n_undefined
        const propConcrete = totalNouns > 0 ? metrics.n_concrete / totalNouns : null

        if (meanFreq == null || meanSdl == null || meanCwpc == null || propConcrete == null) {
            const sentence = this._data.sentences[sentenceIndex]
            return { score: sentence?.lint_score ?? null, level: sentence?.difficulty_level ?? null }
        }

        const C = EditorController.COEFFICIENTS
        const raw = C.constant
            + C.freq_log * meanFreq
            + C.max_sdl * meanSdl
            + C.content_words_per_clause * meanCwpc
            + C.proportion_concrete * propConcrete
        const score = Math.min(100, Math.max(0, 100 - raw))
        const level = score < 34 ? 1 : score < 46 ? 2 : score < 58 ? 3 : 4
        return { score, level }
    }

    /**
     * Like getEffectiveSentenceLevel but returns one {score, level} per
     * resulting sentence. Length > 1 means the accepted suggestion splits
     * the sentence (sdl_values/cwpc_values have multiple entries).
     */
    getEffectiveSentenceLevels(sentenceIndex) {
        const metrics = this._getEffectiveMetrics(sentenceIndex)
        const meanFreq = metrics.word_freq_count > 0
            ? metrics.word_freq_sum / metrics.word_freq_count : null
        const totalNouns = metrics.n_concrete + metrics.n_abstract + metrics.n_undefined
        const propConcrete = totalNouns > 0 ? metrics.n_concrete / totalNouns : null
        const count = Math.max(metrics.sdl_values.length, metrics.cwpc_values.length, 1)
        const sentence = this._data.sentences[sentenceIndex]

        return Array.from({ length: count }, (_, i) => {
            const sdl = metrics.sdl_values[i] ?? null
            const cwpc = metrics.cwpc_values[i] ?? null

            if (meanFreq == null || sdl == null || cwpc == null || propConcrete == null) {
                return { score: sentence?.lint_score ?? null, level: sentence?.difficulty_level ?? null }
            }

            const C = EditorController.COEFFICIENTS
            const raw = C.constant
                + C.freq_log * meanFreq
                + C.max_sdl * sdl
                + C.content_words_per_clause * cwpc
                + C.proportion_concrete * propConcrete
            const score = Math.min(100, Math.max(0, 100 - raw))
            const level = score < 34 ? 1 : score < 46 ? 2 : score < 58 ? 3 : 4
            return { score, level }
        })
    }

    /**
     * Get effective metrics for a sentence, using accepted suggestion metrics
     * when available, otherwise original.
     */
    _getEffectiveMetrics(sentenceIndex) {
        const accepted = this.getSuggestionsForSentence(sentenceIndex)
            .filter(s => this._suggestionStates.get(s.id) === 'accepted')

        if (accepted.length > 0) {
            // Use the first accepted suggestion that has precomputed metrics
            const withMetrics = accepted.find(s => s.new_sentence_metrics)
            if (withMetrics) return withMetrics.new_sentence_metrics
        }

        return this._originalSentenceMetrics[sentenceIndex]
    }

    /**
     * Apply the LiNT-II formula. Returns { score, level }.
     */
    _lintScore(freqLog, maxSdl, cwpc, propConcrete) {
        if (freqLog == null || maxSdl == null || cwpc == null || propConcrete == null) {
            return { score: this._originalDocumentScore, level: this._originalDocumentLevel }
        }
        const C = EditorController.COEFFICIENTS
        const raw = C.constant
            + C.freq_log * freqLog
            + C.max_sdl * maxSdl
            + C.content_words_per_clause * cwpc
            + C.proportion_concrete * propConcrete
        const score = Math.min(100, Math.max(0, 100 - raw))
        const level = score < 34 ? 1 : score < 46 ? 2 : score < 58 ? 3 : 4
        return { score, level }
    }

    // ── End Score Recomputation ─────────────────────────────

    /**
     * Get all suggestions
     */
    get suggestions() {
        return this._data.suggestions?.suggestions || []
    }

    /**
     * Get suggestion counts by status
     */
    get counts() {
        const counts = { pending: 0, accepted: 0, ignored: 0, total: 0 }
        for (const status of this._suggestionStates.values()) {
            counts[status]++
            counts.total++
        }
        return counts
    }

    /**
     * Counts by the unit the user actually sees and acts on: the cluster (one
     * highlighted span, one decision), not the raw suggestion. Clustered
     * alternatives collapse into one highlight, so the raw count over-reports
     * what is visible ("3 aangekondigd, ik zie er 1" — Henk zin 2). Clusters
     * with no highlightable word are excluded, since they cannot be clicked.
     */
    get clusterCounts() {
        const counts = { pending: 0, accepted: 0, ignored: 0, total: 0 }
        for (const [clusterId, cluster] of this._clusters) {
            if (cluster.wordIndices.size === 0) continue
            counts[this.getClusterState(clusterId)]++
            counts.total++
        }
        return counts
    }

    /**
     * Get state of a specific suggestion
     */
    getState(suggestionId) {
        return this._suggestionStates.get(suggestionId) || 'pending'
    }

    /**
     * Accept a suggestion.
     *
     * Auto-ignores other pending suggestions in the SAME cluster: clustered
     * suggestions share affected word spans, so they are competing alternative
     * rewrites of the same text — once one is accepted the others no longer
     * apply. Suggestions elsewhere in the sentence (in other clusters) are
     * left untouched, so independent fixes remain available.
     */
    accept(suggestionId) {
        if (!this._suggestionStates.has(suggestionId)) return
        this._suggestionStates.set(suggestionId, 'accepted')

        const cluster = this.getClusterForSuggestion(suggestionId)
        if (cluster) {
            for (const sid of cluster.suggestionIds) {
                if (sid !== suggestionId && this._suggestionStates.get(sid) === 'pending') {
                    this._suggestionStates.set(sid, 'ignored')
                }
            }
        }

        // Sentence-scoped exclusivity: a whole-sentence rewrite and any other
        // edit on the same sentence cannot be composed by the diff-splice merge
        // (it interleaves them into garbage — Henk Pander Maat, zin 15/16/22/24).
        // So accepting a sentence-scoped suggestion drops every other suggestion
        // in that sentence, and accepting a word-level edit drops any
        // sentence-scoped rewrite there. Two word-level edits (neither scoped)
        // skip this and still co-apply, as they compose cleanly.
        const accepted = this.getSuggestion(suggestionId)
        if (accepted) {
            const acceptedScoped = SENTENCE_SCOPED_TYPES.has(accepted.type)
            for (const other of this.getSuggestionsForSentence(accepted.sentence_index)) {
                if (other.id === suggestionId) continue
                if (acceptedScoped || SENTENCE_SCOPED_TYPES.has(other.type)) {
                    if (this._suggestionStates.get(other.id) !== 'ignored') {
                        this._suggestionStates.set(other.id, 'ignored')
                    }
                }
            }
        }

        this._dispatchChange(suggestionId, 'accepted')
    }

    /**
     * Ignore a suggestion
     */
    ignore(suggestionId) {
        if (this._suggestionStates.has(suggestionId)) {
            this._suggestionStates.set(suggestionId, 'ignored')
            this._dispatchChange(suggestionId, 'ignored')
        }
    }

    /**
     * Reset a suggestion to pending
     */
    reset(suggestionId) {
        if (this._suggestionStates.has(suggestionId)) {
            this._suggestionStates.set(suggestionId, 'pending')
            this._dispatchChange(suggestionId, 'pending')
        }
    }

    /**
     * Add a single suggestion incrementally (for progressive delivery).
     * The suggestion must already be present in this._data.suggestions.suggestions.
     */
    addSuggestion(suggestion) {
        this._suggestionStates.set(suggestion.id, 'pending')
        this._clusters.clear()
        this._suggestionToCluster.clear()
        this._wordToCluster.clear()
        this._buildClusters()
    }

    /**
     * Get suggestion by ID
     */
    getSuggestion(suggestionId) {
        return this.suggestions.find(s => s.id === suggestionId)
    }

    /**
     * Get suggestions for a specific sentence
     */
    getSuggestionsForSentence(sentenceIndex) {
        return this.suggestions.filter(s => s.sentence_index === sentenceIndex)
    }

    /**
     * Return the current text of a sentence after applying all accepted
     * suggestion diffs. Used by the popup to show an up-to-date "Origineel".
     */
    getCurrentSentenceText(sentenceIndex) {
        const sentence = this._data.sentences[sentenceIndex]
        if (!sentence) return ''

        const accepted = this.getSuggestionsForSentence(sentenceIndex)
            .filter(s => this._suggestionStates.get(s.id) === 'accepted')

        const filtered = sentence.word_features.filter(
            t => t.pos !== 'PUNCT' || 'punctuation' in t
        )
        const words = filtered.map(
            t => (t.punctuation?.leading || '') + t.text + (t.punctuation?.trailing || '')
        )

        if (accepted.length === 0) return words.join(' ')

        const strip = t => t.replace(/[.,;:!?()"'“”]/g, '').toLowerCase()
        const origBare = filtered.map(t => strip(t.text))

        const allRegions = []
        for (const s of accepted) {
            const sugText = s.suggested_text.replace(/^[""“]+|[""”]+$/g, '').trim()
            const sugTokens = sugText.split(/\s+/).filter(Boolean)
            const sugBare = sugTokens.map(strip)
            for (const region of this._computeWordDiff(origBare, sugBare, sugTokens)) {
                allRegions.push(region)
            }
        }
        allRegions.sort((a, b) => b.insertBeforeIdx - a.insertBeforeIdx)

        for (const region of allRegions) {
            if (region.origIndices.length > 0) {
                words.splice(region.origIndices[0], region.origIndices.length, ...region.newTexts)
            } else {
                words.splice(region.insertBeforeIdx, 0, ...region.newTexts)
            }
        }

        return words.join(' ')
    }

    /**
     * Compute the current text of just the span a suggestion covers, after all
     * OTHER accepted suggestions for the same sentence are applied. Returns
     * suggestion.original_text unchanged when no other suggestion affects that span.
     */
    getCurrentOriginalForSuggestion(suggestionId) {
        const suggestion = this.getSuggestion(suggestionId)
        if (!suggestion) return ''

        const sentence = this._data.sentences[suggestion.sentence_index]
        if (!sentence) return suggestion.original_text

        const otherAccepted = this.getSuggestionsForSentence(suggestion.sentence_index)
            .filter(s => s.id !== suggestionId && this._suggestionStates.get(s.id) === 'accepted')

        if (otherAccepted.length === 0) return suggestion.original_text

        const strip = t => t.replace(/[.,;:!?()"'“”]/g, '').toLowerCase()
        const filtered = sentence.word_features.filter(
            t => t.pos !== 'PUNCT' || 'punctuation' in t
        )
        const origBare = filtered.map(t => strip(t.text))

        // Locate suggestion.original_text tokens in the original filtered list via LCS
        const thisTokens = suggestion.original_text.trim().split(/\s+/).filter(Boolean)
        const thisBare = thisTokens.map(strip)
        const m = thisTokens.length, n = origBare.length
        const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))
        for (let i = 1; i <= m; i++)
            for (let j = 1; j <= n; j++)
                dp[i][j] = thisBare[i - 1] === origBare[j - 1]
                    ? dp[i - 1][j - 1] + 1
                    : Math.max(dp[i - 1][j], dp[i][j - 1])

        const matchedOrig = []
        {
            let i = m, j = n
            while (i > 0 || j > 0) {
                if (i > 0 && j > 0 && thisBare[i - 1] === origBare[j - 1]) {
                    matchedOrig.push(j - 1); i--; j--
                } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
                    j--
                } else {
                    i--
                }
            }
            matchedOrig.reverse()
        }

        if (matchedOrig.length === 0) return suggestion.original_text
        const origStart = matchedOrig[0]
        const origEnd = matchedOrig[matchedOrig.length - 1]

        // Build trackedWords carrying original filtered indices so we can map
        // the span through splices made by other accepted suggestions.
        const trackedWords = filtered.map((t, idx) => ({
            text: (t.punctuation?.leading || '') + t.text + (t.punctuation?.trailing || ''),
            origIdx: idx
        }))

        const allRegions = []
        for (const s of otherAccepted) {
            const sugText = s.suggested_text.replace(/^["""]+|["""]+$/g, '').trim()
            const sugTokens = sugText.split(/\s+/).filter(Boolean)
            const sugBare = sugTokens.map(strip)
            for (const region of this._computeWordDiff(origBare, sugBare, sugTokens)) {
                allRegions.push(region)
            }
        }
        allRegions.sort((a, b) => b.insertBeforeIdx - a.insertBeforeIdx)

        for (const region of allRegions) {
            const insertPos = region.origIndices.length > 0
                ? region.origIndices[0]
                : region.insertBeforeIdx
            trackedWords.splice(insertPos, region.origIndices.length,
                ...region.newTexts.map(text => ({ text, origIdx: -1 })))
        }

        // Find the current range [currentStart..currentEnd] in trackedWords.
        // Range starts just after the last element with origIdx < origStart,
        // and ends just before the first element with origIdx > origEnd.
        let lastBefore = -1
        for (let k = trackedWords.length - 1; k >= 0; k--) {
            if (trackedWords[k].origIdx !== -1 && trackedWords[k].origIdx < origStart) {
                lastBefore = k; break
            }
        }
        const currentStart = lastBefore + 1

        const boundaryAfterIdx = trackedWords.findIndex(
            w => w.origIdx !== -1 && w.origIdx > origEnd
        )
        const currentEnd = boundaryAfterIdx === -1 ? trackedWords.length - 1 : boundaryAfterIdx - 1

        if (currentStart > currentEnd) return suggestion.original_text

        // Compare with the baseline (unmodified) span to detect actual changes
        const baselineSpan = filtered
            .slice(origStart, origEnd + 1)
            .map(t => (t.punctuation?.leading || '') + t.text + (t.punctuation?.trailing || ''))
            .join(' ')
        const currentSpan = trackedWords.slice(currentStart, currentEnd + 1).map(w => w.text).join(' ')

        return baselineSpan === currentSpan ? suggestion.original_text : currentSpan
    }

    /**
     * Compute the edited text with accepted suggestions applied
     */
    getEditedText() {
        const blocks = this._data.blocks

        // Fallback (no block layout): original behaviour, sentences space-joined.
        if (!Array.isArray(blocks) || blocks.length === 0) {
            const out = []
            for (let i = 0; i < this._data.sentences.length; i++) {
                out.push(this._sentenceOutputText(i))
            }
            return out.join(" ")
        }

        // Reconstruct the document preserving structure (H3): headings and
        // blank lines each on their own line; consecutive sentences (one prose
        // paragraph) space-joined on a single line.
        const lines = []
        let paragraph = []
        const flush = () => {
            if (paragraph.length) { lines.push(paragraph.join(" ")); paragraph = [] }
        }
        for (const block of blocks) {
            if (block.type === "sentence") {
                paragraph.push(this._sentenceOutputText(block.sentence_index))
            } else if (block.type === "list_item") {
                flush()
                const marker = block.ordered ? `${block.number}. ` : "- "
                const body = block.sentence_indices
                    .map(i => this._sentenceOutputText(i)).join(" ")
                lines.push(marker + body)
            } else if (block.type === "quote") {
                flush()
                lines.push("> " + block.text)
            } else if (block.type === "heading") {
                flush()
                lines.push(block.text)
            } else if (block.type === "blank") {
                flush()
                lines.push("")
            }
        }
        flush()
        return lines.join("\n")
    }

    /**
     * Output text for one prose sentence: the first accepted suggestion's
     * rewrite if any, otherwise the reconstructed original.
     */
    _sentenceOutputText(idx) {
        const sentence = this._data.sentences[idx]
        const accepted = this.getSuggestionsForSentence(idx).filter(
            s => this._suggestionStates.get(s.id) === "accepted"
        )
        if (accepted.length === 0) {
            return this._reconstructSentenceText(sentence)
        }
        if (accepted.length === 1) {
            return accepted[0].suggested_text
        }
        // Multiple accepted suggestions must all reach the exported text,
        // composed the same way the display composes them.
        return this._composeAcceptedText(sentence, accepted)
    }

    /**
     * Compose the output text for a sentence with several accepted
     * suggestions: diff each suggestion against the original tokens
     * (word-diff.js, same algorithm as the display path) and apply all
     * change regions right-to-left, tolerating interleaved regions.
     */
    _composeAcceptedText(sentence, accepted) {
        const tokens = sentence.word_features
            .filter(wf => wf.pos !== 'PUNCT' || 'punctuation' in wf)
            .map(wf => (wf.punctuation?.leading || '') + wf.text + (wf.punctuation?.trailing || ''))
        const origBare = tokens.map(stripToken)

        const allRegions = []
        for (const suggestion of accepted) {
            const sugTokens = suggestionTokens(suggestion.suggested_text)
            const sugBare = sugTokens.map(stripToken)
            allRegions.push(...computeWordDiff(origBare, sugBare, sugTokens, tokens))
        }
        allRegions.sort((a, b) => b.insertBeforeIdx - a.insertBeforeIdx)

        const removed = new Array(tokens.length).fill(false)
        const inserts = Array.from({ length: tokens.length + 1 }, () => [])

        for (const region of allRegions) {
            const texts = [...region.newTexts]
            if (region.origIndices.length > 0 && texts.length > 0) {
                // Preserve punctuation from the replaced original words, as
                // the display path does.
                const lastOrig = tokens[region.origIndices.at(-1)]
                const trailMatch = lastOrig.match(/([.,;:!?]+)$/)
                if (trailMatch && !texts.at(-1).match(/[.,;:!?]$/)) {
                    texts[texts.length - 1] += trailMatch[1]
                }
                const firstOrig = tokens[region.origIndices[0]]
                const leadMatch = firstOrig.match(/^([("'“]+)/)
                if (leadMatch && !texts[0].match(/^[("'“]/)) {
                    texts[0] = leadMatch[1] + texts[0]
                }
                // Only re-capitalize at the true sentence start; mid-sentence
                // this would undo a deliberate LLM lowercasing (Henk, zin 8).
                const firstOrigLetter = firstOrig.replace(/^[("'“]+/, '').charAt(0)
                if (region.origIndices[0] === 0 && firstOrigLetter && firstOrigLetter === firstOrigLetter.toUpperCase()
                    && firstOrigLetter !== firstOrigLetter.toLowerCase()) {
                    texts[0] = capitalizeToken(texts[0])
                }
            }

            // Mirror the display's anchor walk: an earlier-applied region may
            // have removed this region's anchor token.
            let anchor = region.insertBeforeIdx
            while (anchor < tokens.length && removed[anchor]) anchor++
            inserts[anchor].push(...texts)

            for (const i of region.origIndices) {
                removed[i] = true
            }
        }

        const out = []
        for (let i = 0; i <= tokens.length; i++) {
            out.push(...inserts[i])
            if (i < tokens.length && !removed[i]) out.push(tokens[i])
        }
        if (out.length > 0) {
            out[0] = capitalizeToken(out[0])
        }
        return out.join(' ')
    }

    /**
     * Reconstruct original sentence text from word features
     */
    _reconstructSentenceText(sentence) {
        let text = ''
        for (const wf of sentence.word_features) {
            if (wf.punctuation?.leading) {
                text += wf.punctuation.leading
            }
            if (wf.pos !== 'PUNCT' || wf.punctuation) {
                text += wf.text
            }
            if (wf.punctuation?.trailing) {
                text += wf.punctuation.trailing
            }
            // Add space after word (simplified)
            if (!wf.punctuation?.trailing?.match(/[.!?,;:]/)) {
                text += ' '
            }
        }
        return text.trim()
    }

    /**
     * Listen for editor changes
     */
    addEventListener(type, listener) {
        this._eventTarget.addEventListener(type, listener)
    }

    removeEventListener(type, listener) {
        this._eventTarget.removeEventListener(type, listener)
    }

    _dispatchChange(suggestionId, newStatus) {
        this._eventTarget.dispatchEvent(new CustomEvent('editor-change', {
            detail: { suggestionId, status: newStatus, counts: this.counts }
        }))
    }
}
