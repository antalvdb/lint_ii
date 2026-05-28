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

        const indices = new Set()
        for (const region of regions) {
            for (const idx of region.origIndices) {
                indices.add(idx)
            }
        }
        return indices
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
     * Get state of a specific suggestion
     */
    getState(suggestionId) {
        return this._suggestionStates.get(suggestionId) || 'pending'
    }

    /**
     * Accept a suggestion. Auto-ignores other pending suggestions for the
     * same sentence, since they are based on the original text which no
     * longer applies once a rewrite is accepted.
     */
    accept(suggestionId) {
        if (!this._suggestionStates.has(suggestionId)) return

        const accepted = this._data.suggestions.suggestions.find(s => s.id === suggestionId)
        const sentenceIdx = accepted?.sentence_index ?? -1

        this._suggestionStates.set(suggestionId, 'accepted')

        // Auto-ignore other pending suggestions for the same sentence
        for (const s of this._data.suggestions.suggestions) {
            if (s.id !== suggestionId && s.sentence_index === sentenceIdx) {
                if (this._suggestionStates.get(s.id) === 'pending') {
                    this._suggestionStates.set(s.id, 'ignored')
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
     * Compute the edited text with accepted suggestions applied
     */
    getEditedText() {
        const sentences = this._data.sentences
        const result = []

        for (let i = 0; i < sentences.length; i++) {
            const sentence = sentences[i]
            const sentenceSuggestions = this.getSuggestionsForSentence(i)

            // Get accepted suggestions for this sentence
            const acceptedSuggestions = sentenceSuggestions.filter(
                s => this._suggestionStates.get(s.id) === 'accepted'
            )

            if (acceptedSuggestions.length === 0) {
                // No accepted suggestions, use original text
                result.push(this._reconstructSentenceText(sentence))
            } else {
                // Apply the first accepted suggestion's text
                // (for simplicity, we use the whole sentence replacement)
                const suggestion = acceptedSuggestions[0]
                result.push(suggestion.suggested_text)
            }
        }

        return result.join(' ')
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
