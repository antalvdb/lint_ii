import { css } from './core/stylesheet.js?v=9'
import { PopupController } from './core/popup.js'
import { WheelHandlerMixin } from './core/wheel-handler.js'
import { StatsData, StatsSpecs } from './core/stats.js?v=2'
import { EditorController } from './core/editor.js?v=10'
import { SuggestionPopupController } from './core/suggestion-popup.js?v=2'


export class LintIIVisualizer extends HTMLElement {
    static get observedAttributes() {
        return ['view', 'mode']
    }

    constructor() {
        super()
        this.attachShadow({ mode: "open" })
        this._currentView = 'sentences'
        this._mode = 'analysis'  // 'analysis' or 'editor'
        this._editorController = null
        this._suggestionPopupController = null
    }

    connectedCallback() {
        if (this._data) {
            this.render()
            this.applyWheelHandling()
        }
    }

    disconnectedCallback() {
        this.removeWheelHandling()
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'view' && newValue) {
            this._currentView = newValue
            if (this.isConnected) {
                this.updateViewVisibility()
                this.updateToggleButton()
            }
        }
        if (name === 'mode' && newValue) {
            this._mode = newValue
            if (this.isConnected && this._data) {
                this.render()
            }
        }
    }

    get mode() {
        return this._mode
    }

    get isEditorMode() {
        return this._mode === 'editor'
    }

    applyWheelHandling() {
        const contentArea = this.shadowRoot.querySelector('#content-area')
        if (contentArea) {
            Object.assign(contentArea, WheelHandlerMixin)
            contentArea.addWheelHandling()
        }
    }

    removeWheelHandling() {
        const contentArea = this.shadowRoot?.querySelector('#content-area')
        if (contentArea?.removeWheelHandling) {
            contentArea.removeWheelHandling()
        }
    }

    set data(value) {
        this._data = value
        this._editorController = null

        if (value.suggestions?.suggestions?.length > 0) {
            this._editorController = new EditorController(value)
        }

        if (this.isConnected) {
            this.render()
        }
    }

    async loadFromUrl(url) {
        const response = await fetch(url)
        this.data = await response.json()
    }

    /**
     * Add one suggestion progressively without a full re-render.
     * On the first call (switching from analysis→editor mode) a full render
     * is needed to add the toolbar and wire event listeners; subsequent calls
     * only re-render the affected sentence.
     */
    addSuggestion(suggestion) {
        if (!this._data.suggestions) {
            this._data.suggestions = { suggestions: [], triggers_found: 0, triggers_processed: 0, model: '' }
        }
        this._data.suggestions.suggestions.push(suggestion)

        const wasEditorMode = this.isEditorMode
        this._mode = 'editor'

        const scrollTop = this.shadowRoot.querySelector('#content-area')?.scrollTop ?? 0

        if (!wasEditorMode || !this._editorController) {
            this._editorController = new EditorController(this._data)
            this.render()
        } else {
            this._editorController.addSuggestion(suggestion)
            this._rerenderSentence(suggestion.sentence_index)
            this.updateEditorToolbar()
        }

        const contentArea = this.shadowRoot.querySelector('#content-area')
        if (contentArea) contentArea.scrollTop = scrollTop
    }

    _rerenderSentence(sentenceIdx) {
        const view = this.shadowRoot.querySelector('[data-view="sentences"]')
        if (!view) return
        const sentenceEls = view.querySelectorAll(':scope > .sentence')
        if (!sentenceEls[sentenceIdx]) return
        const tmp = document.createElement('span')
        tmp.innerHTML = this.renderSentence(this._data.sentences[sentenceIdx], sentenceIdx)
        view.replaceChild(tmp.firstElementChild, sentenceEls[sentenceIdx])
    }

    switchView(view) {
        this._currentView = view
        this.updateViewVisibility()
        this.shadowRoot.querySelector('.popup').classList.remove('visible')

        const toggle = this.shadowRoot.querySelector('.view-toggle')
        const targetView = view === 'sentences' ? 'stats' : 'sentences'
        toggle.dataset.targetView = targetView
        toggle.textContent = view === 'sentences' ? 'Σ' : '¶'
    }

    updateViewVisibility() {
        this.shadowRoot.querySelectorAll('[data-view]').forEach(view => {
            view.hidden = view.dataset.view !== this._currentView
        })
    }

    updateToggleButton() {
        const toggle = this.shadowRoot.querySelector('.view-toggle')
        if (toggle) {
            const targetView = this._currentView === 'sentences' ? 'stats' : 'sentences'
            toggle.dataset.targetView = targetView
            toggle.textContent = this._currentView === 'sentences' ? 'Σ' : '¶'
        }
    }

    setupEventListeners() {
        // View toggle buttons
        this.shadowRoot.querySelector('.view-toggle').addEventListener('click', (e) => {
            this.switchView(e.target.dataset.targetView)
        })

        const contentArea = this.shadowRoot.querySelector('#content-area')
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                ${css}
            </style>
            <header>
                <h1>
                    <span style="--index: 0">i</span><span style="--index: 1">N</span><span style="--index: 2">T</span>-<span style="--index: 0">I</span><span style="--index: 2">I</span>
                </h1>
                ${this.renderDocumentScores()}
                <button class="view-toggle" data-target-view="${this._currentView === 'sentences' ? 'stats' : 'sentences'}">
                    ${this._currentView === 'sentences' ? 'Σ' : '¶'}
                </button>
            </header>
            ${this.isEditorMode ? this.renderEditorToolbar() : ''}
            <div id="content-area">
                <div data-view="sentences">
                    ${this.renderBlocks()}
                </div>
                <div data-view="stats"></div>
            </div>
            <div class="popup"></div>
            ${this.isEditorMode ? '<div class="suggestion-popup"></div>' : ''}
        `
        this.updateViewVisibility()
        this.setupEventListeners()
        this.renderStats()

        // Setup suggestion popup controller in editor mode
        if (this.isEditorMode && this._editorController) {
            const suggestionPopup = this.shadowRoot.querySelector('.suggestion-popup')
            this._suggestionPopupController = new SuggestionPopupController(
                suggestionPopup,
                this._editorController
            )
            this.setupEditorEventListeners()
        }
    }

    renderEditorToolbar() {
        if (!this._editorController) return ''

        const counts = this._editorController.counts
        return `
            <div class="editor-toolbar">
                <div class="suggestion-counts">
                    <span class="count-item pending">
                        <span class="count-badge">${counts.pending}</span>
                        <span class="count-label">in behandeling</span>
                    </span>
                    <span class="count-item accepted">
                        <span class="count-badge">${counts.accepted}</span>
                        <span class="count-label">geaccepteerd</span>
                    </span>
                    <span class="count-item ignored">
                        <span class="count-badge">${counts.ignored}</span>
                        <span class="count-label">genegeerd</span>
                    </span>
                </div>
                <button class="sem-type-toggle" title="Toon/verberg woordsoorten">
                    Woordsoorten
                </button>
                <button class="copy-result-btn" title="Kopieer bewerkte tekst">
                    Kopieer resultaat
                </button>
            </div>
        `
    }

    updateEditorToolbar() {
        if (!this._editorController) return

        const counts = this._editorController.counts
        const toolbar = this.shadowRoot.querySelector('.editor-toolbar')
        if (!toolbar) return

        const pendingBadge = toolbar.querySelector('.pending .count-badge')
        const acceptedBadge = toolbar.querySelector('.accepted .count-badge')
        const ignoredBadge = toolbar.querySelector('.ignored .count-badge')

        if (pendingBadge) pendingBadge.textContent = counts.pending
        if (acceptedBadge) acceptedBadge.textContent = counts.accepted
        if (ignoredBadge) ignoredBadge.textContent = counts.ignored
    }

    updateDocumentScore() {
        if (!this._editorController) return
        const { score, level } = this._editorController.computeUpdatedScore()

        const scoreEl = this.shadowRoot.querySelector('.lint-score-value')
        const levelBadge = this.shadowRoot.querySelector(
            '.document-scores [data-level] .level-badge'
        )
        const levelContainer = this.shadowRoot.querySelector(
            '.document-scores [data-level]'
        )

        if (scoreEl && score != null) scoreEl.textContent = score.toFixed(1)
        if (levelBadge && level != null) levelBadge.textContent = level
        if (levelContainer && level != null) levelContainer.dataset.level = level
    }

    updateSentenceScore(sentenceIndex) {
        if (!this._editorController) return

        // Remove split-sentence elements from a previous accept of this sentence
        this.shadowRoot.querySelectorAll(
            `[data-sentence-index="${sentenceIndex}"][data-split-part]`
        ).forEach(el => el.remove())

        const levels = this._editorController.getEffectiveSentenceLevels(sentenceIndex)
        const sentenceEl = this.shadowRoot.querySelector(
            `[data-sentence-index="${sentenceIndex}"]:not([data-split-part])`
        )
        if (!sentenceEl) return

        const { level } = levels[0]
        sentenceEl.dataset.level = level != null ? String(level) : ""
        const badge = sentenceEl.querySelector(".level-badge")
        if (badge && level != null) badge.textContent = level

        if (levels.length > 1) {
            this._applySentenceSplit(sentenceEl, sentenceIndex, levels)
        }
    }

    _applySentenceSplit(primaryEl, sentenceIndex, levels) {
        const wordEls = Array.from(primaryEl.querySelectorAll(".word"))
        if (wordEls.length < 2) return

        // Find split boundaries: word ending .!? followed by uppercase-initial word
        const splitPoints = []
        for (let i = 0; i < wordEls.length - 1; i++) {
            const text = wordEls[i].textContent.trim()
            const nextText = wordEls[i + 1].textContent.trim()
            if (/[.!?]$/.test(text) && /^[A-Z\u00C0-\u00D6\u00D8-\u00DE]/.test(nextText)) {
                splitPoints.push(i)
                if (splitPoints.length >= levels.length - 1) break
            }
        }
        if (splitPoints.length === 0) return

        let insertAfter = primaryEl
        for (let p = 0; p < splitPoints.length; p++) {
            const partLevel = levels[p + 1] != null ? levels[p + 1].level : null
            const nextSplit = splitPoints[p + 1] != null ? splitPoints[p + 1] : wordEls.length - 1
            const partWords = wordEls.slice(splitPoints[p] + 1, nextSplit + 1)

            const newSentence = document.createElement("span")
            newSentence.className = "sentence"
            newSentence.dataset.level = partLevel != null ? String(partLevel) : ""
            newSentence.dataset.sentenceIndex = sentenceIndex
            newSentence.dataset.splitPart = p + 2

            const startGroup = document.createElement("span")
            startGroup.className = "sent-start-group"
            startGroup.innerHTML = "<span class=\"sent-start\"></span>"
            newSentence.appendChild(startGroup)

            for (const wordEl of partWords) {
                newSentence.appendChild(wordEl)
            }

            const endGroup = document.createElement("span")
            endGroup.className = "sent-end-group"
            endGroup.innerHTML = "<span class=\"sent-end\"></span>"
            const lvBadge = document.createElement("span")
            lvBadge.className = "level-badge"
            lvBadge.textContent = partLevel != null ? String(partLevel) : "?"
            endGroup.appendChild(lvBadge)
            newSentence.appendChild(endGroup)

            insertAfter.after(newSentence)
            insertAfter = newSentence
        }
    }

    setupEditorEventListeners() {
        // Semantic type toggle
        const semToggle = this.shadowRoot.querySelector('.sem-type-toggle')
        if (semToggle) {
            semToggle.addEventListener('click', () => {
                if (this.dataset.showSemTypes) {
                    delete this.dataset.showSemTypes
                    semToggle.classList.remove('active')
                } else {
                    this.dataset.showSemTypes = ''
                    semToggle.classList.add('active')
                }
            })
        }

        // Copy result button
        const copyBtn = this.shadowRoot.querySelector('.copy-result-btn')
        if (copyBtn) {
            copyBtn.addEventListener('click', async () => {
                const editedText = this._editorController.getEditedText()
                try {
                    // Requires HTTPS or localhost
                    await navigator.clipboard.writeText(editedText)
                } catch {
                    // Fallback for plain HTTP contexts
                    const ta = document.createElement('textarea')
                    ta.value = editedText
                    ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none'
                    document.body.appendChild(ta)
                    ta.select()
                    document.execCommand('copy')
                    document.body.removeChild(ta)
                }
                copyBtn.classList.add('copied')
                copyBtn.textContent = 'Gekopieerd!'
                setTimeout(() => {
                    copyBtn.classList.remove('copied')
                    copyBtn.textContent = 'Kopieer resultaat'
                }, 2000)
            })
        }

        // Listen for editor changes to update UI
        this._editorController.addEventListener('editor-change', (e) => {
            const { suggestionId, status } = e.detail
            this.updateSuggestionStatus(suggestionId, status)
            this.updateEditorToolbar()
            this.updateDocumentScore()
            const suggestion = this._editorController.getSuggestion(suggestionId)
            if (suggestion) this.updateSentenceScore(suggestion.sentence_index)
        })

        // Suggestion hover handling (cluster-aware)
        const contentArea = this.shadowRoot.querySelector('#content-area')
        contentArea.addEventListener('mouseover', (e) => {
            const wordEl = e.target.closest('[data-cluster-id]')
            if (wordEl && this._suggestionPopupController) {
                const clusterId = wordEl.dataset.clusterId
                this._suggestionPopupController.showCluster(clusterId, wordEl)
            }
        })

        contentArea.addEventListener('mouseout', (e) => {
            const wordEl = e.target.closest('[data-cluster-id]')
            if (wordEl && this._suggestionPopupController) {
                this._suggestionPopupController.hide()
            }
        })

        // Show popup on tap (touch devices) — click also fires on desktop so this
        // complements hover without breaking it.
        contentArea.addEventListener('click', (e) => {
            const wordEl = e.target.closest('[data-cluster-id]')
            if (wordEl && this._suggestionPopupController) {
                this._suggestionPopupController.showCluster(wordEl.dataset.clusterId, wordEl)
            }
        })

        // Dismiss popup when tapping anywhere that is not a suggestion word or the popup itself.
        this.shadowRoot.addEventListener('click', (e) => {
            if (!this._suggestionPopupController) return
            if (!e.target.closest('[data-cluster-id]') && !e.target.closest('.suggestion-popup')) {
                this._suggestionPopupController._hideNow()
            }
        })
    }

    updateSuggestionStatus(suggestionId, status) {
        const suggestion = this._editorController.getSuggestion(suggestionId)
        if (!suggestion) return

        const cluster = this._editorController.getClusterForSuggestion(suggestionId)
        if (!cluster) return

        const sentenceEl = this.shadowRoot.querySelector(
            `[data-sentence-index="${cluster.sentenceIdx}"]:not([data-split-part])`
        )
        if (!sentenceEl) return

        // Store original HTML on first modification
        if (!sentenceEl.dataset.originalHtml) {
            sentenceEl.dataset.originalHtml = sentenceEl.innerHTML
        }

        // Always restore to original first
        sentenceEl.innerHTML = sentenceEl.dataset.originalHtml

        // Tag every word with its cluster info (before applying diffs)
        const wordEls = Array.from(sentenceEl.querySelectorAll('.word'))
        for (let idx = 0; idx < wordEls.length; idx++) {
            const wCluster = this._editorController.getClusterForWord(cluster.sentenceIdx, idx)
            if (wCluster) {
                const cState = this._editorController.getClusterState(wCluster.id)
                const primaryId = [...wCluster.suggestionIds][0]
                wordEls[idx].dataset.clusterId = wCluster.id
                wordEls[idx].dataset.suggestionId = primaryId
                wordEls[idx].dataset.suggestionStatus = cState
            }
        }

        // Apply text diffs for all accepted suggestions in this sentence
        const allSuggestions = this._editorController
            .getSuggestionsForSentence(cluster.sentenceIdx)
        const accepted = allSuggestions.filter(s =>
            this._editorController.getState(s.id) === 'accepted'
        )

        if (accepted.length === 0) {
            delete sentenceEl.dataset.originalHtml
        } else {
            this._applyAcceptedDiffs(sentenceEl, accepted)
        }
    }

    /**
     * Compute word-level diffs for each accepted suggestion against the
     * original sentence using LCS, then apply all change regions.
     * Regions are applied right-to-left so index shifts don't interfere.
     * New spans and surviving cluster words are tagged with cluster state.
     */
    _applyAcceptedDiffs(sentenceEl, acceptedSuggestions) {
        const wordEls = Array.from(sentenceEl.querySelectorAll('.word'))
        const strip = t => t.replace(/[,;:()"'\u201c\u201d]/g, "").toLowerCase()
        const origBare = wordEls.map(el => strip(el.textContent))

        // Collect all change regions from all accepted suggestions
        const allRegions = []
        for (const suggestion of acceptedSuggestions) {
            const sugText = suggestion.suggested_text
                .replace(/^[""\u201c]+|[""\u201d]+$/g, '').trim()
            const sugTokens = sugText.split(/\s+/).filter(Boolean)

            // Capitalize the first suggested token to match sentence start,
            // so capitalization differences never produce a spurious diff region.
            if (sugTokens.length > 0) {
                const first = sugTokens[0]
                const lead = first.match(/^[("'\u201c]*/)
                const off = lead ? lead[0].length : 0
                if (off < first.length) {
                    sugTokens[0] = first.slice(0, off) + first.charAt(off).toUpperCase() + first.slice(off + 1)
                }
            }

            const sugBare = sugTokens.map(strip)

            const regions = this._computeWordDiff(origBare, sugBare, sugTokens)
            for (const region of regions) {
                region.suggestion = suggestion
            }
            allRegions.push(...regions)
        }

        // Sort right-to-left by insertion point
        allRegions.sort((a, b) => b.insertBeforeIdx - a.insertBeforeIdx)

        for (const region of allRegions) {
            const insertBefore = region.insertBeforeIdx < wordEls.length
                ? wordEls[region.insertBeforeIdx]
                : sentenceEl.querySelector('.sent-end-group')

            // Preserve punctuation and capitalization from deleted original words.
            const texts = [...region.newTexts]
            if (region.origIndices.length > 0 && texts.length > 0) {
                const lastOrig = wordEls[region.origIndices.at(-1)].textContent
                const trailMatch = lastOrig.match(/([.,;:!?]+)$/)
                if (trailMatch && !texts.at(-1).match(/[.,;:!?]$/)) {
                    texts[texts.length - 1] += trailMatch[1]
                }
                const firstOrig = wordEls[region.origIndices[0]].textContent
                const leadMatch = firstOrig.match(/^([("'\u201c]+)/)
                if (leadMatch && !texts[0].match(/^[("'\u201c]/)) {
                    texts[0] = leadMatch[1] + texts[0]
                }

                // Preserve capitalization: if first deleted word started
                // uppercase, capitalize the first new word to match.
                const firstOrigLetter = firstOrig.replace(/^[("'\u201c]+/, '').charAt(0)
                if (firstOrigLetter && firstOrigLetter === firstOrigLetter.toUpperCase() && firstOrigLetter !== firstOrigLetter.toLowerCase()) {
                    const firstNew = texts[0]
                    const leadPunct = firstNew.match(/^[("'\u201c]*/)
                    const offset = leadPunct ? leadPunct[0].length : 0
                    if (offset < firstNew.length) {
                        texts[0] = firstNew.slice(0, offset) + firstNew.charAt(offset).toUpperCase() + firstNew.slice(offset + 1)
                    }
                }
            }

            // Get cluster for this suggestion to tag new spans
            const cluster = this._editorController.getClusterForSuggestion(region.suggestion.id)

            for (const text of texts) {
                const span = document.createElement('span')
                span.className = 'word suggestion-changed'
                span.textContent = text
                span.dataset.suggestionId = region.suggestion.id
                span.dataset.suggestionStatus = 'accepted'
                if (cluster) span.dataset.clusterId = cluster.id
                sentenceEl.insertBefore(span, insertBefore)
            }

            for (const idx of region.origIndices) {
                wordEls[idx].remove()
            }
        }

    }

    /**
     * LCS-based word diff that returns multiple independent change regions.
     * Each region: { origIndices: [...], newTexts: [...], insertBeforeIdx }
     */
    _computeWordDiff(origBare, sugBare, sugTokens) {
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

        // Group contiguous non-keep operations into change regions
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

            // Determine DOM insertion point
            const insertBeforeIdx = origIndices.length > 0
                ? origIndices[origIndices.length - 1] + 1
                : (idx < ops.length ? ops[idx].origIdx : origBare.length)

            regions.push({ origIndices, newTexts, insertBeforeIdx })
        }

        return regions
    }

    renderContent() {
        const contentArea = this.shadowRoot.querySelector('#content-area')

        if (this._currentView === 'sentences') {
            contentArea.innerHTML = `
                <div data-view="sentences">
                    ${this.renderBlocks()}
                </div>
            `
            this.applyWheelHandling()
        } else {
            contentArea.innerHTML = '<div id="stats-container"></div>'
            this.renderStats()
        }
    }

    renderDocumentScores() {
        const totalWords = this._data.sentences.reduce((sum, s) => sum + s.word_features.length, 0)

        return `<dl class="document-scores">
            <div class="doc-stat">
                <dt>zinnen</dt>
                <dd>${this._data.sentences.length}</dd>
            </div>
            <div class="doc-stat">
                <dt>lint score</dt>
                <dd class="lint-score-value">${this._data.document_lint_score != null ? this._data.document_lint_score.toFixed(1) : '—'}</dd>
            </div>
            ${this.renderDocumentLevel()}
        </dl>`
    }

    renderDocumentLevel() {
        return `<div data-level="${this._data.document_difficulty_level}"><span class="level-badge">${this._data.document_difficulty_level}</span></div>`
    }

    renderSentence(sentence, idx) {
        const tokens = sentence.word_features.filter(token => {
            if (token.pos !== 'PUNCT') return true
            return 'punctuation' in token
        })

        return `<span class="sentence" data-level="${sentence.difficulty_level}" data-sentence-index="${idx}">
            <span class="sent-start-group">
                <span class="sent-idx">${idx + 1}</span>
                <span class="sent-start"></span>
            </span>
            ${tokens.map((item, wordIdx) => this.renderWord(item, idx, wordIdx)).join('')}
            <span class="sent-end-group">
                <span class="sent-end"></span>
                <span class="level-badge"
                    data-length="${tokens.length}"
                    data-score="${sentence.lint_score}"
                    data-max-sdl="${sentence.max_sdl}"
                    data-mean-freq="${sentence.mean_log_word_frequency}"
                    data-concrete-prop="${sentence.proportion_of_concrete_nouns}">
                    ${sentence.difficulty_level ?? '?'}
                </span>
            </span>
        </span>`
    }

    /**
     * Render the document in block order (H3): prose sentences interleaved
     * with non-prose headings and blank-line separators. Falls back to the
     * old sentences-only rendering if the payload carries no block layout.
     */
    renderBlocks() {
        const blocks = this._data.blocks
        if (!Array.isArray(blocks) || blocks.length === 0) {
            return this._data.sentences.map((s, idx) => this.renderSentence(s, idx)).join("")
        }
        return blocks.map(block => {
            if (block.type === "sentence") {
                const idx = block.sentence_index
                return this.renderSentence(this._data.sentences[idx], idx)
            }
            if (block.type === "heading") {
                return `<div class="doc-heading">${this._escapeHtml(block.text)}</div>`
            }
            if (block.type === "blank") {
                return `<div class="doc-blank"></div>`
            }
            return ""
        }).join("")
    }

    _escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]))
    }

    renderWord(wf, sentenceIdx = null, wordIdx = null) {
        const leading = wf.punctuation?.leading || ''
        const trailing = wf.punctuation?.trailing || ''
        const displayText = leading + wf.text + trailing

        if (!this.isEditorMode) {
            return `<span class="word">${displayText}</span>`
        }

        const attrs = [
            `data-pos="${wf.pos}"`,
            `data-tag="${wf.tag}"`,
            wf.super_sem_type && `data-sem-type="${wf.super_sem_type}"`,
            wf.word_frequency && `data-freq="${wf.word_frequency}"`,
            wf.dep_length > 0 && `data-dep-length="${wf.dep_length}"`
        ].filter(Boolean)

        if (this._editorController && sentenceIdx !== null) {
            const cluster = this._editorController.getClusterForWord(sentenceIdx, wordIdx)
            if (cluster) {
                const clusterState = this._editorController.getClusterState(cluster.id)
                const primaryId = [...cluster.suggestionIds][0]
                attrs.push(`data-cluster-id="${cluster.id}"`)
                attrs.push(`data-suggestion-id="${primaryId}"`)
                attrs.push(`data-suggestion-status="${clusterState}"`)
            }
        }

        return `<span class="word" ${attrs.join(' ')}>${displayText}</span>`
    }

    async renderStats() {
        const styles = getComputedStyle(this)
        const colors = {
            concrete: styles.getPropertyValue('--color-concrete').trim(),
            abstract: styles.getPropertyValue('--color-abstract').trim(),
            undefined: styles.getPropertyValue('--color-undefined').trim(),
            unknown: styles.getPropertyValue('--color-unknown').trim(),
            currentColor: getComputedStyle(this).color,
        }
        const statsData = new StatsData(this._data)

        const spec = StatsSpecs.createStatsVisualization({
                wordFreqs: statsData.getWordFrequencies(),
                sentScores: statsData.getSentenceScores(),
                nounCounts: statsData.getNounCountsByType(),
                depLengths: statsData.getDependencyLengths(),
                contentWordsPerClause: statsData.getContentWordsPerClause(),
            }, colors
        )

        const container = this.shadowRoot.querySelector('[data-view="stats"]')
        await vegaEmbed(container, spec, { actions: false })
    }
}

window.customElements.define('lint-ii-visualizer', LintIIVisualizer)
