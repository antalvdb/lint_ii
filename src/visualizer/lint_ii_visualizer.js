import { css } from './core/stylesheet.js?v=22'
import { PopupController } from './core/popup.js'
import { WheelHandlerMixin } from './core/wheel-handler.js'
import { StatsData, StatsSpecs } from './core/stats.js?v=2'
import { EditorController } from './core/editor.js?v=20'
import { SuggestionPopupController } from './core/suggestion-popup.js?v=9'
import { computeWordDiff, stripToken, suggestionTokens, capitalizeToken } from './core/word-diff.js?v=2'


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
        if (this._docClickHandler) {
            document.removeEventListener('click', this._docClickHandler)
            this._docClickHandler = null
        }
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
     * Number of actionable suggestions the user can see and act on (highlighted
     * clusters), for the "N suggesties gevonden" headline. Falls back to 0 when
     * there is no editor (analysis-only result).
     */
    get actionableSuggestionCount() {
        return this._editorController ? this._editorController.clusterCounts.total : 0
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
                ${this.renderLevelLegend()}
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

        const counts = this._editorController.clusterCounts
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

        const counts = this._editorController.clusterCounts
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
        const origScore = this._data.document_lint_score
        const origLevel = this._data.document_difficulty_level

        const scoreEl = this.shadowRoot.querySelector('.lint-score-value')
        const scoreOrig = this.shadowRoot.querySelector('.lint-score-orig')
        const levelBadge = this.shadowRoot.querySelector(
            '.document-scores [data-level] .level-badge'
        )
        const levelContainer = this.shadowRoot.querySelector(
            '.document-scores [data-level]'
        )
        const levelOrig = this.shadowRoot.querySelector('.level-orig')

        if (scoreEl && score != null) scoreEl.textContent = this._fmtScore(score)
        if (scoreOrig) {
            scoreOrig.textContent = `was ${this._fmtScore(origScore)}`
            scoreOrig.hidden = !(score != null && origScore != null && Math.abs(score - origScore) >= 0.05)
        }
        if (levelBadge && level != null) levelBadge.textContent = level
        if (levelContainer && level != null) levelContainer.dataset.level = level
        if (levelOrig) {
            levelOrig.textContent = `was ${origLevel}`
            levelOrig.hidden = !(level != null && origLevel != null && level !== origLevel)
        }
    }

    /**
     * Pop a transient score-delta badge next to the just-accepted change, so
     * the improvement registers even though the score bar in the header is
     * scrolled out of view while editing.
     */
    _flashScoreDelta(suggestionId, suggestion, delta, levelFrom, levelTo) {
        this.shadowRoot.querySelectorAll('.score-flash').forEach(el => el.remove())

        let anchor = this.shadowRoot.querySelector(
            `.suggestion-changed[data-suggestion-id="${suggestionId}"]`
        )
        if (!anchor && suggestion) {
            anchor = this.shadowRoot.querySelector(
                `[data-sentence-index="${suggestion.sentence_index}"]:not([data-split-part])`
            )
        }
        if (!anchor) return
        const rect = anchor.getBoundingClientRect()

        const sign = delta < 0 ? '−' : '+'
        const mag = Math.abs(delta).toFixed(1).replace('.', ',')
        const levelChanged = levelFrom != null && levelTo != null && levelFrom !== levelTo

        const flash = document.createElement('div')
        flash.className = 'score-flash ' + (delta < 0 ? 'improve' : 'worse')
        flash.innerHTML =
            `<span class="score-flash-delta">${sign}${mag}</span>` +
            (levelChanged ? `<span class="score-flash-level">niveau ${levelFrom}→${levelTo}</span>` : '')
        this.shadowRoot.appendChild(flash)

        flash.style.top = `${Math.max(8, rect.top - 6)}px`
        flash.style.left = `${Math.min(window.innerWidth - 130, rect.right + 8)}px`
        flash.addEventListener('animationend', () => flash.remove())
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
                    this.dataset.showSemTypes = '1'
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

        // Baseline for score-delta flashes; reflects the current accepted
        // state so a re-render doesn't produce a spurious flash.
        const base = this._editorController.computeUpdatedScore()
        this._lastDocScore = base.score
        this._lastDocLevel = base.level

        // Listen for editor changes to update UI
        this._editorController.addEventListener('editor-change', (e) => {
            const { suggestionId, status } = e.detail
            const beforeScore = this._lastDocScore
            const beforeLevel = this._lastDocLevel
            try {
                this.updateSuggestionStatus(suggestionId, status)
            } catch (err) {
                // A diff-application failure on one sentence must not abort
                // the toolbar/score/split updates below, or the view is left
                // inconsistent (stale split parts render as duplicated text).
                console.error('updateSuggestionStatus failed:', err)
            }
            this.updateEditorToolbar()
            this.updateDocumentScore()
            const suggestion = this._editorController.getSuggestion(suggestionId)
            if (suggestion) this.updateSentenceScore(suggestion.sentence_index)
            // A rewrite of a merge's first sentence composes into the merged
            // text; if that merge is accepted, re-render it to reflect the graft.
            if (suggestion && suggestion.type !== 'connective') {
                const c = this._editorController.getConnectiveForFirstSentence(suggestion.sentence_index)
                if (c && this._editorController.getState(c.id) === 'accepted') {
                    try {
                        this._renderMergedSentence(c.merges_sentences[0], c)
                        this.updateSentenceScore(c.merges_sentences[0])
                    } catch (err) { console.error('merge recompose failed:', err) }
                }
            }
            // Reconcile connective chips/merged views with current state: an
            // accept elsewhere may have auto-ignored (or an undo revived) a
            // connective whose own change wasn't dispatched to this sentence.
            try { this._refreshConnectiveMarkers() } catch (err) {
                console.error('connective refresh failed:', err)
            }

            const after = this._editorController.computeUpdatedScore()
            // Suppress the delta flash for a connective merge: fusing sentences
            // raises the score (a longer sentence), which the flash would read as
            // "you made it worse". Coherence isn't captured by the LiNT metrics;
            // the popup explains this instead. Ordinary rewrites still flash.
            if (status === 'accepted' && suggestion?.type !== 'connective'
                && beforeScore != null && after.score != null
                && Math.abs(after.score - beforeScore) >= 0.05) {
                try {
                    this._flashScoreDelta(suggestionId, suggestion, after.score - beforeScore, beforeLevel, after.level)
                } catch (err) {
                    console.error('score flash failed:', err)
                }
            }
            this._lastDocScore = after.score
            this._lastDocLevel = after.level
        })

        // Suggestion hover handling (cluster-aware, plus connective markers)
        const contentArea = this.shadowRoot.querySelector('#content-area')
        contentArea.addEventListener('mouseover', (e) => {
            if (!this._suggestionPopupController) return
            const connEl = e.target.closest('[data-connective-id]')
            if (connEl) {
                this._suggestionPopupController.showConnective(connEl.dataset.connectiveId, connEl)
                return
            }
            const wordEl = e.target.closest('[data-cluster-id]')
            if (wordEl) {
                this._suggestionPopupController.showCluster(wordEl.dataset.clusterId, wordEl)
            }
        })

        contentArea.addEventListener('mouseout', (e) => {
            const el = e.target.closest('[data-cluster-id],[data-connective-id]')
            if (el && this._suggestionPopupController) {
                this._suggestionPopupController.hide()
            }
        })

        // Show popup on tap (touch devices) — click also fires on desktop so this
        // complements hover without breaking it.
        contentArea.addEventListener('click', (e) => {
            if (!this._suggestionPopupController) return
            const connEl = e.target.closest('[data-connective-id]')
            if (connEl) {
                this._suggestionPopupController.showConnective(connEl.dataset.connectiveId, connEl)
                return
            }
            const wordEl = e.target.closest('[data-cluster-id]')
            if (wordEl) {
                this._suggestionPopupController.showCluster(wordEl.dataset.clusterId, wordEl)
            }
        })

        // Dismiss popup when tapping anywhere that is not a suggestion word/marker or the popup itself.
        this.shadowRoot.addEventListener('click', (e) => {
            if (!this._suggestionPopupController) return
            if (!e.target.closest('[data-cluster-id]') && !e.target.closest('[data-connective-id]')
                && !e.target.closest('.suggestion-popup')) {
                this._suggestionPopupController._hideNow()
            }
        })

        // The shadowRoot listener never sees clicks outside the component
        // (e.g. on the page background), so those left the popup stuck open.
        if (this._docClickHandler) {
            document.removeEventListener('click', this._docClickHandler)
        }
        this._docClickHandler = (e) => {
            if (!this._suggestionPopupController) return
            if (e.composedPath().includes(this)) return
            this._suggestionPopupController._hideNow()
        }
        document.addEventListener('click', this._docClickHandler)
    }

    updateSuggestionStatus(suggestionId, status) {
        const suggestion = this._editorController.getSuggestion(suggestionId)
        if (!suggestion) return

        if (suggestion.type === 'connective') {
            this._updateConnectiveStatus(suggestion, status)
            return
        }

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
        const origTokens = wordEls.map(el => el.textContent)
        const origBare = origTokens.map(stripToken)

        // Collect all change regions from all accepted suggestions
        const allRegions = []
        for (const suggestion of acceptedSuggestions) {
            const sugTokens = suggestionTokens(suggestion.suggested_text)
            const sugBare = sugTokens.map(stripToken)

            const regions = computeWordDiff(origBare, sugBare, sugTokens, origTokens)
            for (const region of regions) {
                region.suggestion = suggestion
            }
            allRegions.push(...regions)
        }

        // Sort right-to-left by insertion point
        allRegions.sort((a, b) => b.insertBeforeIdx - a.insertBeforeIdx)

        for (const region of allRegions) {
            // Regions from different accepted suggestions can interleave: an
            // earlier-applied region may have removed this region's anchor
            // word (insertBefore on a detached node throws). Walk forward to
            // the first still-attached word.
            let insertBefore = null
            for (let idx = region.insertBeforeIdx; idx < wordEls.length; idx++) {
                if (wordEls[idx].isConnected) {
                    insertBefore = wordEls[idx]
                    break
                }
            }
            if (!insertBefore) {
                insertBefore = sentenceEl.querySelector('.sent-end-group')
            }

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

                // Preserve capitalization only at the sentence start: if the
                // FIRST word of the sentence is being replaced and it was
                // uppercase, capitalize the first new word to match. Applying
                // this mid-sentence would re-uppercase a word the LLM
                // deliberately lowercased (Henk: stray "Onze" in zin 8); the
                // real sentence start is fixed up again at the end of this method.
                const firstOrigLetter = firstOrig.replace(/^[("'\u201c]+/, '').charAt(0)
                if (region.origIndices[0] === 0 && firstOrigLetter && firstOrigLetter === firstOrigLetter.toUpperCase() && firstOrigLetter !== firstOrigLetter.toLowerCase()) {
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

        // A region may have deleted the original sentence-initial words,
        // promoting a mid-sentence word to the front (e.g. "het Openbaar
        // Ministerie ..."). Make sure the sentence starts with a capital.
        const firstWord = sentenceEl.querySelector('.word')
        if (firstWord) {
            firstWord.textContent = capitalizeToken(firstWord.textContent)
        }
    }

    /**
     * Apply/undo a connective merge in the DOM. On accept, the first sentence
     * is rebuilt to show the merged text and the second (absorbed) sentence is
     * hidden. On ignore/reset, both sentences are re-rendered from current
     * editor state (which drops the marker if ignored, restores it if pending).
     * Kept separate from the per-sentence diff path, which can't span sentences.
     */
    _updateConnectiveStatus(suggestion, status) {
        const m = suggestion.merges_sentences || []
        if (m.length < 2) return
        const first = m[0], second = m[m.length - 1]

        if (status === 'accepted') {
            this.shadowRoot
                .querySelectorAll(`[data-sentence-index="${second}"]`)
                .forEach(el => el.classList.add('connective-absorbed'))
            this._renderMergedSentence(first, suggestion)
        } else {
            this._replaceSentenceEl(first)
            this.shadowRoot
                .querySelectorAll(`[data-sentence-index="${second}"][data-split-part]`)
                .forEach(el => el.remove())
            this.shadowRoot
                .querySelectorAll(`[data-sentence-index="${second}"]`)
                .forEach(el => el.classList.remove('connective-absorbed'))
            this._replaceSentenceEl(second)
        }
    }

    /** Reconcile every connective's on-screen state with the editor state.
     *  Handles cross-sentence side effects that the single dispatched change
     *  can't reach: a connective auto-ignored by accepting a competing rewrite
     *  (drop its chip / undo its merged view) or revived by an undo (restore its
     *  chip). A connective the user is directly acting on is already handled by
     *  _updateConnectiveStatus, so those cases are no-ops here. */
    _refreshConnectiveMarkers() {
        if (!this._editorController) return
        const ec = this._editorController
        for (const c of ec.connectiveSuggestions) {
            const state = ec.getState(c.id)
            const m = c.merges_sentences || []
            const second = m.length >= 2 ? m[m.length - 1] : null
            if (second == null) continue

            const secondAbsorbed = this.shadowRoot.querySelector(
                `[data-sentence-index="${second}"].connective-absorbed`)

            if (state === 'accepted') {
                // Ensure the merged view exists (normally applied on its own
                // accept; restore if a revive/relayout dropped it).
                if (!secondAbsorbed) this._updateConnectiveStatus(c, 'accepted')
                continue
            }

            // pending / ignored must NOT be in the merged view.
            if (secondAbsorbed) {
                this._updateConnectiveStatus(c, state)  // restores both sentences
                continue
            }
            const chip = this.shadowRoot.querySelector(
                `.connective-marker[data-connective-id="${c.id}"]`)
            if (state === 'pending' && !chip) {
                this._replaceSentenceEl(second)          // bring the chip back
            } else if (state === 'ignored' && chip) {
                chip.remove()
            }
        }
    }

    /** Replace a sentence element in place with a fresh render reflecting the
     *  current editor state (used to restore after an undone merge). */
    _replaceSentenceEl(idx) {
        const el = this.shadowRoot.querySelector(
            `[data-sentence-index="${idx}"]:not([data-split-part])`)
        if (!el) return
        const tmp = document.createElement('span')
        tmp.innerHTML = this.renderSentence(this._data.sentences[idx], idx)
        const fresh = tmp.firstElementChild
        if (fresh) el.replaceWith(fresh)
    }

    /** Rebuild the first sentence's element to display the merged sentence text.
     *  Every word carries data-connective-id so the whole merged sentence stays
     *  clickable (to reach the popup and undo); only the inserted connective is
     *  highlighted as the change, and it anchors the score flash. The level
     *  badge is refined by updateSentenceScore. */
    _renderMergedSentence(idx, suggestion) {
        const el = this.shadowRoot.querySelector(
            `[data-sentence-index="${idx}"]:not([data-split-part])`)
        if (!el) return
        const text = this._editorController._composedMergeText
            ? this._editorController._composedMergeText(suggestion)
            : suggestion.suggested_text
        const tokens = text.trim().split(/\s+/).filter(Boolean)
        const strip = t => t.replace(/[.,;:!?()"'“”‘’]/g, '').toLowerCase()
        const connWord = this._connectiveWord(suggestion)
        let marked = false
        const wordsHtml = tokens.map(t => {
            const isConn = !marked && strip(t) === connWord
            if (isConn) marked = true
            const cls = isConn ? 'word suggestion-changed' : 'word'
            const statusAttr = isConn ? ' data-suggestion-status="accepted"' : ''
            return `<span class="${cls}" data-connective-id="${suggestion.id}"` +
                ` data-suggestion-id="${suggestion.id}"${statusAttr}>${this._escapeHtml(t)}</span>`
        }).join(' ')
        const level = this._data.sentences[idx]?.difficulty_level ?? '?'
        el.innerHTML =
            `<span class="sent-start-group"><span class="sent-idx">${idx + 1}</span>` +
            `<span class="sent-start"></span></span>${wordsHtml}` +
            `<span class="sent-end-group"><span class="sent-end"></span>` +
            `<span class="level-badge">${level}</span></span>`
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

    _fmtScore(v) {
        return v != null ? v.toFixed(1).replace('.', ',') : '—'
    }

    renderDocumentScores() {
        const origScore = this._data.document_lint_score
        const origLevel = this._data.document_difficulty_level
        let curScore = origScore, curLevel = origLevel
        if (this._editorController) {
            const u = this._editorController.computeUpdatedScore()
            if (u.score != null) curScore = u.score
            if (u.level != null) curLevel = u.level
        }
        const scoreChanged = curScore != null && origScore != null && Math.abs(curScore - origScore) >= 0.05

        return `<dl class="document-scores">
            <div class="doc-stat">
                <dt>zinnen</dt>
                <dd>${this._data.sentences.length}</dd>
            </div>
            <div class="doc-stat">
                <dt>lint score</dt>
                <dd>
                    <span class="lint-score-value">${this._fmtScore(curScore)}</span>
                    <span class="lint-score-orig"${scoreChanged ? '' : ' hidden'}>was ${this._fmtScore(origScore)}</span>
                </dd>
            </div>
            ${this.renderDocumentLevel(curLevel, origLevel)}
        </dl>`
    }

    renderDocumentLevel(curLevel = this._data.document_difficulty_level, origLevel = this._data.document_difficulty_level) {
        const changed = curLevel != null && origLevel != null && curLevel !== origLevel
        return `<div data-level="${curLevel}">
            <span class="level-badge">${curLevel}</span>
            <span class="level-orig"${changed ? '' : ' hidden'}>was ${origLevel}</span>
        </div>`
    }

    /** 2x2 legend of the four LiNT levels with their colours and a short label. */
    renderLevelLegend() {
        const levels = [[1, "Makkelijk"], [2, "Gemiddeld"], [3, "Moeilijk"], [4, "Zeer moeilijk"]]
        return `<div class="level-legend" aria-label="LiNT-niveaus">
            ${levels.map(([n, label]) => `<div class="legend-item">
                <span class="legend-badge" style="background-color: var(--color-level-${n})">${n}</span>
                <span class="legend-label">${label}</span>
            </div>`).join("")}
        </div>`
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
            ${this._connectiveMarker(idx)}
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
        const excludedTip = "Dit tekstblok wordt niet geanalyseerd of aangepast."
        return blocks.map(block => {
            if (block.type === "sentence") {
                const idx = block.sentence_index
                return this.renderSentence(this._data.sentences[idx], idx)
            }
            if (block.type === "list_item") {
                const marker = block.ordered ? `${block.number}.` : "•"
                const inner = block.sentence_indices
                    .map(i => this.renderSentence(this._data.sentences[i], i)).join("")
                return `<div class="doc-list-item"><span class="list-marker">${this._escapeHtml(marker)}</span>${inner}</div>`
            }
            if (block.type === "quote") {
                return `<div class="doc-quote" data-tip="${excludedTip}" aria-label="${excludedTip}">${this._escapeHtml(block.text)}</div>`
            }
            if (block.type === "heading") {
                return `<div class="doc-heading" data-tip="${excludedTip}" aria-label="${excludedTip}">${this._escapeHtml(block.text)}</div>`
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

    /**
     * A pending connective merge is rendered as a small clickable chip at the
     * start of the SECOND sentence ("↰ want"), where the join would appear.
     * Only shown while pending; on accept the two sentences are visually merged
     * (see _updateConnectiveStatus), on ignore the chip is dropped.
     */
    _connectiveMarker(sentenceIdx) {
        if (!this.isEditorMode || !this._editorController) return ''
        const c = this._editorController.getConnectiveForSecondSentence(sentenceIdx)
        if (!c || this._editorController.getState(c.id) !== 'pending') return ''
        const word = this._connectiveWord(c)
        return `<span class="connective-marker" data-connective-id="${c.id}"` +
            ` data-suggestion-status="pending"` +
            ` title="Verbind met de vorige zin met &quot;${this._escapeHtml(word)}&quot;">` +
            `↰&nbsp;${this._escapeHtml(word)}</span>`
    }

    /** The connective the merge inserts: the first suggested word that is not in
     *  the original sentence pair. Falls back to the relation label. */
    _connectiveWord(suggestion) {
        const strip = t => t.replace(/[.,;:!?()"'“”‘’]/g, '').toLowerCase()
        const orig = new Set((suggestion.original_text || '').split(/\s+/).map(strip))
        for (const tok of (suggestion.suggested_text || '').split(/\s+/)) {
            const b = strip(tok)
            if (b && !orig.has(b)) return b
        }
        return suggestion.relation || 'verbind'
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
        // Vega is loaded by the host page; if it failed to load, degrade to a
        // message instead of throwing on every render.
        if (typeof vegaEmbed === "undefined") {
            const container = this.shadowRoot.querySelector('[data-view="stats"]')
            if (container) {
                container.innerHTML = '<p style="opacity:.7">Statistieken zijn niet beschikbaar (grafiekbibliotheek niet geladen).</p>'
            }
            return
        }
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
