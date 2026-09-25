/**
 * SuggestionPopupController manages the suggestion popup UI.
 *
 * Shows cluster suggestions with original text, suggested replacement,
 * explanation in Dutch, and Accept/Ignore buttons per suggestion.
 */
// Plain, transparent labels for rewrite variants (not the jargon
// "Behoudend"/"Volledig"). The conservative and intermediate variants carry a
// sentence-count contract the backend enforces, so their labels are fixed; the
// full variant is labelled by its actual count ("Drie zinnen"), which also
// keeps it honest when it collapses to two sentences. Falls back to the fixed
// labels when a result predates the n_sentences metric.
const FIXED_VARIANT_LABELS = {
    conservative: 'Eén zin, niet gesplitst',
    intermediate: 'Twee zinnen',
    full: 'Opgesplitst',
    // A single model rewrite becomes a choice once the user edits it.
    suggestion: 'Suggestie',
    edited: 'Eigen versie',
}
const COUNT_WORDS = ['', 'Eén', 'Twee', 'Drie', 'Vier', 'Vijf', 'Zes', 'Zeven', 'Acht']

function sentenceCountLabel(n) {
    const word = COUNT_WORDS[n] || String(n)
    return n === 1 ? `${word} zin` : `${word} zinnen`
}

const CONTRACT_COUNTS = { conservative: 1, intermediate: 2 }

export function variantLabels(variants) {
    return variants.map(v => {
        const n = v.new_sentence_metrics?.n_sentences
        if (v.key !== 'full' || !Number.isInteger(n) || n < 1) {
            return FIXED_VARIANT_LABELS[v.key] || v.label || ''
        }
        // A full variant with the same count as another offered variant is a
        // different rewrite of that length; say so, or the two read as duplicates.
        const clash = variants.some(o => o !== v && CONTRACT_COUNTS[o.key] === n)
        return clash ? `${sentenceCountLabel(n)}, verder herschreven` : sentenceCountLabel(n)
    })
}

const fmtScore = v => v.toFixed(1).replace('.', ',')

/**
 * The popup's one-line summary of how an accepted change moved its sentence's
 * LiNT score: { text, trend } with trend 'easier' | 'harder' | 'same' |
 * 'scoring' (an edit whose score is still being computed) | 'failed'. A lower score is
 * easier to read. Returns null when the sentence has no score.
 */
export function sentenceScoreText({ before, after, scoring, failed, reason }) {
    if (failed) {
        return {
            text: `Deze zin: score kon niet worden berekend${reason ? ` (${reason})` : ''}; `
                + 'de score telt de oorspronkelijke zin nog.',
            trend: 'failed',
        }
    }
    if (scoring) return { text: 'Deze zin: score wordt berekend…', trend: 'scoring' }
    if (before?.score == null || after?.score == null) return null
    const delta = after.score - before.score
    if (Math.abs(delta) < 0.05) {
        return { text: `Deze zin: ${fmtScore(after.score)}, geen merkbare verandering`, trend: 'same' }
    }
    const levels = before.level != null && after.level != null && before.level !== after.level
        ? ` · niveau ${before.level} → ${after.level}` : ''
    return {
        text: `Deze zin: ${fmtScore(before.score)} → ${fmtScore(after.score)} `
            + `(${delta < 0 ? 'makkelijker' : 'moeilijker'})${levels}`,
        trend: delta < 0 ? 'easier' : 'harder',
    }
}

export class SuggestionPopupController {
    constructor(popupElement, editorController) {
        this._popup = popupElement
        this._editor = editorController
        this._currentClusterId = null
        this._hideTimeout = null
        // { suggestionId } while the user is editing a suggestion's text. The
        // popup is pinned meanwhile: hover-out, other words and outside clicks
        // must not throw away what they typed.
        this._editing = null

        this._setupEventListeners()
    }

    _setupEventListeners() {
        // Handle button clicks within the popup
        this._popup.addEventListener('click', (e) => {
            const button = e.target.closest('button')
            if (!button) return

            const suggestionId = button.dataset.suggestionId
            if (!suggestionId) return

            // Edit actions first: the apply/cancel buttons also carry the
            // accept/ignore classes for their styling. Both re-render the popup,
            // detaching the clicked button; the page's outside-click handler
            // would then see a target outside the popup and close it, so the
            // click must not reach it.
            if (button.classList.contains('edit-btn')) {
                e.stopPropagation()
                this._startEditing(suggestionId, button.dataset.variantKey)
                return
            }
            if (button.classList.contains('edit-apply-btn')) {
                e.stopPropagation()
                this._applyEdit(suggestionId)
                return
            }
            if (button.classList.contains('edit-cancel-btn')) {
                this._stopEditing()
                this._hideNow()
                return
            }

            if (button.classList.contains('accept-btn')) {
                // A variant rewrite: pick the chosen alternative, then accept.
                if (button.dataset.variantKey) {
                    this._editor.chooseVariant(suggestionId, button.dataset.variantKey)
                }
                this._editor.accept(suggestionId)
                this._hideNow()
            } else if (button.classList.contains('ignore-btn')) {
                this._editor.ignore(suggestionId)
                this._hideNow()
            } else if (button.classList.contains('reset-btn')) {
                this._editor.reset(suggestionId)
                this._hideNow()
            }
        })

        // In the edit box: Escape cancels, Ctrl/Cmd+Enter applies.
        this._popup.addEventListener('keydown', (e) => {
            if (!this._editing || !e.target.classList?.contains('edit-text')) return
            if (e.key === 'Escape') {
                e.preventDefault()
                this._stopEditing()
                this._hideNow()
            } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault()
                this._applyEdit(this._editing.suggestionId)
            }
        })

        // Keep popup alive while hovered; schedule hide on leave
        this._popup.addEventListener('mouseenter', () => {
            this._cancelHide()
            this._popup.dataset.hovered = 'true'
        })
        this._popup.addEventListener('mouseleave', () => {
            delete this._popup.dataset.hovered
            this.hide()
        })
    }

    /**
     * Show popup for all suggestions in a cluster.
     * Cancels any pending hide so the popup stays visible.
     */
    showCluster(clusterId, targetElement) {
        if (this._editing) return
        this._cancelHide()

        const suggestions = this._editor.getClusterSuggestions(clusterId)
        if (!suggestions.length) return

        this._currentClusterId = clusterId

        this._popup.innerHTML = this._renderClusterContent(suggestions)
        this._popup.classList.add('visible')

        // Position below the target element
        const rect = targetElement.getBoundingClientRect()
        this._popup.style.top = `${rect.bottom + 8}px`
        this._popup.style.left = `${rect.left}px`

        // Ensure popup stays within viewport
        const popupRect = this._popup.getBoundingClientRect()
        if (popupRect.right > window.innerWidth - 16) {
            this._popup.style.left = `${window.innerWidth - popupRect.width - 16}px`
        }
        if (popupRect.bottom > window.innerHeight - 16) {
            // Flip above the target, but never past the top edge — a tall
            // multi-alternative popup scrolls internally (max-height in CSS).
            this._popup.style.top = `${Math.max(8, rect.top - popupRect.height - 8)}px`
        }
    }

    /**
     * Show the popup for a connective (coherence) suggestion. Connectives are
     * not clustered — they merge two whole sentences — so they get their own
     * entry point and a dedicated render that shows the sentence pair and the
     * merged result directly (no per-sentence LCS relocation).
     */
    showConnective(suggestionId, targetElement) {
        if (this._editing) return
        this._cancelHide()
        const suggestion = this._editor.getSuggestion(suggestionId)
        if (!suggestion) return

        this._currentClusterId = null
        this._popup.innerHTML = this._renderConnectiveSuggestion(suggestion)
        this._popup.classList.add('visible')

        const rect = targetElement.getBoundingClientRect()
        this._popup.style.top = `${rect.bottom + 8}px`
        this._popup.style.left = `${rect.left}px`

        const popupRect = this._popup.getBoundingClientRect()
        if (popupRect.right > window.innerWidth - 16) {
            this._popup.style.left = `${window.innerWidth - popupRect.width - 16}px`
        }
        if (popupRect.bottom > window.innerHeight - 16) {
            this._popup.style.top = `${Math.max(8, rect.top - popupRect.height - 8)}px`
        }
    }

    _renderConnectiveSuggestion(suggestion) {
        const status = this._editor.getState(suggestion.id)
        const { statusHTML, buttonsHTML } = this._statusAndButtons(suggestion.id, status)
        const relation = suggestion.relation
            ? `<span class="suggestion-category">${this._escapeHtml(this._relationLabel(suggestion.relation))}</span>`
            : ''
        // Reflect an already-accepted first-sentence rewrite in both strings, so
        // the popup shows the current first clause, not the stale original.
        const before = this._editor.connectiveOriginalPair
            ? this._editor.connectiveOriginalPair(suggestion) : suggestion.original_text
        const after = this._editor._composedMergeText
            ? this._editor._composedMergeText(suggestion) : suggestion.suggested_text
        const { origHtml, sugHtml } = this._renderDiff(before, after)

        return `
            <div class="suggestion-popup-content">
                <div class="suggestion-header">
                    <span class="suggestion-type">Verbindingswoord</span>
                    ${relation}
                    ${statusHTML}
                </div>

                <div class="suggestion-comparison">
                    <div class="original">
                        <span class="label">Twee zinnen:</span>
                        <span class="text">${origHtml}</span>
                    </div>
                    <div class="suggested">
                        <span class="label">Samengevoegd:</span>
                        <span class="text">${sugHtml}</span>
                    </div>
                </div>

                ${suggestion.explanation ? `
                    <div class="suggestion-explanation">
                        <span class="label">Uitleg:</span>
                        <span class="text">${this._escapeHtml(this._stripBrackets(suggestion.explanation))}</span>
                    </div>
                ` : ''}

                ${this._suggestionNote(suggestion)}

                <div class="suggestion-actions">
                    ${buttonsHTML}
                </div>
            </div>
        `
    }

    /**
     * Show the popup for an enumeration (bullet-list) suggestion. Like a
     * connective it is block-level, not clustered, so it gets its own entry
     * point and a render that previews the lead-in + bullet items.
     */
    showEnumeration(suggestionId, targetElement) {
        if (this._editing) return
        this._cancelHide()
        const suggestion = this._editor.getSuggestion(suggestionId)
        if (!suggestion) return

        this._currentClusterId = null
        this._popup.innerHTML = this._renderEnumerationSuggestion(suggestion)
        this._popup.classList.add('visible')

        const rect = targetElement.getBoundingClientRect()
        this._popup.style.top = `${rect.bottom + 8}px`
        this._popup.style.left = `${rect.left}px`

        const popupRect = this._popup.getBoundingClientRect()
        if (popupRect.right > window.innerWidth - 16) {
            this._popup.style.left = `${window.innerWidth - popupRect.width - 16}px`
        }
        if (popupRect.bottom > window.innerHeight - 16) {
            this._popup.style.top = `${Math.max(8, rect.top - popupRect.height - 8)}px`
        }
    }

    _renderEnumerationSuggestion(suggestion) {
        const status = this._editor.getState(suggestion.id)
        const { statusHTML, buttonsHTML } = this._statusAndButtons(suggestion.id, status)
        const intro = this._escapeHtml(suggestion.list_intro || '')
        const items = (suggestion.list_items || [])
            .map(it => `<li>${this._escapeHtml(it)}</li>`).join('')

        return `
            <div class="suggestion-popup-content">
                <div class="suggestion-header">
                    <span class="suggestion-type">Opsomming als lijst</span>
                    ${statusHTML}
                </div>

                <div class="suggestion-comparison">
                    <div class="original">
                        <span class="label">Origineel:</span>
                        <span class="text">${this._escapeHtml(suggestion.original_text)}</span>
                    </div>
                    <div class="suggested">
                        <span class="label">Als lijst:</span>
                        <span class="text">
                            <span class="enum-preview-intro">${intro}</span>
                            <ul class="enum-preview">${items}</ul>
                        </span>
                    </div>
                </div>

                ${suggestion.explanation ? `
                    <div class="suggestion-explanation">
                        <span class="label">Uitleg:</span>
                        <span class="text">${this._escapeHtml(this._stripBrackets(suggestion.explanation))}</span>
                    </div>
                ` : ''}

                <div class="suggestion-note">
                    Let op: een opsomming als lijst verandert de zinslengte, dus de LiNT-score kan iets stijgen of dalen; de lezer overziet de losse punten wel makkelijker.
                </div>

                <div class="suggestion-actions">
                    ${buttonsHTML}
                </div>
            </div>
        `
    }

    /**
     * Backwards-compatible: show popup for a single suggestion ID.
     */
    show(suggestionId, targetElement) {
        const cluster = this._editor.getClusterForSuggestion(suggestionId)
        if (cluster) {
            this.showCluster(cluster.id, targetElement)
        }
    }

    /**
     * Schedule hiding the popup after a short delay.
     * The delay allows the user to move from the word to the popup
     * (or between cluster words) without the popup disappearing.
     */
    hide() {
        if (this._editing) return
        if (this._hideTimeout) return // already scheduled
        this._hideTimeout = setTimeout(() => {
            this._hideTimeout = null
            if (!this._popup.dataset.hovered) {
                this._popup.classList.remove('visible')
                this._currentClusterId = null
            }
        }, 300)
    }

    /** Cancel a pending scheduled hide. */
    _cancelHide() {
        if (this._hideTimeout) {
            clearTimeout(this._hideTimeout)
            this._hideTimeout = null
        }
    }

    /** Hide immediately (used after button actions and outside clicks). Does
     *  nothing while editing: only Toepassen or Annuleren ends an edit. */
    _hideNow() {
        if (this._editing) return
        this._cancelHide()
        delete this._popup.dataset.hovered
        this._popup.classList.remove('visible')
        this._currentClusterId = null
    }

    /**
     * Replace the popup content with an edit box holding the chosen rewrite
     * (or the variant whose "Bewerk" was clicked).
     */
    _startEditing(suggestionId, variantKey) {
        const s = this._editor.getSuggestion(suggestionId)
        if (!s || !this._editor.constructor.isEditable?.(s)) return
        const v = variantKey ? (s.variants || []).find(x => x.key === variantKey) : null
        const startText = v ? v.suggested_text : s.suggested_text
        this._cancelHide()
        this._editing = { suggestionId }
        this._popup.innerHTML = this._renderEditPanel(s, startText)
        const ta = this._popup.querySelector('.edit-text')
        if (ta) {
            ta.focus()
            ta.setSelectionRange(ta.value.length, ta.value.length)
        }
    }

    _applyEdit(suggestionId) {
        const ta = this._popup.querySelector('.edit-text')
        if (!ta) return
        if (!this._editor.setEditedText(suggestionId, ta.value)) {
            const msg = this._popup.querySelector('.edit-message')
            if (msg) {
                msg.textContent = ta.value.trim()
                    ? 'Je versie is gelijk aan de huidige zin; pas de tekst eerst aan.'
                    : 'Vul eerst een tekst in.'
                msg.hidden = false
            }
            return
        }
        this._stopEditing()
        this._editor.accept(suggestionId)
        this._showResult(suggestionId)
    }

    /** After applying an edit, keep the popup open on the accepted suggestion
     *  so its sentence-score line is visible (on a phone the document score is
     *  usually scrolled out of view). refreshFor updates it once the edit's
     *  score arrives. */
    _showResult(suggestionId) {
        const cluster = this._editor.getClusterForSuggestion(suggestionId)
        if (!cluster) {
            this._hideNow()
            return
        }
        this._currentClusterId = cluster.id
        this._popup.innerHTML = this._renderClusterContent(this._editor.getClusterSuggestions(cluster.id))
        this._popup.scrollTop = 0
    }

    /** Re-render the open popup when a suggestion it shows has changed (for
     *  instance its score arrived). Never while the user is editing. */
    refreshFor(suggestionId) {
        if (this._editing || this._currentClusterId == null) return
        if (!this._popup.classList.contains('visible')) return
        const shown = this._editor.getClusterSuggestions(this._currentClusterId)
        if (!shown.some(s => s.id === suggestionId)) return
        this._popup.innerHTML = this._renderClusterContent(shown)
    }

    /** The sentence-score line, directly under an accepted suggestion's header:
     *  a phone shows only the top of a long popup, which scrolls internally. */
    _sentenceScoreLine(suggestion, status) {
        if (status !== 'accepted' || !this._editor.getSentenceScoreChange) return ''
        const line = sentenceScoreText(this._editor.getSentenceScoreChange(suggestion.sentence_index))
        if (!line) return ''
        return `<div class="sentence-score sentence-score-${line.trend}">${this._escapeHtml(line.text)}</div>`
    }

    _stopEditing() {
        this._editing = null
    }

    _renderEditPanel(suggestion, startText) {
        const currentOriginal = this._editor.getCurrentOriginalForSuggestion(suggestion.id)
        const origLabel = currentOriginal !== suggestion.original_text ? 'Huidig:' : 'Origineel:'
        const id = suggestion.id
        return `
            <div class="suggestion-popup-content edit-panel">
                <div class="suggestion-header">
                    <span class="suggestion-type">Eigen versie</span>
                </div>
                <div class="suggestion-comparison">
                    <div class="original">
                        <span class="label">${origLabel}</span>
                        <span class="text">${this._escapeHtml(currentOriginal)}</span>
                    </div>
                </div>
                <textarea class="edit-text" rows="4" lang="nl" spellcheck="true"
                    aria-label="Eigen versie van de zin">${this._escapeHtml(startText)}</textarea>
                <div class="edit-message" role="alert" hidden></div>
                <div class="suggestion-note">
                    Je eigen versie wordt niet door het taalmodel gecontroleerd. De score wordt opnieuw berekend zodra je hem toepast.
                </div>
                <div class="suggestion-actions">
                    <button class="accept-btn edit-apply-btn" data-suggestion-id="${id}" title="Toepassen (Ctrl+Enter)">Toepassen</button>
                    <button class="ignore-btn edit-cancel-btn" data-suggestion-id="${id}" title="Annuleren (Esc)">Annuleren</button>
                </div>
            </div>`
    }

    /** A "Bewerk" button, only on editable suggestions that are still pending. */
    _editButton(suggestion, status, variantKey = null) {
        if (status !== 'pending' || !this._editor.constructor.isEditable?.(suggestion)) return ''
        const keyAttr = variantKey ? ` data-variant-key="${variantKey}"` : ''
        return `<button class="edit-btn" data-suggestion-id="${suggestion.id}"${keyAttr} title="Deze tekst zelf aanpassen">Bewerk</button>`
    }


    /**
     * Render popup content for a cluster of suggestions.
     */
    _renderClusterContent(suggestions) {
        if (suggestions.length === 1) {
            return this._renderSingleSuggestion(suggestions[0])
        }

        const sections = suggestions.map((s, i) =>
            this._renderSuggestionSection(s, i + 1, suggestions.length)
        ).join('')

        return `<div class="suggestion-popup-content">${sections}</div>`
    }

    /**
     * Render a single suggestion (no cluster header).
     */
    _renderSingleSuggestion(suggestion) {
        if (this._editor.constructor.hasVariants?.(suggestion)) {
            return this._renderVariantChoice(suggestion)
        }
        const status = this._editor.getState(suggestion.id)
        const typeLabel = this._suggestionLabel(suggestion)
        const categoryLabel = suggestion.error_category ? this._errorCategoryLabel(suggestion.error_category) : null
        const { statusHTML, buttonsHTML } = this._statusAndButtons(suggestion.id, status, suggestion)
        const currentOriginal = this._editor.getCurrentOriginalForSuggestion(suggestion.id)
        const origLabel = currentOriginal !== suggestion.original_text ? 'Huidig:' : 'Origineel:'
        const { origHtml, sugHtml } = this._renderDiff(currentOriginal, suggestion.suggested_text)

        return `
            <div class="suggestion-popup-content">
                <div class="suggestion-header">
                    <span class="suggestion-type">${typeLabel}</span>
                    ${categoryLabel ? `<span class="suggestion-category">${categoryLabel}</span>` : ''}
                    ${statusHTML}
                </div>
                ${this._sentenceScoreLine(suggestion, status)}

                <div class="suggestion-comparison">
                    <div class="original">
                        <span class="label">${origLabel}</span>
                        <span class="text">${origHtml}</span>
                    </div>
                    <div class="suggested">
                        <span class="label">Suggestie:</span>
                        <span class="text">${sugHtml}</span>
                    </div>
                </div>

                ${suggestion.explanation ? `
                    <div class="suggestion-explanation">
                        <span class="label">Uitleg:</span>
                        <span class="text">${this._escapeHtml(this._stripBrackets(suggestion.explanation))}</span>
                    </div>
                ` : ''}

                ${this._suggestionNote(suggestion)}

                <div class="suggestion-actions">
                    ${buttonsHTML}
                </div>
            </div>
        `
    }

    /**
     * Render a sentence rewrite offered as a choice between a conservative
     * (one-sentence), an intermediate (two-sentence) and a full (possibly split)
     * variant — whichever of them survived. Each variant shows its
     * own diff and a "Kies deze" button; the whole suggestion has one Negeren /
     * Ongedaan maken. Choosing a variant points the suggestion at that rewrite.
     */
    _renderVariantChoice(suggestion) {
        const status = this._editor.getState(suggestion.id)
        const typeLabel = this._suggestionLabel(suggestion)
        const currentOriginal = this._editor.getCurrentOriginalForSuggestion(suggestion.id)
        const origLabel = currentOriginal !== suggestion.original_text ? 'Huidig:' : 'Origineel:'
        const chosen = this._editor.getChosenVariantKey(suggestion.id)
        const labels = variantLabels(suggestion.variants)

        const variantsHtml = suggestion.variants.map((v, i) => {
            const { sugHtml } = this._renderDiff(currentOriginal, v.suggested_text)
            const isChosen = v.key === chosen
            // Only mark a variant as "chosen" once accepted; while pending both
            // options are equal (styled the same, emphasised on hover).
            const markChosen = isChosen && status === 'accepted'
            const action = status === 'pending'
                ? `<button class="accept-btn" data-suggestion-id="${suggestion.id}" data-variant-key="${v.key}" title="Deze herschrijving kiezen">Kies deze</button>
                   ${this._editButton(suggestion, status, v.key)}`
                : (isChosen ? '<span class="status-badge accepted">Gekozen</span>' : '')
            return `
                <div class="variant${markChosen ? ' variant-chosen' : ''}">
                    <div class="variant-head">
                        <span class="variant-label">${this._escapeHtml(labels[i])}</span>
                    </div>
                    <span class="text">${sugHtml}</span>
                    <div class="variant-action">${action}</div>
                </div>`
        }).join('')

        const statusHTML = status === 'accepted'
            ? '<span class="status-badge accepted">Geaccepteerd</span>'
            : status === 'ignored'
                ? '<span class="status-badge ignored">Genegeerd</span>'
                : '<span class="status-badge pending">In behandeling</span>'
        const footer = status === 'pending'
            ? `<button class="ignore-btn" data-suggestion-id="${suggestion.id}" title="Suggestie negeren">Negeren</button>`
            : `<button class="reset-btn" data-suggestion-id="${suggestion.id}" title="Ongedaan maken">Ongedaan maken</button>`

        return `
            <div class="suggestion-popup-content">
                <div class="suggestion-header">
                    <span class="suggestion-type">${typeLabel}</span>
                    ${statusHTML}
                </div>
                ${this._sentenceScoreLine(suggestion, status)}
                <div class="suggestion-comparison">
                    <div class="original">
                        <span class="label">${origLabel}</span>
                        <span class="text">${this._escapeHtml(currentOriginal)}</span>
                    </div>
                </div>
                <div class="variant-choice">
                    <div class="variant-choice-label">Kies een herschrijving:</div>
                    ${variantsHtml}
                </div>
                ${suggestion.explanation ? `
                    <div class="suggestion-explanation">
                        <span class="label">Uitleg:</span>
                        <span class="text">${this._escapeHtml(this._stripBrackets(suggestion.explanation))}</span>
                    </div>` : ''}
                <div class="suggestion-actions">${footer}</div>
            </div>`
    }

    /**
     * Render one section within a multi-suggestion cluster popup.
     */
    _renderSuggestionSection(suggestion, index, total) {
        if (this._editor.constructor.hasVariants?.(suggestion)) {
            return this._renderVariantChoice(suggestion)
        }
        const status = this._editor.getState(suggestion.id)
        const typeLabel = this._suggestionLabel(suggestion)
        const categoryLabel = suggestion.error_category ? this._errorCategoryLabel(suggestion.error_category) : null
        const { statusHTML, buttonsHTML } = this._statusAndButtons(suggestion.id, status, suggestion)
        const currentOriginal = this._editor.getCurrentOriginalForSuggestion(suggestion.id)
        const origLabel = currentOriginal !== suggestion.original_text ? 'Huidig:' : 'Origineel:'
        const { origHtml, sugHtml } = this._renderDiff(currentOriginal, suggestion.suggested_text)

        return `
            <div class="suggestion-section">
                <div class="suggestion-header">
                    <span class="suggestion-type">${typeLabel} (${index}/${total})</span>
                    ${categoryLabel ? `<span class="suggestion-category">${categoryLabel}</span>` : ''}
                    ${statusHTML}
                </div>
                ${this._sentenceScoreLine(suggestion, status)}

                <div class="suggestion-comparison">
                    <div class="original">
                        <span class="label">${origLabel}</span>
                        <span class="text">${origHtml}</span>
                    </div>
                    <div class="suggested">
                        <span class="label">Suggestie:</span>
                        <span class="text">${sugHtml}</span>
                    </div>
                </div>

                ${suggestion.explanation ? `
                    <div class="suggestion-explanation">
                        <span class="label">Uitleg:</span>
                        <span class="text">${this._escapeHtml(this._stripBrackets(suggestion.explanation))}</span>
                    </div>
                ` : ''}

                ${this._suggestionNote(suggestion)}

                <div class="suggestion-actions">
                    ${buttonsHTML}
                </div>
            </div>
        `
    }

    /**
     * Heading label for a suggestion. A consolidated sentence_rewrite names the
     * underlying LiNT signals it addressed (from component_types) rather than
     * the generic "Zinsverbetering", so the user can see which measured feature
     * each rewrite targets.
     */
    _suggestionLabel(suggestion) {
        if (suggestion.type === 'sentence_rewrite'
            && Array.isArray(suggestion.component_types)
            && suggestion.component_types.length) {
            const signals = suggestion.component_types
                .map(t => this._typeLabel(t))
                .join(', ')
            return `Zinsverbetering: ${signals}`
        }
        return this._typeLabel(suggestion.type)
    }

    /**
     * Optional advisory note under a suggestion. For a pure passive-voice
     * suggestion, a passive can be a deliberate choice, so nudge the user to
     * decide rather than accept blindly (Henk Pander Maat, zin 13).
     */
    _suggestionNote(suggestion) {
        if (suggestion.type === 'passive') {
            return `
                <div class="suggestion-note">
                    Let op: een passieve vorm kan een bewuste keuze zijn — beoordeel zelf of herschrijven hier nodig is.
                </div>`
        }
        if (suggestion.type === 'connective') {
            return `
                <div class="suggestion-note">
                    Let op: het verband tussen de zinnen is een interpretatie — beoordeel zelf of het klopt voordat je samenvoegt.
                    Samenvoegen maakt de zin langer, waardoor de LiNT-score iets kan stijgen; een verbindingswoord verbetert de samenhang, en dat meet de LiNT-score niet.
                </div>`
        }
        if (suggestion.type === 'word_frequency'
            && typeof suggestion.replacement_word === 'string'
            && suggestion.replacement_word.trim().includes(' ')) {
            return `
                <div class="suggestion-note">
                    Let op: een lange samenstelling opsplitsen maakt de zin langer, waardoor de LiNT-score iets kan stijgen; toch wordt de tekst begrijpelijker doordat een zeldzaam woord plaatsmaakt voor gewone woorden.
                </div>`
        }
        return ''
    }

    _typeLabel(type) {
        const typeLabels = {
            'word_frequency': 'Weinig gebruikt woord',
            'max_sdl': 'Zinsstructuur',
            'content_words_per_clause': 'Informatiedichtheid',
            'abstract_nouns': 'Abstracte taal',
            'spelling': 'Spelling/grammatica',
            'passive': 'Passieve zin',
            'subordinate_clause': 'Bijzinsstructuur',
            'sentence_length': 'Lange zin',
            'sentence_rewrite': 'Zinsverbetering',
            'connective': 'Verbindingswoord',
            'enumeration': 'Opsomming als lijst',
        }
        return typeLabels[type] || type
    }

    _relationLabel(relation) {
        const labels = {
            'reden': 'reden', 'oorzaak': 'oorzaak', 'gevolg': 'gevolg',
            'tegenstelling': 'tegenstelling', 'toelichting': 'toelichting',
            'opsomming': 'opsomming',
        }
        return labels[relation] || relation
    }

    _errorCategoryLabel(category) {
        const labels = {
            'spelling': 'Spelfout',
            'grammar': 'Grammaticafout'
        }
        return labels[category] || null
    }

    _statusAndButtons(suggestionId, status, suggestion = null) {
        let statusHTML = ''
        let buttonsHTML = ''

        if (status === 'pending') {
            statusHTML = '<span class="status-badge pending">In behandeling</span>'
            buttonsHTML = `
                <button class="accept-btn" data-suggestion-id="${suggestionId}" title="Suggestie accepteren">Accepteren</button>
                ${suggestion ? this._editButton(suggestion, status) : ''}
                <button class="ignore-btn" data-suggestion-id="${suggestionId}" title="Suggestie negeren">Negeren</button>
            `
        } else if (status === 'accepted') {
            statusHTML = '<span class="status-badge accepted">Geaccepteerd</span>'
            buttonsHTML = `
                <button class="reset-btn" data-suggestion-id="${suggestionId}" title="Ongedaan maken">Ongedaan maken</button>
            `
        } else if (status === 'ignored') {
            statusHTML = '<span class="status-badge ignored">Genegeerd</span>'
            buttonsHTML = `
                <button class="reset-btn" data-suggestion-id="${suggestionId}" title="Ongedaan maken">Ongedaan maken</button>
            `
        }

        return { statusHTML, buttonsHTML }
    }

    /**
     * Compute a word-level diff between original and suggested text.
     * Returns two HTML strings: origHtml with <del> on removed words,
     * sugHtml with <ins> on added words; all other words plain.
     */
    _renderDiff(originalText, suggestedText) {
        const strip = t => t.replace(/[.,;:!?()"'“”‘’]/g, '').toLowerCase()
        const escape = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

        const origTokens = originalText.trim().split(/\s+/).filter(Boolean)
        const sugTokens = suggestedText.trim().split(/\s+/).filter(Boolean)
        const origBare = origTokens.map(strip)
        const sugBare = sugTokens.map(strip)

        const m = origBare.length, n = sugBare.length
        const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))
        for (let i = 1; i <= m; i++)
            for (let j = 1; j <= n; j++)
                dp[i][j] = origBare[i - 1] === sugBare[j - 1]
                    ? dp[i - 1][j - 1] + 1
                    : Math.max(dp[i - 1][j], dp[i][j - 1])

        const origStatus = new Array(m).fill('keep')
        const sugStatus = new Array(n).fill('keep')
        let i = m, j = n
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && origBare[i - 1] === sugBare[j - 1]) {
                i--; j--
            } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
                sugStatus[j - 1] = 'insert'; j--
            } else {
                origStatus[i - 1] = 'delete'; i--
            }
        }

        const origHtml = origTokens.map((t, idx) =>
            origStatus[idx] === 'delete' ? `<del>${escape(t)}</del>` : escape(t)
        ).join(' ')

        const sugHtml = sugTokens.map((t, idx) =>
            sugStatus[idx] === 'insert' ? `<ins>${escape(t)}</ins>` : escape(t)
        ).join(' ')

        return { origHtml, sugHtml }
    }

    _escapeHtml(text) {
        const div = document.createElement('div')
        div.textContent = text
        return div.innerHTML
    }

    /** The model sometimes echoes the prompt's placeholder brackets around
     *  its explanation; the server strips them for new analyses, this covers
     *  results that were cached before that fix. */
    _stripBrackets(text) {
        const t = (text || '').trim()
        if (t.length >= 2 && t.startsWith('[') && t.endsWith(']')) {
            return t.slice(1, -1).trim()
        }
        return t
    }
}
