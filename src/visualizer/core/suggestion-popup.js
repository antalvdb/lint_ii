/**
 * SuggestionPopupController manages the suggestion popup UI.
 *
 * Shows cluster suggestions with original text, suggested replacement,
 * explanation in Dutch, and Accept/Ignore buttons per suggestion.
 */
export class SuggestionPopupController {
    constructor(popupElement, editorController) {
        this._popup = popupElement
        this._editor = editorController
        this._currentClusterId = null
        this._hideTimeout = null

        this._setupEventListeners()
    }

    _setupEventListeners() {
        // Handle button clicks within the popup
        this._popup.addEventListener('click', (e) => {
            const button = e.target.closest('button')
            if (!button) return

            const suggestionId = button.dataset.suggestionId
            if (!suggestionId) return

            if (button.classList.contains('accept-btn')) {
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
            this._popup.style.top = `${rect.top - popupRect.height - 8}px`
        }
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

    /** Hide immediately (used after button actions). */
    _hideNow() {
        this._cancelHide()
        delete this._popup.dataset.hovered
        this._popup.classList.remove('visible')
        this._currentClusterId = null
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
        const status = this._editor.getState(suggestion.id)
        const typeLabel = this._typeLabel(suggestion.type)
        const categoryLabel = suggestion.error_category ? this._errorCategoryLabel(suggestion.error_category) : null
        const { statusHTML, buttonsHTML } = this._statusAndButtons(suggestion.id, status)
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

                <div class="suggestion-actions">
                    ${buttonsHTML}
                </div>
            </div>
        `
    }

    /**
     * Render one section within a multi-suggestion cluster popup.
     */
    _renderSuggestionSection(suggestion, index, total) {
        const status = this._editor.getState(suggestion.id)
        const typeLabel = this._typeLabel(suggestion.type)
        const categoryLabel = suggestion.error_category ? this._errorCategoryLabel(suggestion.error_category) : null
        const { statusHTML, buttonsHTML } = this._statusAndButtons(suggestion.id, status)
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

                <div class="suggestion-actions">
                    ${buttonsHTML}
                </div>
            </div>
        `
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
        }
        return typeLabels[type] || type
    }

    _errorCategoryLabel(category) {
        const labels = {
            'spelling': 'Spelfout',
            'grammar': 'Grammaticafout'
        }
        return labels[category] || null
    }

    _statusAndButtons(suggestionId, status) {
        let statusHTML = ''
        let buttonsHTML = ''

        if (status === 'pending') {
            statusHTML = '<span class="status-badge pending">In behandeling</span>'
            buttonsHTML = `
                <button class="accept-btn" data-suggestion-id="${suggestionId}" title="Suggestie accepteren">Accepteren</button>
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
