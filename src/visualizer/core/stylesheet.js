export const css = `
    :host {
        color-scheme: light dark;
        --background: light-dark(hsl(60, 100%, 98%), hsl(60, 3%, 7%));

        --color-concrete: hsl(60, 80%, 50%);
        --color-abstract: hsl(195, 53%, 79%);
        --color-undefined: hsl(350, 90%, 83%);
        --color-unknown: hsl(0, 20%, 90%);

        --color-level-1: hsl(153, 53%, 53%);
        --color-level-2: hsl(198, 100%, 70%);
        --color-level-3: hsl(42, 100%, 53%);
        --color-level-4: hsl(348, 100%, 70%);

        display: grid;
        grid-template-rows: auto auto;
        border-top: 1px solid currentColor;
        border-bottom: 1px solid currentColor;
    }
    header {
        --gap: 3em;
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid currentColor;
        gap: var(--gap);

        h1 {
            display: none;
            user-select: none;
            line-height: 1.5em;
            padding-left: .35em;
            letter-spacing: 0.2em;
            white-space: nowrap;
            font-family: arial;
            :nth-child(4) {
                letter-spacing: 0.125em;
            }
            border-left: .225em solid currentColor;
            border-bottom: .2225em solid currentColor;

            span {
                font-size: calc(1em + var(--index) * 0.1em);
            }
            margin-block: .25em;
            margin-left: 1rem;
        }

        .document-scores {
            display: flex;
            align-items: center;
            gap: var(--gap);

            .doc-stat {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1.5em;

                dt {
                    display: inline;
                    font-size: .9em;
                    opacity: .7;
                }
                dd {
                    display: inline;
                    margin: 0;
                    font-family: monospace;
                    font-size: 1.25em;
                }
            }
        }
    }
    .view-toggle {
        position: absolute;
        font-family: monospace;
        top: 0;
        right: 0;
        width: 1.5em;
        height: 1.5em;
        color: currentColor;
        background: transparent;
        border: 1px solid currentColor;
        border-top: none;
        border-bottom-left-radius: .25em;
        cursor: pointer;
        transition: filter 0.2s ease;
        z-index: 100;

        &:hover {
            background-color: color-mix(in oklch, currentColor 10%, transparent);
        }
    }

    #content-area {
        overflow-y: visible;
        }

    [data-view][hidden] {
        display: none;
    }
    [data-view="sentences"] {
        display: flex;
        flex-wrap: wrap;
        row-gap: 0.25em;
        align-items: center;
        padding-inline: .5em;
        padding-bottom: 1em;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }
    [data-view="stats"] {
        margin-block: .5em;
    }

    .level-badge {
        display: grid;
        place-items: center;
        width: 1.5em;
        height: 1.5em;
        border-radius: 50%;
        margin-right: .5em;
        font-family: monospace;
        line-height: 1em;
        color: white;
        user-select: none;
    }
    header .level-badge {
        font-size: 2em;
    }
    [data-level="null"], [data-level=""] {
        .level-badge {
            background-color: hsl(0, 0%, 65%);
        }
        .sent-idx,
        .sent-start::before,
        .sent-end::after {
            color: hsl(0, 0%, 65%);
        }
    }
    [data-level="1"] {
        .level-badge {
            background-color: var(--color-level-1);
        }
        .sent-idx,
        .sent-start::before,
        .sent-end::after {
            color: var(--color-level-1);
        }
    }
    [data-level="2"] {
        .level-badge {
            background-color: var(--color-level-2);
        }
        .sent-idx,
        .sent-start::before,
        .sent-end::after {
            color: var(--color-level-2);
        }
    }
    [data-level="3"] {
        .level-badge {
            background-color: var(--color-level-3);
        }
        .sent-idx,
        .sent-start::before,
        .sent-end::after {
            color: var(--color-level-3);
        }
    }
    [data-level="4"] {
        .level-badge {
            background-color: var(--color-level-4);
        }
        .sent-idx,
        .sent-start::before,
        .sent-end::after {
            color: var(--color-level-4);
        }
    }

    .sentence {
        --scale: 1.25;
        display: contents;

        .sent-start-group,
        .sent-end-group {
            position: relative;
            display: inline-flex;
            align-items: center;
            white-space: nowrap;
            transition: transform 0.2s ease;
        }

        &:has(.sent-start-group:hover, .sent-end-group:hover) .sent-start-group,
        &:has(.sent-start-group:hover, .sent-end-group:hover) .sent-end-group {
            transform: scale(var(--scale));
        }

        .sent-idx {
            font-size: .7em;
            position: absolute;
            top: 1em;
            right: 100%;
            margin-right: -.5em;
            text-align: right;
            font-family: monospace;
        }
        .sent-start::before,
        .sent-end::after {
            font-family: monospace;
            font-size: 2.4em;
            vertical-align: middle;
            /* Keep the tall bracket glyph from inflating the line box, so line
               spacing follows the text rather than the brackets. */
            line-height: 0;
        }
        .sent-start::before {
            content: '[';
            padding-right: 0.2em;
        }
        .sent-end::after {
            content: ']';
            padding-left: 0.2em;
        }
    }
    .word {
        padding-inline: .55em;
        padding-block: .3em;
        margin-inline: .1em;
        border-radius: .25em;
        cursor: default;

        transition: filter 0.2s ease;

        &:hover {
            filter: brightness(0.85);
        }

        &:not([data-sem-type]):hover {
            background-color: color-mix(in oklch, currentColor 15%, transparent);
        }

        &[data-sem-type] {
            color: black;
        }
        &[data-sem-type="concrete"] {
            background-color: var(--color-concrete);
        }
        &[data-sem-type="abstract"] {
            background-color: var(--color-abstract);
        }
        &[data-sem-type="undefined"] {
            background-color: var(--color-undefined);
        }
        &[data-sem-type="unknown"] {
            background-color: var(--color-unknown);
        }

    }
    .popup {
        display: none;
        pointer-events: none;
        position: fixed;
        padding: 0.5em 1em;
        border: 1px solid currentColor;
        border-radius: .25em;
        background: var(--vscode-notebook-editorBackground, var(--background));
        z-index: 1000;

        .label {
            font-size: .8125em;
        }
        .value {
            font-family: monospace;
            font-size: 1.25em;
        }
        &.visible {
            display: grid;
            column-gap: 1rem;
            grid-template-columns: auto 1fr;
            align-items: center;
        }
    }

    /* Editor mode styles */
    :host([mode="editor"]) {
        --color-suggestion-pending: rgba(255, 176, 176, 0.75);
        --color-suggestion-accepted: rgba(157, 239, 182, 0.75);
        --color-suggestion-ignored: hsla(0, 0%, 50%, 0.1);
    }

    /* Hide semantic type colors in editor mode by default */
    :host([mode="editor"]) .word[data-sem-type] {
        background-color: transparent;
        color: inherit;
    }

    /* Show them when toggled on */
    :host([mode="editor"][data-show-sem-types]) .word[data-sem-type="concrete"] {
        background-color: var(--color-concrete);
        color: black;
    }
    :host([mode="editor"][data-show-sem-types]) .word[data-sem-type="abstract"] {
        background-color: var(--color-abstract);
        color: black;
    }
    :host([mode="editor"][data-show-sem-types]) .word[data-sem-type="undefined"] {
        background-color: var(--color-undefined);
        color: black;
    }
    :host([mode="editor"][data-show-sem-types]) .word[data-sem-type="unknown"] {
        background-color: var(--color-unknown);
        color: black;
    }

    .editor-toolbar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid currentColor;
        background: color-mix(in oklch, currentColor 5%, transparent);

        .suggestion-counts {
            display: flex;
            gap: 1rem;
            font-size: 0.875em;

            .count-item {
                display: flex;
                align-items: center;
                gap: 0.25rem;

                &.pending .count-badge { background: var(--color-suggestion-pending); }
                &.accepted .count-badge { background: var(--color-suggestion-accepted); }
                &.ignored .count-badge { opacity: 0.5; }
            }

            .count-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 1.5em;
                height: 1.5em;
                padding: 0 0.25em;
                border-radius: 0.75em;
                font-family: monospace;
                font-size: 0.875em;
            }
        }

        .sem-type-toggle {
            padding: 0.375rem 0.75rem;
            font-size: 0.875em;
            color: currentColor;
            background: transparent;
            border: 1px solid currentColor;
            border-radius: 0.25rem;
            cursor: pointer;
            opacity: 0.5;
            transition: all 0.2s;

            &:hover {
                opacity: 0.8;
            }

            &.active {
                opacity: 1;
                background: color-mix(in oklch, currentColor 10%, transparent);
            }
        }

        .copy-result-btn {
            margin-left: auto;
            padding: 0.375rem 0.75rem;
            font-size: 0.875em;
            color: currentColor;
            background: transparent;
            border: 1px solid currentColor;
            border-radius: 0.25rem;
            cursor: pointer;
            transition: background-color 0.2s;

            &:hover {
                background: color-mix(in oklch, currentColor 10%, transparent);
            }

            &.copied {
                background: var(--color-suggestion-accepted);
            }
        }
    }

    /* Suggestion highlights */
    .word[data-suggestion-id] {
        position: relative;
        cursor: pointer;
    }

    .word[data-suggestion-status="pending"] {
        background: var(--color-suggestion-pending) !important;
        outline: 2px solid hsla(50, 100%, 50%, 0.5);
        outline-offset: 1px;
    }

    .word[data-suggestion-status="accepted"] {
        background: var(--color-suggestion-accepted) !important;
    }

    .word.suggestion-changed {
        background: var(--color-suggestion-accepted) !important;
        border-bottom: 2px solid hsl(120, 60%, 40%);
        color: inherit;
    }

    .word[data-suggestion-status="ignored"] {
        background: transparent !important;
        outline: none !important;
        opacity: 0.5;
    }

    /* Sentence-level suggestion highlights are applied per-word
       via data-suggestion-id on each .word element, using the
       same .word[data-suggestion-status] rules above. */

    /* Suggestion popup */
    .suggestion-popup {
        display: none;
        pointer-events: auto;
        position: fixed;
        max-width: 500px;
        padding: 1rem;
        border: 2px solid #b3b3ab;
        border-radius: 10px;
        background: white;
        color: #525347;
        font-family: Verdana, sans-serif;
        font-size: 12px;
        box-shadow: 5px 5px 8px rgba(0,0,0,0.2);
        z-index: 1001;

        &.visible {
            display: block;
        }
    }

    .suggestion-popup-content {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .suggestion-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;

        .suggestion-type {
            font-weight: 600;
            font-size: 0.9em;
        }

        .suggestion-category {
            font-size: 0.75em;
            padding: 0.125rem 0.5rem;
            border-radius: 1rem;
            background: color-mix(in oklch, currentColor 10%, transparent);
            opacity: 0.8;
        }
    }

    .status-badge {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 0.05em;

        &.pending {
            background: var(--color-suggestion-pending);
        }
        &.accepted {
            background: var(--color-suggestion-accepted);
        }
        &.ignored {
            background: var(--color-suggestion-ignored);
        }
    }

    .suggestion-comparison {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 0.75rem;
        background: color-mix(in oklch, currentColor 5%, transparent);
        border-radius: 0.25rem;

        .original, .suggested {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .label {
            font-size: 0.75em;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .text {
            line-height: 1.5;

            del {
                text-decoration: line-through;
                opacity: 0.6;
                text-decoration-color: currentColor;
            }

            ins {
                text-decoration: none;
                font-style: normal;
                background: color-mix(in oklch, oklch(0.7 0.18 145) 30%, transparent);
                border-radius: 0.2em;
                padding: 0.05em 0.15em;
            }
        }
    }

    .suggestion-explanation {
        font-size: 0.9em;
        line-height: 1.5;
        padding: 0.5rem;
        background: color-mix(in oklch, currentColor 3%, transparent);
        border-radius: 0.25rem;
        border-left: 3px solid currentColor;

        .label {
            font-size: 0.75em;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: block;
            margin-bottom: 0.25rem;
        }
    }

    .suggestion-actions {
        display: flex;
        gap: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid color-mix(in oklch, currentColor 20%, transparent);

        button {
            flex: 1;
            padding: 0.5rem 1rem;
            font-size: 0.875em;
            color: currentColor;
            background: transparent;
            border: 1px solid currentColor;
            border-radius: 0.25rem;
            cursor: pointer;
            transition: all 0.2s;

            &:hover {
                background: color-mix(in oklch, currentColor 10%, transparent);
            }
        }

        .accept-btn {
            background: #9defb6;
            border-color: #5cb87a;
            color: #2a5c38;

            &:hover {
                background: #7de0a0;
            }
        }

        .ignore-btn {
            opacity: 0.7;

            &:hover {
                opacity: 1;
            }
        }

        .reset-btn {
            font-size: 0.8em;
        }
    }

    .suggestion-section {
        padding-bottom: 0.75rem;
        border-bottom: 1px solid color-mix(in oklch, currentColor 15%, transparent);

        &:last-child {
            padding-bottom: 0;
            border-bottom: none;
        }
    }

    .suggestion-section + .suggestion-section {
        padding-top: 0.75rem;
    }

    /* Narrow screens: the header, document scores and toolbar are flex rows
       with a large gap and no wrap, which overflow horizontally on a phone.
       Shrink the gap and let them wrap so everything stays within the viewport. */
    @media (max-width: 600px) {
        :host { --gap: 1em; max-width: 100%; }
        #content-area { max-width: 100%; overflow-x: hidden; }
        header {
            flex-wrap: wrap;
            gap: 0.75em;
            padding-right: 1.75em; /* room for the absolute view-toggle */
        }
        .document-scores { flex-wrap: wrap; gap: 1em; }
        .document-scores .doc-stat { gap: 0.5em; }
        header .level-badge { font-size: 1.4em; }
        .editor-toolbar { flex-wrap: wrap; gap: 0.5rem; }
        .suggestion-counts { gap: 0.6rem; }
        [data-view="sentences"] { font-size: 0.95em; padding-inline: .25em; }
        .word { padding-inline: .4em; }
        /* The closing bracket + level badge are the widest trailing unit and
           poked past the right edge; shrink the brackets and the badge's
           trailing margin so the sentence-end group stays inside the viewport. */
        .sent-start::before, .sent-end::after { font-size: 1.7em; }
        .level-badge { margin-right: .25em; }
        .suggestion-popup { max-width: calc(100vw - 1rem); }
    }
`
