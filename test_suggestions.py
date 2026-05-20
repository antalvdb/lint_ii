#!/usr/bin/env python3
"""
Test script for LLM-powered rewriting suggestions.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY=sk-...

    # Run the script
    python test_suggestions.py

    # Or pass the API key directly
    python test_suggestions.py --api-key sk-...
"""

import sys
import os
import argparse
import logging

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(description='Test LLM suggestion generation')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic', 'ollama'],
                        help='LLM provider to use')
    parser.add_argument('--model', help='Model to use (provider-specific default if not set)')
    parser.add_argument('--max-suggestions', type=int, default=3,
                        help='Maximum number of suggestions to generate')
    parser.add_argument('--text', help='Custom Dutch text to analyze')
    parser.add_argument('--show-triggers-only', action='store_true',
                        help='Only show identified triggers without calling LLM')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show debug output including raw LLM responses')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='    %(levelname)s [%(name)s] %(message)s',
    )

    # Sample Dutch text with spelling errors and grammatical mistakes
    # Errors: "word" → "wordt" (dt-fout), "geprobeert" → "geprobeerd",
    #          "aanpassingen" is correct but "aanbevelingen" → "aanbeveligen" (typo),
    #          "vind" → "vindt" (dt-fout)
    sample_text = args.text or (
        "De onderzoeker word al jaren geconfronteerd met een tekort aan financiële middelen, "
        "waardoor het steeds moeilijker wordt om grootschalige experimenten op te zetten die "
        "voldoen aan de strenge eisen van de wetenschappelijke gemeenschap. "
        "Het team heeft geprobeert om de resultaten van het vorige experiment te repliceren, "
        "maar de meetapparatuur bleek onvoldoende gekalibreerd te zijn en de omgevingscondities "
        "verschilden aanzienlijk van de oorspronkelijke opzet. "
        "In het rapport worden verschillende aanbeveligen gedaan voor de verbetering van het "
        "onderwijscurriculum, waaronder het integreren van digitale leermiddelen en het trainen "
        "van docenten in moderne didactische methoden. "
        "De minister vind dat de nieuwe regelgeving voldoende waarborgen biedt voor de "
        "bescherming van de persoonlijke levenssfeer van burgers, ondanks de kritiek van "
        "verschillende mensenrechtenorganisaties die zich zorgen maken over de reikwijdte "
        "van de voorgestelde maatregelen. "
        "Hoewel de gemeente herhaaldelijk heeft aangegeven dat de renovatie van het historische "
        "pand binnen het vastgestelde budget zou blijven, blijkt uit recente berekeningen dat de "
        "kosten inmiddels met meer dan dertig procent zijn gestegen ten opzichte van de "
        "oorspronkelijke raming."
    )

    print("=" * 60)
    print("LiNT-II Suggestion Generation Test")
    print("=" * 60)

    # Step 1: Analyze the text
    print("\n[1] Analyzing text...")
    from lint_ii import ReadabilityAnalysis

    analysis = ReadabilityAnalysis.from_text(sample_text)

    print(f"    Sentences: {len(analysis.sentences)}")
    print(f"    LiNT Score: {analysis.lint.score:.1f}")
    print(f"    Difficulty Level: {analysis.lint.level}")
    print(f"    Mean Word Frequency: {analysis.mean_log_word_frequency:.2f}")
    print(f"    Mean Max SDL: {analysis.mean_max_sdl:.1f}")

    # Step 2: Identify triggers
    print("\n[2] Identifying suggestion triggers...")
    from lint_ii.llm.suggestions import SuggestionEngine, DEFAULT_THRESHOLDS

    engine = SuggestionEngine()
    triggers = engine.identify_triggers(analysis)

    print(f"    Found {len(triggers)} triggers:")
    for i, trigger in enumerate(triggers, 1):
        print(f"    {i}. [{trigger.type.value}] Sentence {trigger.sentence_index + 1}")
        print(f"       Value: {trigger.feature_value:.2f} (threshold: {trigger.threshold})")
        if trigger.word:
            print(f"       Word: '{trigger.word}'")

    if args.show_triggers_only:
        print("\n[--show-triggers-only] Stopping before LLM call")
        return

    # Step 3: Generate suggestions
    print("\n[3] Generating suggestions via LLM...")

    # Build LLM config
    llm_config = {'provider': args.provider}

    if args.api_key:
        llm_config['api_key'] = args.api_key
    elif args.provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
        print("    ERROR: No API key provided.")
        print("    Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)
    elif args.provider == 'anthropic' and not os.environ.get('ANTHROPIC_API_KEY'):
        print("    ERROR: No API key provided.")
        print("    Set ANTHROPIC_API_KEY env var or use --api-key")
        sys.exit(1)

    if args.model:
        llm_config['model'] = args.model

    try:
        suggestions = analysis.generate_suggestions(
            llm_config=llm_config,
            max_suggestions=args.max_suggestions,
        )

        print(f"    Model: {suggestions.model}")
        print(f"    Triggers found: {suggestions.triggers_found}")
        print(f"    Triggers processed: {suggestions.triggers_processed}")
        print(f"    Suggestions generated: {len(suggestions.suggestions)}")

        # Step 4: Display suggestions
        print("\n[4] Generated suggestions:")
        for i, suggestion in enumerate(suggestions.suggestions, 1):
            print(f"\n    --- Suggestion {i} ({suggestion.type.value}) ---")
            print(f"    Original: {suggestion.original_text[:80]}...")
            print(f"    Suggested: {suggestion.suggested_text[:80]}...")
            if suggestion.explanation:
                print(f"    Explanation: {suggestion.explanation}")

        # Step 5: Show how to use with_suggestions
        print("\n[5] Creating visualization object...")
        viz = analysis.with_suggestions(suggestions)
        print(f"    Mode: {viz.mode}")
        print(f"    Data keys: {list(viz.as_dict().keys())}")

        # Step 6: Export as JSON for testing in browser
        print("\n[6] Exporting to JSON for browser testing...")
        import json
        output_path = 'test_suggestions_output.json'
        with open(output_path, 'w') as f:
            json.dump(viz.as_dict(), f, indent=2, ensure_ascii=False)
        print(f"    Saved to: {output_path}")
        print(f"    Open index.html and load this file to test the editor UI")

    except ImportError as e:
        print(f"    ERROR: Missing dependency: {e}")
        print("    Install with: pip install lint_ii[llm]")
        sys.exit(1)
    except Exception as e:
        print(f"    ERROR: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
