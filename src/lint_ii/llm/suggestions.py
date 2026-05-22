"""
Suggestion engine for LiNT-II readability improvements.

Identifies triggers from linguistic analysis and generates suggestions
using LLM providers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING
import logging
import uuid

from lint_ii.llm.providers import LLMProvider, create_provider
from lint_ii.llm.prompts import format_prompt, parse_llm_response, parse_spelling_response

logger = logging.getLogger(__name__)


class _AuthenticationError(Exception):
    """Raised when LLM authentication fails, to abort early."""


if TYPE_CHECKING:
    from lint_ii.core.readability_analysis import ReadabilityAnalysis
    from lint_ii.core.sentence_analysis import SentenceAnalysis


class SuggestionType(str, Enum):
    """Types of suggestions based on linguistic features."""
    WORD_FREQUENCY = "word_frequency"
    MAX_SDL = "max_sdl"
    CONTENT_WORDS_PER_CLAUSE = "content_words_per_clause"
    ABSTRACT_NOUNS = "abstract_nouns"
    SPELLING = "spelling"
    PASSIVE = "passive"
    SUBORDINATE_CLAUSE = "subordinate_clause"
    SENTENCE_LENGTH = "sentence_length"


# Default thresholds for triggering suggestions
DEFAULT_THRESHOLDS: dict[str, float] = {
    "word_frequency": 3.0,           # Zipf frequency below this triggers suggestion
    "max_sdl": 5,                    # SDL above this triggers suggestion
    "content_words_per_clause": 7,   # Content words/clause above this triggers
    "abstract_noun_ratio": 0.7,      # Abstract ratio above this (concrete < 30%)
    "sentence_length": 25,           # Words above this triggers suggestion
    "n_subordinate_clauses": 1,      # More than this many subordinate clauses triggers
}


@dataclass
class SuggestionTrigger:
    """A detected issue that may warrant a suggestion."""
    type: SuggestionType
    sentence_index: int
    sentence_text: str
    feature_value: float
    threshold: float
    # Additional context based on type
    word: str | None = None          # For word_frequency
    word_index: int | None = None    # Token index within sentence
    context: str | None = None       # Surrounding text for context
    abstract_nouns: list[str] = field(default_factory=list)  # For abstract_nouns
    passives: list[str] = field(default_factory=list)         # For passive


@dataclass
class Suggestion:
    """A generated suggestion for improving readability."""
    id: str
    type: SuggestionType
    sentence_index: int
    original_text: str
    suggested_text: str
    explanation: str
    # For word-level suggestions
    word: str | None = None
    word_index: int | None = None
    replacement_word: str | None = None
    # Metadata
    model: str | None = None
    error_category: str | None = None  # "spelling" or "grammar" for spelling suggestions
    # Precomputed metrics for score recomputation
    new_sentence_metrics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize suggestion to dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "sentence_index": self.sentence_index,
            "original_text": self.original_text,
            "suggested_text": self.suggested_text,
            "explanation": self.explanation,
        }
        if self.word is not None:
            result["word"] = self.word
        if self.word_index is not None:
            result["word_index"] = self.word_index
        if self.replacement_word is not None:
            result["replacement_word"] = self.replacement_word
        if self.error_category is not None:
            result["error_category"] = self.error_category
        if self.new_sentence_metrics is not None:
            result["new_sentence_metrics"] = self.new_sentence_metrics
        return result


@dataclass
class SuggestionsResult:
    """Result of suggestion generation."""
    suggestions: list[Suggestion]
    triggers_found: int
    triggers_processed: int
    model: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "suggestions": [s.as_dict() for s in self.suggestions],
            "triggers_found": self.triggers_found,
            "triggers_processed": self.triggers_processed,
            "model": self.model,
        }


class SuggestionEngine:
    """
    Engine for identifying triggers and generating suggestions.

    The engine analyzes ReadabilityAnalysis results to find potential
    readability issues, then uses an LLM to generate improvement suggestions.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize the suggestion engine.

        Args:
            provider: LLM provider for generating suggestions
            thresholds: Custom thresholds for triggers (uses defaults if not specified)
        """
        self._provider = provider
        self._thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def identify_triggers(
        self,
        analysis: "ReadabilityAnalysis",
    ) -> list[SuggestionTrigger]:
        """
        Identify potential readability issues that could benefit from suggestions.

        Args:
            analysis: ReadabilityAnalysis object to analyze

        Returns:
            List of SuggestionTrigger objects describing found issues
        """
        triggers: list[SuggestionTrigger] = []

        for sent_idx, sent_analysis in enumerate(analysis.sentence_analyses):
            sentence_text = sent_analysis.doc.text

            # Check for low-frequency words
            triggers.extend(
                self._check_word_frequency(sent_analysis, sent_idx, sentence_text)
            )

            # Check for high syntactic dependency length
            trigger = self._check_max_sdl(sent_analysis, sent_idx, sentence_text)
            if trigger:
                triggers.append(trigger)

            # Check for high content density
            trigger = self._check_content_density(sent_analysis, sent_idx, sentence_text)
            if trigger:
                triggers.append(trigger)

            # Check for high abstract noun ratio
            trigger = self._check_abstract_nouns(sent_analysis, sent_idx, sentence_text)
            if trigger:
                triggers.append(trigger)

            # Check for passive constructions
            trigger = self._check_passive(sent_analysis, sent_idx, sentence_text)
            if trigger:
                triggers.append(trigger)

            # Check for many subordinate clauses
            trigger = self._check_subordinate_clauses(sent_analysis, sent_idx, sentence_text)
            if trigger:
                triggers.append(trigger)

            # Check for long sentences
            trigger = self._check_sentence_length(sent_analysis, sent_idx, sentence_text)
            if trigger:
                triggers.append(trigger)

        return triggers

    def _check_word_frequency(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> list[SuggestionTrigger]:
        """Check for words with low frequency."""
        triggers = []
        threshold = self._thresholds["word_frequency"]

        for word_idx, wf in enumerate(sent_analysis.word_features):
            freq = wf.word_frequency
            if freq is not None and freq < threshold:
                # Get context (surrounding words)
                context = sentence_text
                triggers.append(
                    SuggestionTrigger(
                        type=SuggestionType.WORD_FREQUENCY,
                        sentence_index=sent_idx,
                        sentence_text=sentence_text,
                        feature_value=freq,
                        threshold=threshold,
                        word=wf.text,
                        word_index=word_idx,
                        context=context,
                    )
                )

        return triggers

    def _check_max_sdl(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> SuggestionTrigger | None:
        """Check for high syntactic dependency length."""
        threshold = self._thresholds["max_sdl"]
        max_sdl = sent_analysis.max_sdl

        if max_sdl is not None and max_sdl > threshold:
            return SuggestionTrigger(
                type=SuggestionType.MAX_SDL,
                sentence_index=sent_idx,
                sentence_text=sentence_text,
                feature_value=max_sdl,
                threshold=threshold,
            )
        return None

    def _check_content_density(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> SuggestionTrigger | None:
        """Check for high content word density."""
        threshold = self._thresholds["content_words_per_clause"]
        cwpc = sent_analysis.content_words_per_clause

        if cwpc is not None and cwpc > threshold:
            return SuggestionTrigger(
                type=SuggestionType.CONTENT_WORDS_PER_CLAUSE,
                sentence_index=sent_idx,
                sentence_text=sentence_text,
                feature_value=cwpc,
                threshold=threshold,
            )
        return None

    def _check_abstract_nouns(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> SuggestionTrigger | None:
        """Check for high proportion of abstract nouns."""
        threshold = self._thresholds["abstract_noun_ratio"]
        proportion_concrete = sent_analysis.proportion_of_concrete_nouns

        # If proportion_concrete is None, there are no categorizable nouns
        if proportion_concrete is None:
            return None

        # High abstract ratio means low concrete proportion
        if proportion_concrete < (1 - threshold):
            abstract_nouns = [wf.text for wf in sent_analysis.abstract_nouns]
            if abstract_nouns:  # Only trigger if there are actually abstract nouns
                return SuggestionTrigger(
                    type=SuggestionType.ABSTRACT_NOUNS,
                    sentence_index=sent_idx,
                    sentence_text=sentence_text,
                    feature_value=proportion_concrete,
                    threshold=1 - threshold,
                    abstract_nouns=abstract_nouns,
                    context=sentence_text,
                )
        return None

    def _check_passive(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> SuggestionTrigger | None:
        """Check for passive constructions."""
        if not sent_analysis.has_passive:
            return None
        passives = [span.text for span in sent_analysis.passives]
        return SuggestionTrigger(
            type=SuggestionType.PASSIVE,
            sentence_index=sent_idx,
            sentence_text=sentence_text,
            feature_value=float(len(passives)),
            threshold=0,
            passives=passives,
        )

    def _check_subordinate_clauses(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> SuggestionTrigger | None:
        """Check for sentences with many subordinate clauses."""
        threshold = self._thresholds["n_subordinate_clauses"]
        n = sent_analysis.n_subordinate_clauses

        if n > threshold:
            return SuggestionTrigger(
                type=SuggestionType.SUBORDINATE_CLAUSE,
                sentence_index=sent_idx,
                sentence_text=sentence_text,
                feature_value=float(n),
                threshold=threshold,
            )
        return None

    def _check_sentence_length(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> SuggestionTrigger | None:
        """Check for sentences that are too long."""
        threshold = self._thresholds["sentence_length"]
        length = sent_analysis.sent_length

        if length > threshold:
            return SuggestionTrigger(
                type=SuggestionType.SENTENCE_LENGTH,
                sentence_index=sent_idx,
                sentence_text=sentence_text,
                feature_value=float(length),
                threshold=threshold,
            )
        return None

    @staticmethod
    def _prioritize_triggers(
        triggers: list[SuggestionTrigger],
        max_suggestions: int | None,
    ) -> list[SuggestionTrigger]:
        """
        Prioritize triggers to ensure variety across suggestion types.

        Takes one of each type first (sentence-level before word-level),
        then fills remaining slots with additional triggers.
        """
        if max_suggestions is None:
            return triggers

        # Sentence-level types first, then word-level
        type_priority = [
            SuggestionType.SENTENCE_LENGTH,
            SuggestionType.PASSIVE,
            SuggestionType.SUBORDINATE_CLAUSE,
            SuggestionType.MAX_SDL,
            SuggestionType.CONTENT_WORDS_PER_CLAUSE,
            SuggestionType.ABSTRACT_NOUNS,
            SuggestionType.WORD_FREQUENCY,
        ]

        by_type: dict[SuggestionType, list[SuggestionTrigger]] = {}
        for trigger in triggers:
            by_type.setdefault(trigger.type, []).append(trigger)

        result: list[SuggestionTrigger] = []

        # Round 1: one of each type
        for typ in type_priority:
            if len(result) >= max_suggestions:
                break
            if typ in by_type and by_type[typ]:
                result.append(by_type[typ].pop(0))

        # Round 2+: fill remaining slots
        for typ in type_priority:
            while len(result) < max_suggestions and by_type.get(typ):
                result.append(by_type[typ].pop(0))

        return result

    def generate_spelling_suggestions(
        self,
        analysis: "ReadabilityAnalysis",
        provider: "LLMProvider",
    ) -> list[Suggestion]:
        """
        Run a single LLM call to identify spelling and grammar errors
        across the full document.

        Returns:
            List of Suggestion objects with type=SPELLING.
        """
        # Build numbered text from all sentences
        sentence_texts: list[str] = []
        for idx, sent_analysis in enumerate(analysis.sentence_analyses):
            sentence_texts.append(f"{idx + 1}. {sent_analysis.doc.text}")
        full_text = "\n".join(sentence_texts)

        system_prompt, user_prompt = format_prompt("spelling", text=full_text)

        try:
            response = provider.complete(user_prompt, system_prompt)
        except Exception as e:
            err_str = str(e).lower()
            if "authentication" in err_str or "401" in err_str or "api_key" in err_str:
                raise _AuthenticationError(e) from e
            logger.error("Failed to generate spelling suggestions: %s", e, exc_info=True)
            return []

        logger.debug("Spelling LLM response:\n%s", response.content)
        parsed_errors = parse_spelling_response(response.content)

        suggestions: list[Suggestion] = []
        for error in parsed_errors:
            try:
                sent_num = int(error.get("ZIN_NUMMER", "0")) - 1  # 1-based → 0-based
            except (ValueError, TypeError):
                continue

            if sent_num < 0 or sent_num >= len(analysis.sentence_analyses):
                continue

            word = error.get("WOORD", "").strip()
            correction = error.get("CORRECTIE", "").strip()
            category_raw = error.get("CATEGORIE", "").strip().lower()
            explanation = error.get("UITLEG", "").strip()

            if not word or not correction:
                continue

            # Map category to normalized value
            if "spel" in category_raw:
                error_category = "spelling"
            else:
                error_category = "grammar"

            sent_text = analysis.sentence_analyses[sent_num].doc.text
            # Build the corrected sentence by replacing the first occurrence
            suggested_text = sent_text.replace(word, correction, 1)

            # Find the word index in the sentence tokens
            word_index = None
            for wi, wf in enumerate(analysis.sentence_analyses[sent_num].word_features):
                if wf.text == word:
                    word_index = wi
                    break

            suggestions.append(Suggestion(
                id=str(uuid.uuid4())[:8],
                type=SuggestionType.SPELLING,
                sentence_index=sent_num,
                original_text=sent_text,
                suggested_text=suggested_text,
                explanation=explanation,
                word=word,
                word_index=word_index,
                replacement_word=correction,
                model=response.model,
                error_category=error_category,
            ))

        return suggestions

    def generate_suggestions(
        self,
        analysis: "ReadabilityAnalysis",
        max_suggestions: int | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> SuggestionsResult:
        """
        Generate suggestions for improving readability.

        Args:
            analysis: ReadabilityAnalysis object to generate suggestions for
            max_suggestions: Maximum number of suggestions to generate (None for all)
            llm_config: LLM configuration if no provider was set in __init__

        Returns:
            SuggestionsResult with generated suggestions
        """
        # Get or create provider
        provider = self._provider
        if provider is None and llm_config:
            provider = create_provider(**llm_config)
        if provider is None:
            raise ValueError(
                "No LLM provider configured. Pass llm_config or set provider in __init__."
            )

        import time

        # Step 1: Spelling/grammar pass (single call for entire document)
        t0 = time.perf_counter()
        spelling_suggestions = self.generate_spelling_suggestions(analysis, provider)
        t1 = time.perf_counter()
        logger.info("TIMING spelling_pass=%.2fs (%d suggestions)", t1 - t0, len(spelling_suggestions))

        # Step 2: Find readability triggers
        triggers = self.identify_triggers(analysis)
        triggers_to_process = self._prioritize_triggers(triggers, max_suggestions)
        t2 = time.perf_counter()
        logger.info("TIMING trigger_detection=%.2fs (%d triggers → %d to process)", t2 - t1, len(triggers), len(triggers_to_process))

        # Step 3: Generate readability suggestions for each trigger
        suggestions: list[Suggestion] = list(spelling_suggestions)
        for trigger in triggers_to_process:
            t_trigger = time.perf_counter()
            try:
                suggestion = self._generate_suggestion_for_trigger(trigger, provider)
            except _AuthenticationError as e:
                raise RuntimeError(
                    f"LLM authentication failed: {e}. Check your API key."
                ) from e
            logger.info("TIMING trigger_%s=%.2fs", trigger.type.value, time.perf_counter() - t_trigger)
            if suggestion:
                suggestions.append(suggestion)

        return SuggestionsResult(
            suggestions=suggestions,
            triggers_found=len(triggers),
            triggers_processed=len(triggers_to_process),
            model=provider.model_name,
        )

    @staticmethod
    def _analyze_suggested_text(text: str) -> dict[str, Any] | None:
        """Analyze the suggested text to precompute sentence metrics for score recomputation."""
        try:
            import time
            from lint_ii.core.readability_analysis import ReadabilityAnalysis
            t0 = time.perf_counter()
            analysis = ReadabilityAnalysis.from_text(text)
            logger.info("TIMING _analyze_suggested_text spacy=%.2fs", time.perf_counter() - t0)

            word_freqs = [
                f for feat in analysis.word_features
                if (f := feat.word_frequency) is not None
            ]
            # Store per-sentence values so the JS side can correctly
            # adjust document-level means when a single original sentence
            # is replaced by potentially multiple new sentences.
            return {
                "word_freq_sum": sum(word_freqs),
                "word_freq_count": len(word_freqs),
                "sdl_values": [s.max_sdl for s in analysis.sentences if s.max_sdl is not None],
                "cwpc_values": [
                    s.content_words_per_clause for s in analysis.sentences
                    if s.content_words_per_clause is not None
                ],
                "n_concrete": len(analysis.concrete_nouns),
                "n_abstract": len(analysis.abstract_nouns),
                "n_undefined": len(analysis.undefined_nouns),
            }
        except Exception as e:
            logger.warning("Failed to analyze suggested text for metrics: %s", e)
            return None

    def _generate_suggestion_for_trigger(
        self,
        trigger: SuggestionTrigger,
        provider: LLMProvider,
    ) -> Suggestion | None:
        """Generate a suggestion for a single trigger."""
        try:
            # Format the prompt based on trigger type
            if trigger.type == SuggestionType.WORD_FREQUENCY:
                system_prompt, user_prompt = format_prompt(
                    "word_frequency",
                    word=trigger.word,
                    context=trigger.context,
                    frequency=trigger.feature_value,
                )
            elif trigger.type == SuggestionType.MAX_SDL:
                system_prompt, user_prompt = format_prompt(
                    "max_sdl",
                    sentence=trigger.sentence_text,
                    max_sdl=int(trigger.feature_value),
                )
            elif trigger.type == SuggestionType.CONTENT_WORDS_PER_CLAUSE:
                system_prompt, user_prompt = format_prompt(
                    "content_words_per_clause",
                    sentence=trigger.sentence_text,
                    content_words_per_clause=trigger.feature_value,
                )
            elif trigger.type == SuggestionType.ABSTRACT_NOUNS:
                system_prompt, user_prompt = format_prompt(
                    "abstract_nouns",
                    context=trigger.context or trigger.sentence_text,
                    abstract_nouns=", ".join(trigger.abstract_nouns),
                )
            elif trigger.type == SuggestionType.PASSIVE:
                system_prompt, user_prompt = format_prompt(
                    "passive",
                    sentence=trigger.sentence_text,
                    passives=", ".join(f'"{p}"' for p in trigger.passives),
                )
            elif trigger.type == SuggestionType.SUBORDINATE_CLAUSE:
                system_prompt, user_prompt = format_prompt(
                    "subordinate_clause",
                    sentence=trigger.sentence_text,
                    n_subordinate_clauses=int(trigger.feature_value),
                )
            elif trigger.type == SuggestionType.SENTENCE_LENGTH:
                system_prompt, user_prompt = format_prompt(
                    "sentence_length",
                    sentence=trigger.sentence_text,
                    sent_length=int(trigger.feature_value),
                )
            else:
                return None

            # Call LLM
            response = provider.complete(user_prompt, system_prompt)

            # Parse response
            logger.debug(
                "LLM response for %s trigger (sentence %d):\n%s",
                trigger.type.value, trigger.sentence_index, response.content,
            )
            parsed = parse_llm_response(response.content, trigger.type.value)
            logger.debug("Parsed fields: %s", list(parsed.keys()))

            # Extract suggestion from parsed response
            suggested_text = parsed.get("HERSCHRIJVING", "")
            explanation = parsed.get("UITLEG", "")
            replacement_word = parsed.get("VERVANGING")

            if not suggested_text:
                logger.warning(
                    "No HERSCHRIJVING found in LLM response for %s trigger. "
                    "Parsed fields: %s. Raw response:\n%s",
                    trigger.type.value, parsed, response.content,
                )
                return None

            new_metrics = self._analyze_suggested_text(suggested_text)

            return Suggestion(
                id=str(uuid.uuid4())[:8],
                type=trigger.type,
                sentence_index=trigger.sentence_index,
                original_text=trigger.sentence_text,
                suggested_text=suggested_text,
                explanation=explanation,
                word=trigger.word,
                word_index=trigger.word_index,
                replacement_word=replacement_word,
                model=response.model,
                new_sentence_metrics=new_metrics,
            )

        except Exception as e:
            # Detect auth errors and abort early instead of retrying
            err_str = str(e).lower()
            if "authentication" in err_str or "401" in err_str or "api_key" in err_str:
                raise _AuthenticationError(e) from e

            logger.error(
                "Failed to generate suggestion for %s trigger (sentence %d): %s",
                trigger.type.value, trigger.sentence_index, e,
                exc_info=True,
            )
            return None
