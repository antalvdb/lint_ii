"""
Suggestion engine for LiNT-II readability improvements.

Identifies triggers from linguistic analysis and generates suggestions
using LLM providers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING
import logging
import os
import re
import uuid

from lint_ii.llm.providers import LLMProvider, LLMTimeoutError, create_provider
from lint_ii.llm.prompts import format_prompt, parse_block_response, parse_llm_response, parse_spelling_response

logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


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
    # Consolidated per-sentence rewrite addressing several sentence-level issues
    # at once (see _plan_jobs / sentence_rewrite prompt). Phase 1 scaffolding.
    SENTENCE_REWRITE = "sentence_rewrite"
    # Cross-sentence coherence: add a missing connective between two adjacent
    # sentences (may merge them). Detected per paragraph, LLM-driven. Gated
    # behind LINT_II_CONNECTIVES; frontend accept path is not wired yet.
    CONNECTIVE = "connective"
    # Re-present a long in-sentence enumeration as a bulleted list (lead-in +
    # items). Structural output, not an inline rewrite. Gated behind
    # LINT_II_ENUMERATIONS; frontend block-accept path is not wired yet (phase 1).
    ENUMERATION = "enumeration"


# Default thresholds for triggering suggestions
DEFAULT_THRESHOLDS: dict[str, float] = {
    "word_frequency": 3.0,           # Zipf frequency below this triggers suggestion
    "max_sdl": 5,                    # SDL above this triggers suggestion
    "content_words_per_clause": 7,   # Content words/clause above this triggers
    "abstract_noun_ratio": 0.7,      # Abstract ratio above this (concrete < 30%)
    "sentence_length": 25,           # Words above this triggers suggestion
    "n_subordinate_clauses": 1,      # More than this many subordinate clauses triggers
    "enumeration_min_items": 3,      # Coordinated items needed to suggest a bullet list
    "enumeration_min_span_words": 12, # Token span the enumeration must cover
}


def _enumerations_enabled() -> bool:
    """Whether the enumeration→bullet-list pass is on (LINT_II_ENUMERATIONS)."""
    return os.environ.get("LINT_II_ENUMERATIONS", "0").lower() in ("1", "true", "yes", "on")


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
    # For a consolidated sentence_rewrite: the LiNT trigger types it merged, so
    # the UI can name the underlying signals instead of a generic label.
    component_types: list[str] = field(default_factory=list)
    # For a connective suggestion: the sentence indices it spans (may be two,
    # when it merges a pair) and the discourse relation it makes explicit.
    merges_sentences: list[int] = field(default_factory=list)
    relation: str | None = None
    # Precomputed metrics for score recomputation
    new_sentence_metrics: dict[str, Any] | None = None
    # For a connective: exact metrics of the merged sentence when composed with an
    # accepted full rewrite of the first sentence, keyed by that rewrite's id. Lets
    # the UI score a composed merge precisely instead of reusing new_sentence_metrics.
    composed_metrics: dict[str, Any] = field(default_factory=dict)
    # For an enumeration: the lead-in (ending ":") and the list items. suggested_text
    # carries a plain-text rendering for the copy/export fallback.
    list_intro: str | None = None
    list_items: list[str] = field(default_factory=list)
    # For a consolidated sentence_rewrite offered as a choice: the alternative
    # rewrites, each {key, label, suggested_text, new_sentence_metrics}. Present
    # (>=2) only when a conservative one-sentence variant and a fuller (possibly
    # split) variant meaningfully differ; the top-level suggested_text mirrors the
    # full variant so variant-unaware code still works.
    variants: list[dict[str, Any]] = field(default_factory=list)

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
        if self.component_types:
            result["component_types"] = self.component_types
        if self.merges_sentences:
            result["merges_sentences"] = self.merges_sentences
        if self.relation is not None:
            result["relation"] = self.relation
        if self.new_sentence_metrics is not None:
            result["new_sentence_metrics"] = self.new_sentence_metrics
        if self.composed_metrics:
            result["composed_metrics"] = self.composed_metrics
        if self.list_intro is not None:
            result["list_intro"] = self.list_intro
        if self.list_items:
            result["list_items"] = self.list_items
        if self.variants:
            result["variants"] = self.variants
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


# Trigger types that rewrite a whole sentence. When consolidation is enabled,
# all of these for one sentence are merged into a single sentence_rewrite job.
# word_frequency is excluded — it stays a precise word-level suggestion.
SENTENCE_LEVEL_TRIGGER_TYPES = frozenset({
    SuggestionType.MAX_SDL,
    SuggestionType.CONTENT_WORDS_PER_CLAUSE,
    SuggestionType.ABSTRACT_NOUNS,
    SuggestionType.PASSIVE,
    SuggestionType.SUBORDINATE_CLAUSE,
    SuggestionType.SENTENCE_LENGTH,
})


# Word-frequency triggers per bundled LLM call. Response budget is ~150-200
# tokens per item (VERVANGING + short UITLEG + full rewritten fragment).
_WORDFREQ_BUNDLE_SIZE = 8


@dataclass
class SuggestionJob:
    """One planned LLM call.

    kind == "single":          one trigger, generated with that trigger's own
                               type-specific prompt (one suggestion).
    kind == "consolidated":    several sentence-level triggers for one sentence,
                               addressed together via the sentence_rewrite
                               prompt (one suggestion).
    kind == "wordfreq_bundle": up to _WORDFREQ_BUNDLE_SIZE word_frequency
                               triggers answered by one LLM call (one
                               suggestion per trigger).
    """
    kind: str
    sentence_index: int
    triggers: list[SuggestionTrigger]


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
        consolidate_sentence_rewrites: bool | None = None,
    ):
        """
        Initialize the suggestion engine.

        Args:
            provider: LLM provider for generating suggestions
            thresholds: Custom thresholds for triggers (uses defaults if not specified)
            consolidate_sentence_rewrites: when True, merge all sentence-level
                triggers for a sentence into a single rewrite (design #1). When
                None (default), read the LINT_CONSOLIDATE_REWRITES env var,
                defaulting to True if unset. Set the env var to a falsy value
                (0/false/no/off) to fall back to the per-trigger path for A/B.
        """
        self._provider = provider
        self._thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        if consolidate_sentence_rewrites is None:
            env = os.environ.get("LINT_CONSOLIDATE_REWRITES")
            # Consolidation is the default; an explicit env value can turn it off.
            consolidate_sentence_rewrites = (
                True if env is None else env.lower() in ("1", "true", "yes", "on")
            )
        self._consolidate_sentence_rewrites = consolidate_sentence_rewrites

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

            # Check for a long in-sentence enumeration (gated, off by default).
            if _enumerations_enabled():
                trigger = self._check_enumeration(sent_analysis, sent_idx, sentence_text)
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

    def _check_enumeration(
        self,
        sent_analysis: "SentenceAnalysis",
        sent_idx: int,
        sentence_text: str,
    ) -> SuggestionTrigger | None:
        """Detect a long in-sentence enumeration: a coordination of >= N phrase-
        level conjuncts spanning a meaningful part of the sentence. Precise by
        design (short natural lists like "koffie, thee en water" are skipped via
        the item-count and span gates). Fail-open on any parse issue."""
        min_items = int(self._thresholds["enumeration_min_items"])
        min_span = int(self._thresholds["enumeration_min_span_words"])
        try:
            doc = sent_analysis.doc
            # Group conjuncts by the base (first) member of their coordination.
            # Key by the base Token itself, not token.i: sent_analysis.doc is a
            # Span whose tokens carry document-global indices, so doc[token.i]
            # overflows the span for any sentence after the first ([E1002]).
            chains: dict = {}
            for tok in doc:
                if tok.dep_ != "conj":
                    continue
                base = tok.head
                guard = 0
                while base.dep_ == "conj" and guard < 50:
                    base = base.head
                    guard += 1
                chains.setdefault(base, []).append(tok)

            best = None  # (n_items, span_words)
            for base, conjs in chains.items():
                members = [base] + conjs
                n_items = len(members)
                if n_items < min_items:
                    continue
                # Phrase-level only: reject if any conjunct head is a FINITE verb
                # (coordinated clauses are max_sdl / subordinate_clause's job). A
                # nominalized infinitive ("het voorkomen van ...") is nominal and
                # stays eligible.
                if any(
                    m.pos_ in ("VERB", "AUX") and "Fin" in m.morph.get("VerbForm")
                    for m in members
                ):
                    continue
                idxs = [m.i for m in members]
                span_words = max(idxs) - min(idxs) + 1
                if span_words < min_span:
                    continue
                if best is None or n_items > best[0]:
                    best = (n_items, span_words)

            if best is None:
                return None
            return SuggestionTrigger(
                type=SuggestionType.ENUMERATION,
                sentence_index=sent_idx,
                sentence_text=sentence_text,
                feature_value=float(best[0]),
                threshold=float(min_items),
            )
        except Exception as e:  # never break trigger detection
            logger.warning("Enumeration check failed on sentence %d: %s", sent_idx, e)
            return None

    @staticmethod
    def _prioritize_triggers(
        triggers: list[SuggestionTrigger],
        max_suggestions: int | None,
    ) -> list[SuggestionTrigger]:
        """
        Prioritize triggers to ensure coverage across sentences and variety
        across suggestion types.

        Round-robins across sentences so every sentence with a trigger gets
        a suggestion before any sentence gets a second one. Within a sentence,
        higher-priority (sentence-level) types are chosen first. This prevents
        a single dense sentence from consuming the whole max_suggestions quota
        and starving later sentences.
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
        type_rank = {typ: i for i, typ in enumerate(type_priority)}

        # Group triggers by sentence; within each sentence, order by type priority
        by_sentence: dict[int, list[SuggestionTrigger]] = {}
        for trigger in triggers:
            by_sentence.setdefault(trigger.sentence_index, []).append(trigger)
        for sent_triggers in by_sentence.values():
            sent_triggers.sort(key=lambda t: type_rank.get(t.type, len(type_priority)))

        # Round-robin across sentences in document order
        result: list[SuggestionTrigger] = []
        sentence_order = sorted(by_sentence.keys())
        while len(result) < max_suggestions:
            progressed = False
            for sent_idx in sentence_order:
                if len(result) >= max_suggestions:
                    break
                if by_sentence[sent_idx]:
                    result.append(by_sentence[sent_idx].pop(0))
                    progressed = True
            if not progressed:
                break

        return result

    @staticmethod
    def _plan_jobs(
        triggers: list[SuggestionTrigger],
        max_suggestions: int | None,
        consolidate: bool,
    ) -> list["SuggestionJob"]:
        """
        Plan the set of LLM calls (jobs) to run, capped at max_suggestions jobs.

        consolidate=False (legacy): every selected trigger becomes its own
        "single" job, using the same round-robin selection as
        _prioritize_triggers, so behaviour is unchanged.

        consolidate=True: all sentence-level triggers for a sentence merge into
        one "consolidated" job; word_frequency triggers stay individual "single"
        jobs. Consolidated rewrites are scheduled first (one per sentence, in
        document order), then word_frequency jobs fill the remaining budget
        round-robin across sentences.

        max_suggestions caps the number of jobs (= LLM calls); None means no cap.
        """
        type_priority = [
            SuggestionType.SENTENCE_LENGTH,
            SuggestionType.PASSIVE,
            SuggestionType.SUBORDINATE_CLAUSE,
            SuggestionType.MAX_SDL,
            SuggestionType.CONTENT_WORDS_PER_CLAUSE,
            SuggestionType.ABSTRACT_NOUNS,
            SuggestionType.WORD_FREQUENCY,
        ]
        type_rank = {typ: i for i, typ in enumerate(type_priority)}

        if not consolidate:
            selected = SuggestionEngine._prioritize_triggers(triggers, max_suggestions)
            return [
                SuggestionJob(kind="single", sentence_index=t.sentence_index, triggers=[t])
                for t in selected
            ]

        # Consolidated mode: separate sentence-level rewrites from word_frequency
        rewrite_by_sentence: dict[int, list[SuggestionTrigger]] = {}
        wordfreq_by_sentence: dict[int, list[SuggestionTrigger]] = {}
        enumeration_triggers: list[SuggestionTrigger] = []
        for trigger in triggers:
            if trigger.type in SENTENCE_LEVEL_TRIGGER_TYPES:
                rewrite_by_sentence.setdefault(trigger.sentence_index, []).append(trigger)
            elif trigger.type == SuggestionType.WORD_FREQUENCY:
                wordfreq_by_sentence.setdefault(trigger.sentence_index, []).append(trigger)
            elif trigger.type == SuggestionType.ENUMERATION:
                # Structural output (a list), never consolidated into a prose rewrite.
                enumeration_triggers.append(trigger)
            # other types (e.g. spelling) are handled outside this planner

        # Order each sentence's rewrite triggers by type priority — this only
        # affects the order issues are presented to the model, not the outcome.
        for sent_triggers in rewrite_by_sentence.values():
            sent_triggers.sort(key=lambda t: type_rank.get(t.type, len(type_priority)))

        jobs: list[SuggestionJob] = []
        cap = max_suggestions if max_suggestions is not None else float("inf")

        # Round 1: one rewrite job per sentence with sentence-level issues, in
        # document order. A sentence with only ONE such issue keeps its targeted
        # per-type "single" job — a multi-fix rewrite over-rewrites simple
        # sentences into ungrammatical/meaning-shifted Dutch (observed on the
        # level-3 text). Only sentences with >=2 issues get a consolidated rewrite.
        for sent_idx in sorted(rewrite_by_sentence.keys()):
            if len(jobs) >= cap:
                break
            sent_triggers = rewrite_by_sentence[sent_idx]
            if len(sent_triggers) >= 2:
                jobs.append(SuggestionJob(
                    kind="consolidated",
                    sentence_index=sent_idx,
                    triggers=sent_triggers,
                ))
            else:
                jobs.append(SuggestionJob(
                    kind="single",
                    sentence_index=sent_idx,
                    triggers=[sent_triggers[0]],
                ))

        # Round 1b: one single job per enumeration (its own structured prompt).
        for trigger in sorted(enumeration_triggers, key=lambda t: t.sentence_index):
            if len(jobs) >= cap:
                break
            jobs.append(SuggestionJob(
                kind="single", sentence_index=trigger.sentence_index, triggers=[trigger],
            ))

        # Round 2+: word_frequency triggers, round-robin across sentences.
        # Selection (and the cap, which counts SUGGESTIONS) is unchanged, but
        # the selected triggers are bundled into groups of up to
        # _WORDFREQ_BUNDLE_SIZE, each answered by ONE LLM call that returns a
        # block per word — same suggestions, far fewer calls.
        selected_wf: list[SuggestionTrigger] = []
        wf_order = sorted(wordfreq_by_sentence.keys())
        while len(jobs) + len(selected_wf) < cap:
            progressed = False
            for sent_idx in wf_order:
                if len(jobs) + len(selected_wf) >= cap:
                    break
                bucket = wordfreq_by_sentence[sent_idx]
                if bucket:
                    selected_wf.append(bucket.pop(0))
                    progressed = True
            if not progressed:
                break

        for i in range(0, len(selected_wf), _WORDFREQ_BUNDLE_SIZE):
            group = selected_wf[i:i + _WORDFREQ_BUNDLE_SIZE]
            jobs.append(SuggestionJob(
                kind="wordfreq_bundle",
                sentence_index=group[0].sentence_index,
                triggers=group,
            ))

        return jobs

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
        # Chunk the document: a single whole-document call risks truncating
        # its response at max_tokens on long texts (errors beyond the cutoff
        # are silently lost). Sentences keep their GLOBAL 1-based numbers so
        # ZIN_NUMMER maps back directly regardless of chunk.
        _CHUNK_SENTENCES = 25
        sentence_analyses = analysis.sentence_analyses
        parsed_errors: list[dict[str, str]] = []
        for start in range(0, len(sentence_analyses), _CHUNK_SENTENCES):
            chunk = sentence_analyses[start:start + _CHUNK_SENTENCES]
            chunk_text = "\n".join(
                f"{start + i + 1}. {sa.doc.text}" for i, sa in enumerate(chunk)
            )
            system_prompt, user_prompt = format_prompt("spelling", text=chunk_text)

            try:
                response = provider.complete(user_prompt, system_prompt, max_tokens=1024)
            except LLMTimeoutError:
                # A wedged/timed-out provider must fail the whole job visibly,
                # not degrade it to an analysis with fewer suggestions.
                raise
            except Exception as e:
                err_str = str(e).lower()
                if "authentication" in err_str or "401" in err_str or "api_key" in err_str:
                    raise _AuthenticationError(e) from e
                logger.error(
                    "Spelling pass failed for sentences %d-%d: %s",
                    start + 1, start + len(chunk), e, exc_info=True,
                )
                continue

            logger.debug("Spelling LLM response (sentences %d-%d):\n%s",
                         start + 1, start + len(chunk), response.content)
            parsed_errors.extend(parse_spelling_response(response.content))

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

            # The model sometimes writes the correction in "oud → nieuw
            # *(of alternatief)*" notation or as a mini-rewrite with
            # commentary. A correction must be a drop-in replacement for
            # WOORD: reject meta-notation outright, and corrections that are
            # far longer than the word they replace.
            if any(marker in correction for marker in ("→", "->", "(of ", "*(")):
                logger.info(
                    "Spelling suggestion discarded: meta-notation in correction '%s'",
                    correction[:80],
                )
                continue
            if len(correction.split()) > max(4, 2 * len(word.split()) + 2):
                logger.info(
                    "Spelling suggestion discarded: correction too long for '%s': '%s'",
                    word, correction[:80],
                )
                continue

            # Map category to normalized value
            if "spel" in category_raw:
                error_category = "spelling"
            else:
                error_category = "grammar"

            sent_text = analysis.sentence_analyses[sent_num].doc.text
            # Build the corrected sentence by replacing the first occurrence
            suggested_text = sent_text.replace(word, correction, 1)

            # Skip if the replacement had no effect (word not found in text)
            if suggested_text == sent_text:
                logger.debug("Spelling suggestion skipped: '%s' not found in sentence", word)
                continue

            # Skip if the LLM inserted a placeholder marker instead of a real correction
            if "niet toegepast" in suggested_text.lower():
                logger.info("Spelling suggestion discarded: placeholder marker in suggested text for '%s'", word)
                continue

            # Find the word index in the sentence tokens
            word_index = None
            for wi, wf in enumerate(analysis.sentence_analyses[sent_num].word_features):
                if wf.text == word:
                    word_index = wi
                    break

            # Skip if word not found in tokens (index mismatch would break highlighting)
            if word_index is None:
                logger.debug("Spelling suggestion skipped: '%s' not found in token list", word)
                continue

            # Discard suggestions where both the original and the correction are valid
            # Dutch words (per Hunspell) and they differ by more than 2 characters.
            # This catches false positives like "over" → "voor" where the LLM
            # substitutes one valid word for another rather than fixing a real error.
            # Edit-distance ≤ 2 is kept to preserve form changes like word/wordt.
            try:
                from lint_ii.llm.hunspell_spelling import _get_dictionary as _get_hunspell
                _hd = _get_hunspell()
                if _hd.lookup(word) and _hd.lookup(correction):
                    _dist = _levenshtein(word.lower(), correction.lower())
                    if _dist > 2:
                        logger.info(
                            "Spelling suggestion discarded: both '%s' and '%s' are valid "
                            "Dutch words (edit distance %d)", word, correction, _dist,
                        )
                        continue
            except Exception:
                pass

            # For spelling (not grammar) suggestions, skip if the correction is not
            # more frequent than the original — this filters LLM hallucinations where
            # a correctly-spelled but rare word is "corrected" to an equally rare one.
            # (Kept as a strict > rather than a frequency-band test: a real spelling
            # fix between same-band words is still worth applying.)
            if error_category == "spelling":
                from lint_ii.linguistic_data.wordlists import FREQ_DATA
                zero_count_freq = 1.359228547196266
                original_freq = FREQ_DATA.get(word.lower(), zero_count_freq)
                correction_freq = FREQ_DATA.get(correction.lower(), zero_count_freq)
                if correction_freq <= original_freq:
                    logger.info(
                        "Spelling suggestion skipped: correction '%s' (%.2f) not more "
                        "frequent than original '%s' (%.2f)",
                        correction, correction_freq, word, original_freq,
                    )
                    continue

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
                model=provider.model_name,
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

        # Step 1a: LLM spelling/grammar pass (single call for entire document)
        t0 = time.perf_counter()
        spelling_suggestions = self.generate_spelling_suggestions(analysis, provider)
        t1 = time.perf_counter()
        logger.info("TIMING spelling_llm=%.2fs (%d suggestions)", t1 - t0, len(spelling_suggestions))

        # Step 1b: Hunspell spelling pass — high-precision rule-based check,
        # skips words already flagged by the LLM to avoid duplicates
        from lint_ii.llm.hunspell_spelling import generate_hunspell_suggestions
        llm_covered = {
            (s.sentence_index, s.word_index)
            for s in spelling_suggestions
            if s.word_index is not None
        }
        hunspell_suggestions = generate_hunspell_suggestions(analysis, llm_covered)
        spelling_suggestions = spelling_suggestions + hunspell_suggestions
        logger.info("TIMING spelling_hunspell=%.2fs (%d suggestions)", time.perf_counter() - t1, len(hunspell_suggestions))

        # Step 2: Find readability triggers and plan the LLM calls (jobs).
        # When consolidation is on, sentence-level triggers for a sentence are
        # merged into one rewrite job; otherwise each trigger is its own job.
        triggers = self.identify_triggers(analysis)
        jobs = self._plan_jobs(triggers, max_suggestions, self._consolidate_sentence_rewrites)
        t2 = time.perf_counter()
        logger.info(
            "TIMING trigger_detection=%.2fs (%d triggers → %d jobs, consolidate=%s)",
            t2 - t1, len(triggers), len(jobs), self._consolidate_sentence_rewrites,
        )

        # Step 3: Generate a readability suggestion for each planned job
        document_level = getattr(analysis.lint, "level", None)
        suggestions: list[Suggestion] = list(spelling_suggestions)
        for job in jobs:
            t_job = time.perf_counter()
            try:
                if job.kind == "wordfreq_bundle":
                    new_suggestions = self._generate_wordfreq_bundle(job, provider, document_level)
                else:
                    single = self._generate_suggestion_for_job(job, provider, document_level)
                    new_suggestions = [single] if single else []
            except _AuthenticationError as e:
                raise RuntimeError(
                    f"LLM authentication failed: {e}. Check your API key."
                ) from e
            if job.kind == "consolidated":
                label = "consolidated"
            elif job.kind == "wordfreq_bundle":
                label = f"wordfreq_bundle_{len(job.triggers)}"
            else:
                label = job.triggers[0].type.value
            logger.info("TIMING job_%s=%.2fs", label, time.perf_counter() - t_job)
            suggestions.extend(new_suggestions)

        # An enumeration suggestion IS the rewrite for its sentence: a bulleted
        # list. Drop competing full-sentence prose rewrites for the same
        # sentence so the list is the single clear offer -- the alternative
        # ("Ten eerste... Ten tweede...") reads as wordy and tends to raise the
        # LiNT score. Only sentences where the enumeration LLM actually produced
        # a list are affected; a detector that fired but returned no list
        # (SuggestionType.ENUMERATION absent) leaves the prose rewrite intact.
        # Runs before the connective pass so it never composes with a rewrite
        # that is about to be removed.
        enum_sentences = {
            s.sentence_index
            for s in suggestions
            if s.type == SuggestionType.ENUMERATION
        }
        if enum_sentences:
            before = len(suggestions)
            suggestions = [
                s
                for s in suggestions
                if not (
                    s.type in self._FULL_REWRITE_TYPES
                    and s.sentence_index in enum_sentences
                )
            ]
            if len(suggestions) != before:
                logger.info(
                    "enumeration suppressed %d full-rewrite suggestion(s) on sentence(s) %s",
                    before - len(suggestions),
                    sorted(enum_sentences),
                )

        # Step 4: cross-sentence coherence pass (gated behind LINT_II_CONNECTIVES,
        # fail-open). Adds connective suggestions that span sentence pairs.
        t_conn = time.perf_counter()
        connective_suggestions = self.generate_connective_suggestions(
            analysis, provider, existing=suggestions,
        )
        if connective_suggestions:
            logger.info(
                "TIMING connective_pass=%.2fs (%d suggestions)",
                time.perf_counter() - t_conn, len(connective_suggestions),
            )
        suggestions.extend(connective_suggestions)

        return SuggestionsResult(
            suggestions=suggestions,
            triggers_found=len(triggers),
            triggers_processed=sum(len(j.triggers) for j in jobs),
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

    @staticmethod
    def _append_level_constraint(system_prompt: str, document_level: int | None) -> str:
        """Append the 'aim one level lower, don't over-simplify' instruction."""
        if document_level is None:
            return system_prompt
        target_level = max(1, document_level - 1)
        return system_prompt + (
            f"\n\nDe tekst heeft LiNT-niveau {document_level} (schaal 1–4, waarbij 4 het moeilijkst is). "
            f"Streef naar een herschrijving die de complexiteit met één niveau verlaagt (naar niveau {target_level}). "
            f"Vereenvoudig niet verder dan nodig — behoud de toon, stijl en vakinhoud van de originele tekst zo veel mogelijk."
        )

    # Clause-coordinating conjunctions whose mid-sentence join should not be
    # broken: they encode an argumentative relation (contrast/reason/consequence)
    # that is lost when the clauses are split into separate sentences. en/of are
    # excluded — they are often phrase-level ("koffie en thee"), so guarding them
    # would reject too many legitimate edits.
    _CLAUSE_CONJUNCTIONS = ("maar", "want", "dus")

    @classmethod
    def _breaks_clause_conjunction(cls, original: str, suggested: str) -> str | None:
        """Return the conjunction whose ', <conj> ' join the rewrite broke, else None.

        Deterministic backstop for the prompt guideline: if the original joins
        two clauses with ', maar ' (or want/dus) and the suggestion no longer
        contains that exact join, the rewrite split the clauses or swapped the
        conjunction (e.g. maar -> echter), which we reject.
        """
        orig_l = original.lower()
        sug_l = suggested.lower()
        for conj in cls._CLAUSE_CONJUNCTIONS:
            join = f", {conj} "
            if join in orig_l and join not in sug_l:
                return conj
        return None

    # URLs and e-mail addresses are not prose: a rewrite must keep them
    # byte-for-byte. Trailing sentence punctuation is ignored when comparing so
    # an original "...phishing" still matches a kept "...phishing".
    _URL_RE = re.compile(
        r"(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+\.[\w.-]+",
        re.IGNORECASE,
    )
    _URL_TRAIL_PUNCT = ".,;:!?)]"

    @classmethod
    def _alters_url(cls, original: str, suggested: str) -> str | None:
        """Return a URL/e-mail the rewrite failed to keep verbatim, else None.

        Deterministic backstop for the prompt guideline: every URL or e-mail
        address in the original must appear unchanged in the suggestion. If one
        is dropped, truncated or reworded, we reject the rewrite.
        """
        for match in cls._URL_RE.finditer(original):
            url = match.group(0).rstrip(cls._URL_TRAIL_PUNCT)
            if url and url not in suggested:
                return url
        return None

    @staticmethod
    def _in_higher_freq_band(candidate_freq: float, original_freq: float) -> bool:
        """True if candidate sits in a higher Zipf frequency band than original.

        Zipf scores are rounded to the nearest integer band, so words that
        differ only marginally count as "equally frequent" and do not warrant a
        substitution. This guards against swaps that are barely more common but
        semantically wrong (e.g. "uitstoot" 2.86 → "uitlaat" 3.38, both band 3),
        while still allowing genuine jumps (e.g. band 2 → band 3).
        """
        return round(candidate_freq) > round(original_freq)

    # A single-token word at least this long is treated as a likely compound that
    # may be split into a clearer word group (see _replacement_passes_band).
    _COMPOUND_MIN_LEN = 15

    def _replacement_passes_band(
        self, trigger_word: str, replacement_word: str | None, original_freq: float
    ) -> bool:
        """Whether a word_frequency replacement is worth keeping.

        A single-word replacement must sit in a higher Zipf band than the
        original (a marginal or equal-band swap isn't worth the risk). A
        MULTI-word replacement is a rephrasing, not a swap — a phrase has no
        single frequency to band-check — and is accepted only when the original
        is a long compound with no simple synonym, which is exactly the case a
        split ("levensmiddelendistributiecentrum" -> "centrum voor de distributie
        van levensmiddelen") is meant to solve. Other backstops (misspelling,
        clause-join, URL) still apply to the rewrite."""
        if not replacement_word:
            return True
        rep = replacement_word.strip()
        if " " in rep:
            return len(trigger_word or "") >= self._COMPOUND_MIN_LEN
        from lint_ii.linguistic_data.wordlists import FREQ_DATA
        zero_count_freq = 1.359228547196266
        return self._in_higher_freq_band(
            FREQ_DATA.get(rep.lower(), zero_count_freq), original_freq
        )

    # Split a rewrite into word tokens, stripping surrounding punctuation.
    _TOKEN_TRIM_RE = re.compile(r"^[^0-9A-Za-zÀ-ſ]+|[^0-9A-Za-zÀ-ſ]+$")

    @classmethod
    def _word_tokens(cls, text: str) -> list[str]:
        return [cls._TOKEN_TRIM_RE.sub("", t) for t in text.split()]

    @classmethod
    def _introduces_misspelling(cls, original: str, suggested: str) -> str | None:
        """Return a NEW misspelled token the rewrite introduced, else None.

        Deterministic safety net against gross corruption: a rewrite must not
        introduce a token that is absent from the original, unknown to both the
        Dutch Hunspell dictionary and SUBTLEX-NL, and not a likely proper noun.
        Conservative on purpose (skips capitalised, short, non-alphabetic and
        known-frequent tokens) so it never rejects a legitimate rewrite.
        """
        try:
            from lint_ii.llm.hunspell_spelling import _get_dictionary
            from lint_ii.linguistic_data.wordlists import FREQ_DATA
            dictionary = _get_dictionary()
        except Exception as e:  # dictionary/data unavailable — fail open
            logger.warning("Misspelling backstop unavailable: %s", e)
            return None

        original_tokens = {t.lower() for t in cls._word_tokens(original)}
        for raw in cls._word_tokens(suggested):
            low = raw.lower()
            if not raw or low in original_tokens:
                continue
            if len(raw) < 3 or not raw.isalpha():
                continue
            if raw[0].isupper():          # likely a proper noun
                continue
            if low in FREQ_DATA:          # a known Dutch word by frequency
                continue
            if dictionary.lookup(raw):    # valid per Hunspell
                continue
            return raw
        return None

    # A definite/demonstrative/possessive determiner makes an -e adjective
    # correct, so its presence suppresses the de/het check below.
    _DEFINITE_DET_TAG_PREFIXES = ("LID|bep", "VNW|aanw", "VNW|bez")

    @classmethod
    def _dehet_disagreement(cls, text: str) -> str | None:
        """Return an adjective+noun pair with wrong de/het inflection, else None.

        Conservative check for the one unambiguous over-inflection direction:
        a positive-degree prenominal adjective carrying -e ('met-e') on an
        INDEFINITE, SINGULAR, NEUTER noun, where Dutch requires the bare form
        (e.g. 'buitenlandse bezit' → should be 'buitenlands bezit', Henk zin 5).
        The reverse direction is deliberately left alone — materials/-en
        adjectives ('houten', 'gouden') are legitimately uninflected — so this
        does not reject good rewrites. Validated to 0 false positives on a
        battery of correct sentences.
        """
        try:
            from lint_ii.core.readability_analysis import ReadabilityAnalysis
            analysis = ReadabilityAnalysis.from_text(text)
        except Exception as e:  # parse failure — fail open, never reject
            logger.warning("de/het backstop parse failed: %s", e)
            return None

        for sent in analysis.sentence_analyses:
            for tok in sent.doc:
                if tok.pos_ != "ADJ" or tok.dep_ != "amod":
                    continue
                if not {"prenom", "basis", "met-e"} <= set(tok.tag_.split("|")):
                    continue
                noun = tok.head
                if noun.pos_ != "NOUN" or noun.i < tok.i:
                    continue
                if not {"ev", "onz"} <= set(noun.tag_.split("|")):
                    continue
                if any(
                    child.dep_ in ("det", "nmod:poss")
                    and any(child.tag_.startswith(p) for p in cls._DEFINITE_DET_TAG_PREFIXES)
                    for child in noun.children
                ):
                    continue
                return f"{tok.text} {noun.text}"
        return None

    # ── Connective (coherence) pass ──────────────────────────────────────
    # Discourse connectives, lowercased. Used to (a) skip a boundary whose
    # second sentence already opens with one, and (b) allow these words to be
    # added by a rewrite without tripping the "invented content" backstop.
    _CONNECTIVE_LEXICON = frozenset({
        "want", "omdat", "doordat", "aangezien", "zodat", "waardoor", "dus",
        "daarom", "hierdoor", "daardoor", "maar", "echter", "toch", "hoewel",
        "terwijl", "immers", "namelijk", "bovendien", "daarnaast", "verder",
        "ook", "vervolgens", "daarna", "kortom", "derhalve", "bijgevolg",
        "desondanks", "niettemin", "integendeel", "sterker",
    })

    @staticmethod
    def _connective_paragraphs(analysis: "ReadabilityAnalysis") -> list[list[int]]:
        """Group the document into paragraphs: maximal runs of consecutive
        sentence entries in the layout (never crossing a heading, blank, list
        item or quote). Returns lists of global sentence indices."""
        paragraphs: list[list[int]] = []
        current: list[int] = []
        for entry in analysis.layout:
            if entry.get("type") == "sentence":
                current.append(entry["sentence_index"])
            else:
                if current:
                    paragraphs.append(current)
                    current = []
        if current:
            paragraphs.append(current)
        return paragraphs

    @classmethod
    def _connective_candidates(cls, analysis: "ReadabilityAnalysis", para: list[int]) -> list[int]:
        """Return the 0-based positions p in `para` where the boundary between
        para[p] and para[p+1] is a candidate for a connective: both sentences
        are declarative and long enough, and the second doesn't already open
        with a connective. Cheap deterministic pre-filter to bound LLM calls."""
        def declarative_and_long(sent) -> bool:
            text = sent.doc.text.strip()
            if text.endswith("?"):
                return False
            content = [t for t in sent.doc if not t.is_punct and not t.is_space]
            return len(content) >= 4

        def opens_with_connective(sent) -> bool:
            for t in sent.doc:
                if t.is_punct or t.is_space:
                    continue
                return t.lower_ in cls._CONNECTIVE_LEXICON
            return False

        candidates = []
        for p in range(len(para) - 1):
            a = analysis.sentences[para[p]]
            b = analysis.sentences[para[p + 1]]
            if declarative_and_long(a) and declarative_and_long(b) and not opens_with_connective(b):
                candidates.append(p)
        return candidates

    @staticmethod
    def _is_recompound(word: str, orig_tokens: set[str]) -> bool:
        """True if ``word`` is exactly two original tokens concatenated — a
        separable verb the merge glued back together ("kapot" + "ging" ->
        "kapotging", "kapot" + "gemaakt" -> "kapotgemaakt"). Merging with a
        subordinating connective forces verb-final order, which routinely
        recompounds separable verbs; such a word introduces no new content, so
        it must not trip the invented-content guard. Both halves must be real
        original tokens of >=2 chars, keeping this from matching arbitrary
        substrings of a genuinely new word."""
        for i in range(2, len(word) - 1):
            if word[:i] in orig_tokens and word[i:] in orig_tokens:
                return True
        return False

    @classmethod
    def _connective_adds_content(cls, original: str, suggested: str) -> str | None:
        """Return a content word the rewrite introduced that is neither in the
        original pair nor an allowed connective, else None. Guards against the
        model inventing content or smuggling in a relation's facts."""
        orig = {t.lower() for t in cls._word_tokens(original)}
        for raw in cls._word_tokens(suggested):
            low = raw.lower()
            if low in orig or low in cls._CONNECTIVE_LEXICON:
                continue
            if len(raw) < 4 or not raw.isalpha():   # allow short function words
                continue
            if raw[0].isupper():                     # proper noun / sentence start
                continue
            if cls._is_recompound(low, orig):        # separable verb re-glued
                continue
            return raw
        return None

    def generate_connective_suggestions(
        self,
        analysis: "ReadabilityAnalysis",
        provider: LLMProvider,
        existing: list[Suggestion] | None = None,
    ) -> list[Suggestion]:
        """Cross-sentence coherence pass: per paragraph, one LLM call proposes
        connectives for the candidate boundaries. Gated behind LINT_II_CONNECTIVES
        (default off) and fail-open — any error yields no connective suggestions
        rather than breaking the main pass.

        ``existing`` are the suggestions already generated this run; a connective
        uses them to precompute exact metrics for composing with a first-sentence
        rewrite (see _build_connective_suggestion)."""
        if os.environ.get("LINT_II_CONNECTIVES", "0").lower() not in ("1", "true", "yes", "on"):
            return []

        suggestions: list[Suggestion] = []
        try:
            paragraphs = self._connective_paragraphs(analysis)
        except Exception as e:
            logger.warning("Connective pass: paragraph grouping failed: %s", e)
            return []

        for para in paragraphs:
            if len(para) < 2:
                continue
            try:
                candidates = self._connective_candidates(analysis, para)
                if not candidates:
                    continue
                numbered = "\n".join(
                    f"{i + 1}. {analysis.sentences[g].doc.text}" for i, g in enumerate(para)
                )
                boundaries = ", ".join(str(p + 1) for p in candidates)
                system_prompt, user_prompt = format_prompt(
                    "connective", paragraph=numbered, boundaries=boundaries,
                )
                response = provider.complete(user_prompt, system_prompt)
                blocks = parse_block_response(
                    response.content,
                    fields=["NA_ZIN", "RELATIE", "HERSCHRIJVING", "UITLEG"],
                    required="NA_ZIN",
                )
                for block in blocks:
                    sug = self._build_connective_suggestion(
                        analysis, para, set(candidates), block, response.model,
                        existing or [],
                    )
                    if sug:
                        suggestions.append(sug)
            except LLMTimeoutError:
                raise
            except Exception as e:
                logger.warning("Connective pass failed for a paragraph: %s", e)
                continue

        return suggestions

    # Full-sentence rewrite types a connective can compose with (mirrors the
    # frontend SENTENCE_SCOPED_TYPES). A rewrite of the first sentence carries
    # the connective, so its composed merge can be scored exactly.
    _FULL_REWRITE_TYPES = frozenset({
        SuggestionType.SENTENCE_REWRITE,
        SuggestionType.MAX_SDL,
        SuggestionType.CONTENT_WORDS_PER_CLAUSE,
        SuggestionType.ABSTRACT_NOUNS,
        SuggestionType.PASSIVE,
        SuggestionType.SUBORDINATE_CLAUSE,
        SuggestionType.SENTENCE_LENGTH,
    })

    @classmethod
    def _connective_inserted_word(cls, original_pair: str, suggested: str) -> str | None:
        """The word the connective inserts: the first word of the merge not in the
        original pair (mirrors the frontend _connectiveWord)."""
        strip = lambda t: cls._TOKEN_TRIM_RE.sub("", t).lower()
        orig = {strip(t) for t in original_pair.split()}
        for tok in suggested.split():
            b = strip(tok)
            if b and b not in orig:
                return b
        return None

    @classmethod
    def _compose_merge_text(
        cls, original_pair: str, suggested: str, rewrite_text: str
    ) -> str | None:
        """Graft a first-sentence rewrite onto the connective: rewritten first
        clause (minus terminal punctuation) + the connective tail. Mirrors the
        frontend _composedMergeText. Returns None when the join can't be located."""
        cw = cls._connective_inserted_word(original_pair, suggested)
        if not cw:
            return None
        sep = suggested.lower().find(", " + cw + " ")
        if sep < 0:
            return None
        tail = suggested[sep:]
        first_clause = re.sub(r"[.!?]+$", "", rewrite_text.strip())
        return first_clause + tail

    def _build_connective_suggestion(
        self,
        analysis: "ReadabilityAnalysis",
        para: list[int],
        candidate_positions: set[int],
        block: dict[str, str],
        model: str | None,
        existing: list[Suggestion],
    ) -> Suggestion | None:
        """Validate one parsed connective block and build a Suggestion, or None."""
        m = re.search(r"\d+", block.get("NA_ZIN", ""))
        if not m:
            return None
        pos = int(m.group()) - 1
        if pos not in candidate_positions or pos + 1 >= len(para):
            return None

        n, n1 = para[pos], para[pos + 1]
        suggested = block.get("HERSCHRIJVING", "").strip().strip('"“”')
        if not suggested or "niet toegepast" in suggested.lower():
            return None

        original_pair = f"{analysis.sentences[n].doc.text} {analysis.sentences[n1].doc.text}"
        if self._connective_adds_content(original_pair, suggested):
            logger.info("Connective discarded: introduced content, sentences %d-%d", n, n1)
            return None
        if self._introduces_misspelling(original_pair, suggested):
            return None
        if self._dehet_disagreement(suggested):
            return None

        # Precompute exact metrics for composing with each full rewrite of the
        # FIRST sentence, so the UI scores that combination precisely rather than
        # reusing the (original-based) merge metrics.
        composed: dict[str, Any] = {}
        for s in existing:
            if s.sentence_index != n or s.type not in self._FULL_REWRITE_TYPES:
                continue
            composed_text = self._compose_merge_text(original_pair, suggested, s.suggested_text)
            if not composed_text:
                continue
            metrics = self._analyze_suggested_text(composed_text)
            if metrics is not None:
                composed[s.id] = metrics

        return Suggestion(
            id=str(uuid.uuid4())[:8],
            type=SuggestionType.CONNECTIVE,
            sentence_index=n,
            original_text=original_pair,
            suggested_text=suggested,
            explanation=block.get("UITLEG", ""),
            model=model,
            merges_sentences=[n, n1],
            relation=(block.get("RELATIE") or None),
            new_sentence_metrics=self._analyze_suggested_text(suggested),
            composed_metrics=composed,
        )

    # ── Enumeration (bullet-list) pass ────────────────────────────────────

    @staticmethod
    def _parse_enumeration_response(content: str) -> tuple[str | None, list[str], str]:
        """Parse an enumeration response into (intro, items, uitleg). Collects the
        repeated ITEM: lines in order; tolerant of stray list markers."""
        intro: str | None = None
        items: list[str] = []
        uitleg = ""
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("---"):
                continue
            low = line.lower()
            if low.startswith("inleiding:"):
                intro = line.split(":", 1)[1].strip().strip('"“”')
            elif low.startswith("item:"):
                it = line.split(":", 1)[1].strip().lstrip("-•*").strip().strip('"“”')
                if it:
                    items.append(it)
            elif low.startswith("uitleg:"):
                uitleg = line.split(":", 1)[1].strip()
        return intro, items, uitleg

    @classmethod
    def _enumeration_adds_content(
        cls, original: str, intro: str, items: list[str]
    ) -> str | None:
        """Guard against a fabricated list ITEM, while allowing the paraphrasing
        the tool exists to do. Each item must stay anchored to the original by
        sharing at least one content word (>= 4 letters) with it; an item with no
        such anchor is treated as invented and returned. The intro is a lead-in
        that legitimately rephrases ("... richt zich op vier punten:") and light
        synonym simplification inside an anchored item ("periodiek" -> "regelmatig")
        is fine, so neither is word-for-word contained. Returns the first
        unanchored item, else None."""
        orig = {t.lower() for t in cls._word_tokens(original)}
        for item in items:
            anchored = any(
                len(raw) >= 4 and raw.isalpha() and raw.lower() in orig
                for raw in cls._word_tokens(item)
            )
            if not anchored:
                return item
        return None

    def _generate_enumeration_suggestion(
        self, trigger: SuggestionTrigger, provider: LLMProvider
    ) -> Suggestion | None:
        """Re-present a long enumeration as a bullet list. Fail-open; a containment
        backstop rejects any invented content."""
        try:
            system_prompt, user_prompt = format_prompt(
                "enumeration", sentence=trigger.sentence_text,
            )
            response = provider.complete(user_prompt, system_prompt)
        except LLMTimeoutError:
            raise
        except Exception as e:
            logger.warning("Enumeration generation failed: %s", e)
            return None

        intro, items, uitleg = self._parse_enumeration_response(response.content)
        min_items = int(self._thresholds["enumeration_min_items"])
        if not intro or len(items) < min_items:
            return None
        if intro and not intro.rstrip().endswith(":"):
            intro = intro.rstrip() + ":"

        original = trigger.sentence_text or ""
        invented = self._enumeration_adds_content(original, intro, items)
        if invented:
            logger.info(
                "Enumeration discarded: introduced content (%r), sentence %d",
                invented, trigger.sentence_index,
            )
            return None

        plain = intro + "\n" + "\n".join(f"- {it}" for it in items)
        flattened = intro + " " + " ".join(items)
        return Suggestion(
            id=str(uuid.uuid4())[:8],
            type=SuggestionType.ENUMERATION,
            sentence_index=trigger.sentence_index,
            original_text=original,
            suggested_text=plain,
            explanation=uitleg,
            model=response.model,
            list_intro=intro,
            list_items=items,
            new_sentence_metrics=self._analyze_suggested_text(flattened),
        )

    @staticmethod
    def _format_issue(trigger: SuggestionTrigger) -> str | None:
        """Render one sentence-level trigger as a Dutch bullet for the rewrite prompt."""
        t = trigger.type
        if t == SuggestionType.SENTENCE_LENGTH:
            return f"De zin is lang ({int(trigger.feature_value)} woorden)."
        if t == SuggestionType.PASSIVE:
            if trigger.passives:
                joined = ", ".join(f'"{p}"' for p in trigger.passives)
                return f"De zin bevat passieve constructie(s): {joined}."
            return "De zin bevat een passieve constructie."
        if t == SuggestionType.SUBORDINATE_CLAUSE:
            return f"De zin bevat {int(trigger.feature_value)} bijzin(nen)."
        if t == SuggestionType.MAX_SDL:
            return (
                f"De zin heeft een complexe structuur met lange afhankelijkheden tussen woorden "
                f"(maximale afhankelijkheidslengte {int(trigger.feature_value)})."
            )
        if t == SuggestionType.CONTENT_WORDS_PER_CLAUSE:
            return (
                f"De zin heeft een hoge informatiedichtheid "
                f"({trigger.feature_value:.1f} inhoudswoorden per deelzin)."
            )
        if t == SuggestionType.ABSTRACT_NOUNS:
            if trigger.abstract_nouns:
                return f"De zin bevat abstracte woorden: {', '.join(trigger.abstract_nouns)}."
            return "De zin bevat abstracte taal."
        return None

    def _generate_suggestion_for_job(
        self,
        job: "SuggestionJob",
        provider: LLMProvider,
        document_level: int | None = None,
    ) -> Suggestion | None:
        """Dispatch a planned job to the right generation path."""
        if job.kind == "consolidated":
            return self._generate_consolidated_suggestion(job, provider, document_level)
        if job.triggers and job.triggers[0].type == SuggestionType.ENUMERATION:
            return self._generate_enumeration_suggestion(job.triggers[0], provider)
        return self._generate_suggestion_for_trigger(job.triggers[0], provider, document_level)

    @staticmethod
    def _clean_variant(sentence_text: str, raw: str) -> str:
        """Strip wrapping quotes/markdown emphasis the LLM may add, unless the
        original had them (e.g. a VOLLEDIG value returned as *...* )."""
        quotes = "\"\u201c\u201d\u2018\u2019\u0027*_"
        text = (raw or "").strip()
        if text and not sentence_text.startswith(tuple(quotes)):
            text = text.lstrip(quotes)
        if text and not sentence_text.endswith(tuple(quotes)):
            text = text.rstrip(quotes)
        return text.strip()

    def _rewrite_backstop_failure(self, sentence_text: str, candidate: str) -> str | None:
        """The first hard-backstop reason a rewrite candidate fails, else None.
        Shared by both consolidated variants so each is validated independently."""
        if not candidate:
            return "empty"
        if "niet toegepast" in candidate.lower():
            return "placeholder marker"
        conj = self._breaks_clause_conjunction(sentence_text, candidate)
        if conj:
            return f"broke ', {conj} ' clause join"
        url = self._alters_url(sentence_text, candidate)
        if url:
            return f"URL not preserved ({url})"
        typo = self._introduces_misspelling(sentence_text, candidate)
        if typo:
            return f"introduced misspelling '{typo}'"
        disagreement = self._dehet_disagreement(candidate)
        if disagreement:
            return f"de/het disagreement '{disagreement}'"
        return None

    def _generate_consolidated_suggestion(
        self,
        job: "SuggestionJob",
        provider: LLMProvider,
        document_level: int | None = None,
    ) -> Suggestion | None:
        """Generate one rewrite for a sentence addressing all its bundled issues."""
        if not job.triggers:
            return None
        sentence_text = job.triggers[0].sentence_text

        issue_lines = [
            line for trigger in job.triggers
            if (line := self._format_issue(trigger)) is not None
        ]
        if not issue_lines:
            return None
        issues = "\n".join(f"- {line}" for line in issue_lines)

        try:
            system_prompt, user_prompt = format_prompt(
                "sentence_rewrite", sentence=sentence_text, issues=issues,
            )
            system_prompt = self._append_level_constraint(system_prompt, document_level)

            response = provider.complete(user_prompt, system_prompt)
            logger.debug(
                "LLM response for consolidated rewrite (sentence %d):\n%s",
                job.sentence_index, response.content,
            )
            parsed = parse_llm_response(response.content, "sentence_rewrite")
            explanation = parsed.get("UITLEG", "")

            # Two offered rewrites: BEHOUDEND (one sentence, no split) and
            # VOLLEDIG (may split). Fall back to the old single HERSCHRIJVING
            # field when the model does not produce the pair.
            conservative = self._clean_variant(sentence_text or "", parsed.get("BEHOUDEND", ""))
            full = self._clean_variant(
                sentence_text or "", parsed.get("VOLLEDIG") or parsed.get("HERSCHRIJVING", ""))

            def _norm(t):
                return " ".join(t.lower().split())

            survivors = []
            seen = set()
            for key, label, text in (
                ("full", "Volledig", full),
                ("conservative", "Behoudend", conservative),
            ):
                if not text:
                    continue
                reason = self._rewrite_backstop_failure(sentence_text, text)
                if reason:
                    logger.info(
                        "Consolidated %s variant discarded (%s) for sentence %d",
                        key, reason, job.sentence_index,
                    )
                    continue
                norm = _norm(text)
                if norm in seen:
                    continue
                seen.add(norm)
                survivors.append({
                    "key": key,
                    "label": label,
                    "suggested_text": text,
                    "new_sentence_metrics": self._analyze_suggested_text(text),
                })

            if not survivors:
                logger.warning(
                    "Consolidated rewrite: no usable variant for sentence %d. Raw:\n%s",
                    job.sentence_index, response.content,
                )
                return None

            # Primary (applied by variant-unaware code / accept-all) = the full
            # variant when present, else the sole survivor. Offer a choice only
            # when both a conservative and a full variant survived and differ.
            primary = next((s for s in survivors if s["key"] == "full"), survivors[0])
            variants = (
                sorted(survivors, key=lambda s: 0 if s["key"] == "conservative" else 1)
                if len(survivors) >= 2 else []
            )

            return Suggestion(
                id=str(uuid.uuid4())[:8],
                type=SuggestionType.SENTENCE_REWRITE,
                sentence_index=job.sentence_index,
                original_text=sentence_text,
                suggested_text=primary["suggested_text"],
                explanation=explanation,
                model=response.model,
                component_types=list(dict.fromkeys(t.type.value for t in job.triggers)),
                new_sentence_metrics=primary["new_sentence_metrics"],
                variants=variants,
            )
        except LLMTimeoutError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "authentication" in err_str or "401" in err_str or "api_key" in err_str:
                raise _AuthenticationError(e) from e
            logger.error(
                "Failed to generate consolidated rewrite for sentence %d: %s",
                job.sentence_index, e, exc_info=True,
            )
            return None

    def _generate_wordfreq_bundle(
        self,
        job: SuggestionJob,
        provider: LLMProvider,
        document_level: int | None = None,
    ) -> list[Suggestion]:
        """Generate word-swap suggestions for several word_frequency triggers
        with ONE LLM call. Each trigger yields its own suggestion and passes
        the same validation as the per-trigger path."""
        triggers = job.triggers
        items = "\n\n".join(
            f'{i + 1}. WOORD: "{t.word}" (frequentie {t.feature_value:.2f})\n'
            f'   FRAGMENT: "{t.context or t.sentence_text}"'
            for i, t in enumerate(triggers)
        )
        try:
            system_prompt, user_prompt = format_prompt(
                "word_frequency_bundle", n_items=len(triggers), items=items,
            )
            system_prompt = self._append_level_constraint(system_prompt, document_level)
            response = provider.complete(
                user_prompt, system_prompt,
                max_tokens=min(2048, 200 * len(triggers) + 150),
            )
        except LLMTimeoutError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "authentication" in err_str or "401" in err_str or "api_key" in err_str:
                raise _AuthenticationError(e) from e
            logger.error("Failed to generate word-frequency bundle: %s", e, exc_info=True)
            return []

        logger.debug("Word-frequency bundle response:\n%s", response.content)
        blocks = parse_block_response(
            response.content,
            fields=["NUMMER", "VERVANGING", "UITLEG", "HERSCHRIJVING"],
            required="NUMMER",
        )
        suggestions: list[Suggestion] = []
        seen: set[int] = set()
        for block in blocks:
            match = re.search(r"\d+", block.get("NUMMER", ""))
            if match is None:
                continue
            item_idx = int(match.group()) - 1
            if item_idx < 0 or item_idx >= len(triggers) or item_idx in seen:
                continue
            seen.add(item_idx)
            suggestion = self._build_wordfreq_suggestion(
                triggers[item_idx], block, provider.model_name,
            )
            if suggestion:
                suggestions.append(suggestion)
        if len(suggestions) < len(triggers):
            logger.info(
                "Word-frequency bundle: %d of %d items yielded a suggestion",
                len(suggestions), len(triggers),
            )
        return suggestions

    def _build_wordfreq_suggestion(
        self,
        trigger: SuggestionTrigger,
        parsed: dict[str, str],
        model_name: str,
    ) -> Suggestion | None:
        """Validate one parsed word-swap block into a Suggestion, applying the
        same filters as the per-trigger word_frequency path (frequency band,
        placeholder markers, clause-conjunction and URL preservation)."""
        suggested_text = parsed.get("HERSCHRIJVING", "")
        original = trigger.sentence_text or ""
        _quotes = '"“”‘’\''
        if suggested_text and not original.startswith(tuple(_quotes)):
            suggested_text = suggested_text.lstrip(_quotes)
        if suggested_text and not original.endswith(tuple(_quotes)):
            suggested_text = suggested_text.rstrip(_quotes)
        explanation = parsed.get("UITLEG", "")
        replacement_word = parsed.get("VERVANGING")

        if not suggested_text:
            logger.warning(
                "No HERSCHRIJVING in bundle block for word %r", trigger.word,
            )
            return None
        if "niet toegepast" in suggested_text.lower():
            return None

        if replacement_word and not self._replacement_passes_band(
            trigger.word, replacement_word, trigger.feature_value
        ):
            logger.info(
                "Dropping bundled word_frequency suggestion: %r -> %r not in a higher band",
                trigger.word, replacement_word,
            )
            return None

        if self._breaks_clause_conjunction(original, suggested_text):
            return None
        if self._alters_url(original, suggested_text):
            return None
        if self._introduces_misspelling(original, suggested_text):
            return None

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
            model=model_name,
            new_sentence_metrics=self._analyze_suggested_text(suggested_text),
        )

    def _generate_suggestion_for_trigger(
        self,
        trigger: SuggestionTrigger,
        provider: LLMProvider,
        document_level: int | None = None,
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

            # Append level constraint so the LLM aims one level lower, not maximally simpler
            system_prompt = self._append_level_constraint(system_prompt, document_level)

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
            original = trigger.sentence_text or ""
            _quotes = '"""''\''
            if suggested_text and not original.startswith(tuple(_quotes)):
                suggested_text = suggested_text.lstrip(_quotes)
            if suggested_text and not original.endswith(tuple(_quotes)):
                suggested_text = suggested_text.rstrip(_quotes)
            explanation = parsed.get("UITLEG", "")
            replacement_word = parsed.get("VERVANGING")

            # For word_frequency suggestions, require the replacement to sit in a
            # higher Zipf frequency band than the original. A merely marginal gain
            # (same band) is not worth the risk of a semantically wrong but
            # slightly-more-common swap (e.g. "uitstoot" → "uitlaat"), and an
            # equally-rare swap (e.g. "beslistermijnen" → "beslissingstermijnen")
            # is worthless.
            if (trigger.type == SuggestionType.WORD_FREQUENCY and replacement_word
                    and not self._replacement_passes_band(
                        trigger.word, replacement_word, trigger.feature_value)):
                logger.info(
                    "Dropping word_frequency suggestion: replacement '%s' for '%s' (%.2f) "
                    "not in a higher frequency band (and not a compound split)",
                    replacement_word, trigger.word, trigger.feature_value,
                )
                return None

            if not suggested_text:
                logger.warning(
                    "No HERSCHRIJVING found in LLM response for %s trigger. "
                    "Parsed fields: %s. Raw response:\n%s",
                    trigger.type.value, parsed, response.content,
                )
                return None

            if "niet toegepast" in suggested_text.lower():
                logger.info(
                    "Trigger suggestion discarded: placeholder marker in HERSCHRIJVING for %s trigger",
                    trigger.type.value,
                )
                return None

            broken_conj = self._breaks_clause_conjunction(trigger.sentence_text or "", suggested_text)
            if broken_conj:
                logger.info(
                    "Trigger suggestion discarded: %s rewrite broke ', %s ' clause join",
                    trigger.type.value, broken_conj,
                )
                return None

            altered_url = self._alters_url(trigger.sentence_text or "", suggested_text)
            if altered_url:
                logger.info(
                    "Trigger suggestion discarded: %s rewrite did not preserve URL (%s)",
                    trigger.type.value, altered_url,
                )
                return None

            typo = self._introduces_misspelling(trigger.sentence_text or "", suggested_text)
            if typo:
                logger.info(
                    "Trigger suggestion discarded: %s rewrite introduced misspelling '%s'",
                    trigger.type.value, typo,
                )
                return None

            disagreement = self._dehet_disagreement(suggested_text)
            if disagreement:
                logger.info(
                    "Trigger suggestion discarded: %s rewrite has de/het disagreement '%s'",
                    trigger.type.value, disagreement,
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

        except LLMTimeoutError:
            raise
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
