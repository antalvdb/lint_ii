from operator import itemgetter
from functools import cached_property
from typing import Any, TypedDict, TYPE_CHECKING
import re
import statistics

from lint_ii.core.preprocessor import preprocess_text, fix_quotemarks, extract_text_from_node
from lint_ii.core.word_features import WordFeatures
from lint_ii.core.sentence_analysis import SentenceAnalysis
from lint_ii.core.lint_scorer import LintScorer
from lint_ii.core.sentence_analysis import SentenceAnalysis, SentenceAnalysisDict

from lint_ii.visualization.html import LintIIVisualizer

if TYPE_CHECKING:
    from lint_ii.llm.suggestions import SuggestionsResult


class DocumentStatsDict(TypedDict):
    sentence_count: int
    document_lint_score: float | None
    document_difficulty_level: int | None
    min_lint_score: float | None
    max_lint_score: float | None


class ReadabilityAnalysisDict(TypedDict):
    sentences: list[SentenceAnalysisDict]
    blocks: list[dict[str, Any]]
    document_lint_score: float | None
    document_difficulty_level: int | None
    sentence_count: int
    min_lint_score: float | None
    max_lint_score: float | None


# --- Structure-aware segmentation (H3: preserve document structure) ----------
# A block is treated as prose only if it ends like a sentence (final
# punctuation, ignoring trailing quotes/brackets). Headings, salutations,
# labels and captions usually lack sentence-final punctuation; they are kept
# verbatim as non-prose so they are neither merged into a neighbouring
# sentence, rewritten, nor split. Blank lines are preserved as separators.
# Quote chars below are written as escapes to avoid editor quote mangling:
# "=double quote, '=apostrophe, ”/’=curly closers.
_SENTENCE_FINAL = (".", "!", "?", "…")
_CLOSERS = "\u0022\u0027)]\u201d\u2019 "


# A URL or e-mail address at the very end is a valid sentence ending that
# carries no final punctuation. A line ending this way is prose (so a whole
# paragraph that happens to end in a link is still analysed in full), and the
# URL itself must be left untouched — never append a period to it.
_URL_OR_EMAIL_END_RE = re.compile(
    r"(?:https?://|www\.)\S+$|[\w.+-]+@[\w-]+\.[\w.-]+$",
    re.IGNORECASE,
)
# An internal sentence boundary (final punctuation followed by whitespace) means
# the line already contains at least one complete sentence, so it is prose even
# if it ends on a colon — e.g. a line that introduces a list or a link.
_INTERNAL_SENTENCE_RE = re.compile(r"[.!?…]\s")


def _ends_like_sentence(line: str) -> bool:
    stripped = line.rstrip(_CLOSERS)
    if not stripped:
        return False
    if stripped[-1] in _SENTENCE_FINAL:
        return True
    if _INTERNAL_SENTENCE_RE.search(line):
        return True
    # A sentence whose final token is a URL/e-mail (no closing period) is prose
    # only when there is real text before the link; a bare URL on its own line
    # is not prose and stays excluded.
    match = _URL_OR_EMAIL_END_RE.search(stripped)
    return bool(match) and bool(stripped[: match.start()].strip())


def _segment_blocks(text: str) -> list[dict[str, Any]]:
    """Split raw text into ordered structural blocks without flattening it.

    Each block is one of: {"type": "prose", "text": ...} (fed to spaCy),
    {"type": "heading", "text": ...} (non-prose, kept verbatim), or
    {"type": "blank"} (a paragraph separator). Runs of blank lines collapse to
    a single separator; leading/trailing blanks are dropped.
    """
    blocks: list[dict[str, Any]] = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            if blocks and blocks[-1]["type"] != "blank":
                blocks.append({"type": "blank"})
            continue
        norm = fix_quotemarks(re.sub(r"[ \t]+", " ", raw_line.strip()))
        kind = "prose" if _ends_like_sentence(norm) else "heading"
        blocks.append({"type": kind, "text": norm})
    while blocks and blocks[-1]["type"] == "blank":
        blocks.pop()
    return blocks


# Runs of letters (no digits/underscore) — used to tell a real (if unscoreable)
# sentence from a fragment.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _is_unscoreable_fragment(text: str) -> bool:
    """Whether an unscoreable line is a non-prose fragment rather than body text.

    Used only for sentences the scorer returns no level for. Such a line is a
    fragment when it has no alphabetic word (a bare number), ends in a colon (a
    label or list/intro), or is very short (<= 3 words). Longer unscoreable
    prose (e.g. a plain sentence the LiNT formula cannot score) is kept.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return True
    if text.rstrip().endswith(":"):
        return True
    return len(words) <= 3


def _markdown_block_text(node: dict) -> str:
    """Flatten a Markdown AST node to clean inline text."""
    raw = extract_text_from_node(node)
    return fix_quotemarks(re.sub(r"\s+", " ", raw)).strip()


def _is_bold_only_paragraph(node: dict) -> bool:
    """True if a paragraph is entirely bold (a single strong span, modulo
    whitespace). Many documents style a heading with bold rather than a Heading
    style, so pandoc emits it as a **...** paragraph rather than a # heading;
    we treat such a paragraph as a heading."""
    has_strong = False
    for child in node.get("children", []):
        ctype = child.get("type")
        if ctype == "strong":
            has_strong = True
        elif ctype == "text" and not child.get("raw", "").strip():
            continue
        else:
            return False
    return has_strong


def _segment_blocks_from_markdown(md_text: str) -> list[dict[str, Any]]:
    """Segment Markdown into the same block list as _segment_blocks, but driven
    by the structure encoded in the Markdown (headings, lists, block quotes)
    rather than line heuristics. Intended for input produced by converting a
    source document (e.g. a .docx via pandoc), where the real structure exists.

    Paragraphs and list items become prose (analysed; list items keep their
    bullet/number marker). Headings and block quotes are kept verbatim as
    excluded non-prose blocks. Code blocks, horizontal rules and raw HTML are
    dropped.
    """
    import mistune
    ast = mistune.create_markdown(renderer="ast")(md_text)

    blocks: list[dict[str, Any]] = []

    def add_blank() -> None:
        if blocks and blocks[-1]["type"] != "blank":
            blocks.append({"type": "blank"})

    for node in ast:
        kind = node.get("type")
        if kind == "paragraph":
            text = _markdown_block_text(node)
            if not text:
                continue
            # A paragraph that is entirely bold, or that does not read like a
            # sentence (e.g. a salutation ending in a comma), is a heading the
            # source styled visually rather than with a Heading style — exclude
            # it the way the plain-text path would.
            if _is_bold_only_paragraph(node) or not _ends_like_sentence(text):
                blocks.append({"type": "heading", "text": text})
            else:
                blocks.append({"type": "prose", "text": text})
        elif kind == "heading":
            text = _markdown_block_text(node)
            if text:
                blocks.append({"type": "heading", "text": text})
        elif kind == "list":
            ordered = node.get("attrs", {}).get("ordered", False)
            for number, item in enumerate(node.get("children", []), start=1):
                text = _markdown_block_text(item)
                if text:
                    blocks.append({
                        "type": "list_item", "text": text,
                        "ordered": ordered, "number": number,
                    })
            add_blank()
        elif kind == "block_quote":
            text = _markdown_block_text(node)
            if text:
                blocks.append({"type": "quote", "text": text})
        elif kind == "blank_line":
            add_blank()
        # block_code, thematic_break, block_html, etc. are ignored.

    while blocks and blocks[-1]["type"] == "blank":
        blocks.pop()
    return blocks


class ReadabilityAnalysis(LintIIVisualizer):
    """
    Document-level readability analysis for Dutch texts using the LiNT-II formula.

    This class analyzes documents by aggregating sentence-level features and 
    computing readability scores based on four linguistic features: word frequency, 
    syntactic dependency length, content words per clause, and proportion of concrete nouns.

    Parameters
    ----------
    sentences : list[SentenceAnalysis]
        List of sentence-level analysis objects. Each sentence must be a 
        SentenceAnalysis instance containing linguistic features and metadata.

    Attributes & Properties
    -----------------------
    sentences : list[SentenceAnalysis]
        The input sentence analyses.
    word_features : list[WordFeatures]
        Flattened list of all word features across sentences.
    concrete_nouns : list[WordFeatures]
        All concrete nouns in the document.
    abstract_nouns : list[WordFeatures]
        All abstract nouns in the document.
    undefined_nouns : list[WordFeatures]
        All undefined nouns in the document (have both a concrete and an abstract meaning).
    mean_log_word_frequency : float | None
        Document-level mean log frequency of content words (excluding proper nouns).
        Returns None if there are no frequencies on the sentence-level. Cached property.
    mean_max_sdl : float | None
        Mean of maximum syntactic dependency lengths across sentences.
        Returns None if there are no SDLs on the sentence-level. Cached property.
    mean_content_words_per_clause : float | None
        Mean content words per clause across sentences.
        Returns None if there are no content words / clause on the sentence-level. Cached property.
    proportion_of_concrete_nouns : float | None
        Proportion of concrete nouns out of the total nouns in the document.
        Nouns of type `unknown` (not in the list) are excluded from the totals count.
        Returns None if totals are 0, i.e. there are no nouns or only `unknown` nouns in the sentence. Cached property.
    lint : LintScorer
        LintScorer object that contains the score (lint.score) and the difficulty level (lint.level) for the document. Cached property.
    lint_scores_per_sentence : list[float]
        Individual LiNT scores for each sentence in the document. Cached property.
    min_lint_score : float | None
        Lowest sentence-level score in the document.
        Returns None if there are no sentence-level scores. Cached property.
    max_lint_score : float | None
        Highest sentence-level score in the document.
        Returns None if there are no sentence-level scores. Cached property.
    entities_and_situations : list[WordFeatures]
        Bag of entities and situations for the document.
    contextually_new : list[WordFeatures]
        Bag of contextually new words in the document.

    Methods
    -------
    from_text(text: str) -> ReadabilityAnalysis
        Create analysis from text string. Preprocesses text and applies spaCy NLP pipeline.
    get_top_n_least_frequent -> list[tuple[WordFeatures, float]]
        Get the top n least frequent words in the document.
    calculate_document_stats() -> DocumentStatsDict
        Generate summary statistics including sentence count, mean/min/max scores.
    get_detailed_analysis() -> dict[str, Any]
        Return comprehensive analysis with both document and sentence-level details.
    as_dict() -> ReadabilityAnalysisDict
        Serialize analysis to dictionary format (used in the LiNT-II visualizer).

    Examples
    --------
    >>> from lint_ii import ReadabilityAnalysis
    >>> text = "Jip zit bij de kapper. Knip, knap, zegt de schaar."
    >>> analysis = ReadabilityAnalysis.from_text(text)
    >>> analysis.lint.score
    21.9
    >>> analysis.lint.level
    1
    >>> stats = analysis.calculate_document_stats()
    >>> stats['sentence_count']
    2

    See Also
    --------
    SentenceAnalysis : Sentence-level readability analysis
    WordFeatures : Token-level linguistic feature extraction
    LintScorer : LiNT scoring algorithms
    """

    def __init__(
        self,
        sentences: list[SentenceAnalysis],
        layout: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sentences = sentences
        # Ordered document layout interleaving prose sentences (referenced by
        # index) with non-prose headings and blank-line separators. Defaults to
        # the sentences in order when not supplied (e.g. tests constructing the
        # object directly), keeping old callers working.
        self.layout = layout if layout is not None else [
            {"type": "sentence", "sentence_index": i}
            for i in range(len(sentences))
        ]
        for sent in self.sentences:
            sent.readability_analysis = self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({[repr(s.doc) for s in self.sentences]})"

    @classmethod
    def from_text(cls, text: str) -> 'ReadabilityAnalysis':
        """Create analysis from plain text, detecting structure with line
        heuristics (headings, blank lines) — see _segment_blocks."""
        return cls._from_blocks(_segment_blocks(text))

    @classmethod
    def from_markdown(cls, md_text: str) -> 'ReadabilityAnalysis':
        """Create analysis from Markdown, using the structure encoded in the
        Markdown (headings, lists, block quotes) instead of line heuristics.
        Intended for input produced by converting a source document such as a
        .docx via pandoc — see _segment_blocks_from_markdown."""
        return cls._from_blocks(_segment_blocks_from_markdown(md_text))

    @classmethod
    def _from_blocks(cls, blocks: list[dict[str, Any]]) -> 'ReadabilityAnalysis':
        """Build the analysis from an ordered block list. Prose blocks and list
        items are sentence-segmented with spaCy (list items keep their marker);
        headings, block quotes and blanks are preserved verbatim as excluded
        layout."""
        from lint_ii.linguistic_data.nlp_model import NLP_MODEL

        sentence_final = {".", "!", "?"}
        sentences: list[SentenceAnalysis] = []
        layout: list[dict[str, Any]] = []

        def segment(text: str) -> list[int]:
            """spaCy-segment a block, append its sentences, return their indices.
            The short-line merge runs only WITHIN the block, so it can never glue
            a heading onto a following sentence the way a whole-document pass would."""
            doc = NLP_MODEL(text)
            raw_sents = list(doc.sents)
            merged = []
            i = 0
            while i < len(raw_sents):
                sent = raw_sents[i]
                real_toks = [t for t in sent if not t.is_punct and not t.is_space]
                if len(real_toks) <= 2 and sent[-1].text not in sentence_final and i + 1 < len(raw_sents):
                    merged.append(doc[sent.start:raw_sents[i + 1].end])
                    i += 2
                else:
                    merged.append(sent)
                    i += 1
            indices = []
            for span in merged:
                indices.append(len(sentences))
                sentences.append(SentenceAnalysis(span))
            return indices

        for block in blocks:
            bt = block["type"]
            if bt == "prose":
                for idx in segment(block["text"]):
                    layout.append({"type": "sentence", "sentence_index": idx})
            elif bt == "list_item":
                indices = segment(block["text"])
                if indices:
                    layout.append({
                        "type": "list_item",
                        "ordered": block.get("ordered", False),
                        "number": block.get("number", 1),
                        "sentence_indices": indices,
                    })
            elif bt == "blank":
                layout.append({"type": "blank"})
            elif bt == "quote":
                # Quotes are someone else's words — kept verbatim, not analysed.
                layout.append({"type": "quote", "text": block["text"]})
            else:  # heading and any other excluded text block
                layout.append({"type": "heading", "text": block["text"]})

        analysis = cls(sentences, layout=layout)
        analysis._exclude_unscoreable_fragments()
        return analysis

    def _exclude_unscoreable_fragments(self) -> None:
        """Reclassify unscoreable prose fragments as excluded non-prose blocks.

        A prose sentence the scorer cannot level AND that looks like a fragment
        (see ``_is_unscoreable_fragment`` — bare number, colon label, or <= 3
        words) is moved out of ``sentences`` into the layout as a non-prose
        block, so it renders as excluded text instead of a broken grey "?"
        sentence, gets no suggestions, and does not affect document scores.
        Longer unscoreable prose is left as a normal sentence.
        """
        drop = {
            i for i, sent in enumerate(self.sentences)
            if sent.lint.level is None and _is_unscoreable_fragment(sent.doc.text)
        }
        if not drop:
            return

        # Remap kept sentences to their new indices.
        remap: dict[int, int] = {}
        new_sentences: list[SentenceAnalysis] = []
        for i, sent in enumerate(self.sentences):
            if i not in drop:
                remap[i] = len(new_sentences)
                new_sentences.append(sent)

        new_layout: list[dict[str, Any]] = []
        for entry in self.layout:
            t = entry.get("type")
            if t == "sentence":
                i = entry["sentence_index"]
                if i in drop:
                    new_layout.append({"type": "heading", "text": self.sentences[i].doc.text})
                else:
                    new_layout.append({"type": "sentence", "sentence_index": remap[i]})
            elif t == "list_item":
                kept = [remap[i] for i in entry["sentence_indices"] if i not in drop]
                if kept:
                    new_entry = dict(entry)
                    new_entry["sentence_indices"] = kept
                    new_layout.append(new_entry)
                else:
                    text = " ".join(self.sentences[i].doc.text for i in entry["sentence_indices"])
                    new_layout.append({"type": "heading", "text": text})
            else:
                new_layout.append(entry)
        self.layout = new_layout
        self.sentences = new_sentences

    @property
    def word_features(self) -> list[WordFeatures]:
        """Bag of word features for the document."""
        return [
            feat
            for sentence in self.sentences
            for feat in sentence.word_features
        ]

    @property
    def concrete_nouns(self) -> list[WordFeatures]:
        """Bag of concrete nouns for the document."""
        return [
            noun
            for sentence in self.sentences
            for noun in sentence.concrete_nouns
        ]

    @property
    def abstract_nouns(self) -> list[WordFeatures]:
        """Bag of abstract nouns for the document."""
        return [
            noun
            for sentence in self.sentences
            for noun in sentence.abstract_nouns
        ]
    
    @property
    def undefined_nouns(self) -> list[WordFeatures]:
        """Bag of undefined nouns for the document."""
        return [
            noun
            for sentence in self.sentences
            for noun in sentence.undefined_nouns
        ]

    @cached_property
    def mean_log_word_frequency(self) -> float | None:
        """
        Mean log word frequency for the document.
        Returns None if there are no frequencies on the sentence-level.
        """
        frequencies = [
            freq
            for feat in self.word_features
            if (freq := feat.word_frequency) is not None
        ]
        if not frequencies:
            return None
        return statistics.mean(frequencies)

    @cached_property
    def mean_max_sdl(self) -> float | None:
        """
        Mean value of sentence-level maximum dependency lengths.
        Returns None if there are no SDLs on the sentence-level.
        """
        sdls = [s.max_sdl for s in self.sentences if s.max_sdl is not None]
        if not sdls:
            return None
        return statistics.mean(sdls)

    @cached_property
    def mean_content_words_per_clause(self) -> float | None:
        """
        Mean value of sentence-level content words per clause.
        Returns None if there are no content words / clause on the sentence-level.
        """
        content_words_per_clause = [
            s.content_words_per_clause
            for s in self.sentences
            if s.content_words_per_clause is not None
        ]
        if not content_words_per_clause:
            return None
        return statistics.mean(content_words_per_clause)

    @cached_property
    def proportion_of_concrete_nouns(self) -> float | None:
        """
        Proportion of concrete nouns out of the total nouns in the document.
        Nouns of type `unknown` (not in the list) are excluded from the totals count.
        Returns None if totals are 0, i.e. there are no nouns or only `unknown` nouns in the sentence.
        """
        n_concrete_nouns = len(self.concrete_nouns)
        n_abstract_nouns = len(self.abstract_nouns)
        n_undefined_nouns = len(self.undefined_nouns)
        total_nouns = n_concrete_nouns + n_abstract_nouns + n_undefined_nouns
        if total_nouns == 0:
            return None
        return n_concrete_nouns / total_nouns

    @cached_property
    def lint(self) -> LintScorer:
        return LintScorer(
            freq_log = self.mean_log_word_frequency,
            max_sdl = self.mean_max_sdl,
            content_words_per_clause = self.mean_content_words_per_clause,
            proportion_concrete = self.proportion_of_concrete_nouns,
        )

    @cached_property
    def lint_scores_per_sentence(self) -> list[float]:
        return [
            sent.lint.score
            for sent in self.sentences
            if sent.lint.score is not None
        ]

    @cached_property
    def min_lint_score(self) -> float | None:
        """
        Lowest sentence-level score in the document.
        Returns None if there are no sentence-level scores.
        """
        return min(self.lint_scores_per_sentence, default=None)

    @cached_property
    def max_lint_score(self) -> float | None:
        """
        Highest sentence-level score in the document.
        Returns None if there are no sentence-level scores.
        """
        return max(self.lint_scores_per_sentence, default=None)

    @property
    def entities_and_situations(self) -> list[WordFeatures]:
        """Bag of entities and situations for the document."""
        return [feat for feat in self.word_features if feat.is_entity_or_situation]

    @property
    def contextually_new(self) -> list[WordFeatures]:
        """Bag of contextually new words in the document."""
        return [feat for feat in self.word_features if feat.is_contextually_new]

    def get_top_n_least_frequent(self, n: int = 5) -> list[tuple[WordFeatures, float]]:
        """Get the top n least frequent words in the document."""
        frequencies = {
            feat:freq
            for feat in self.word_features
            if (freq := feat.word_frequency) is not None
        }
        if n == -1:
            return sorted(frequencies.items(), key=itemgetter(1))
        return sorted(frequencies.items(), key=itemgetter(1))[:n]

    def calculate_document_stats(self) -> DocumentStatsDict:
        """
        Statistics on a document level (sentence count, document LiNT score, document difficulty level, min LiNT score, max LiNT score).
        """
        return {
            'sentence_count': len(self.sentences),
            'document_lint_score': self.lint.score,
            'document_difficulty_level': self.lint.level,
            'min_lint_score': self.min_lint_score,
            'max_lint_score': self.max_lint_score,
        }
    
    def get_detailed_analysis(self, n: int = 5) -> dict[str, Any]:
        """Get detailed readability analysis per sentence in the document."""
        return {
            'document_stats': self.calculate_document_stats(),
            'sentence_stats': [
                sent.get_detailed_analysis(n=n)
                for sent in self.sentences
            ],
            'contextually_new_words': [feat.text for feat in self.contextually_new],
        }

    def as_dict(self) -> ReadabilityAnalysisDict:
        doc_stats = self.calculate_document_stats()
        return {
            'sentences': [sent.as_dict() for sent in self.sentences],
            "blocks": self.layout,
            'document_lint_score': doc_stats['document_lint_score'],
            'document_difficulty_level': doc_stats['document_difficulty_level'],
            'sentence_count': doc_stats['sentence_count'],
            'min_lint_score': doc_stats['min_lint_score'],
            'max_lint_score': doc_stats['max_lint_score'],
        }

    @property
    def sentence_analyses(self) -> list[SentenceAnalysis]:
        """Alias for sentences property, used by SuggestionEngine."""
        return self.sentences

    def generate_suggestions(
        self,
        llm_config: dict[str, Any] | None = None,
        thresholds: dict[str, float] | None = None,
        max_suggestions: int | None = None,
    ) -> "SuggestionsResult":
        """
        Generate LLM-powered suggestions for improving readability.

        This method analyzes the text to identify potential readability issues
        and uses an LLM to generate specific improvement suggestions.

        Parameters
        ----------
        llm_config : dict, optional
            LLM provider configuration. Keys:
            - provider: 'openai', 'anthropic', or 'ollama' (default: 'openai')
            - api_key: API key (uses env var if not provided)
            - model: Model name (uses provider default if not provided)
        thresholds : dict, optional
            Custom thresholds for triggering suggestions. Keys:
            - word_frequency: Zipf frequency below this triggers (default: 3.0)
            - max_sdl: SDL above this triggers (default: 5)
            - content_words_per_clause: Above this triggers (default: 7)
            - abstract_noun_ratio: Abstract ratio above this triggers (default: 0.7)
        max_suggestions : int, optional
            Maximum number of suggestions to generate. None for all triggers.

        Returns
        -------
        SuggestionsResult
            Object containing list of suggestions and metadata.

        Examples
        --------
        >>> analysis = ReadabilityAnalysis.from_text("Dutch text...")
        >>> suggestions = analysis.generate_suggestions(
        ...     llm_config={'provider': 'openai', 'api_key': 'sk-...'}
        ... )
        >>> len(suggestions.suggestions)
        3

        Notes
        -----
        Requires the `llm` optional dependencies: pip install lint_ii[llm]

        See Also
        --------
        with_suggestions : Display analysis with suggestions in editor mode
        """
        from lint_ii.llm.suggestions import SuggestionEngine

        engine = SuggestionEngine(thresholds=thresholds)
        return engine.generate_suggestions(
            analysis=self,
            max_suggestions=max_suggestions,
            llm_config=llm_config,
        )

    def as_dict_with_suggestions(
        self,
        suggestions: "SuggestionsResult",
    ) -> dict[str, Any]:
        """
        Serialize analysis to dictionary with suggestions included.

        Parameters
        ----------
        suggestions : SuggestionsResult
            Suggestions generated by generate_suggestions()

        Returns
        -------
        dict
            Analysis dictionary with 'suggestions' key added
        """
        result = dict(self.as_dict())
        result['suggestions'] = suggestions.as_dict()
        return result

    def with_suggestions(
        self,
        suggestions: "SuggestionsResult",
    ) -> "ReadabilityAnalysisWithSuggestions":
        """
        Create a visualization-ready object with suggestions included.

        This returns an object that displays in editor mode when rendered
        in Jupyter notebooks, showing suggestions as interactive highlights.

        Parameters
        ----------
        suggestions : SuggestionsResult
            Suggestions generated by generate_suggestions()

        Returns
        -------
        ReadabilityAnalysisWithSuggestions
            Object with _repr_html_() for Jupyter display in editor mode

        Examples
        --------
        >>> analysis = ReadabilityAnalysis.from_text("Dutch text...")
        >>> suggestions = analysis.generate_suggestions(llm_config={...})
        >>> analysis.with_suggestions(suggestions)  # Displays in Jupyter

        See Also
        --------
        generate_suggestions : Generate suggestions for the analysis
        """
        return ReadabilityAnalysisWithSuggestions(self, suggestions)


class ReadabilityAnalysisWithSuggestions(LintIIVisualizer):
    """
    Wrapper class for displaying ReadabilityAnalysis with suggestions.

    This class provides a _repr_html_() method that renders the analysis
    in editor mode, with suggestion highlights and interactive controls.
    """

    def __init__(
        self,
        analysis: ReadabilityAnalysis,
        suggestions: "SuggestionsResult",
    ):
        self._analysis = analysis
        self._suggestions = suggestions

    def as_dict(self) -> dict[str, Any]:
        """Return the combined analysis and suggestions data."""
        return self._analysis.as_dict_with_suggestions(self._suggestions)

    @property
    def mode(self) -> str:
        """Return the visualizer mode for this object."""
        return "editor"
