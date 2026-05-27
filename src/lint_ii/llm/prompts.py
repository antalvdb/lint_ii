"""
Dutch-language prompt templates for LiNT-II suggestion generation.

Each prompt template is designed to generate specific types of readability
improvements based on the linguistic features analyzed by LiNT-II.
"""

import re
from typing import TypedDict


class PromptTemplate(TypedDict):
    """Structure for a prompt template."""
    system: str
    user: str


SYSTEM_PROMPT_BASE = """Je bent een expert in begrijpelijk Nederlands schrijven. Je taak is om teksten leesbaarder te maken door gerichte, bescheiden aanpassingen.

Belangrijke richtlijnen:
- Blijf zo dicht mogelijk bij de originele zin — verander alleen wat echt nodig is
- Behoud de oorspronkelijke betekenis volledig
- Behoud de formele schrijfstijl en het register van de originele tekst
- Voeg geen nieuwe informatie, voorbeelden, metaforen of vergelijkingen toe
- Herschrijf niet creatief — pas aan, vervang of splits waar nodig, maar verzin niets nieuws
- Gebruik gangbaar Nederlands, maar pas het taalniveau aan aan de originele tekst
- Schrijf uitleg in eenvoudige, korte zinnen — vermijd vakjargon
- Antwoord altijd in het Nederlands"""


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "word_frequency": PromptTemplate(
        system=SYSTEM_PROMPT_BASE + """

Je richt je specifiek op het vervangen van infrequente of moeilijke woorden door meer gangbare synoniemen.""",
        user="""Herschrijf het volgende tekstfragment door het onderstreepte woord "{word}" te vervangen door een frequenter, begrijpelijker synoniem.

Tekstfragment: "{context}"

Het woord "{word}" heeft een lage woordfrequentie ({frequency:.2f} Zipf), wat betekent dat veel lezers dit woord mogelijk niet kennen.

Belangrijk: als de vervanging gevolgen heeft voor de grammaticale context (bijv. adjectief\u00adverbuiging, lidwoord de/het, meervoud/enkelvoud, werkwoordsvervoeging), pas dan ook de omringende woorden aan zodat de zin grammaticaal correct blijft.

Geef je antwoord in het volgende formaat:
VERVANGING: [het nieuwe woord of de nieuwe woordgroep]
UITLEG: [één korte, eenvoudige zin die uitlegt waarom het nieuwe woord makkelijker is — schrijf alsof je het aan een leek uitlegt]
HERSCHRIJVING: [het volledige herschreven tekstfragment met alle noodzakelijke grammaticale aanpassingen]"""
    ),

    "max_sdl": PromptTemplate(
        system=SYSTEM_PROMPT_BASE + """

Je richt je specifiek op het vereenvoudigen van zinnen met complexe zinsstructuren en lange afhankelijkheden tussen woorden.""",
        user="""Herschrijf de volgende zin om de zinsstructuur te vereenvoudigen. De zin heeft een hoge syntactische complexiteit (maximale afhankelijkheidslengte: {max_sdl}).

Zin: "{sentence}"

Lange afhankelijkheden tussen woorden maken het moeilijker om de zin te begrijpen. Probeer:
- Woorden die bij elkaar horen dichter bij elkaar te plaatsen
- De zin eventueel op te splitsen in kortere zinnen
- Een directere woordvolgorde te gebruiken

Geef je antwoord in het volgende formaat:
PROBLEEM: [korte beschrijving van wat de zin complex maakt]
HERSCHRIJVING: [de vereenvoudigde zin of zinnen]
UITLEG: [één korte zin die alleen beschrijft wat er structureel gedaan is, zoals herstructureren of opsplitsen — geen uitleg over het waarom, geen vakjargon]"""
    ),

    "content_words_per_clause": PromptTemplate(
        system=SYSTEM_PROMPT_BASE + """

Je richt je specifiek op het opsplitsen van zinnen met een te hoge informatiedichtheid.""",
        user="""Herschrijf de volgende zin door deze op te splitsen in meerdere, kortere zinnen. De zin bevat te veel inhoudswoorden per deelzin ({content_words_per_clause:.1f} woorden/deelzin).

Zin: "{sentence}"

Een hoge informatiedichtheid maakt tekst moeilijker te verwerken. Probeer:
- De informatie over meerdere zinnen te verdelen
- Elke zin één hoofdgedachte te laten bevatten
- Verbindingswoorden te gebruiken voor samenhang

Geef je antwoord in het volgende formaat:
PROBLEEM: [welke informatie is samengeperst in deze zin]
HERSCHRIJVING: [de opgesplitste zinnen]
UITLEG: [één korte zin die alleen beschrijft wat er structureel gedaan is, zoals opsplitsen of herformuleren — geen uitleg over het waarom, geen vakjargon]"""
    ),

    "abstract_nouns": PromptTemplate(
        system=SYSTEM_PROMPT_BASE + """

Je richt je specifiek op het concreter maken van abstracte taal.""",
        user="""Herschrijf het volgende tekstfragment om het begrijpelijker te maken. Het fragment bevat abstracte zelfstandige naamwoorden.

Tekstfragment: "{context}"

Abstracte woorden in dit fragment: {abstract_nouns}

Abstracte woorden zijn moeilijker te begrijpen. Probeer ze te omschrijven of te vervangen door concretere alternatieven — maar uitsluitend op basis van wat er al in de tekst staat. Voeg geen nieuwe informatie, voorbeelden of inhoud toe die niet in het origineel staan.

Geef je antwoord in het volgende formaat:
ABSTRACTIES: [welke abstracte begrippen je hebt aangepakt]
HERSCHRIJVING: [het herschreven tekstfragment]
UITLEG: [één korte, eenvoudige zin over hoe de tekst begrijpelijker is geworden — geen vakjargon]"""
    ),

    "passive": PromptTemplate(
        system=SYSTEM_PROMPT_BASE + """

Je richt je specifiek op het omzetten van passieve zinnen naar actieve zinnen.""",
        user="""Herschrijf de volgende zin door de passieve constructie(s) om te zetten naar actieve zinnen.

Zin: "{sentence}"

Passieve constructie(s): {passives}

Actieve zinnen zijn makkelijker te begrijpen omdat duidelijk is wie de handeling uitvoert. Probeer:
- Te benoemen wie de handeling uitvoert
- De actieve werkwoordsvorm te gebruiken
- De zinsstructuur zo min mogelijk te veranderen

Geef je antwoord in het volgende formaat:
PROBLEEM: [welke passieve constructie(s) zijn aangepakt]
HERSCHRIJVING: [de actieve versie van de zin]
UITLEG: [één korte, eenvoudige zin over de verbetering — geen vakjargon]"""
    ),

    "subordinate_clause": PromptTemplate(
        system=SYSTEM_PROMPT_BASE + """

Je richt je specifiek op het vereenvoudigen van zinnen met veel ingebedde bijzinnen.""",
        user="""Herschrijf de volgende zin door de bijzinnen te vereenvoudigen of op te splitsen. De zin bevat {n_subordinate_clauses} bijzin(nen), wat de zin complex maakt.

Zin: "{sentence}"

Veel bijzinnen maken een zin moeilijker te volgen. Probeer:
- De bijzin(nen) om te zetten naar afzonderlijke zinnen
- De hoofdgedachte voorop te stellen
- Verbindingswoorden te gebruiken voor samenhang

Geef je antwoord in het volgende formaat:
PROBLEEM: [welke bijzinsstructuur de zin complex maakt]
HERSCHRIJVING: [de vereenvoudigde zin of zinnen]
UITLEG: [één korte zin die alleen beschrijft wat er structureel gedaan is, zoals opsplitsen of herstructureren — geen uitleg over het waarom, geen vakjargon]"""
    ),

    "sentence_length": PromptTemplate(
        system=SYSTEM_PROMPT_BASE + """

Je richt je specifiek op het opsplitsen van lange zinnen in kortere, beter te verwerken zinnen.""",
        user="""Herschrijf de volgende lange zin ({sent_length} woorden) door deze op te splitsen in kortere zinnen.

Zin: "{sentence}"

Lange zinnen zijn moeilijker te verwerken. Probeer:
- De zin op te splitsen in twee of drie kortere zinnen
- Elke zin één hoofdgedachte te laten bevatten
- Verbindingswoorden te gebruiken voor samenhang

Geef je antwoord in het volgende formaat:
PROBLEEM: [wat de zin lang maakt]
HERSCHRIJVING: [de kortere zinnen]
UITLEG: [één korte zin die alleen beschrijft wat er structureel gedaan is, zoals opsplitsen — geen uitleg over het waarom, geen vakjargon]"""
    ),

    "spelling": PromptTemplate(
        system="""Je bent een expert Nederlandse taalkundige en corrector. Je taak is om suggesties te geven om Nederlandse tekst leesbaarder te maken, door spelfouten en contextuele zinsbouw- en grammaticafouten te identificeren.

Belangrijke richtlijnen:
- Identificeer echte spelfouten (tikfouten, verkeerd gespelde woorden)
- Identificeer contextuele grammatica- en zinsbouwfouten (bijv. "ik wordt" → "ik word", "hij loop" → "hij loopt", dt-fouten, verkeerd lidwoord de/het, onduidelijke of foutieve zinsconstructies)
- Negeer stilistische keuzes — richt je alleen op objectieve fouten
- Negeer woorden die correct gespeld zijn, ook als ze technisch, formeel of ongebruikelijk zijn — meld ze niet als fout
- Geef voor elke fout de categorie aan: "spelfout" of "grammatica"
- Schrijf uitleg in eenvoudige, korte zinnen die voor iedereen te begrijpen zijn — vermijd vakjargon
- Antwoord altijd in het Nederlands""",
        user="""Controleer de volgende tekst op spelfouten en contextuele grammaticafouten. De tekst bestaat uit genummerde zinnen.

Tekst:
{text}

Geef voor elke gevonden fout het volgende gestructureerde formaat (één blok per fout):

---
WOORD: [het foutieve woord]
ZIN_NUMMER: [het nummer van de zin waarin de fout staat]
CORRECTIE: [het gecorrigeerde woord]
CATEGORIE: [spelfout, grammatica of zinsbouw]
UITLEG: [één korte, eenvoudige zin die uitlegt wat er fout is — schrijf alsof je het aan een leek uitlegt]
---

Als er geen fouten zijn, antwoord dan met:
GEEN_FOUTEN"""
    ),
}


def format_prompt(
    template_name: str,
    **kwargs,
) -> tuple[str, str]:
    """
    Format a prompt template with the given parameters.

    Args:
        template_name: Name of the template ('word_frequency', 'max_sdl', etc.)
        **kwargs: Parameters to fill in the template

    Returns:
        Tuple of (system_prompt, user_prompt)

    Raises:
        KeyError: If template_name is not found
        KeyError: If required template parameters are missing
    """
    if template_name not in PROMPT_TEMPLATES:
        raise KeyError(
            f"Unknown template: {template_name}. "
            f"Available: {', '.join(PROMPT_TEMPLATES.keys())}"
        )

    template = PROMPT_TEMPLATES[template_name]
    return template["system"], template["user"].format(**kwargs)


def parse_llm_response(response: str, template_name: str) -> dict[str, str]:
    """
    Parse a structured LLM response into its components.

    Args:
        response: Raw response text from the LLM
        template_name: Name of the template used (determines expected fields)

    Returns:
        Dictionary with parsed fields (e.g., 'VERVANGING', 'UITLEG', 'HERSCHRIJVING')
    """
    result: dict[str, str] = {}
    current_field: str | None = None
    current_content: list[str] = []

    # Define expected fields per template
    expected_fields = {
        "word_frequency": ["VERVANGING", "UITLEG", "HERSCHRIJVING"],
        "max_sdl": ["PROBLEEM", "HERSCHRIJVING", "UITLEG"],
        "content_words_per_clause": ["PROBLEEM", "HERSCHRIJVING", "UITLEG"],
        "abstract_nouns": ["ABSTRACTIES", "HERSCHRIJVING", "UITLEG"],
        "spelling": ["WOORD", "ZIN_NUMMER", "CORRECTIE", "CATEGORIE", "UITLEG"],
        "passive": ["PROBLEEM", "HERSCHRIJVING", "UITLEG"],
        "subordinate_clause": ["PROBLEEM", "HERSCHRIJVING", "UITLEG"],
        "sentence_length": ["PROBLEEM", "HERSCHRIJVING", "UITLEG"],
    }

    fields = expected_fields.get(template_name, [])

    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Strip markdown formatting: **FIELD:** → FIELD:, ## FIELD: → FIELD:
        clean_line = re.sub(r'^[#*_\s]*', '', line)
        clean_line = re.sub(r'\*+', '', clean_line)

        # Check if this line starts a new field
        found_field = False
        for field in fields:
            if clean_line.upper().startswith(field + ":"):
                # Save previous field if exists
                if current_field:
                    result[current_field] = " ".join(current_content).strip()
                # Start new field
                current_field = field
                content = clean_line[len(field) + 1:].strip()
                current_content = [content] if content else []
                found_field = True
                break

        if not found_field and current_field:
            current_content.append(line)

    # Save last field
    if current_field:
        result[current_field] = " ".join(current_content).strip()

    return result


def parse_spelling_response(response: str) -> list[dict[str, str]]:
    """
    Parse a multi-error spelling/grammar response into a list of error dicts.

    Each error block is delimited by '---' lines and contains fields:
    WOORD, ZIN_NUMMER, CORRECTIE, CATEGORIE, UITLEG.

    Returns:
        List of dicts, each with keys matching the field names.
    """
    # Normalize "ZIN NUMMER" (space) to "ZIN_NUMMER" (underscore) for model variations
    response = re.sub(r'\bZIN\s+NUMMER\b', 'ZIN_NUMMER', response, flags=re.IGNORECASE)

    # Only short-circuit if there are no error blocks at all
    if "GEEN_FOUTEN" in response and "WOORD:" not in response.upper():
        return []

    fields = ["WOORD", "ZIN_NUMMER", "CORRECTIE", "CATEGORIE", "UITLEG"]
    errors: list[dict[str, str]] = []

    current: dict[str, str] = {}
    current_field: str | None = None
    current_content: list[str] = []

    def _flush_field():
        nonlocal current_field, current_content
        if current_field:
            current[current_field] = " ".join(current_content).strip()
            current_field = None
            current_content = []

    def _flush_block():
        nonlocal current
        _flush_field()
        if current and "WOORD" in current:
            errors.append(current)
        current = {}

    for line in response.split("\n"):
        stripped = line.strip()

        # Block delimiter
        if stripped.startswith("---"):
            _flush_block()
            continue

        if not stripped:
            continue

        # Strip markdown formatting
        clean = re.sub(r'^[#*_\s]*', '', stripped)
        clean = re.sub(r'\*+', '', clean)

        # Check if this line starts a new field
        found = False
        for f in fields:
            if clean.upper().startswith(f + ":"):
                _flush_field()
                current_field = f
                content = clean[len(f) + 1:].strip()
                current_content = [content] if content else []
                found = True
                break

        if not found and current_field:
            current_content.append(stripped)

    # Flush any remaining block
    _flush_block()

    return errors
