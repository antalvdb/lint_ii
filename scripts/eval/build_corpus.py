#!/usr/bin/env python3
"""Build the LiNT-II self-diagnosis corpus (corpus.json).

Authoring the 100 labelled Dutch inputs as Python data is far less error-prone
than raw JSON; this script validates and emits corpus.json next to itself.

Labels are the ground truth: `should_suggest` drives presence/absence
(precision/recall); `phenomena` names the expected suggestion kinds for
positives; `must_not` names prohibited behaviours for guard cases. All text is
synthetic (no PII).
"""
import json
import os

# (id, [phenomena], text) — a suggestion is expected.
POS = [
    ("passive-1", ["passive"], "De aanvraag is door de behandelend ambtenaar zorgvuldig beoordeeld en zal binnen enkele weken schriftelijk aan de aanvrager worden teruggekoppeld."),
    ("passive-2", ["passive"], "Het besluit werd door het college genomen nadat alle betrokken partijen waren gehoord."),
    ("passive-3", ["passive"], "De schade aan het gebouw is door een onafhankelijk expert vastgesteld en wordt door de verzekeraar volledig vergoed."),
    ("passive-4", ["passive"], "De nieuwe regels zullen vanaf volgend jaar door de handhaving strikt worden gecontroleerd."),
    ("passive-5", ["passive"], "Er is door de directie besloten dat de kantoren voortaan op vrijdag gesloten zullen zijn."),
    ("passive-6", ["passive"], "De klachten van de omwonenden zijn door de gemeente serieus genomen en worden op dit moment nader onderzocht."),
    ("passive-7", ["passive"], "Het rapport is door de commissie opgesteld en zal volgende maand aan de raad worden aangeboden."),
    ("passive-8", ["passive"], "De subsidie is aan de vereniging toegekend nadat de aanvraag volledig en tijdig was ingediend."),

    ("long-1", ["max_sdl", "sentence_length"], "Omdat de gemeente de afgelopen jaren onvoldoende heeft geïnvesteerd in de digitale beveiliging van haar systemen, wat mede het gevolg was van een structureel tekort aan gekwalificeerd personeel, heeft het college nu besloten om een extern bureau in te schakelen dat de bestaande risico's grondig in kaart moet brengen."),
    ("long-2", ["max_sdl", "sentence_length"], "De bewoners van de wijk, die al maandenlang klagen over de aanhoudende geluidsoverlast van het nabijgelegen bedrijventerrein, hebben inmiddels een gezamenlijke brief gestuurd naar de wethouder met het dringende verzoek om op korte termijn passende maatregelen te nemen."),
    ("long-3", ["max_sdl", "sentence_length"], "Hoewel de aannemer had toegezegd dat het project voor het einde van het jaar zou worden afgerond, is door onvoorziene vertragingen in de levering van materialen en door het slechte weer de oplevering nu uitgesteld tot het voorjaar."),
    ("long-4", ["max_sdl", "sentence_length"], "De commissie die is ingesteld om de gang van zaken rond de aanbesteding te onderzoeken, heeft na het bestuderen van een groot aantal documenten en na gesprekken met verschillende betrokkenen een aantal stevige conclusies getrokken."),
    ("long-5", ["max_sdl", "sentence_length"], "Nadat de inspectie had vastgesteld dat de brandveiligheid in het gebouw op meerdere punten niet op orde was, heeft de eigenaar op eigen kosten een reeks aanpassingen laten uitvoeren die inmiddels allemaal zijn goedgekeurd."),
    ("long-6", ["max_sdl", "sentence_length"], "Doordat het aantal aanvragen de laatste maanden veel sterker is gestegen dan verwacht, en doordat er tegelijkertijd enkele medewerkers langdurig ziek zijn, lopen de wachttijden bij de afdeling op dit moment flink op."),
    ("long-7", ["max_sdl", "sentence_length"], "De minister heeft in een uitgebreide brief aan de Kamer uiteengezet waarom de invoering van de nieuwe wet, die aanvankelijk voor dit jaar gepland stond, met minstens twaalf maanden moet worden uitgesteld."),
    ("long-8", ["max_sdl", "sentence_length"], "Aangezien de kosten van het onderhoud van de oude brug de afgelopen jaren fors zijn opgelopen en de verkeersdrukte bovendien sterk is toegenomen, onderzoekt de provincie nu of vervanging door een nieuwe brug op termijn voordeliger zou zijn."),

    ("enum-1", ["enumeration"], "Het verbeterplan van de organisatie richt zich op het aanscherpen van de interne controles, het bijscholen van het personeel op het gebied van privacy, het vastleggen van duidelijke afspraken met leveranciers en het periodiek evalueren van de gemaakte keuzes."),
    ("enum-2", ["enumeration"], "De nieuwe aanpak bestaat uit het vergroten van de capaciteit van de opvang, het verkorten van de wachttijden aan de balie, het verbeteren van de digitale dienstverlening en het uitbreiden van de openingstijden in het weekend."),
    ("enum-3", ["enumeration"], "De maatregelen omvatten het isoleren van de daken, het vervangen van de oude ketels, het plaatsen van zonnepanelen op de gemeentelijke gebouwen en het stimuleren van bewoners om zelf te verduurzamen."),
    ("enum-4", ["enumeration"], "Het onderzoek richt zich op het in kaart brengen van de risico's, het beoordelen van de bestaande procedures, het interviewen van de betrokken medewerkers en het opstellen van concrete aanbevelingen."),
    ("enum-5", ["enumeration"], "De cursus behandelt het schrijven van heldere brieven, het opstellen van correcte besluiten, het beoordelen van lastige aanvragen en het voeren van moeilijke gesprekken met burgers."),
    ("enum-6", ["enumeration"], "Het beleid is gericht op het terugdringen van de administratieve lasten, het versnellen van de vergunningverlening, het verbeteren van de samenwerking tussen afdelingen en het vergroten van de tevredenheid van inwoners."),
    ("enum-7", ["enumeration"], "De renovatie omvat het vernieuwen van de installaties, het herinrichten van de kantoren, het verduurzamen van de gevel en het toegankelijker maken van de ingang voor mensen met een beperking."),
    ("enum-8", ["enumeration"], "De taken van de commissie bestaan uit het beoordelen van de ingediende plannen, het adviseren van het bestuur, het bewaken van de voortgang en het rapporteren van de resultaten aan de raad."),

    ("conn-1", ["connective"], "Het treinverkeer lag die ochtend urenlang volledig stil. Een belangrijk sein langs het spoor was defect geraakt."),
    ("conn-2", ["connective"], "De weg blijft dit hele weekend afgesloten. Er wordt groot onderhoud aan het wegdek uitgevoerd."),
    ("conn-3", ["connective"], "De vergadering is verplaatst naar volgende week. Verschillende leden zijn deze week verhinderd."),
    ("conn-4", ["connective"], "De kosten van de verbouwing vallen hoger uit dan begroot. De prijzen van bouwmaterialen zijn het afgelopen jaar sterk gestegen."),
    ("conn-5", ["connective"], "Veel bewoners wilden graag deelnemen aan de informatieavond. Er waren in de zaal veel te weinig stoelen beschikbaar."),
    ("conn-6", ["connective"], "De nieuwe regeling is bedoeld om ondernemers te helpen. Veel kleine bedrijven vinden de aanvraag veel te ingewikkeld."),
    ("conn-7", ["connective"], "Het plan klinkt op papier veelbelovend. De uitvoering blijkt in de praktijk erg lastig."),
    ("conn-8", ["connective"], "De brug over het kanaal is wegens onderhoud afgesloten. Het verkeer wordt omgeleid via de provinciale weg."),
    ("conn-9", ["connective"], "Het aantal besmettingen is de afgelopen week sterk gedaald. De maatregelen kunnen geleidelijk worden versoepeld."),
    ("conn-10", ["connective"], "De subsidiepot voor dit jaar is inmiddels helemaal leeg. Nieuwe aanvragen worden pas volgend jaar weer in behandeling genomen."),

    ("compound-1", ["word_frequency"], "De werkloosheidsuitkeringsaanvraag van de betrokkene is vorige week door de afdeling in behandeling genomen en wordt op dit moment inhoudelijk beoordeeld."),
    ("compound-2", ["word_frequency"], "Het uitstroombevorderingsbeleid van de gemeente heeft het afgelopen jaar helaas nauwelijks resultaat opgeleverd."),
    ("compound-3", ["word_frequency"], "De verkeersveiligheidsmaatregelen in de schoolomgeving worden na de zomervakantie stap voor stap ingevoerd."),
    ("compound-4", ["word_frequency"], "Het klanttevredenheidsonderzoek van vorig kwartaal gaf een wisselend beeld van de dienstverlening."),
    ("compound-5", ["word_frequency"], "De arbeidsmarktparticipatie van ouderen blijft ondanks alle inspanningen een hardnekkig aandachtspunt."),
    ("compound-6", ["word_frequency"], "De gezondheidszorgvoorzieningen in de regio staan door de vergrijzing onder toenemende druk."),
    ("compound-7", ["word_frequency"], "Het levensmiddelendistributiecentrum aan de rand van de stad kampt al maanden met ernstige personeelstekorten."),
    ("compound-8", ["word_frequency"], "De arbeidsongeschiktheidsverzekering dekt de kosten van langdurige uitval niet volledig."),

    ("abstract-1", ["abstract_nouns"], "De implementatie van de nieuwe procedure heeft geleid tot een verbetering van de doorstroming en een vermindering van de wachttijden bij de balie."),
    ("abstract-2", ["abstract_nouns"], "Na de invoering van het systeem was er sprake van een toename van het aantal klachten over de bereikbaarheid."),
    ("abstract-3", ["abstract_nouns"], "De realisatie van de doelstellingen vereist een intensivering van de samenwerking tussen de betrokken partijen."),
    ("abstract-4", ["abstract_nouns"], "Er is behoefte aan een verduidelijking van de regels en een vereenvoudiging van de aanvraagprocedure."),
    ("abstract-5", ["abstract_nouns"], "De uitvoering van het onderhoud leidde tot een tijdelijke onderbreking van de dienstverlening aan de bewoners."),
    ("abstract-6", ["abstract_nouns"], "De beoordeling van de aanvragen vindt plaats op basis van een afweging van de beschikbare middelen."),
    ("abstract-7", ["abstract_nouns"], "Het uitgangspunt is een vergroting van de zelfredzaamheid en een versterking van de eigen verantwoordelijkheid."),
    ("abstract-8", ["abstract_nouns"], "De stijging van de kosten is een gevolg van de toename van de vraag en de schaarste op de arbeidsmarkt."),

    ("wordfreq-1", ["word_frequency"], "De inspecteur constateerde tijdens het bezoek diverse discrepanties tussen de aangeleverde administratie en de feitelijke voorraad in het magazijn."),
    ("wordfreq-2", ["word_frequency"], "De aanvraag werd afgewezen wegens een manifeste omissie in de vereiste documentatie."),
    ("wordfreq-3", ["word_frequency"], "De gemeente ambieert een substantiële reductie van de uitstoot binnen het komende decennium."),
    ("wordfreq-4", ["word_frequency"], "De commissie achtte de aangevoerde argumenten niet plausibel en verzocht om een nadere onderbouwing."),
    ("wordfreq-5", ["word_frequency"], "Het bestuur benadrukte de noodzaak van een adequate en expeditieuze afhandeling van de openstaande dossiers."),
    ("wordfreq-6", ["word_frequency"], "De uitkomst van de procedure was voor beide partijen weinig satisfactoir."),
    ("wordfreq-7", ["word_frequency"], "De maatregel beoogt een mitigatie van de nadelige effecten voor kwetsbare huishoudens."),
    ("wordfreq-8", ["word_frequency"], "De ambtenaar wees op de precaire financiële situatie van de betrokken instelling."),

    ("multi-1", ["sentence_rewrite", "passive"], "Het is door de leverancier schriftelijk bevestigd dat de bestelde goederen met een aanzienlijke vertraging zullen worden geleverd aan het centrale magazijn van de organisatie."),
    ("multi-2", ["sentence_rewrite", "passive"], "Aan de betrokken bewoners is door de gemeente meegedeeld dat de drinkwatervoorziening tijdelijk zal worden onderbroken wegens noodzakelijke werkzaamheden aan het leidingnet."),
    ("multi-3", ["sentence_rewrite", "passive"], "Door de behandelend arts is aan de patiënt te kennen gegeven dat het aanvullende onderzoek naar de aanhoudende klachten inmiddels volledig is afgerond."),
    ("multi-4", ["sentence_rewrite", "passive"], "Er is door de directie na uitvoerig overleg besloten dat de openingstijden van de vestigingen met ingang van volgend kwartaal ingrijpend zullen worden gewijzigd."),
    ("multi-5", ["sentence_rewrite", "passive"], "Het is door de rechtbank aan de aanvrager uitdrukkelijk toegestaan om binnen een termijn van zes weken schriftelijk bezwaar te maken tegen het genomen besluit."),
    ("multi-6", ["sentence_rewrite", "passive"], "Door de afdeling is aan de aanvrager nadrukkelijk verzocht om de nog ontbrekende gegevens zo spoedig mogelijk alsnog per e-mail aan te leveren."),
    ("multi-7", ["sentence_rewrite", "passive"], "Het is door de commissie na een zorgvuldige afweging van alle belangen besloten dat de vergunning onder een aantal aanvullende voorwaarden zal worden verleend."),

]

# (id, [must_not], text) — NO suggestion expected (or a specific behaviour forbidden).
NEG = [
    ("clean-1", [], "De bibliotheek is open van maandag tot en met vrijdag. U kunt boeken lenen met uw pas. Het lenen is gratis."),
    ("clean-2", [], "Wij hebben uw brief ontvangen. Wij danken u voor uw bericht. U hoort binnen twee weken van ons."),
    ("clean-3", [], "Het weer is vandaag mooi. De zon schijnt de hele dag. Het is niet koud."),
    ("clean-5", [], "U kunt met de bus naar het station. De bus vertrekt elk half uur. De rit duurt tien minuten."),
    ("clean-6", [], "Het zwembad heeft een groot bad en een klein bad. Kinderen zwemmen in het kleine bad. Er is altijd een badmeester aanwezig."),
    ("clean-7", [], "De cursus begint in september. De lessen zijn op dinsdag. U kunt zich nu aanmelden."),
    ("clean-8", [], "Wij verhuizen naar een nieuw kantoor. Het nieuwe adres is Dorpsstraat 5. Het telefoonnummer blijft hetzelfde."),
    ("clean-9", [], "De markt is elke woensdag op het plein. U vindt er groente, fruit en kaas. De markt duurt tot vier uur."),
    ("clean-10", [], "Het park is de hele dag open. Honden mogen mee aan de lijn. Er zijn genoeg banken om te zitten."),
    ("clean-12", [], "U kunt online een afspraak maken. Kies een dag en een tijd. U krijgt daarna een bevestiging."),
    ("clean-14", [], "De trein naar Utrecht vertrekt van spoor drie. U hoeft niet over te stappen. De reis duurt een uur."),
    ("clean-15", [], "Wij zoeken vrijwilligers voor het festival. U helpt een dagdeel mee. Wij zorgen voor eten en drinken."),
    ("clean-17", [], "Het spreekuur is op donderdagochtend. U hoeft geen afspraak te maken. Neem uw legitimatie mee."),
    ("clean-18", [], "De speeltuin is opnieuw ingericht. Er staan nieuwe toestellen. Kinderen zijn van harte welkom."),

    ("shortlist-1", ["enumeration"], "In de kantine zijn koffie, thee en water verkrijgbaar. De automaat staat naast de ingang."),
    ("shortlist-2", ["enumeration"], "U kunt betalen met pin, contant of een tegoedbon. Aan de kassa helpen wij u graag verder."),
    ("shortlist-3", ["enumeration"], "De doos bevat een lader, een kabel en een handleiding. Bewaar de bon goed."),
    ("shortlist-4", ["enumeration"], "Op het formulier vult u uw naam, adres en geboortedatum in. Onderteken het formulier daarna."),

    ("conj-1", ["split ', maar '"], "De maatregel geldt voor alle inwoners, maar niet voor bedrijven."),
    ("conj-2", ["split ', maar '"], "U kunt vandaag nog langskomen, maar dan wel vóór vijf uur."),
    ("conj-3", ["split ', dus '"], "De aanvraag is helemaal compleet, dus wij nemen hem in behandeling."),
    ("conj-4", ["split ', want '"], "Het evenement gaat gewoon door, want het weer wordt goed."),

    ("url-1", ["alter URL"], "Meer informatie over de regeling vindt u op https://www.voorbeeld.nl/regeling zonder dat u zich vooraf hoeft aan te melden."),
    ("url-2", ["alter URL"], "Aanmelden kan via www.gemeente-voorbeeld.nl/aanmelden tot uiterlijk volgende week vrijdag."),
    ("url-3", ["alter URL"], "Stuur uw vragen naar info@voorbeeld.nl en u krijgt binnen drie werkdagen antwoord."),

    ("fragment-1", ["suggest on non-prose"], "Kenmerk: 2024-XY-0031\nDatum: 3 maart\nAfdeling: Vergunningen\nContact: balie 4"),
    ("fragment-2", ["suggest on non-prose"], "Openingstijden:\nMaandag 9 tot 17 uur\nDinsdag 9 tot 17 uur\nWoensdag gesloten"),
    ("fragment-3", ["suggest on non-prose"], "Agenda\n1. Opening\n2. Mededelingen\n3. Rondvraag\n4. Sluiting"),

    ("good-1", [], "Bedankt voor uw aanmelding. Wij zien u graag op de bijeenkomst."),
    ("good-2", [], "Uw pakket is bezorgd bij de buren op nummer 12."),

    # Reverted to negative (2026-07-17, run 4): already-clear paragraphs the
    # now-conservative connective pass correctly leaves alone.
    ("clean-4", [], "De winkel is dicht op zondag. Op andere dagen is de winkel open. De openingstijden staan op de deur."),
    ("clean-11", [], "De school heeft een nieuwe directeur. Zij begint na de zomer. Ouders krijgen binnenkort een brief."),
    ("clean-13", [], "Het museum is gratis voor kinderen. Volwassenen betalen tien euro. De kaartjes koopt u bij de ingang."),
    ("clean-16", [], "De container wordt op maandag geleegd. Zet hem de avond ervoor buiten. Doe de deksel goed dicht."),
    ("good-3", [], "De betaling is gelukt. U ontvangt de kaartjes per e-mail."),
]

items = []
for id_, phen, text in POS:
    items.append({"id": id_, "should_suggest": True, "phenomena": phen, "text": text})
for id_, must_not, text in NEG:
    items.append({"id": id_, "should_suggest": False, "phenomena": [], "must_not": must_not, "text": text})

corpus = {
    "description": "LiNT-II self-diagnosis corpus. Paragraph-length Dutch inputs spanning the phenomena the demo should act on, plus negatives that should stay untouched and guard cases it must not break. should_suggest = ground truth for presence/absence (precision/recall); phenomena = expected suggestion kinds for positives; must_not = prohibited behaviours for guard cases. Synthetic text, no PII.",
    "items": items,
}

out = os.path.join(os.path.dirname(__file__), "corpus.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=1)

n_pos = sum(1 for i in items if i["should_suggest"])
print(f"wrote {len(items)} items ({n_pos} positive, {len(items) - n_pos} negative) -> {out}")
