#!/usr/bin/env python3
"""Build corpus4.json — a fourth held-out LiNT-II eval set.

Same structure and label scheme as build_corpus.py .. build_corpus3.py, but
again entirely new texts and domains. First set evaluated against the Qwen
backend (sets 1-3 ran against Mistral-large / early Qwen), and the first
authored AFTER the word-family frequency guard (cb49c56): the wordfreq
positives were checked to have rare FAMILIES (not just rare surface forms),
and a new family-* guard group holds rare surface forms of common families
(koelere-style) that the pre-guard code fired on and current code must not.
Synthetic, no PII.

    python3 scripts/eval/build_corpus4.py
    python3 scripts/eval/run_eval.py --corpus scripts/eval/corpus4.json --results scripts/eval/results4.json --fresh
"""
import json
import os

POS = [
    ("passive-1", ["passive"], "De gestrande zeehond is door de dierenambulance nog dezelfde ochtend naar de opvang gebracht."),
    ("passive-2", ["passive"], "Het monumentale orgel wordt door een gespecialiseerde restaurator pijp voor pijp schoongemaakt."),
    ("passive-3", ["passive"], "De lekkage in de hoofdleiding is door het drinkwaterbedrijf binnen enkele uren verholpen."),
    ("passive-4", ["passive"], "De jaarrekening van de stichting werd door een externe accountant zonder voorbehoud goedgekeurd."),
    ("passive-5", ["passive"], "De zwerm bijen is door een plaatselijke imker veilig uit de schoorsteen weggehaald."),
    ("passive-6", ["passive"], "Het vervallen pakhuis aan de kade wordt door de woningcorporatie tot appartementen verbouwd."),
    ("passive-7", ["passive"], "De dijk langs de rivier zal door het waterschap komende zomer over de volle lengte worden versterkt."),
    ("passive-8", ["passive"], "De verloren gewaande brieven zijn door een medewerker van het archief bij toeval teruggevonden."),

    ("long-1", ["max_sdl", "sentence_length"], "Omdat het aantal vrijwilligers bij de reddingsbrigade de afgelopen jaren zo sterk is teruggelopen dat de bezetting van de strandposten in het weekend niet langer gegarandeerd kan worden, overweegt het bestuur om betaalde krachten in te zetten tijdens het hoogseizoen."),
    ("long-2", ["max_sdl", "sentence_length"], "De violiste, die haar opleiding aan het conservatorium ooit moest afbreken vanwege een hardnekkige blessure aan haar linkerhand, maakte gisteravond in een uitverkochte zaal een omjubeld debuut met het orkest waar zij jaren geleden als invaller begon."),
    ("long-3", ["max_sdl", "sentence_length"], "Hoewel de vissers al bij het eerste ochtendlicht waren uitgevaren en de weersverwachting voor de hele dag gunstig leek, dwong een plotseling opstekende storm de complete vloot om halverwege de middag onverrichter zake terug te keren naar de haven."),
    ("long-4", ["max_sdl", "sentence_length"], "Nadat de bewoners van de flat maandenlang tevergeefs bij de verhuurder hadden aangedrongen op het herstel van de kapotte lift, schakelden zij gezamenlijk de huurcommissie in, die de verhuurder uiteindelijk tot een forse huurverlaging veroordeelde."),
    ("long-5", ["max_sdl", "sentence_length"], "Doordat de kwekerij haar planten dit voorjaar veel eerder dan gebruikelijk naar buiten had verplaatst en een late nachtvorst precies in die week toesloeg, ging een aanzienlijk deel van de jonge oogst in één enkele koude nacht verloren."),
    ("long-6", ["max_sdl", "sentence_length"], "De conservator legde tijdens de rondleiding uit dat het schilderij, dat decennialang onopgemerkt in het depot van het museum had gehangen, na een grondige technische analyse vrijwel zeker aan een beroemde zeventiende-eeuwse meester kan worden toegeschreven."),
    ("long-7", ["max_sdl", "sentence_length"], "Aangezien de wachttijden bij de apotheek in het centrum tijdens de middagpauze zo ver waren opgelopen dat klanten hun medicijnen soms pas na een uur meekregen, is er nu een aparte balie geopend voor het ophalen van herhaalrecepten."),
    ("long-8", ["max_sdl", "sentence_length"], "Hoewel de nieuwe dienstregeling was opgesteld om de drukte in de spits beter te spreiden en de aansluitingen op het busvervoer te verbeteren, klagen veel forensen juist dat hun dagelijkse reis er langer en onvoorspelbaarder op is geworden."),

    ("enum-1", ["enumeration"], "Het jubileumprogramma van het theater bestaat uit het opvoeren van drie premières, het uitnodigen van internationale gezelschappen, het organiseren van workshops voor jong talent en het openstellen van de repetities voor publiek."),
    ("enum-2", ["enumeration"], "De renovatie van het zwembad omvat het vervangen van het volledige leidingwerk, het vernieuwen van de tegelvloeren, het plaatsen van energiezuinige pompen en het toegankelijk maken van de kleedkamers voor rolstoelen."),
    ("enum-3", ["enumeration"], "Het actieplan tegen wateroverlast voorziet in het vergroten van de riolering, het aanleggen van groene daken, het afkoppelen van regenpijpen en het graven van een extra opvangvijver aan de rand van de wijk."),
    ("enum-4", ["enumeration"], "De voorbereiding op de vogeltelling behelst het uitdelen van telformulieren, het instrueren van de vrijwilligers, het uitzetten van vaste looproutes en het controleren van de verrekijkers en veldgidsen."),
    ("enum-5", ["enumeration"], "Het veiligheidsprotocol van het laboratorium schrijft het dragen van beschermende kleding, het registreren van alle bezoekers, het wekelijks testen van de nooddouches en het gescheiden afvoeren van chemisch afval voor."),
    ("enum-6", ["enumeration"], "De opknapbeurt van het schoolplein draait om het verwijderen van de oude tegels, het planten van schaduwbomen, het plaatsen van nieuwe speeltoestellen en het aanleggen van een moestuin voor de onderbouw."),
    ("enum-7", ["enumeration"], "Het inwerkprogramma voor nieuwe medewerkers omvat het volgen van een introductiecursus, het meelopen met een ervaren collega, het bezoeken van alle vestigingen en het bespreken van de huisregels met de leidinggevende."),
    ("enum-8", ["enumeration"], "De wintervoorbereiding van de camping bestaat uit het aftappen van de waterleidingen, het stallen van het meubilair, het snoeien van de overhangende takken en het controleren van de verlichting langs de paden."),

    ("conn-1", ["connective"], "Het concert in het park is afgelast. De organisatie kreeg de veiligheid van het podium niet op orde."),
    ("conn-2", ["connective"], "De bakker op de hoek stopt ermee. Een opvolger is ondanks lang zoeken niet gevonden."),
    ("conn-3", ["connective"], "Het waterpeil in de sloten staat extreem laag. Het heeft al twee maanden nauwelijks geregend."),
    ("conn-4", ["connective"], "De bibliotheek had gerekend op minder bezoekers. De leeszaal zat de hele winter juist vol."),
    ("conn-5", ["connective"], "De veerboot vaart vandaag niet uit. De kapitein vertrouwt de sterke stroming niet."),
    ("conn-6", ["connective"], "Het nieuwe medicijn bleek zeer effectief. De vergoeding ervan is nog steeds niet geregeld."),
    ("conn-7", ["connective"], "De jonge schaker had zich nauwelijks voorbereid. Hij won het toernooi met overmacht."),
    ("conn-8", ["connective"], "De sneeuwval hield de hele nacht aan. De eerste lessen op school begonnen een uur later."),
    ("conn-9", ["connective"], "De aardbeienoogst viel dit jaar erg vroeg. Het voorjaar was uitzonderlijk warm en zonnig."),
    ("conn-10", ["connective"], "Het restaurant kreeg een tweede ster. Reserveren kan voorlopig alleen nog maanden vooruit."),

    ("compound-1", ["word_frequency"], "De funderingsherstelwerkzaamheden aan de oude grachtenpanden beginnen na de bouwvak."),
    ("compound-2", ["word_frequency"], "Het bodemsaneringsprogramma voor het voormalige fabrieksterrein duurt zeker vier jaar."),
    ("compound-3", ["word_frequency"], "De legionellapreventiemaatregelen in het zwembad worden elk kwartaal gecontroleerd."),
    ("compound-4", ["word_frequency"], "Het faunapassageproject onder de provinciale weg moet de otterpopulatie beschermen."),
    ("compound-5", ["word_frequency"], "De studiefinancieringsregels veranderen voor studenten die in het buitenland stage lopen."),
    ("compound-6", ["word_frequency"], "Het monumentenvergunningstraject voor de verbouwing van de watertoren is eindelijk afgerond."),
    ("compound-7", ["word_frequency"], "De geluidsisolatienormen voor woningen langs het spoor zijn opnieuw aangescherpt."),
    ("compound-8", ["word_frequency"], "Het rioolvervangingsproject in de binnenstad zorgt maandenlang voor omleidingen."),

    ("abstract-1", ["abstract_nouns"], "De reorganisatie mikt op een versnelling van de besluitvorming en een vermindering van de overlegdruk."),
    ("abstract-2", ["abstract_nouns"], "Het convenant regelt een verruiming van de openingstijden en een verscherping van het toezicht."),
    ("abstract-3", ["abstract_nouns"], "De pilot moet leiden tot een verkorting van de wachttijden en een verhoging van de tevredenheid."),
    ("abstract-4", ["abstract_nouns"], "Het akkoord voorziet in een matiging van de huren en een versnelling van de nieuwbouw."),
    ("abstract-5", ["abstract_nouns"], "De campagne beoogt een vergroting van het bewustzijn en een verandering van het gedrag."),
    ("abstract-6", ["abstract_nouns"], "De fusie betekent een bundeling van de expertise en een verbreding van de dienstverlening."),
    ("abstract-7", ["abstract_nouns"], "Het programma streeft naar een versterking van de weerbaarheid en een vermindering van de uitval."),
    ("abstract-8", ["abstract_nouns"], "De maatregel zorgt voor een verlichting van de administratie en een verruiming van de bevoegdheden."),

    ("wordfreq-1", ["word_frequency"], "De rechter gaf de advocaat een stevige reprimande wegens zijn late stukken."),
    ("wordfreq-2", ["word_frequency"], "De gemeente stelde een moratorium in op de bouw van nieuwe distributiecentra."),
    ("wordfreq-3", ["word_frequency"], "Volgens de reclassering is het risico op recidive bij deze aanpak aanzienlijk kleiner."),
    ("wordfreq-4", ["word_frequency"], "De accountant wees op een opvallend hiaat in de administratie van de vereniging."),
    ("wordfreq-5", ["word_frequency"], "De toezichthouder hanteert sinds dit jaar aanzienlijk stringentere regels voor kleine fondsen."),
    ("wordfreq-6", ["word_frequency"], "De financiering van het jeugdhonk blijft volgens de wethouder uiterst precair."),
    ("wordfreq-7", ["word_frequency"], "De naburige camping staat in de regio bekend als een notoire bron van nachtelijke overlast."),
    ("wordfreq-8", ["word_frequency"], "De reder noemde de gevraagde liggelden ronduit exorbitant."),

    ("multi-1", ["sentence_rewrite", "passive"], "Aan de omwonenden is door de provincie toegezegd dat de geluidswal nog vóór de opening van de nieuwe rijbaan volledig zal worden afgewerkt."),
    ("multi-2", ["sentence_rewrite", "passive"], "Door het bestuur van de coöperatie is na lang beraad besloten dat de winstuitkering dit jaar volledig in het zonnepark zal worden geïnvesteerd."),
    ("multi-3", ["sentence_rewrite", "passive"], "Het is door de keuringsdienst vastgesteld dat bij twee kramen op de markt de koeling van verse vis niet op orde was."),
    ("multi-4", ["sentence_rewrite", "passive"], "Aan de leden is door de penningmeester beloofd dat de contributie ondanks de gestegen zaalhuur volgend seizoen niet zal worden verhoogd."),
    ("multi-5", ["sentence_rewrite", "passive"], "Door de curator is aan de schuldeisers gemeld dat de failliete drukkerij hoogstwaarschijnlijk in afgeslankte vorm zal worden voortgezet."),
    ("multi-6", ["sentence_rewrite", "passive"], "Het is door de examencommissie bevestigd dat de fout in de opgave geen enkele kandidaat in het eindcijfer zal worden aangerekend."),
    ("multi-7", ["sentence_rewrite", "passive"], "Aan de reizigers is door de vervoerder gemeld dat de nachtbus op vrijdag en zaterdag voortaan een uur langer zal doorrijden."),
]

NEG = [
    ("clean-1", [], "De kinderboerderij is elke dag open. De geitjes mogen gevoerd worden. Voer koopt u bij de ingang."),
    ("clean-2", [], "Het wijkcentrum zoekt een kok. Het gaat om twee dagen per week. Bel voor meer informatie."),
    ("clean-3", [], "De glasbak is verplaatst. Hij staat nu bij de supermarkt. De oude plek wordt een plantsoen."),
    ("clean-4", [], "Het spreekuur begint om negen uur. Neem uw pas mee. Zonder afspraak kan het druk zijn."),
    ("clean-5", [], "De boot naar het eiland vaart elk uur. Honden mogen gratis mee. Fietsen kost een euro extra."),
    ("clean-6", [], "De cursus start in september. Er zijn nog vier plekken vrij. Inschrijven kan aan de balie."),
    ("clean-7", [], "Het gemaal draait weer volop. Het water zakt langzaam. De kade blijft nog even afgezet."),
    ("clean-8", [], "De boekenkast bij het station is vernieuwd. U mag boeken meenemen en achterlaten. Ruilen is het idee."),
    ("clean-9", [], "De moestuinvereniging houdt zaterdag open dag. Er zijn plantjes te koop. De koffie is gratis."),
    ("clean-10", [], "Het loket is tijdelijk dicht. U kunt terecht in het stadskantoor. Vergeet uw afspraakbevestiging niet."),
    ("clean-11", [], "De vuurtoren is weer te beklimmen. Boven is het uitzicht prachtig. Kaartjes zijn er alleen online."),
    ("clean-12", [], "Het repaircafé is elke eerste zaterdag. Vrijwilligers maken uw spullen. Een kleine gift is welkom."),
    ("clean-13", [], "De schaatsbaan gaat vrijdag open. Schaatsen huren kan ter plekke. Kinderen betalen de helft."),

    ("family-1", ["word_frequency on rare form of common family"], "Veel mensen zoeken in de zomer de wat koelere bossen op. Daar is het aangenaam."),
    ("family-2", ["word_frequency on rare form of common family"], "De warmere dagen komen eraan. Het terras gaat weer vroeg open."),
    ("family-3", ["word_frequency on rare form of common family"], "In de leeshoek staan lage stoeltjes voor de kleintjes. De boeken liggen lekker laag."),
    ("family-4", ["word_frequency on rare form of common family"], "Langs het nieuwe fietspad zijn jonge boompjes geplant. Over tien jaar geven ze schaduw."),
    ("family-5", ["word_frequency on rare form of common family"], "Ouders mogen de eerste kilometer meefietsen naar school. Daarna gaan de kinderen zelf verder."),

    ("shortlist-1", ["enumeration"], "In de rugzak zitten een lunch, een regenjas en een routekaart. Meer is niet nodig."),
    ("shortlist-2", ["enumeration"], "Bij de lunch serveren we soep, brood of een salade. Alles komt uit eigen keuken."),
    ("shortlist-3", ["enumeration"], "Voor de knutselmiddag zijn schaar, lijm en oud papier genoeg. De rest ligt klaar."),
    ("shortlist-4", ["enumeration"], "De EHBO-post heeft pleisters, verband en een hartstarter. De post is de hele dag bemand."),

    ("conj-1", ["split ', maar '"], "Het zwembad is vandaag open, maar het buitenbad blijft dicht."),
    ("conj-2", ["split ', dus '"], "De parkeerplaats is vol, dus kom liever met de fiets."),
    ("conj-3", ["split ', want '"], "Wees er snel bij, want de kaarten gaan hard."),
    ("conj-4", ["split ', maar '"], "De reparatie is klaar, maar de rekening volgt nog per post."),

    ("url-1", ["alter URL"], "Het volledige programma staat op https://www.voorbeeldfestival.nl/programma en hangt ook bij de ingang."),
    ("url-2", ["alter URL"], "Openingstijden en tarieven vindt u op www.voorbeeldbad.nl onder het kopje bezoek."),
    ("url-3", ["alter URL"], "Vragen over uw aanslag stuurt u naar heffingen@voorbeeld-waterschap.nl met uw kenmerk erbij."),

    ("fragment-1", ["suggest on non-prose"], "Reserveringsnummer: 88412\nDatum: 14 augustus\nPersonen: 6\nOpmerking: allergie noten"),
    ("fragment-2", ["suggest on non-prose"], "Vertrektijden pont:\nMaandag t/m vrijdag elk half uur\nZaterdag elk uur\nZondag geen dienst"),
    ("fragment-3", ["suggest on non-prose"], "Checklist verhuizing\n1. Adreswijziging doorgeven\n2. Meterstanden noteren\n3. Sleutels inleveren"),

    ("good-1", [], "Uw pakket ligt klaar bij de balie. Neem uw legitimatie mee."),
    ("good-2", [], "De storing is verholpen. Excuses voor het ongemak."),
    ("good-3", [], "Uw inschrijving is gelukt. U hoort snel van ons."),
]

items = []
for id_, phen, text in POS:
    items.append({"id": id_, "should_suggest": True, "phenomena": phen, "text": text})
for id_, must_not, text in NEG:
    items.append({"id": id_, "should_suggest": False, "phenomena": [], "must_not": must_not, "text": text})

corpus = {
    "description": "LiNT-II held-out eval set 4 — independent of corpus.json/corpus2.json/corpus3.json (new texts/domains), same label scheme. First set for the Qwen backend and the word-family frequency guard; family-* negatives are rare surface forms of common families that must not fire word_frequency. Synthetic, no PII.",
    "items": items,
}
out = os.path.join(os.path.dirname(__file__), "corpus4.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=1)
n_pos = sum(1 for i in items if i["should_suggest"])
print(f"wrote {len(items)} items ({n_pos} positive, {len(items) - n_pos} negative) -> {out}")
