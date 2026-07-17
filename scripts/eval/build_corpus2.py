#!/usr/bin/env python3
"""Build corpus2.json — a held-out LiNT-II eval set (independent of corpus.json).

Same structure and label scheme as build_corpus.py, but entirely new texts and
domains, so it measures the tuned system without the development-set bias (the
connective/abstract thresholds were calibrated on corpus.json). Synthetic, no PII.

    python3 scripts/eval/build_corpus2.py
    python3 scripts/eval/run_eval.py --corpus scripts/eval/corpus2.json --results scripts/eval/results2.json --fresh
"""
import json
import os

POS = [
    ("passive-1", ["passive"], "De vergunning is door de gemeente verleend nadat alle bezwaren zorgvuldig waren afgewogen."),
    ("passive-2", ["passive"], "Het voorstel werd tijdens de vergadering door een ruime meerderheid aangenomen."),
    ("passive-3", ["passive"], "De patiënt wordt na de operatie enkele dagen door het verplegend personeel nauwlettend gevolgd."),
    ("passive-4", ["passive"], "De gegevens zijn per abuis door een medewerker naar het verkeerde adres verstuurd."),
    ("passive-5", ["passive"], "Het terrein zal komende maand door een gespecialiseerd bedrijf volledig worden gesaneerd."),
    ("passive-6", ["passive"], "De contributie moet vóór het einde van het jaar door alle leden worden voldaan."),
    ("passive-7", ["passive"], "De resultaten van het examen zullen volgende week per post aan de kandidaten worden toegezonden."),
    ("passive-8", ["passive"], "Het gebouw is na de brand door de brandweer als onveilig bestempeld."),

    ("long-1", ["max_sdl", "sentence_length"], "Omdat het aantal inwoners in de gemeente de afgelopen tien jaar sneller is gegroeid dan voorzien, en omdat de bestaande scholen daardoor uit hun jasje zijn gegroeid, heeft de raad besloten om op twee locaties nieuwe schoolgebouwen te laten bouwen."),
    ("long-2", ["max_sdl", "sentence_length"], "De ondernemer, die al jaren met succes een winkel in het centrum runt maar de laatste tijd steeds meer concurrentie ondervindt van webwinkels, overweegt nu om ook zelf een online verkoopkanaal te openen."),
    ("long-3", ["max_sdl", "sentence_length"], "Hoewel de artsen aanvankelijk dachten dat een operatie onvermijdelijk zou zijn, bleek na een reeks aanvullende onderzoeken dat de klachten ook met medicijnen en fysiotherapie goed onder controle te krijgen waren."),
    ("long-4", ["max_sdl", "sentence_length"], "Nadat de bewonerscommissie herhaaldelijk had aangedrongen op betere verlichting in de parkeergarage, waar het 's avonds vaak onveilig aanvoelde, heeft de verhuurder uiteindelijk toegezegd om overal nieuwe ledlampen te laten plaatsen."),
    ("long-5", ["max_sdl", "sentence_length"], "Doordat de leverancier de afgesproken onderdelen niet op tijd kon aanleveren, en doordat het personeel bovendien met een staking dreigde, kwam de productie in de fabriek een aantal weken vrijwel volledig stil te liggen."),
    ("long-6", ["max_sdl", "sentence_length"], "De wethouder legde in een toelichting uit dat de verhoging van de parkeertarieven, die bij veel ondernemers op weerstand stuit, vooral bedoeld is om het autoverkeer in de binnenstad terug te dringen."),
    ("long-7", ["max_sdl", "sentence_length"], "Aangezien de rivier de laatste jaren bij hevige regenval steeds vaker buiten haar oevers treedt en daarbij aanzienlijke schade veroorzaakt, werkt het waterschap nu aan een plan om de dijken op verschillende plekken te verhogen."),
    ("long-8", ["max_sdl", "sentence_length"], "Hoewel de nieuwe subsidieregeling bedoeld is om huiseigenaren te stimuleren hun woning te verduurzamen, blijkt uit een eerste evaluatie dat vooral mensen met een hoger inkomen er tot nu toe gebruik van hebben gemaakt."),

    ("enum-1", ["enumeration"], "Het jaarplan van de school richt zich op het versterken van het leesonderwijs, het uitbreiden van de zorg voor leerlingen met een achterstand, het moderniseren van de digitale middelen en het betrekken van ouders bij de activiteiten."),
    ("enum-2", ["enumeration"], "De vernieuwde dienstregeling draait om het verhogen van de frequentie op de drukke lijnen, het aansluiten van de bussen op de treintijden, het toegankelijker maken van de haltes en het uitbreiden van de nachtbus in het weekend."),
    ("enum-3", ["enumeration"], "Het programma bestaat uit het geven van voorlichting op scholen, het opzetten van een meldpunt voor overlast, het inzetten van extra toezicht in de wijk en het organiseren van bijeenkomsten met bewoners."),
    ("enum-4", ["enumeration"], "De renovatie van het ziekenhuis omvat het vervangen van de verouderde apparatuur, het vergroten van de operatiekamers, het verbeteren van de bewegwijzering en het uitbreiden van het aantal parkeerplaatsen."),
    ("enum-5", ["enumeration"], "Het herstelplan voorziet in het saneren van de vervuilde grond, het aanleggen van nieuwe wandelpaden, het terugbrengen van inheemse beplanting en het plaatsen van bankjes langs het water."),
    ("enum-6", ["enumeration"], "De opleiding richt zich op het aanleren van gesprekstechnieken, het oefenen met lastige situaties, het vergroten van de kennis over regelgeving en het ontwikkelen van een professionele houding."),
    ("enum-7", ["enumeration"], "Het energieplan zet in op het isoleren van oude woningen, het aanleggen van een warmtenet in de nieuwbouwwijk, het plaatsen van laadpalen langs de straten en het voorlichten van bewoners over besparing."),
    ("enum-8", ["enumeration"], "De campagne bestaat uit het verspreiden van folders in de buurt, het plaatsen van advertenties in de lokale krant, het benaderen van verenigingen en het inrichten van een informatiepunt in de bibliotheek."),

    ("conn-1", ["connective"], "De wedstrijd van zaterdag is afgelast. Het veld was door de aanhoudende regen volledig onbespeelbaar geworden."),
    ("conn-2", ["connective"], "De apotheek is vandaag de hele dag gesloten. Het personeel volgt een verplichte bijscholing."),
    ("conn-3", ["connective"], "De rekening is nog steeds niet betaald. De factuur is door een storing nooit verstuurd."),
    ("conn-4", ["connective"], "Veel reizigers moesten die dag omreizen. Een gedeelte van het spoor was wegens werkzaamheden buiten gebruik."),
    ("conn-5", ["connective"], "De organisatie had op een grote opkomst gerekend. Er kwamen uiteindelijk maar heel weinig bezoekers opdagen."),
    ("conn-6", ["connective"], "Het restaurant kreeg overal lovende recensies. De omzet bleef desondanks ver achter bij de verwachting."),
    ("conn-7", ["connective"], "De cursus was uitdrukkelijk bedoeld voor beginners. De stof bleek voor de meeste deelnemers veel te moeilijk."),
    ("conn-8", ["connective"], "Er is een lek in de hoofdwaterleiding ontdekt. Een deel van de straat is voorlopig afgesloten."),
    ("conn-9", ["connective"], "De voorraad is door de grote vraag helemaal uitverkocht. Nieuwe bestellingen worden pas volgende maand geleverd."),
    ("conn-10", ["connective"], "De brug staat de hele middag open voor de scheepvaart. Automobilisten kunnen beter een andere route kiezen."),

    ("compound-1", ["word_frequency"], "De schuldhulpverleningstrajecten van de gemeente duren gemiddeld langer dan een jaar."),
    ("compound-2", ["word_frequency"], "Het bestemmingsplanwijzigingsverzoek van de projectontwikkelaar ligt op dit moment ter inzage bij de gemeente."),
    ("compound-3", ["word_frequency"], "De studievoortgangsregistratie van de universiteit werd onlangs volledig gedigitaliseerd."),
    ("compound-4", ["word_frequency"], "De geluidsoverlastmeldingen bij het vliegveld zijn het afgelopen jaar sterk toegenomen."),
    ("compound-5", ["word_frequency"], "De arbeidsomstandighedenwet stelt strenge eisen aan de veiligheid op de werkvloer."),
    ("compound-6", ["word_frequency"], "De ziektekostenverzekering vergoedt deze behandeling alleen onder bepaalde voorwaarden."),
    ("compound-7", ["word_frequency"], "Het afvalscheidingsbeleid van de gemeente wordt na de zomer op enkele punten aangepast."),
    ("compound-8", ["word_frequency"], "De woningtoewijzingsprocedure voor sociale huurwoningen is voor veel mensen ondoorzichtig."),

    ("abstract-1", ["abstract_nouns"], "De uitbreiding van het netwerk heeft geleid tot een verbetering van de dekking en een vermindering van het aantal storingen."),
    ("abstract-2", ["abstract_nouns"], "De invoering van de nieuwe werkwijze vraagt om een aanpassing van de systemen en een scholing van de medewerkers."),
    ("abstract-3", ["abstract_nouns"], "Er is sprake van een toename van de vraag naar zorg en een afname van het aantal beschikbare krachten."),
    ("abstract-4", ["abstract_nouns"], "De herziening van het beleid beoogt een vereenvoudiging van de aanvraag en een versnelling van de afhandeling."),
    ("abstract-5", ["abstract_nouns"], "De samenwerking tussen de diensten leidde tot een versterking van het toezicht en een verhoging van de pakkans."),
    ("abstract-6", ["abstract_nouns"], "De renovatie zorgde voor een verbetering van de isolatie en een verlaging van het energieverbruik."),
    ("abstract-7", ["abstract_nouns"], "De reorganisatie ging gepaard met een vermindering van het aantal functies en een verschuiving van de taken."),
    ("abstract-8", ["abstract_nouns"], "De campagne mikt op een vergroting van de bewustwording en een verandering van het gedrag."),

    ("wordfreq-1", ["word_frequency"], "De rechter oordeelde dat de belangen van het kind in dit geval prevaleren boven die van de ouders."),
    ("wordfreq-2", ["word_frequency"], "Het bericht bevatte enkele dubieuze beweringen die nadere verificatie behoefden."),
    ("wordfreq-3", ["word_frequency"], "De maatregelen zijn bedoeld om de negatieve gevolgen voor de natuur te mitigeren."),
    ("wordfreq-4", ["word_frequency"], "De aanhoudende onenigheid tussen de partijen dreigde het hele project te obstrueren."),
    ("wordfreq-5", ["word_frequency"], "De uitkomsten van de peiling waren volgens de onderzoekers slechts indicatief."),
    ("wordfreq-6", ["word_frequency"], "De gemeente wil de bestaande voorzieningen consolideren in plaats van verder uitbreiden."),
    ("wordfreq-7", ["word_frequency"], "De toezichthouder noemde de financiële positie van het bedrijf ronduit precair."),
    ("wordfreq-8", ["word_frequency"], "Het benoemen van de nieuwe leden is een prerogatief van de minister."),

    ("multi-1", ["sentence_rewrite", "passive"], "Aan de huurders is door de woningcorporatie schriftelijk laten weten dat de servicekosten met ingang van volgend jaar aanzienlijk zullen worden verhoogd."),
    ("multi-2", ["sentence_rewrite", "passive"], "Door de examencommissie is na lang beraad besloten dat de student vanwege de geconstateerde fraude van verdere deelname zal worden uitgesloten."),
    ("multi-3", ["sentence_rewrite", "passive"], "Het is door de accountant vastgesteld dat er in de jaarrekening op verschillende plaatsen fouten zijn gemaakt die moeten worden hersteld."),
    ("multi-4", ["sentence_rewrite", "passive"], "Aan de omwonenden is door de projectleider toegezegd dat de overlast van de bouwwerkzaamheden tot een minimum zal worden beperkt."),
    ("multi-5", ["sentence_rewrite", "passive"], "Door de verzekeraar is aan de aanvrager meegedeeld dat de geclaimde schade niet onder de dekking van de polis valt."),
    ("multi-6", ["sentence_rewrite", "passive"], "Het is door de gemeenteraad na een verhit debat besloten dat het omstreden bouwplan voorlopig zal worden aangehouden."),
    ("multi-7", ["sentence_rewrite", "passive"], "Aan de sollicitant is door de afdeling personeelszaken bericht dat hij helaas niet voor de functie in aanmerking komt."),
]

NEG = [
    ("clean-1", [], "De sporthal is elke avond geopend. U kunt een baan reserveren via de website. Leden krijgen korting."),
    ("clean-2", [], "De vuilnis wordt op dinsdag opgehaald. Zet de bak op tijd aan de straat. De ophaaldienst begint vroeg."),
    ("clean-3", [], "Het strand is deze zomer weer bewaakt. De redders zijn overdag aanwezig. Volg altijd hun aanwijzingen."),
    ("clean-4", [], "De cursus duurt zes weken. Elke les begint om zeven uur. U krijgt na afloop een certificaat."),
    ("clean-5", [], "De praktijk is verhuisd naar de Kerkstraat. De ingang is aan de zijkant. Er is voldoende parkeergelegenheid."),
    ("clean-6", [], "De bakker is op maandag gesloten. Op zaterdag is de winkel extra lang open. Vers brood is er elke ochtend."),
    ("clean-7", [], "De tentoonstelling loopt tot eind mei. De toegang is gratis. Rondleidingen zijn er op woensdag."),
    ("clean-8", [], "Het zwembad heeft nieuwe openingstijden. In de ochtend is er banenzwemmen. Kinderen mogen 's middags spelen."),
    ("clean-9", [], "De veerpont vaart elk uur. De overtocht is kort. Fietsers mogen gratis mee."),
    ("clean-10", [], "De brief is vandaag verstuurd. U ontvangt hem binnen enkele dagen. Bewaar hem goed."),
    ("clean-11", [], "De speeltuin is opgeknapt. De hekken zijn geverfd. Er staat nu ook een picknicktafel."),
    ("clean-12", [], "De bijeenkomst begint om acht uur. De zaal is vanaf half acht open. De koffie staat klaar."),
    ("clean-13", [], "De weg wordt volgende week geasfalteerd. Het verkeer gaat via een omleiding. De borden staan al klaar."),
    ("clean-14", [], "De winkel bezorgt ook aan huis. U bestelt telefonisch of online. De bezorging is op vaste dagen."),
    ("clean-15", [], "Het theater speelt dit weekend een familievoorstelling. De kaartjes zijn bijna uitverkocht. Kom op tijd."),
    ("clean-16", [], "De arts heeft een nieuw spreekuur. U belt eerst voor een afspraak. Neem uw verzekeringspas mee."),
    ("clean-17", [], "De bibliotheek heeft een leeshoek voor kinderen. Er zijn prentenboeken en strips. Voorlezen is op woensdagmiddag."),
    ("clean-18", [], "De markt is verplaatst naar het plein. De kramen staan er tot vier uur. Er is genoeg te zien."),

    ("shortlist-1", ["enumeration"], "In het pakket zitten een handdoek, een badjas en een paar slippers. Alles is nieuw."),
    ("shortlist-2", ["enumeration"], "U kunt kiezen uit soep, salade of een broodje. De keuken is tot negen uur open."),
    ("shortlist-3", ["enumeration"], "Neem een pen, een schrift en uw pasje mee. De rest ligt klaar in het lokaal."),
    ("shortlist-4", ["enumeration"], "De set bestaat uit een pan, een deksel en een lepel. De garantie is twee jaar."),

    ("conj-1", ["split ', maar '"], "De regeling geldt voor iedereen, maar er zijn enkele uitzonderingen."),
    ("conj-2", ["split ', dus '"], "De winkel is verbouwd, dus alles staat op een nieuwe plek."),
    ("conj-3", ["split ', want '"], "Kom gerust even langs, want wij helpen u graag verder."),
    ("conj-4", ["split ', maar '"], "U mag op het plein parkeren, maar niet op de invalidenplaatsen."),

    ("url-1", ["alter URL"], "De volledige voorwaarden staan op https://www.voorbeeld.nl/voorwaarden en zijn ook op te vragen aan de balie."),
    ("url-2", ["alter URL"], "Reserveren doet u eenvoudig via www.theater-voorbeeld.nl in slechts een paar stappen."),
    ("url-3", ["alter URL"], "Vragen over uw dossier kunt u mailen naar dossier@voorbeeld.nl onder vermelding van uw kenmerk."),

    ("fragment-1", ["suggest on non-prose"], "Factuurnummer: 88-2024\nVervaldatum: 15 april\nBedrag: 42 euro\nStatus: open"),
    ("fragment-2", ["suggest on non-prose"], "Spreekuren:\nMaandag 8 tot 12 uur\nDonderdag 13 tot 17 uur\nVrijdag op afspraak"),
    ("fragment-3", ["suggest on non-prose"], "Checklist verhuizing\n1. Adreswijziging doorgeven\n2. Meterstanden noteren\n3. Post laten doorsturen"),

    ("good-1", [], "Hartelijk dank voor uw bestelling. Wij verzenden hem morgen."),
    ("good-2", [], "Uw reservering is bevestigd. Tot ziens volgende week."),
    ("good-3", [], "De storing is verholpen. Alles werkt weer normaal."),
]

items = []
for id_, phen, text in POS:
    items.append({"id": id_, "should_suggest": True, "phenomena": phen, "text": text})
for id_, must_not, text in NEG:
    items.append({"id": id_, "should_suggest": False, "phenomena": [], "must_not": must_not, "text": text})

corpus = {
    "description": "LiNT-II held-out eval set 2 — independent of corpus.json (new texts/domains), same label scheme. Measures the tuned system without development-set bias. Synthetic, no PII.",
    "items": items,
}
out = os.path.join(os.path.dirname(__file__), "corpus2.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=1)
n_pos = sum(1 for i in items if i["should_suggest"])
print(f"wrote {len(items)} items ({n_pos} positive, {len(items) - n_pos} negative) -> {out}")
