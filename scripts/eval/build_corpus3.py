#!/usr/bin/env python3
"""Build corpus3.json — a third held-out LiNT-II eval set.

Same structure and label scheme as build_corpus.py / build_corpus2.py, but
again entirely new texts and domains. A second independent test set gives a
tighter read on generalisation and lets us watch for variance between held-out
sets rather than trusting a single point estimate. Synthetic, no PII.

    python3 scripts/eval/build_corpus3.py
    python3 scripts/eval/run_eval.py --corpus scripts/eval/corpus3.json --results scripts/eval/results3.json --fresh
"""
import json
import os

POS = [
    ("passive-1", ["passive"], "De subsidie is door het ministerie toegekend na een uitvoerige beoordeling van de aanvraag."),
    ("passive-2", ["passive"], "Het rapport werd door de onderzoekscommissie na maanden van beraad eindelijk gepresenteerd."),
    ("passive-3", ["passive"], "De verdachte is door de politie kort na de inbraak op heterdaad aangehouden."),
    ("passive-4", ["passive"], "De nieuwe brug wordt door een aannemer uit de regio in delen ter plaatse gemonteerd."),
    ("passive-5", ["passive"], "De klachten zijn door de ombudsman zorgvuldig en punt voor punt onderzocht."),
    ("passive-6", ["passive"], "Het contract moet door beide partijen vóór het einde van de maand worden ondertekend."),
    ("passive-7", ["passive"], "De hoofdprijs zal tijdens het gala door de jury aan de winnaar worden uitgereikt."),
    ("passive-8", ["passive"], "Het oude fabriekspand is door de eigenaar aan een groep kunstenaars verhuurd."),

    ("long-1", ["max_sdl", "sentence_length"], "Omdat de wachtlijsten in de jeugdzorg de laatste jaren zo sterk zijn opgelopen dat gezinnen soms maandenlang op hulp moeten wachten, heeft de instelling besloten om extra hulpverleners aan te nemen en de intakeprocedure grondig te vereenvoudigen."),
    ("long-2", ["max_sdl", "sentence_length"], "De directeur, die het bedrijf ruim twintig jaar geleden vanuit een kleine garage heeft opgebouwd tot een internationale onderneming met honderden werknemers, kondigde gisteren onverwacht aan dat hij aan het einde van het jaar zal terugtreden."),
    ("long-3", ["max_sdl", "sentence_length"], "Hoewel de meteorologen al dagen voor zware windstoten hadden gewaarschuwd en de gemeente uit voorzorg verschillende parken had afgesloten, bleef de daadwerkelijke schade aan gebouwen en bomen uiteindelijk gelukkig veel beperkter dan gevreesd."),
    ("long-4", ["max_sdl", "sentence_length"], "Nadat de leden van de vereniging tijdens een roerige vergadering hun onvrede over het gevoerde beleid hadden geuit en om opheldering hadden gevraagd, zag het voltallige bestuur zich genoodzaakt om zijn functie per direct neer te leggen."),
    ("long-5", ["max_sdl", "sentence_length"], "Doordat de kosten voor grondstoffen en energie het afgelopen jaar veel harder zijn gestegen dan bij het opstellen van de begroting was voorzien, ziet de fabrikant zich gedwongen om de prijzen van zijn producten opnieuw te verhogen."),
    ("long-6", ["max_sdl", "sentence_length"], "De hoogleraar zette in haar afscheidsrede uiteen dat het onderzoek naar zeldzame ziekten, dat lange tijd nauwelijks aandacht kreeg van financiers, de laatste jaren juist tot verrassend veel nieuwe en veelbelovende behandelingen heeft geleid."),
    ("long-7", ["max_sdl", "sentence_length"], "Aangezien steeds meer bezoekers het historische centrum in het hoogseizoen als onaangenaam druk ervaren en de bewoners daar herhaaldelijk over klagen, onderzoekt het stadsbestuur nu of het aantal toeristen op piekdagen kan worden beperkt."),
    ("long-8", ["max_sdl", "sentence_length"], "Hoewel het nieuwe digitale aanvraagsysteem was bedoeld om de administratie voor zowel burgers als ambtenaren aanzienlijk te vereenvoudigen, blijkt uit de eerste ervaringen dat veel gebruikers juist vastlopen in de vele verplichte en onduidelijke velden."),

    ("enum-1", ["enumeration"], "Het beleidsplan van de bibliotheek zet in op het uitbreiden van de digitale collectie, het organiseren van cursussen voor senioren, het openen van een studieruimte voor scholieren en het samenwerken met lokale verenigingen."),
    ("enum-2", ["enumeration"], "Het herstelprogramma voor het natuurgebied omvat het dempen van oude sloten, het herstellen van de oorspronkelijke waterhuishouding, het aanplanten van inheemse struiken en het weren van gemotoriseerd verkeer."),
    ("enum-3", ["enumeration"], "De aanpak van het personeelstekort bestaat uit het aantrekken van zij-instromers, het verbeteren van de arbeidsvoorwaarden, het bieden van meer opleidingskansen en het verlichten van de werkdruk op de afdelingen."),
    ("enum-4", ["enumeration"], "Het veiligheidsplan van het festival draait om het aanstellen van extra beveiligers, het aanleggen van bredere vluchtroutes, het plaatsen van voldoende drinkwaterpunten en het instrueren van alle vrijwilligers."),
    ("enum-5", ["enumeration"], "De verduurzaming van het sportcomplex behelst het vervangen van de verlichting door led, het isoleren van de kleedkamers, het plaatsen van zonnepanelen op het dak en het opvangen van regenwater voor de velden."),
    ("enum-6", ["enumeration"], "Het onderwijsplan richt zich op het versterken van de basisvaardigheden, het begeleiden van leerlingen met dyslexie, het inzetten van moderne lesmethoden en het betrekken van ouders bij het huiswerk."),
    ("enum-7", ["enumeration"], "De vernieuwing van het stationsgebied omvat het verbreden van de perrons, het overdekken van de fietsenstalling, het verplaatsen van de bushaltes en het aanleggen van een nieuwe tunnel voor voetgangers."),
    ("enum-8", ["enumeration"], "Het gezondheidsprogramma zet in op het stimuleren van meer beweging, het geven van voorlichting over voeding, het ondersteunen van mensen die willen stoppen met roken en het aanbieden van gratis controles."),

    ("conn-1", ["connective"], "De voorstelling van vanavond gaat niet door. De hoofdrolspeler is onverwacht ziek geworden."),
    ("conn-2", ["connective"], "De weg blijft het hele weekend afgesloten. Er wordt op dit moment een nieuwe riolering aangelegd."),
    ("conn-3", ["connective"], "De verwarming in de school doet het al dagen niet. De lessen zijn tijdelijk naar een ander gebouw verplaatst."),
    ("conn-4", ["connective"], "Het bedrijf boekte dit jaar recordwinst. De werknemers kregen toch geen loonsverhoging."),
    ("conn-5", ["connective"], "De trainer had veel van zijn nieuwe spelers verwacht. De ploeg verloor de eerste drie wedstrijden ruim."),
    ("conn-6", ["connective"], "De hele oplage was binnen een dag uitverkocht. De uitgever besloot meteen een tweede druk te laten maken."),
    ("conn-7", ["connective"], "Het examen was door bijna iedereen als pittig ervaren. De slagingspercentages vielen dit jaar flink lager uit."),
    ("conn-8", ["connective"], "De vlucht had een vertraging van enkele uren opgelopen. Veel reizigers misten daardoor hun aansluiting."),
    ("conn-9", ["connective"], "De winkelketen sloot dit jaar tientallen filialen. De verkoop via internet was sterk gestegen."),
    ("conn-10", ["connective"], "De brand richtte grote schade aan in het magazijn. De levering van bestellingen loopt voorlopig vertraging op."),

    ("compound-1", ["word_frequency"], "De inburgeringscursussen voor nieuwkomers worden voortaan door de gemeente zelf georganiseerd."),
    ("compound-2", ["word_frequency"], "Het gegevensbeschermingsbeleid van het ziekenhuis is onlangs op enkele punten aangescherpt."),
    ("compound-3", ["word_frequency"], "De verkeersveiligheidsmaatregelen rond de school hebben het aantal ongelukken verminderd."),
    ("compound-4", ["word_frequency"], "De pensioenpremieafdrachten van de werkgever worden elke maand automatisch verrekend."),
    ("compound-5", ["word_frequency"], "De klimaatadaptatieplannen van de provincie moeten wateroverlast in de toekomst voorkomen."),
    ("compound-6", ["word_frequency"], "De medicijnvergoedingsregeling verandert volgend jaar voor een grote groep patiënten."),
    ("compound-7", ["word_frequency"], "De onderwijshuisvestingsplannen van de gemeente lopen al jaren vertraging op."),
    ("compound-8", ["word_frequency"], "De brandveiligheidsvoorschriften voor de horeca zijn na de laatste controle strenger geworden."),

    ("abstract-1", ["abstract_nouns"], "De uitbreiding van de opvang leidde tot een verlichting van de druk en een verbetering van de doorstroming."),
    ("abstract-2", ["abstract_nouns"], "De invoering van het systeem vroeg om een aanpassing van de processen en een bijscholing van het personeel."),
    ("abstract-3", ["abstract_nouns"], "Er is sprake van een groei van de bevolking en een krimp van de beschikbare woonruimte."),
    ("abstract-4", ["abstract_nouns"], "De herziening van de regeling beoogt een verlaging van de lasten en een vergroting van de duidelijkheid."),
    ("abstract-5", ["abstract_nouns"], "De samenwerking tussen de scholen leidde tot een verrijking van het aanbod en een verbreding van de keuze."),
    ("abstract-6", ["abstract_nouns"], "De verbouwing zorgde voor een uitbreiding van de ruimte en een verhoging van het comfort."),
    ("abstract-7", ["abstract_nouns"], "De hervorming ging gepaard met een inkrimping van de organisatie en een herverdeling van de verantwoordelijkheden."),
    ("abstract-8", ["abstract_nouns"], "De actie streeft naar een vergroting van de betrokkenheid en een versterking van de onderlinge band."),

    ("wordfreq-1", ["word_frequency"], "De commissie achtte de aangevoerde argumenten onvoldoende steekhoudend."),
    ("wordfreq-2", ["word_frequency"], "De directie besloot de omstreden maatregel voorlopig te prolongeren."),
    ("wordfreq-3", ["word_frequency"], "Zijn optreden werd door de critici als tamelijk pretentieus omschreven."),
    ("wordfreq-4", ["word_frequency"], "De verklaring van de getuige bleek op cruciale punten inconsistent."),
    ("wordfreq-5", ["word_frequency"], "De onderhandelingen tussen de partijen belandden al snel in een impasse."),
    ("wordfreq-6", ["word_frequency"], "De minister noemde de aantijgingen volstrekt ongefundeerd."),
    ("wordfreq-7", ["word_frequency"], "De toon van de brief werd door velen als ronduit denigrerend ervaren."),
    ("wordfreq-8", ["word_frequency"], "Het rapport bevatte een groot aantal discutabele aannames."),

    ("multi-1", ["sentence_rewrite", "passive"], "Aan de gedupeerden is door de verzekeraar toegezegd dat de geleden schade zo snel mogelijk volledig zal worden vergoed."),
    ("multi-2", ["sentence_rewrite", "passive"], "Door de raad van bestuur is na een lange discussie besloten dat de omstreden fusie voorlopig zal worden uitgesteld."),
    ("multi-3", ["sentence_rewrite", "passive"], "Het is door de inspectie geconstateerd dat er in de keuken op meerdere punten de hygiëneregels zijn overtreden."),
    ("multi-4", ["sentence_rewrite", "passive"], "Aan de bewoners is door de aannemer beloofd dat de werkzaamheden nog vóór de winter helemaal zullen worden afgerond."),
    ("multi-5", ["sentence_rewrite", "passive"], "Door de school is aan de ouders laten weten dat de schoolreis vanwege de hoge kosten dit jaar niet zal doorgaan."),
    ("multi-6", ["sentence_rewrite", "passive"], "Het is door de rechtbank bepaald dat de verdachte wegens gebrek aan bewijs zal worden vrijgesproken."),
    ("multi-7", ["sentence_rewrite", "passive"], "Aan de klant is door de webwinkel gemeld dat het bestelde artikel helaas niet meer op voorraad is."),
]

NEG = [
    ("clean-1", [], "Het buurthuis is elke middag open. U kunt er terecht voor een kopje koffie. Iedereen is welkom."),
    ("clean-2", [], "De trein vertrekt om het kwartier. De rit naar de stad duurt kort. Kaartjes koopt u bij de automaat."),
    ("clean-3", [], "De tuin wordt in het voorjaar opnieuw ingezaaid. De paden blijven begaanbaar. Honden mogen aan de lijn mee."),
    ("clean-4", [], "De kapper werkt alleen op afspraak. U belt of komt even langs. De zaak is op dinsdag gesloten."),
    ("clean-5", [], "Het museum heeft een nieuwe vleugel. De entree blijft hetzelfde. Kinderen tot twaalf jaar gaan gratis naar binnen."),
    ("clean-6", [], "De sportclub zoekt nieuwe vrijwilligers. Er is werk in de kantine en op het veld. Aanmelden kan bij de balie."),
    ("clean-7", [], "De weekmarkt staat er op vrijdag. De kramen openen om negen uur. Er is vers fruit en kaas."),
    ("clean-8", [], "De school heeft een nieuwe directeur. Zij begint na de zomer. De ouders krijgen binnenkort een brief."),
    ("clean-9", [], "Het zwemseizoen begint in mei. Het water wordt dan verwarmd. Abonnementen zijn bij de kassa te koop."),
    ("clean-10", [], "De bus rijdt 's avonds minder vaak. Kijk voor de tijden op het bord. Neem een geldig vervoerbewijs mee."),
    ("clean-11", [], "De winkel heeft de openingstijden aangepast. Op zondag is de deur nu ook open. De folder ligt bij de kassa."),
    ("clean-12", [], "De dokterspost is 's nachts bereikbaar. Bel eerst voordat u langskomt. Houd uw gegevens bij de hand."),
    ("clean-13", [], "Het park sluit bij zonsondergang. De hekken gaan dan automatisch dicht. Overdag is iedereen welkom."),
    ("clean-14", [], "De vereniging houdt een open dag. U kunt alle sporten uitproberen. De koffie staat klaar."),
    ("clean-15", [], "De garage doet ook kleine reparaties. Maak vooraf een afspraak. Vervangend vervoer is mogelijk."),
    ("clean-16", [], "De bakkerij bakt elke dag vers. Op bestelling maken ze ook taarten. Bel gerust voor de mogelijkheden."),
    ("clean-17", [], "De veerdienst vaart bij goed weer. Bij storm kan de dienst uitvallen. Kijk voor de zekerheid op de site."),
    ("clean-18", [], "Het dorpshuis verhuurt zalen. Er is ruimte voor feesten en vergaderingen. Vraag gerust naar de tarieven."),

    ("shortlist-1", ["enumeration"], "In de doos zitten een oplader, een kabel en een hoesje. Alles is vooraf getest."),
    ("shortlist-2", ["enumeration"], "Op het menu staan vandaag vis, kip of een vegetarisch gerecht. De keuken sluit om tien uur."),
    ("shortlist-3", ["enumeration"], "Neem je zwemspullen, een handdoek en wat kleingeld mee. De rest is aanwezig."),
    ("shortlist-4", ["enumeration"], "De gereedschapsset bevat een hamer, een tang en een schroevendraaier. De koffer is stevig."),

    ("conj-1", ["split ', maar '"], "De cursus is gratis, maar u moet zich vooraf inschrijven."),
    ("conj-2", ["split ', dus '"], "Het weer wordt slechter, dus neem een paraplu mee."),
    ("conj-3", ["split ', want '"], "Kom vooral even langs, want er is genoeg te doen."),
    ("conj-4", ["split ', maar '"], "De bezorging is meestal snel, maar rond de feestdagen kan het langer duren."),

    ("url-1", ["alter URL"], "Meer informatie vindt u op https://www.voorbeeld-gemeente.nl/afval of anders belt u het algemene nummer."),
    ("url-2", ["alter URL"], "Aanmelden voor de nieuwsbrief kan via www.voorbeeldclub.nl in een handomdraai."),
    ("url-3", ["alter URL"], "Stuur uw vraag naar info@voorbeeld-school.nl en vermeld daarbij het klasnummer."),

    ("fragment-1", ["suggest on non-prose"], "Ordernummer: 4471-B\nLevering: 3 mei\nAantal: 12 stuks\nStatus: verzonden"),
    ("fragment-2", ["suggest on non-prose"], "Openingstijden:\nDinsdag 9 tot 17 uur\nWoensdag gesloten\nZaterdag 10 tot 14 uur"),
    ("fragment-3", ["suggest on non-prose"], "Paklijst kamp\n1. Slaapzak inpakken\n2. Regenjas meenemen\n3. Toiletspullen niet vergeten"),

    ("good-1", [], "Bedankt voor uw melding. Wij pakken het zo snel mogelijk op."),
    ("good-2", [], "Uw afspraak staat genoteerd. Tot volgende week."),
    ("good-3", [], "Het probleem is opgelost. U kunt weer inloggen."),
]

items = []
for id_, phen, text in POS:
    items.append({"id": id_, "should_suggest": True, "phenomena": phen, "text": text})
for id_, must_not, text in NEG:
    items.append({"id": id_, "should_suggest": False, "phenomena": [], "must_not": must_not, "text": text})

corpus = {
    "description": "LiNT-II held-out eval set 3 — independent of corpus.json and corpus2.json (new texts/domains), same label scheme. A second test set to gauge variance and generalisation. Synthetic, no PII.",
    "items": items,
}
out = os.path.join(os.path.dirname(__file__), "corpus3.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=1)
n_pos = sum(1 for i in items if i["should_suggest"])
print(f"wrote {len(items)} items ({n_pos} positive, {len(items) - n_pos} negative) -> {out}")
