#!/usr/bin/env python3
"""Build corpus5.json — a fifth held-out LiNT-II eval set.

Same structure and label scheme as build_corpus.py .. build_corpus4.py, all
new texts and domains. Consolidation set for the 2026-07-30/31 fixes: the
word-family guard (incl. comparative/diminutive/particle stripping), the
two-layer spelling plausibility gate, the connective direction fixes, and
the enumeration surface route. New here:
- a spelling-* POSITIVE group (planted errors) — sets 1-4 never measured
  spelling DETECTION recall, only its false positives;
- enum-* mixes nominalized-infinitive lists (surface route) with plain
  noun-phrase lists (conj route);
- family-* includes 'warmere' (spaCy leaves it unlemmatized — regression
  case for the ADJ surface-strip) and covers lemma/comparative/diminutive/
  particle mechanisms;
- conn-* keeps two temporally-expressed consequences (conn-9/10) to track
  the known model-side GEEN residual honestly.
Synthetic, no PII.

    python3 scripts/eval/build_corpus5.py
    python3 scripts/eval/run_eval.py --corpus scripts/eval/corpus5.json --results scripts/eval/results5.json --fresh
"""
import json
import os

POS = [
    ("passive-1", ["passive"], "De gewonde buizerd is door de vogelopvang met succes teruggezet in het duingebied."),
    ("passive-2", ["passive"], "Het lesrooster van de muziekschool wordt door de nieuwe coördinator volledig omgegooid."),
    ("passive-3", ["passive"], "De nalatenschap is door de notaris volgens de laatste wilsbeschikking verdeeld."),
    ("passive-4", ["passive"], "De speeltoestellen van de kinderopvang worden door een onafhankelijke keurmeester jaarlijks getest."),
    ("passive-5", ["passive"], "De zieke kastanjeboom op het plein is door de gemeente uit voorzorg gekapt."),
    ("passive-6", ["passive"], "De sorteermachine in het postcentrum wordt door twee monteurs in de nachtdienst nagekeken."),
    ("passive-7", ["passive"], "Het lekkende dak van de ijsbaan zal door de verhuurder vóór het nieuwe seizoen worden hersteld."),

    ("long-1", ["max_sdl", "sentence_length"], "Omdat het aantal vluchten op de regionale luchthaven de afgelopen jaren zo sterk is gegroeid dat de omwonenden dagelijks over geluidsoverlast klagen, onderzoekt de provincie nu of er een nachtelijk landingsverbod kan worden ingesteld."),
    ("long-2", ["max_sdl", "sentence_length"], "De dirigente, die het amateurorkest twintig jaar geleden overnam toen het nog uit een handvol enthousiaste blazers bestond, neemt na het jubileumconcert afscheid van een vereniging die inmiddels tot ver buiten de regio bekendheid geniet."),
    ("long-3", ["max_sdl", "sentence_length"], "Hoewel de vrijwilligers van de voedselbank wekenlang extra pakketten hadden klaargezet omdat de feestdagen altijd meer aanvragen opleveren, bleek de vraag dit jaar zo groot dat er halverwege december al tekorten dreigden te ontstaan."),
    ("long-4", ["max_sdl", "sentence_length"], "Nadat de leerlingen van de rijschool bij herhaling hadden geklaagd dat de examenplaatsen in de regio maandenlang van tevoren volgeboekt waren, besloot de branchevereniging om samen met het exameninstituut naar structurele oplossingen te zoeken."),
    ("long-5", ["max_sdl", "sentence_length"], "Doordat de restauratie van het openluchtmuseum aanzienlijk duurder uitviel dan in de oorspronkelijke begroting was voorzien, moest het bestuur een beroep doen op een noodfonds dat eigenlijk voor heel andere doeleinden was bestemd."),
    ("long-6", ["max_sdl", "sentence_length"], "De sterrenkundige legde tijdens de publieksavond uit dat het nieuwe meetinstrument, waaraan haar team bijna tien jaar heeft gewerkt, signalen kan opvangen die tot nu toe volledig aan alle waarnemingen ontsnapten."),
    ("long-7", ["max_sdl", "sentence_length"], "Aangezien de wachtlijst voor een plek in het verzorgingstehuis de afgelopen twee jaar meer dan verdubbeld is en mantelzorgers steeds vaker overbelast raken, dringt de cliëntenraad aan op een versnelde uitbreiding van de capaciteit."),

    ("enum-1", ["enumeration"], "Het jubileumjaar van de dierentuin staat in het teken van het openen van een nieuw vlinderverblijf, het uitbreiden van de educatieve rondleidingen, het renoveren van het historische apenhuis en het aanleggen van een natuurspeelplaats."),
    ("enum-2", ["enumeration"], "De verbouwing van de bouwmarkt omvat het verplaatsen van de kassazone, het verbreden van de gangpaden, het vernieuwen van de drive-in voor bouwmaterialen en het inrichten van een aparte afdeling voor gereedschapsverhuur."),
    ("enum-3", ["enumeration"], "Het preventieplan van de tandartspraktijk draait om het geven van poetsinstructie aan kinderen, het versturen van halfjaarlijkse oproepen, het vroegtijdig signaleren van tandvleesproblemen en het terugdringen van no-shows."),
    ("enum-4", ["enumeration"], "De opleiding tot reddingszwemmer behelst het aanleren van vervoersgrepen, het oefenen van reddingen met kleding aan, het wekelijks trainen van de conditie en het grondig bestuderen van de veiligheidsprotocollen."),
    ("enum-5", ["enumeration"], "Het winterprogramma van de volkstuinvereniging bestaat uit het gezamenlijk onderhouden van de paden, het snoeien van de fruitbomen, het controleren van de waterleidingen en het organiseren van een ruilbeurs voor zaden."),
    ("enum-6", ["enumeration"], "De tentoonstelling over de zeevaart toont oude landkaarten uit de zestiende eeuw, handgeschreven brieven van scheepskapiteins, schaalmodellen van koopvaardijschepen en een unieke verzameling antieke navigatie-instrumenten."),
    ("enum-7", ["enumeration"], "De nieuwe stadswandeling voert langs het voormalige weeshuis aan de gracht, de verborgen binnentuin van het hofje, de oudste gevelsteen van de binnenstad en het gerestaureerde pakhuis van de graanhandel."),

    ("conn-1", ["connective"], "De rondvaartboot blijft vandaag aan de kade. De sluis is voor onderhoud gesloten."),
    ("conn-2", ["connective"], "De cursus kalligrafie gaat definitief door. Er zijn genoeg aanmeldingen binnengekomen."),
    ("conn-3", ["connective"], "Het schoolplein staat vol plassen. De drainage is al jaren aan vervanging toe."),
    ("conn-4", ["connective"], "De kaartverkoop verliep in het begin stroef. De zaal was op de avond zelf toch uitverkocht."),
    ("conn-5", ["connective"], "De imker had dit voorjaar weinig volken verloren. De honingoogst viel opnieuw tegen."),
    ("conn-6", ["connective"], "Het asfalt op de rondweg lag er al twintig jaar. De provincie heeft de weg volledig laten vernieuwen."),
    ("conn-7", ["connective"], "De bibliotheekbus stopt voortaan ook in de kleine dorpen. Veel ouderen kunnen niet meer naar de stad reizen."),
    ("conn-8", ["connective"], "Het dorpsfeest kostte meer dan er binnenkwam. De organisatie zoekt voor volgend jaar extra sponsors."),
    ("conn-9", ["connective"], "De storing in de seinen duurde de hele ochtend. De treinen reden pas na de middag weer op tijd."),
    ("conn-10", ["connective"], "Het nieuwe zwembad trok in de eerste week duizenden bezoekers. Kaartjes zijn voorlopig alleen online te krijgen."),

    ("compound-1", ["word_frequency"], "De perronverlengingswerkzaamheden bij het voorstadsstation duren tot diep in het najaar."),
    ("compound-2", ["word_frequency"], "De drinkwatertransportleiding onder de rivier wordt in fasen vervangen."),
    ("compound-3", ["word_frequency"], "De gevelrenovatiesubsidie voor eigenaren van monumentale panden is opnieuw opengesteld."),
    ("compound-4", ["word_frequency"], "Het muskusrattenbestrijdingsprogramma van het waterschap loopt komend jaar gewoon door."),
    ("compound-5", ["word_frequency"], "De schoolzwemvergoedingsregeling geldt alleen voor gezinnen met een laag inkomen."),
    ("compound-6", ["word_frequency"], "Het fietsparkeerverwijssysteem bij het station wijst automatisch de dichtstbijzijnde vrije plek aan."),
    ("compound-7", ["word_frequency"], "De begraafplaatsonderhoudsploeg werkt in de wintermaanden met een aangepast rooster."),

    ("abstract-1", ["abstract_nouns"], "Het plan voorziet in een verlaging van de drempels en een verbreding van de doelgroep."),
    ("abstract-2", ["abstract_nouns"], "De evaluatie wees op een verschraling van het aanbod en een afname van de deelname."),
    ("abstract-3", ["abstract_nouns"], "Het bestuur belooft een versterking van het toezicht en een verheldering van de procedures."),
    ("abstract-4", ["abstract_nouns"], "De subsidie mikt op een verduurzaming van de gebouwen en een verlaging van het verbruik."),
    ("abstract-5", ["abstract_nouns"], "De nota bepleit een verruiming van de openingstijden en een vereenvoudiging van de aanvraag."),
    ("abstract-6", ["abstract_nouns"], "Het traject leidde tot een verzakelijking van de verhoudingen en een verharding van de standpunten."),
    ("abstract-7", ["abstract_nouns"], "De aanpak zorgt voor een bundeling van de krachten en een verkorting van de lijnen."),

    ("wordfreq-1", ["word_frequency"], "De penningmeester schetste tijdens de vergadering een ronduit penibele financiële situatie."),
    ("wordfreq-2", ["word_frequency"], "De rechter deed de aangevoerde bezwaren af als futiele details."),
    ("wordfreq-3", ["word_frequency"], "De douane stuitte in de haven op een clandestiene handel in beschermde planten."),
    ("wordfreq-4", ["word_frequency"], "De wethouder toonde zich ambivalent over de komst van het distributiecentrum."),
    ("wordfreq-5", ["word_frequency"], "De inspectie verweet de instelling een rigide toepassing van de huisregels."),
    ("wordfreq-6", ["word_frequency"], "De burgemeester reageerde zichtbaar gepikeerd op de kritische vragen van de raad."),
    ("wordfreq-7", ["word_frequency"], "De advocaat sprak van kwalijke insinuaties aan het adres van haar cliënt."),

    ("spelling-1", ["spelling"], "De acomodatie op het eiland was tot in de puntjes verzorgd. Het ontbijt stond elke ochtend klaar."),
    ("spelling-2", ["spelling"], "Ik wordt volgende maand aan mijn knie geopereerd. De revalidatie duurt ongeveer zes weken."),
    ("spelling-3", ["spelling"], "Hij loop elke ochtend een rondje om het meer. Daarna drinkt hij koffie in het clubhuis."),
    ("spelling-4", ["spelling"], "De brandweer kwam onmiddelijk in actie na de melding. De schade bleef daardoor beperkt."),
    ("spelling-5", ["spelling"], "De trainer vond het gedrag van de supporters aggresief overkomen. De club neemt maatregelen."),
    ("spelling-6", ["spelling"], "Zij vind het lastig om nee te zeggen tegen extra diensten. Haar rooster zit al helemaal vol."),

    ("multi-1", ["sentence_rewrite", "passive"], "Aan de leden van de volkstuinvereniging is door het bestuur toegezegd dat de gestegen waterkosten dit jaar niet in de contributie zullen worden doorberekend."),
    ("multi-2", ["sentence_rewrite", "passive"], "Door de directie van de luchthaven is na overleg met de omwonenden besloten dat de proef met stillere aanvliegroutes met een jaar zal worden verlengd."),
    ("multi-3", ["sentence_rewrite", "passive"], "Het is door de arbeidsinspectie vastgesteld dat er in het sorteercentrum structureel te zware pakketten door één persoon werden getild."),
    ("multi-4", ["sentence_rewrite", "passive"], "Aan de ouders is door de kinderopvang gemeld dat de vestiging in de zomerweken vanwege personeelsgebrek eerder zal sluiten."),
    ("multi-5", ["sentence_rewrite", "passive"], "Door de organisatie van de wandelvierdaagse is aan de deelnemers beloofd dat er dit jaar meer rustpunten langs de route zullen worden ingericht."),
    ("multi-6", ["sentence_rewrite", "passive"], "Het is door de dierenbescherming bevestigd dat de verwaarloosde pony's inmiddels bij een gastgezin zijn ondergebracht."),
    ("multi-7", ["sentence_rewrite", "passive"], "Aan de huurders is door de woningstichting gemeld dat het achterstallige schilderwerk komend voorjaar alsnog zal worden uitgevoerd."),
]

NEG = [
    ("clean-1", [], "De speeltuin is weer open. De nieuwe schommel hangt er al. Kom gerust langs met de kinderen."),
    ("clean-2", [], "Het koor zoekt nieuwe leden. Zingen kan iedereen leren. De repetitie is op donderdag."),
    ("clean-3", [], "De boot vaart weer op tijd. De overtocht duurt vijf minuten. Fietsers gaan gratis mee."),
    ("clean-4", [], "De kringloopwinkel haalt grote spullen op. Bel voor een afspraak. Kleine spullen brengt u zelf."),
    ("clean-5", [], "Het consultatiebureau is verhuisd. Het nieuwe adres staat in de brief. Parkeren kan voor de deur."),
    ("clean-6", [], "De ijssalon heeft een nieuwe smaak. Proeven mag altijd. Een bolletje kost twee euro."),
    ("clean-7", [], "Het veldje voor honden is verplaatst. Het ligt nu achter de grote zaal. Zakjes hangen bij het hek."),
    ("clean-8", [], "Het ontbijt in het buurthuis is elke eerste zondag. Iedereen neemt iets mee. De koffie staat om negen uur klaar."),
    ("clean-9", [], "De fietsenmaker werkt ook op zaterdag. Een kleine beurt is meestal dezelfde dag klaar. Bellen mag altijd."),
    ("clean-10", [], "Het theater opent een uur voor de voorstelling. De garderobe is gratis. Wie te laat komt, wacht tot de pauze."),
    ("clean-11", [], "De moskee houdt een open dag. Bezoekers krijgen een rondleiding. Vragen stellen mag altijd."),
    ("clean-12", [], "De strandtent is in de winter dicht. De strandwacht blijft bereikbaar. De vlaggen gelden het hele jaar."),
    ("clean-13", [], "De volksuniversiteit start weer met taalles. Er zijn groepen voor elk niveau. Inschrijven kan online."),

    ("family-1", ["word_frequency on rare form of common family"], "Na de zomer komen de warmere truien weer uit de kast. De winkel hangt de collectie alvast op."),
    ("family-2", ["word_frequency on rare form of common family"], "De nieuwe wasmiddelen beloven schonere sportkleding. De test wijst uit of dat klopt."),
    ("family-3", ["word_frequency on rare form of common family"], "Op het terras staan bordjes met de soep van de dag. De prijzen staan op het krijtbord."),
    ("family-4", ["word_frequency on rare form of common family"], "Wie durft, mag na de les een stukje terugzwemmen naar de steiger. De badmeester kijkt mee."),
    ("family-5", ["word_frequency on rare form of common family"], "De vrolijkere kleuren maken het klaslokaal een stuk lichter. De leerlingen kozen ze zelf."),

    ("shortlist-1", ["enumeration"], "In het pakket zitten zaadjes, aarde en een schepje. De handleiding zit erbij."),
    ("shortlist-2", ["enumeration"], "Bij de brunch horen verse jus, broodjes en een eitje. Koffie schenken we onbeperkt bij."),
    ("shortlist-3", ["enumeration"], "Voor de proef heb je een potlood, papier en geduld nodig. De rest ligt op tafel."),
    ("shortlist-4", ["enumeration"], "De doos in de keuken bevat pleisters, een schaar en verband. Controleer de inhoud elk jaar."),

    ("conj-1", ["split ', maar '"], "Het terras is verwarmd, maar een jas blijft aan te raden."),
    ("conj-2", ["split ', dus '"], "De kassa is defect, dus alleen pinnen is mogelijk."),
    ("conj-3", ["split ', want '"], "Reserveer op tijd, want de vakantieweken lopen snel vol."),
    ("conj-4", ["split ', maar '"], "De bus rijdt vandaag wel, maar de dienstregeling is aangepast."),

    ("url-1", ["alter URL"], "De uitslagen van de wedstrijd staan op https://www.voorbeeldatletiek.nl/uitslagen en hangen ook in de kantine."),
    ("url-2", ["alter URL"], "Een afspraak maken kan het snelst via www.voorbeeldpraktijk.nl of anders telefonisch."),
    ("url-3", ["alter URL"], "Meld een kapotte lantaarnpaal via openbareruimte@voorbeeldstad.nl onder vermelding van het mastnummer."),

    ("fragment-1", ["suggest on non-prose"], "Klantnummer: 20931\nAbonnement: maandelijks\nIngangsdatum: 1 september\nStatus: actief"),
    ("fragment-2", ["suggest on non-prose"], "Trainingsschema week 3:\nDinsdag intervalduurloop\nDonderdag krachttraining\nZondag lange duurloop"),
    ("fragment-3", ["suggest on non-prose"], "Boodschappenlijst weekend\n1. Brood en beleg halen\n2. Groente voor de soep\n3. Verjaardagstaart bestellen"),

    ("good-1", [], "Uw auto staat klaar. De sleutel ligt bij de receptie."),
    ("good-2", [], "De foto's zijn geleverd. Veel plezier ermee."),
    ("good-3", [], "Uw aanvraag is goedgekeurd. U ontvangt de pas deze week."),
]

items = []
for id_, phen, text in POS:
    items.append({"id": id_, "should_suggest": True, "phenomena": phen, "text": text})
for id_, must_not, text in NEG:
    items.append({"id": id_, "should_suggest": False, "phenomena": [], "must_not": must_not, "text": text})

corpus = {
    "description": "LiNT-II held-out eval set 5 — independent of corpus.json..corpus4.json (new texts/domains), same label scheme. Consolidation set for the family guard, spelling plausibility gates, connective direction fixes and enumeration surface route; first set with spelling-* DETECTION positives. Synthetic, no PII.",
    "items": items,
}
out = os.path.join(os.path.dirname(__file__), "corpus5.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=1)
n_pos = sum(1 for i in items if i["should_suggest"])
print(f"wrote {len(items)} items ({n_pos} positive, {len(items) - n_pos} negative) -> {out}")
