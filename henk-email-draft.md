# E-mail aan Henk Pander Maat — concept

**Onderwerp:** LiNT-II — wat we met je feedback hebben gedaan

Beste Henk,

Dank nogmaals voor je uitvoerige en scherpe commentaar op LiNT-II — op de schorsings- en de datalekbrief, en op je latere test met de 31 klassieke complexiteitszinnen. Die opmerkingen hebben het werk van de afgelopen tijd grotendeels gestuurd. De demo staat live op <https://lint-ii.valkuil.net>.

Je grondlijn is daarbij het uitgangspunt gebleven: de tool blijft ambachtelijk-stilistisch en vooral transparant, per zin, zonder aan inhoud of volgorde te komen. De toevoegingen hieronder passen binnen die lijn.

## Keuze binnen een herschrijving (je punt bij zin 7)

Je merkte terecht op dat een gebundelde herschrijving alles-of-niets was: de woordvervanging wilde je wel, de splitsing niet ("maar dat kan niet"). Dat kan nu. Bij een herschrijving toont de tool twee varianten waaruit je kiest:

- *Eén zin, niet gesplitst:* "De gemeente bevestigt dat de persoonsgegevens zijn gelekt en dat zij de inwoners hierover per brief informeert."
- *Opgesplitst:* "De gemeente bevestigt dat de persoonsgegevens zijn gelekt. Zij informeert de inwoners hierover per brief."

Elke variant wordt apart gescoord, zodat je ziet wat de splitsing met de LiNT-score doet.

## Verbindingswoorden (je lacune-observatie)

Je wees erop dat LiNT-II per zin werkt en daardoor de relatie tússen zinnen niet ziet, en dat ontbrekende verbindingswoorden een echte, evidence-based lacune zijn — geen lekentheorie. Er is nu een voorzichtige coherentie-pass die aangrenzende zinnen kan samenvoegen met een natuurlijk voegwoord ("…, want …" of "…, maar …"), en die formele varianten als "namelijk/immers" mijdt.

## Opsomming als lijst

Een lange opsomming binnen één zin kan nu als puntsgewijze lijst worden getoond: "Het plan draait om het aanscherpen van de controles, het bijscholen van personeel … en het evalueren van de keuzes" wordt een inleiding met bullets.

## Lange samenstellingen

Een lange samenstelling zonder eenvoudig synoniem wordt gesplitst in plaats van vervangen, bijvoorbeeld "werkloosheidsuitkeringsaanvraag" → "aanvraag voor een werkloosheidsuitkering" of "uitstroombevorderingsbeleid" → "beleid om uitstroom te bevorderen".

## De inhoudelijke kwesties (passief zin 8, splitsing zin 18, foute woordvervangingen)

Interessant genoeg blijken de semantische misgrepen die je zag — 'onderzoek' → 'studie', of het omdraaien van wie iets doet ("wij hebben bevestigd" terwijl wij de bevestiging juist ontvingen) — grotendeels een beperking van het eerdere, kleinere taalmodel te zijn geweest. Sinds de overstap naar een groter model reproduceren die fouten zich in onze tests niet meer: vaktermen blijven staan en de handelende persoon blijft behouden, ook bij lastige ontvangst-constructies als "wij hebben … te horen gekregen" → "de arts heeft ons verteld dat …". We hebben er wel expliciete instructies aan toegevoegd als vangnet. De houterige aanwijzende herhaling bij splitsen ("Deze gegevens … Deze gegevens …") wordt bovendien opgevangen doordat de "één zin"-variant vaak juist de betrekkelijke bijzin geeft die je voorstelde ("… de maatregelen die zij na overleg heeft goedgekeurd").

## Eerder al

De tool maakt boven en in de uitleg expliciet dat hij alleen per zin stilistisch reviseert en niets aan inhoud of volgorde doet; koppen en structuur (ook uit een geüpload .docx) blijven staan en worden niet meer in zin 1 getrokken; en de valse markeringen in het diff-overzicht zijn verholpen.

Wat bewust *niet* is gebeurd: inhoudelijke gaten vullen — dat wilde je ook uitdrukkelijk niet van een revisietool, en wij evenmin.

Ik ben heel benieuwd of dit klopt met wat je voor ogen had, en of de datalekbrief nu beter uitpakt. Alle verdere feedback is opnieuw welkom.

Met hartelijke groet,
Antal

---

*Concept — nog na te lopen vóór verzending:*

- *Aanhef/afsluiting aanpassen aan hoe jullie corresponderen.*
- *Voorbeelden zijn licht opgeschoond voor de leesbaarheid; vervang ze desgewenst door echte schermafbeeldingen of eigen voorbeelden.*
- *Model niet met naam genoemd ("een groter model"); voeg "Mistral-large" toe als je concreet wilt zijn.*
- *Lengte: kan korter als je liever een kort bericht stuurt dat vooral naar de demo verwijst.*
