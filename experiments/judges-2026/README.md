# Human-judge study (Antal van den Bosch, Florian Kunneman, Daan Kolkman)

Five fictional Dutch government letters (200-250 words), written for the study
and addressed to an individual: two at LiNT level 2, two at level 3 and one at
level 4. The senders, regulations, names, amounts and contact details are
invented, and no RVO text is used (those letters are under an EZK
confidentiality agreement).

| file | genre | LiNT (text mode) | level |
|---|---|---|---|
| A_parkeerkaart | receipt of an application (municipality -> citizen) | 43.0 | 2 |
| B_isolatie_herinnering | reminder to supply documents (municipality -> citizen) | 39.0 | 2 |
| C_subsidie_verlening | grant award decision (agency -> business) | 52.9 | 3 |
| D_voornemen_terugvordering | intention to recover an allowance (agency -> citizen) | 50.4 | 3 |
| E_afwijzing_bezwaar | decision on objection (agency -> business) | 66.5 | 4 |

"Text mode" is how the demo analyses pasted text; `## ` heading markers are
stripped first. The selection may still change after comments.

## Frozen analyses

The study items are frozen demo analyses: each text is analysed once by the
live demo, and the result is stored and served at an unguessable URL,
`https://lint-ii.valkuil.net/frozen/<16 random digits>`. Every judge sees the
same suggestions and scores; nothing is re-run. On the box:

    python3 scripts/freeze_analysis.py experiments/judges-2026/texts/*.txt

This prints one URL per text and appends it to `index.tsv` in the snapshot
directory. Freezing again gives new URLs; old ones keep working.
Freezing again does not re-run the analysis, though: the service's result
cache hands back the same suggestions under the new URL. For a genuinely new
sample, pass `--max-suggestions N` with an N not used before for that text and
at least its `triggers_found` (see the script's docstring). Text A was chosen
this way as the best of 7 samples; B-E are first runs. The chosen URLs are
recorded in `selected.tsv` in the snapshot directory, not in this public repo.

## Interaction logging and LimeSurvey embedding

Everything a participant does on a frozen page is logged: popup opened and
closed, accept / ignore / undo (also "overal accepteren"), edits (start,
applied text, cancelled), copying the result, switching to the statistics
view, every suggestion status change with the document score, and the full
editor state (all statuses + the edited text) whenever the page is hidden or
closed. Events are batched in the browser and appended to
`logs/<id>.jsonl` in the snapshot directory, one JSON object per line, each
with the participant code, a random per-tab session id, the browser's time
(`t`), the server's receive time and the user agent. IP addresses are not
stored. The page tells participants that interactions are recorded.

The participant code comes from the page URL, `?p=<code>`, and is stored
verbatim on every line. Embed each item in LimeSurvey as an iframe in the
question text and let LimeSurvey fill the code in:

    <iframe src="https://lint-ii.valkuil.net/frozen/<id>?p={TOKEN}"
            width="100%" height="900" allow="clipboard-write"></iframe>

`{TOKEN}` needs a survey with a participant list; in an anonymous survey use
`{SAVEDID}` (the response id) instead. Check in a test response that the code
arrives. `allow="clipboard-write"` lets the page's copy button use the
clipboard inside the iframe. A page opened without `?p=` still logs, with
`participant: null`.
