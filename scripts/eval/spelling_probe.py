"""Measure LLM spelling / dt-error detection — backlog item 4 (fixed in 48c8fa8).

    python3 spelling_probe.py 10            # score the live prompt
    VARIANT=v1 python3 spelling_probe.py 10 # score a candidate

WHAT THIS SETTLED. The old backlog text called this a "detection wobble,
~1 of 3 runs". It was neither. At 10 reps the model flagged every planted
error; the failures were all one case, where it returned CORRECTIE identical
to WOORD ("wordt" -> "wordt") and the pipeline then correctly dropped the
suggestion (suggested_text == sent_text). A correction-FORMATION failure that
presents as flaky detection.

Hunspell is not in scope here and cannot be: dt-errors are valid dictionary
words (word/vind/loop all look up clean), so the LLM is their only detector.
Hunspell owns non-word typos and is verified alive separately.

The clean-* controls are the half that matters when changing this prompt -- a
dt-focused instruction is exactly the kind that starts "correcting" correct
verb forms. Note they read RAW model output, before the spelling pass filters,
so they overstate false alarms versus production; the base-vs-variant
comparison is still like-for-like.

Sends the REAL spelling prompt with the same numbered-sentence text the pass
builds, and checks whether the planted error is flagged with the right
correction. Hunspell is not involved: dt-errors are valid dictionary words, so
the LLM is the only detector for them.
"""
import os, re, sys, collections, concurrent.futures as cf  # noqa: E401
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
from lint_ii.llm.prompts import PROMPT_TEMPLATES, parse_block_response

SP = PROMPT_TEMPLATES["spelling"]

# (id, kind, full corpus text, bad word, expected correction)
CASES = [
 ("spelling-1","nonword","De acomodatie op het eiland was tot in de puntjes verzorgd. Het ontbijt stond elke ochtend klaar.","acomodatie","accommodatie"),
 ("spelling-2","dt","Ik wordt volgende maand aan mijn knie geopereerd. De revalidatie duurt ongeveer zes weken.","wordt","word"),
 ("spelling-3","dt","Hij loop elke ochtend een rondje om het meer. Daarna drinkt hij koffie in het dorp.","loop","loopt"),
 ("spelling-4","nonword","De brandweer kwam onmiddelijk in actie na de melding. De schade bleef daardoor beperkt.","onmiddelijk","onmiddellijk"),
 ("spelling-5","nonword","De trainer vond het gedrag van de supporters aggresief overkomen. De club neemt maatregelen.","aggresief","agressief"),
 ("spelling-6","dt","Zij vind het lastig om nee te zeggen tegen extra diensten. Haar rooster zit al helemaal vol.","vind","vindt"),
 # clean controls: must yield GEEN_FOUTEN (false-alarm watch)
 ("clean-A","clean","De speeltuin is weer open. De nieuwe schommel hangt er al. Kom gerust langs met de kinderen.",None,None),
 ("clean-B","clean","Het koor zoekt nieuwe leden. Zingen kan iedereen leren. De repetitie is op donderdag.",None,None),
 ("c4-clean-1","clean","De kinderboerderij is elke dag open. De geitjes mogen gevoerd worden. Voer koopt u bij de ingang.",None,None),
 ("c4-clean-2","clean","Het wijkcentrum zoekt een kok. Het gaat om twee dagen per week. Bel voor meer informatie.",None,None),
 ("c4-clean-3","clean","De glasbak is verplaatst. Hij staat nu bij de supermarkt. De oude plek wordt een plantsoen.",None,None),
 ("c4-clean-4","clean","Het spreekuur begint om negen uur. Neem uw pas mee. Zonder afspraak kan het druk zijn.",None,None),
 ("c4-clean-5","clean","De boot naar het eiland vaart elk uur. Honden mogen gratis mee. Fietsen kost een euro extra.",None,None),
 ("c4-clean-6","clean","De cursus start in september. Er zijn nog vier plekken vrij. Inschrijven kan aan de balie.",None,None),
 ("c4-clean-7","clean","Het gemaal draait weer volop. Het water zakt langzaam. De kade blijft nog even afgezet.",None,None),
 ("c4-clean-8","clean","De boekenkast bij het station is vernieuwd. U mag boeken meenemen en achterlaten. Ruilen is het idee.",None,None),
 ("c4-clean-9","clean","De moestuinvereniging houdt zaterdag open dag. Er zijn plantjes te koop. De koffie is gratis.",None,None),
 ("c4-clean-10","clean","Het loket is tijdelijk dicht. U kunt terecht in het stadskantoor. Vergeet uw afspraakbevestiging niet.",None,None),
 ("c4-clean-11","clean","De vuurtoren is weer te beklimmen. Boven is het uitzicht prachtig. Kaartjes zijn er alleen online.",None,None),
 ("c4-clean-12","clean","Het repaircafé is elke eerste zaterdag. Vrijwilligers maken uw spullen. Een kleine gift is welkom.",None,None),
 ("c4-clean-13","clean","De schaatsbaan gaat vrijdag open. Schaatsen huren kan ter plekke. Kinderen betalen de helft.",None,None),
 ("c4-good-1","clean","Uw pakket ligt klaar bij de balie. Neem uw legitimatie mee.",None,None),
]


# --- variants -------------------------------------------------------------
ANCHOR = "Geef voor elke gevonden fout het volgende gestructureerde formaat"
V1_BLOCK = """Let bij werkwoorden op de persoonsvorm: "ik" krijgt de stam zonder -t, "hij/zij/het" en "jij" (voor het werkwoord) krijgen de stam met -t. Twee voorbeelden:
- "Ik reageert morgen" -> WOORD: reageert, CORRECTIE: reageer
- "Hij reageer nooit" -> WOORD: reageer, CORRECTIE: reageert

CORRECTIE bevat het VERBETERDE woord en moet dus altijd verschillen van WOORD. Kun je geen verbeterde vorm geven, meld het woord dan niet.

"""
VARIANTS = {"base": SP["user"], "v1": SP["user"].replace(ANCHOR, V1_BLOCK + ANCHOR)}
if VARIANTS["v1"] == VARIANTS["base"]:
    raise SystemExit(
        "v1 anchor not found in the live spelling prompt. Since 48c8fa8 shipped "
        "this block, v1 may already BE base -- re-anchor before trusting an A/B."
    )
VARIANT = os.environ.get("VARIANT", "base")

def numbered(t):
    parts=[s.strip() for s in re.split(r'(?<=[.!?])\s+', t.strip()) if s.strip()]
    return "\n".join(f"{i+1}. {s}" for i,s in enumerate(parts))

def call(p):
    import httpx
    r=httpx.post("https://inference.hetzner.com/api/v1/chat/completions",
      headers={"Authorization":f"Bearer {os.environ['HETZNER_API_KEY']}"},
      json={"model":"Qwen/Qwen3.6-35B-A3B-FP8",
            "messages":[{"role":"system","content":SP["system"]},
                        {"role":"user","content":p}],
            "temperature":0.3,"max_tokens":900,
            "chat_template_kwargs":{"enable_thinking":False}}, timeout=150.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""

def probe(c):
    cid,kind,text,bad,corr = c
    try: out = call(VARIANTS[VARIANT].format(text=numbered(text)))
    except Exception as e: return cid,"ERROR",""
    blocks = parse_block_response(out, fields=["WOORD","ZIN_NUMMER","CORRECTIE","CATEGORIE","UITLEG"], required="WOORD")
    found = {(b.get("WOORD") or "").strip().lower().strip('".,') :
             (b.get("CORRECTIE") or "").strip().lower().strip('".,') for b in blocks}
    if kind == "clean":
        return cid, ("SILENT" if not blocks else "FALSE_ALARM"), ",".join(found) or "-"
    if bad.lower() not in found:
        return cid, "MISSED", ",".join(found) or ("GEEN_FOUTEN" if "GEEN_FOUTEN" in out.upper() else "-")
    got = found[bad.lower()]
    return cid, ("CAUGHT" if got == corr.lower() else "WRONG_FIX"), got

REPS = int(sys.argv[1]) if len(sys.argv)>1 else 10
jobs=[]
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    for c in CASES:
        for _ in range(REPS): jobs.append((c, ex.submit(probe,c)))
    res=[(c, f.result()) for c,f in jobs]
per=collections.defaultdict(list)
for c,(cid,v,val) in res: per[cid].append((v,val))

print(f"\n=== spelling detection, variant {VARIANT}, {REPS} reps ===")
by_kind=collections.defaultdict(lambda:[0,0])
for c in CASES:
    cid,kind = c[0],c[1]
    cnt=collections.Counter(v for v,_ in per[cid])
    ok = cnt["SILENT"] if kind=="clean" else cnt["CAUGHT"]
    by_kind[kind][0]+=ok; by_kind[kind][1]+=REPS
    flag = "ok  " if ok==REPS else ("MIX " if ok else "FAIL")
    extra=" ".join(f"{k}={n}" for k,n in sorted(cnt.items()))
    print(f"{flag} {cid:11s} [{kind:7s}] {ok:2d}/{REPS}   {extra}")
    for k,n in collections.Counter(f"{v}:{x}" for v,x in per[cid]).most_common(2):
        print(f"        {n}x {k}")
print("\n--- by kind ---")
for k,(a,b) in by_kind.items():
    print(f"  {k:8s} {a:3d}/{b}  ({100*a/b:.0f}%)")
