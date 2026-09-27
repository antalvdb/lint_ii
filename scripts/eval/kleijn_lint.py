#!/usr/bin/env python3
"""Score Kleijn's 60 LIN texts (x2 versions) with LiNT-II and compare with
cloze scores and with the original LiNT tool. Backlog item 8, first step.

Everything runs locally: the texts are scored with the spaCy pipeline, and no
text is sent to any LLM provider. The texts and data are likely copyrighted
and stay in the gitignored scripts/eval/private/ (this script holds no text).

Inputs:
  --texts    dir of T01_mak.txt ... T60_moei.txt  (default: private/kleijn/texts)
  --cloze    Clozedata2016_3opgeschoond_metTscanAangevuld.csv (one row per
             student response; aggregated here to a mean per text version)
  --lindocs  LINdocs_2023_nosplitting.jasp.csv (the original tool's features
             and LiNT scores per text version; `clozepred_perc` is the cloze
             score CORRECTED for the reader sample's ability, the outcome the
             published formula was fitted to, not a prediction)
Output: private/kleijn/kleijn_scores.csv and a report on stdout.

Design facts, read from the data (not assumed):
  - Tekstversie 1 = *_mak (easy), 2 = *_moei (difficult).
  - Manipulatie (20 texts each) = what the difficult version changes:
      1 connectives REMOVED, 2 word order (longer dependencies),
      3 rarer words.
  - LiNT_score23 is the published formula (Pander Maat, Kleijn & Frissen
    2023, Table 3), fitted on these 120 texts: 100 minus (3.204 + 15.845 freq
    + 13.096 concrete - 1.331 dependency length - 3.829 clause length).
    Its levels with bands 34/46/60 reproduce the paper's Table 6 exactly.
"""

import argparse
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

PRIVATE = os.path.join(HERE, "private", "kleijn")
MANIPULATION = {1: "connectives", 2: "word order", 3: "word frequency"}
FEATURES = {  # ours -> the original tool's column in LINdocs
    "freq_log": "wrd_freq_log_zn_corr",
    "max_sdl": "AL_max",
    "content_words_per_clause": "Inhwrd_dz_zonder_abw",
    "proportion_concrete": "Conc_nw_ruim_p",
}


def score_texts(text_dir: str) -> pd.DataFrame:
    from lint_ii import ReadabilityAnalysis

    rows = []
    for name in sorted(os.listdir(text_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(text_dir, name), encoding="utf-8-sig") as f:
            text = f.read()
        # The texts are Markdown-ish ("### Title"); the Markdown path keeps the
        # title out of the scored prose, as the demo does for .docx input.
        a = ReadabilityAnalysis.from_markdown(text)
        lint = a.lint
        rows.append({
            "file": name,
            "text": int(name[1:3]),
            "version": 1 if "_mak" in name else 2,
            "lint": lint.score,
            "level": lint.level,
            "n_sentences": len(a.sentences),
            **{k: getattr(lint, k) for k in FEATURES},
        })
    return pd.DataFrame(rows)


def cloze_by_text(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                     usecols=["Tekstnr", "Tekstversie", "Manipulatie",
                              "Score_accept", "Score_exact", "PPN"])
    g = df.groupby(["Tekstnr", "Tekstversie"]).agg(
        manipulation=("Manipulatie", "first"),
        cloze_accept=("Score_accept", "mean"),
        cloze_exact=("Score_exact", "mean"),
        n_responses=("Score_accept", "size"),
        n_readers=("PPN", "nunique"),
    ).reset_index().rename(columns={"Tekstnr": "text", "Tekstversie": "version"})
    for c in ("cloze_accept", "cloze_exact"):
        g[c] *= 100  # percent correct
    return g


def reference(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    keep = ["Inputfile", "clozepred_perc", "LiNT_score1", "LiNT_niveau1", "LiNT_score2",
            "LiNT_niveau2", "LiNT_score23", "LiNT23_score_oud", "LiNTformule23_compute",
            *FEATURES.values()]
    d = d[[c for c in keep if c in d.columns]].rename(columns={"Inputfile": "file"})
    return d.rename(columns={v: f"ref_{k}" for k, v in FEATURES.items()})


def corr(x, y) -> str:
    ok = x.notna() & y.notna()
    r, p = stats.pearsonr(x[ok], y[ok])
    rho, _ = stats.spearmanr(x[ok], y[ok])
    return f"r={r:+.2f} (p={p:.1g}), rho={rho:+.2f}, n={ok.sum()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--texts", default=os.path.join(PRIVATE, "texts"))
    ap.add_argument("--cloze", required=True)
    ap.add_argument("--lindocs", required=True)
    ap.add_argument("--out", default=os.path.join(PRIVATE, "kleijn_scores.csv"))
    args = ap.parse_args()

    ours = score_texts(args.texts)
    df = ours.merge(cloze_by_text(args.cloze), on=["text", "version"], how="left")
    df = df.merge(reference(args.lindocs), on="file", how="left")
    df["manipulation"] = df["manipulation"].map(MANIPULATION)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} text versions scored; missing cloze: {df.cloze_accept.isna().sum()}, "
          f"missing reference: {df.ref_freq_log.isna().sum()}")

    print("\n== Which reference column is our formula? (ours vs each, all 120)")
    for c in ("LiNT_score1", "LiNT_score2", "LiNT_score23", "LiNT23_score_oud", "LiNTformule23_compute"):
        if c in df:
            diff = (df.lint - df[c]).abs()
            print(f"  {c:22s} {corr(df.lint, df[c])}; mean |diff| {diff.mean():.2f}, max {diff.max():.2f}")
    # The reference FEATURES through our coefficients isolate formula from pipeline.
    from lint_ii.core.lint_scorer import LintScorer
    df["ref_via_our_formula"] = [
        LintScorer(r.ref_freq_log, r.ref_max_sdl, r.ref_content_words_per_clause,
                   r.ref_proportion_concrete).score
        for r in df.itertuples()]
    for c in ("LiNT_score1", "LiNT_score2", "LiNT_score23", "LiNT23_score_oud", "LiNTformule23_compute"):
        if c in df:
            print(f"  original features via our formula vs {c:22s} "
                  f"mean |diff| {(df.ref_via_our_formula - df[c]).abs().mean():.3f}")

    print("\n== Our features vs the original tool's (same texts)")
    for k in FEATURES:
        d = df[k] - df[f"ref_{k}"]
        print(f"  {k:26s} {corr(df[k], df[f'ref_{k}'])}; mean diff {d.mean():+.3f}, mean |diff| {d.abs().mean():.3f}")

    print("\n== LiNT score vs cloze (higher cloze = easier, so expect r < 0)")
    print(f"  ours vs corrected cloze       {corr(df.lint, df.clozepred_perc)}")
    print(f"  published vs corrected cloze  {corr(df.LiNT_score23, df.clozepred_perc)}  (paper: R=.865)")
    print(f"  ours vs raw cloze, all 120    {corr(df.lint, df.cloze_accept)}")
    for v, lab in ((1, "easy (mak)"), (2, "difficult (moei)")):
        s = df[df.version == v]
        print(f"  ours, {lab:17s}  {corr(s.lint, s.cloze_accept)}")
    for m in MANIPULATION.values():
        s = df[df.manipulation == m]
        print(f"  ours, {m:17s}  {corr(s.lint, s.cloze_accept)}")

    print("\n== Pairs: is the difficult version harder? (per manipulation, 20 pairs each)")
    w = df.pivot_table(index=["text", "manipulation"], columns="version",
                       values=["lint", "cloze_accept"]).reset_index()
    w["d_lint"] = w[("lint", 2)] - w[("lint", 1)]
    w["d_cloze"] = w[("cloze_accept", 2)] - w[("cloze_accept", 1)]
    for m in MANIPULATION.values():
        s = w[w.manipulation == m]
        print(f"  {m:15s} LiNT harder in {int((s.d_lint > 0).sum()):2d}/20 (mean {s.d_lint.mean():+.1f}); "
              f"cloze lower in {int((s.d_cloze < 0).sum()):2d}/20 (mean {s.d_cloze.mean():+.1f} pts)")

    band = lambda cut4: (lambda x: 1 if x < 34 else 2 if x < 46 else 3 if x < cut4 else 4)
    for cut in (58, 60):  # LiNT-II uses 58; the paper uses 60
        ref, mine = df.LiNT_score23.map(band(cut)), df.lint.map(band(cut))
        print(f"\n== Levels, bands 34/46/{cut}: agreement {(ref == mine).mean():.0%}; "
              f"published {ref.value_counts().sort_index().to_dict()}, ours {mine.value_counts().sort_index().to_dict()}")
    print("\n== Level spread (ours, LiNT-II bands)")
    print("  " + str(df.groupby(["version", "level"]).size().unstack(fill_value=0).to_dict("index")))
    print(f"  LiNT range {df.lint.min():.1f}–{df.lint.max():.1f}, median {df.lint.median():.1f}")
    print(f"\nPer-text table: {args.out} (private)")


if __name__ == "__main__":
    main()
