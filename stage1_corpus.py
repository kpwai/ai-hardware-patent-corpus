#!/usr/bin/env python3
"""
STAGE 1 -- Corpus construction from the two-folder Lens export.

Reads every per-year CSV from the application and grant folders, harmonises
schemas, collapses to invention level, applies the two-condition AI hardware
rule, and emits the analysis corpus.

Invention-level means one record per application number. Where an invention has
both a pre-grant publication and a grant, the grant supplies metadata (its CPC
codes are as-issued) while both dates are retained and `is_granted` is set. This
removes the double counting that inflated the previous 1.47M figure and answers
the reviewer's request for family-level normalisation.

Outputs:
    stage1_corpus/corpus.parquet
    stage1_corpus/attrition.csv
    stage1_corpus/ingest_log.csv
    stage1_corpus/bad_lines/*.txt
    stage1_corpus/validation_sample.csv
    stage1_corpus/manifest.json

Run:  python stage1_corpus.py
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

BADDIR = C.STAGE1 / "bad_lines"
BADDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------------
def read_one(path):
    bad = []

    def on_bad(line):
        bad.append(line)
        return None

    # low_memory is not supported with engine='python'; the python engine is
    # required here for the on_bad_lines callback that captures malformed rows.
    df = pd.read_csv(path, engine="python", on_bad_lines=on_bad,
                     quoting=csv.QUOTE_MINIMAL)
    if bad:
        with open(BADDIR / f"{path.stem}.txt", "w", encoding="utf-8") as f:
            for b in bad:
                f.write(str(b) + "\n")
    return df, len(bad)


def load_folder(folder, source):
    folder = Path(folder)
    if not folder.exists():
        sys.exit(f"FATAL: {folder} does not exist. Check config.RAW_*_DIR")
    files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.CSV"))
    if not files:
        sys.exit(f"FATAL: no CSV files in {folder}")

    print(f"\n  {source.upper()}: {len(files)} files in {folder}")
    frames, log = [], []
    for f in files:
        try:
            df, n_bad = read_one(f)
        except Exception as e:
            print(f"    {f.name:30} FAILED: {str(e)[:60]}")
            log.append({"source": source, "file": f.name, "rows": 0,
                        "bad_lines": None, "error": str(e)[:200]})
            continue
        df["_source_folder"] = source
        df["_source_file"] = f.name
        frames.append(df)
        log.append({"source": source, "file": f.name, "rows": len(df),
                    "bad_lines": n_bad, "error": None})
        flag = f"  ({n_bad} bad lines)" if n_bad else ""
        print(f"    {f.name:30} {len(df):>9,} rows{flag}")

    # Harmonise on the intersection of columns, not the union: a column absent
    # from some years would otherwise become silently-NaN for those years.
    colsets = [set(d.columns) for d in frames]
    common = set.intersection(*colsets)
    union = set.union(*colsets)
    dropped = union - common
    if dropped:
        print(f"    SCHEMA DRIFT: {len(dropped)} columns not present in all "
              f"files, dropped: {sorted(dropped)[:8]}"
              f"{' ...' if len(dropped) > 8 else ''}")
    order = [c for c in frames[0].columns if c in common]
    out = pd.concat([d[order] for d in frames], ignore_index=True)
    print(f"    -> {len(out):,} rows, {len(order)} columns")
    return out, log


def clean_appno(s):
    return re.sub(C.APPNO_CLEAN_RE, "", str(s)).upper() if pd.notna(s) else None


def parse_date_col(s):
    """Parse a date column that mixes ISO and DD/MM/YYYY.

    The grant folder exports ISO (2025-09-04); the filing folder exports
    DD/MM/YYYY (4/9/2025). Passing dayfirst=True to everything triggers a
    warning on the ISO rows and passing it to nothing silently turns
    4 September into 9 April. Split on the format instead.
    """
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out.loc[iso] = pd.to_datetime(s.loc[iso], format="%Y-%m-%d",
                                      errors="coerce")
    if (~iso).any():
        out.loc[~iso] = pd.to_datetime(s.loc[~iso], dayfirst=True,
                                       errors="coerce")
    return out


# ---------------------------------------------------------------------------
# FILTER
# ---------------------------------------------------------------------------
def parse_cpc(cell):
    if pd.isna(cell):
        return []
    return sorted({c.strip() for c in str(cell).split(";;") if c.strip()})


def any_prefix(codes, prefixes):
    return any(c.startswith(p) for c in codes for p in prefixes)


def safe_parquet(df, path):
    """Write parquet, tolerating the mixed-type columns Lens exports.

    The '#' column is Lens's row counter and arrives as a mix of int and str,
    which pyarrow refuses. Other object columns can hold mixed types too after
    concatenating files with drifting schemas. Cast them to nullable string.
    """
    out = df.copy()
    junk = [c for c in out.columns if str(c).strip() in ("#", "", "Unnamed: 0")]
    if junk:
        out = out.drop(columns=junk)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string")
    out.to_parquet(path, index=False)
    return out


def main():
    t0 = datetime.now()
    print("=" * 74)
    print("STAGE 1 -- CORPUS CONSTRUCTION (two-folder ingest)")
    print("=" * 74)

    # Ingest is the slow part (~30 min for 3.65M rows). Cache the post-ingest,
    # post-dedup, post-jurisdiction frame so rule iterations take a minute.
    # Pass --refresh to force a re-read of the raw folders.
    cache = C.STAGE1 / "ingest_cache.parquet"
    refresh = "--refresh" in sys.argv
    if cache.exists() and not refresh:
        print(f"\n  Using cached ingest: {cache}")
        print("  (pass --refresh to re-read the raw folders)")
        corpus = pd.read_parquet(cache)
        meta = json.loads((C.STAGE1 / "ingest_cache_meta.json").read_text())
        print(f"  {len(corpus):,} inventions restored "
              f"(from {meta['n_publications']:,} publications)")
        return continue_from_ingest(corpus, meta, t0)

    apps, log_a = load_folder(C.RAW_APP_DIR, "folder_app")
    grants, log_g = load_folder(C.RAW_GRANT_DIR, "folder_grant")
    pd.DataFrame(log_a + log_g).to_csv(C.STAGE1 / "ingest_log.csv", index=False)

    n_app, n_grant = len(apps), len(grants)

    common = [c for c in apps.columns if c in set(grants.columns)]
    only_a = set(apps.columns) - set(grants.columns)
    only_g = set(grants.columns) - set(apps.columns)
    if only_a or only_g:
        print(f"\n  Cross-product schema differences:")
        if only_a:
            print(f"    application-only: {sorted(only_a)}")
        if only_g:
            print(f"    grant-only      : {sorted(only_g)}")
        print(f"    using {len(common)} shared columns")

    df = pd.concat([apps[common], grants[common]], ignore_index=True)
    print(f"\n  Combined raw: {len(df):,} rows "
          f"({n_app:,} from folder A + {n_grant:,} from folder B)")

    # ---- exact publication duplicates across folders ---------------------
    # The two folders overlap: the same Lens ID appears in both. These are not
    # application/grant pairs of one invention -- they are the same publication
    # record twice. Collapse before anything else.
    if C.ID_COL in df.columns:
        n_before = len(df)
        in_both = df.groupby(C.ID_COL)["_source_folder"].nunique()
        overlap = int((in_both > 1).sum())
        df = df.drop_duplicates(C.ID_COL, keep="first")
        print(f"  Exact {C.ID_COL} duplicates removed: {n_before - len(df):,}")
        print(f"    {overlap:,} publications appeared in BOTH folders")
        print(f"  Distinct publications: {len(df):,}")

    # ---- document type from the data, not the folder name ----------------
    # Folder names do not reflect content: the 'grant' folder is mostly
    # applications (Kind=A1). Trust the explicit columns.
    if "Document Type" in df.columns:
        dt = df["Document Type"].astype(str).str.strip().str.lower()

        # Drop non-invention publications: search reports, abstract-only
        # records, and untyped rows. Stage 0 found ~6,000 search reports.
        n_before = len(df)
        vc_all = dt.value_counts()
        keep_mask = dt.isin(C.DOC_TYPES_KEEP)
        dropped_types = vc_all[~vc_all.index.isin(C.DOC_TYPES_KEEP)]
        if len(dropped_types):
            print("\n  Dropping non-invention document types:")
            for k, v in dropped_types.items():
                print(f"    {k:38} {v:>9,}")
        df = df[keep_mask].copy()
        dt = dt[keep_mask]
        print(f"  Document-type filter: {n_before:,} -> {len(df):,}")

        df["_source"] = np.where(dt.str.contains("grant") |
                                 dt.str.contains("limited patent") |
                                 dt.str.contains("amended patent"),
                                 "grant", "application")
    elif "Kind" in df.columns:
        df["_source"] = np.where(
            df["Kind"].astype(str).str.upper().str.startswith("B"),
            "grant", "application")
    else:
        sys.exit("FATAL: cannot determine document type; no Document Type or Kind")

    print(f"  By document type: "
          f"{int((df['_source'] == 'grant').sum()):,} grants, "
          f"{int((df['_source'] == 'application').sum()):,} applications")
    xtab = pd.crosstab(df["_source_folder"], df["_source"])
    print("  Folder vs actual document type:")
    print("    " + xtab.to_string().replace("\n", "\n    "))

    # ---- invention-level collapse ---------------------------------------
    appno_col = next((c for c in df.columns if "Application Number" in c), None)
    if appno_col is None:
        sys.exit("FATAL: no 'Application Number' column; cannot deduplicate")

    df["_appno"] = df[appno_col].map(clean_appno)
    n_missing_appno = int(df["_appno"].isna().sum())
    if n_missing_appno:
        print(f"  {n_missing_appno:,} records lack an application number; "
              f"kept as singletons")
        df.loc[df["_appno"].isna(), "_appno"] = (
            "NOAPP_" + df.index[df["_appno"].isna()].astype(str))

    # ---- jurisdiction ----------------------------------------------------
    if "Jurisdiction" in df.columns:
        jur = df["Jurisdiction"].value_counts()
        print(f"\n  Jurisdictions: {len(jur)}")
        for j, c in jur.head(10).items():
            print(f"    {j:6} {c:>10,} ({100*c/len(df):5.1f}%)")
        if len(jur) > 1:
            print("    >> Multiple offices present. Family-level dedup is")
            print("       essential: one invention filed in N offices otherwise")
            print("       counts N times, and office filing incentives (e.g. the")
            print("       post-2015 CN subsidy surge) mimic technology growth.")
        if C.JURISDICTION_FILTER:
            n_before = len(df)
            df = df[df["Jurisdiction"] == C.JURISDICTION_FILTER]
            print(f"    Filtered to {C.JURISDICTION_FILTER}: "
                  f"{n_before:,} -> {len(df):,}")

    # ---- family key ------------------------------------------------------
    # Earliest priority number identifies the invention across offices.
    # Family key = EARLIEST priority number. Lens lists priorities newest-first,
    # so the earliest is the LAST element, not the first. Taking [0] gives the
    # patent's own application number, which makes every continuation its own
    # family and defeats the deduplication entirely.
    if "Priority Numbers" in df.columns:
        parts = (df["Priority Numbers"].fillna("").astype(str)
                   .str.split(";;"))
        earliest = parts.map(lambda L: next((x for x in reversed(L) if x.strip()),
                                            ""))
        first_prio = earliest.map(clean_appno)
        df["_family"] = first_prio.where(first_prio.notna() & (first_prio != ""),
                                         df["_appno"])
        # sanity: how much does this differ from the buggy first-element key?
        wrong = (parts.map(lambda L: next((x for x in L if x.strip()), ""))
                      .map(clean_appno))
        n_diff = int((wrong != first_prio).sum())
        print(f"  Priority chains where earliest != first-listed: {n_diff:,}")
    else:
        df["_family"] = df["_appno"]

    n_fam = df["_family"].nunique()
    print(f"\n  Distinct application numbers: {df['_appno'].nunique():,}")
    print(f"  Distinct families (earliest priority): {n_fam:,}")
    if df["_appno"].nunique() > n_fam:
        print(f"    -> {df['_appno'].nunique() - n_fam:,} additional records are")
        print(f"       same-family duplicates (continuations, divisionals, or")
        print(f"       parallel filings in other offices)")

    dedup_key = {"publication": C.ID_COL,
                 "application": "_appno",
                 "family": "_family"}.get(C.DEDUP_LEVEL, "_family")
    print(f"\n  Deduplicating at level '{C.DEDUP_LEVEL}' (key: {dedup_key})")

    grouped = df.groupby(dedup_key)["_source"].nunique()
    both = int((grouped > 1).sum())
    print(f"  Units with both an application and a grant record: {both:,}")

    if C.CORPUS_MODE == "invention":
        # prefer the grant record; its CPC codes are as-issued
        df["_pref"] = (df["_source"] == "grant").astype(int)
        corpus = (df.sort_values("_pref", ascending=False)
                    .drop_duplicates(dedup_key, keep="first").copy())
        granted = set(df.loc[df["_source"] == "grant", dedup_key])
        corpus["is_granted"] = corpus[dedup_key].isin(granted)
    elif C.CORPUS_MODE == "grants_only":
        corpus = df[df["_source"] == "grant"].drop_duplicates(dedup_key).copy()
        corpus["is_granted"] = True
    elif C.CORPUS_MODE == "apps_only":
        corpus = df[df["_source"] == "application"].drop_duplicates(dedup_key).copy()
        corpus["is_granted"] = corpus[dedup_key].isin(
            set(df.loc[df["_source"] == "grant", dedup_key]))
    else:
        sys.exit(f"FATAL: unknown CORPUS_MODE {C.CORPUS_MODE!r}")

    n_inv = len(corpus)
    print(f"  Mode '{C.CORPUS_MODE}': {n_inv:,} inventions "
          f"({100*n_inv/len(df):.1f}% of publications)")
    print(f"    granted: {int(corpus['is_granted'].sum()):,} "
          f"({100*corpus['is_granted'].mean():.1f}%)")

    # ---- dates -----------------------------------------------------------
    # Dates are DD/MM/YYYY in this export (confirmed by values like 28/2/2025).
    # Without dayfirst=True pandas silently reads 4/9/2025 as 9 April instead of
    # 4 September, corrupting every record whose day is <= 12.
    for src, dst in (("Publication Date", "pub_year"),
                     ("Application Date", "filing_year"),
                     ("Earliest Priority Date", "priority_year")):
        if src in corpus.columns:
            parsed = parse_date_col(corpus[src])
            corpus[dst] = parsed.dt.year
            corpus[dst.replace("_year", "_date")] = parsed
            miss = 100 * parsed.isna().mean()
            print(f"  {src:24} parsed, {miss:5.1f}% missing/unparseable")
            if miss > 20:
                print(f"    >> HIGH MISSINGNESS. Do not use {dst} as the primary")
                print(f"       time axis. Stage 0 found ~60% missing publication")
                print(f"       dates in older records.")

    if {"pub_year", "filing_year"} <= set(corpus.columns):
        lag = (corpus["pub_year"] - corpus["filing_year"]).dropna()
        lag = lag[lag.between(0, 20)]
        print(f"  Publication lag: median {lag.median():.1f}y  "
              f"mean {lag.mean():.2f}y  p90 {lag.quantile(0.9):.1f}y")
        print("    -> filing year is the primary basis for early-detection claims")

    meta = {"n_publications": int(len(df)), "n_inventions": int(n_inv),
            "n_both": int(both), "n_folder_app": int(n_app),
            "n_folder_grant": int(n_grant), "n_shared_columns": int(len(common))}
    safe_parquet(corpus, C.STAGE1 / "ingest_cache.parquet")
    (C.STAGE1 / "ingest_cache_meta.json").write_text(json.dumps(meta))
    print(f"  Cached ingest -> {C.STAGE1 / 'ingest_cache.parquet'}")
    return continue_from_ingest(corpus, meta, t0)


def continue_from_ingest(corpus, meta, t0):
    """Everything downstream of ingest. Split out so the rule can be re-run
    against the cached frame without re-reading 3.65M rows."""
    n_pub = meta["n_publications"]
    n_inv = meta["n_inventions"]
    # ---- AI hardware filter ---------------------------------------------
    cpc_col = next((c for c in corpus.columns if "CPC" in c), None)
    if cpc_col is None:
        sys.exit("FATAL: no CPC column found")
    codes = corpus[cpc_col].map(parse_cpc)
    corpus["cpc_codes"] = codes.map(lambda L: ";;".join(L))

    title = corpus["Title"].fillna("").astype(str)
    abstract = corpus["Abstract"].fillna("").astype(str)
    corpus["text"] = (title + ". " + abstract).str.strip()

    t1_pref = tuple(C.TIER1_CPC_PREFIXES)
    if C.TIER1_INCLUDE_ARCHITECTURE:
        t1_pref += tuple(C.TIER1_ARCHITECTURE_PREFIXES)

    ev_t1 = codes.map(lambda L: any_prefix(L, t1_pref))
    ev_ai_cpc = codes.map(lambda L: any_prefix(L, C.AI_CPC_PREFIXES)
                          and not any_prefix(L, C.AI_CPC_EXCLUDE))
    ev_hw_cpc = codes.map(lambda L: any_prefix(L, C.HW_CPC_PREFIXES))
    ev_ai_txt = corpus["text"].str.contains(C.AI_TEXT_RE, regex=True)
    ev_ai_mech = corpus["text"].str.contains(C.AI_MECHANISM_RE, regex=True)
    ev_hw_txt = corpus["text"].str.contains(C.HW_TEXT_STRONG_RE, regex=True)
    ev_hw_weak = corpus["text"].str.contains(C.HW_TEXT_WEAK_RE, regex=True)

    cond_ai = ev_ai_cpc | ev_ai_txt
    if C.AI_MECHANISM_COUNTS_AS_AI:
        cond_ai = cond_ai | ev_ai_mech
    # Only STRONG hardware text counts. Weak terms (processor, memory, chip)
    # are software-claim boilerplate and matched a quarter of the whole sweep.
    cond_hw = ev_hw_cpc | ev_hw_txt

    # V7: the definitional code, corroborated by text. Tier 2 is disabled --
    # in the labelled sample it produced no true positives and 19 false ones.
    if C.TIER1_REQUIRE_TEXT_CORROBORATION:
        corroborated = ev_hw_txt | ev_ai_txt | ev_ai_mech
        keep = ev_t1 & corroborated
        n_uncorr = int((ev_t1 & ~corroborated).sum())
        print(f"\n  Tier 1 text corroboration: {int(ev_t1.sum()):,} carry "
              f"G06N3/06, {n_uncorr:,} lack any supporting text and are dropped")
    else:
        keep = ev_t1
    if C.ENABLE_TIER2:
        keep = keep | (cond_ai & cond_hw)

    # Subject-matter veto: quantum computing, and ML applied to chip design or
    # fabrication ("AI for hardware" rather than "hardware for AI").
    ev_veto = pd.Series(False, index=corpus.index)
    if C.APPLY_SUBJECT_VETO:
        scope = corpus["Title"].fillna("").astype(str) \
            if getattr(C, "VETO_SCOPE", "title") == "title" else corpus["text"]
        ev_veto = scope.str.contains(C.EXCLUDE_SUBJECT_RE, regex=True)
        n_vetoed = int((keep & ev_veto).sum())
        n_veto_t1 = int((ev_t1 & keep & ev_veto).sum())
        print(f"\n  Subject veto (quantum / EDA-fab), scope="
              f"{getattr(C, 'VETO_SCOPE', 'title')}: removed {n_vetoed:,} "
              f"otherwise-retained records")
        print(f"    of which Tier 1: {n_veto_t1:,} "
              f"(should be near zero; investigate if not)")
        if n_vetoed:
            ex = corpus.loc[keep & ev_veto, "Title"].head(8)
            print("    examples:")
            for t in ex:
                print(f"      - {str(t)[:78]}")
        keep = keep & ~ev_veto

    n_weak_only = int((ev_hw_weak & ~ev_hw_txt & ~ev_hw_cpc).sum())
    print(f"  Hardware evidence: strong text {int(ev_hw_txt.sum()):,} | "
          f"CPC {int(ev_hw_cpc.sum()):,} | weak-only {n_weak_only:,} (excluded)")

    corpus = corpus.assign(
        ev_tier1=ev_t1, ev_ai_cpc=ev_ai_cpc, ev_hw_cpc=ev_hw_cpc,
        ev_ai_text=ev_ai_txt, ev_ai_mech=ev_ai_mech, ev_hw_text=ev_hw_txt,
        ev_hw_weak=ev_hw_weak, ev_veto=ev_veto,
        cond_ai=cond_ai, cond_hw=cond_hw, in_corpus=keep,
        tier=np.where(keep & ev_t1, "T1",
                      np.where(keep, "T2", "excluded")))

    rows = [
        ("Raw publications (applications + grants)", n_pub),
        (f"Inventions after dedup (mode={C.CORPUS_MODE})", n_inv),
        ("  satisfies AI condition", int(cond_ai.sum())),
        ("  satisfies hardware condition", int(cond_hw.sum())),
        ("  Carries G06N3/06", int(ev_t1.sum())),
        ("  ...with text corroboration", int((ev_t1 & (ev_hw_txt | ev_ai_txt | ev_ai_mech)).sum())),
        ("  Tier 2 (disabled)" if not C.ENABLE_TIER2 else "  Tier 2 (both conditions)",
         int((keep & ~ev_t1).sum())),
        ("RETAINED AI HARDWARE CORPUS", int(keep.sum())),
    ]
    print("\n" + "-" * 74)
    print("ATTRITION")
    print("-" * 74)
    for label, v in rows:
        print(f"  {label:<50} {v:>10,} ({100*v/n_pub:5.2f}%)")

    final = corpus[corpus.in_corpus].copy()

    print("\n  Evidence composition:")
    r = final
    print(f"    AI  cpc-only {int((r.ev_ai_cpc & ~r.ev_ai_text).sum()):>8,}"
          f"  text-only {int((~r.ev_ai_cpc & r.ev_ai_text).sum()):>8,}"
          f"  both {int((r.ev_ai_cpc & r.ev_ai_text).sum()):>8,}")
    mech_only = int((r.ev_ai_mech & ~r.ev_ai_cpc & ~r.ev_ai_text).sum())
    print(f"    AI  mechanism pathway ONLY (no G06N code, no AI vocabulary): "
          f"{mech_only:,}")
    print(f"        = {100*mech_only/max(len(r),1):.1f}% of the corpus. If the")
    print(f"        validation sample shows these are mostly noise, set")
    print(f"        AI_MECHANISM_COUNTS_AS_AI = False and rebuild.")
    print(f"    HW  cpc-only {int((r.ev_hw_cpc & ~r.ev_hw_text).sum()):>8,}"
          f"  text-only {int((~r.ev_hw_cpc & r.ev_hw_text).sum()):>8,}"
          f"  both {int((r.ev_hw_cpc & r.ev_hw_text).sum()):>8,}")
    if "filing_year" in final.columns:
        vc = final["filing_year"].value_counts().sort_index()
        print("\n  Retained corpus by filing year:")
        print(vc[vc.index >= C.YEAR_START].to_string())

    # ---- outputs ---------------------------------------------------------
    pd.DataFrame(rows, columns=["step", "n"]).assign(
        pct_of_raw=lambda d: 100 * d["n"] / n_pub
    ).to_csv(C.STAGE1 / "attrition.csv", index=False)
    safe_parquet(final, C.STAGE1 / "corpus.parquet")

    # Slim record of the ENTIRE sweep, retained and excluded alike. Stage 4 uses
    # this to compute what the trends would have looked like without the filter
    # -- the naive-vs-validated comparison that replaces the firm analysis.
    slim_cols = [c for c in [C.ID_COL, "text", "pub_year", "filing_year",
                             "priority_year", "in_corpus", "tier", "cond_ai",
                             "cond_hw", "ev_ai_mech"] if c in corpus.columns]
    safe_parquet(corpus[slim_cols], C.STAGE1 / "sweep_slim.parquet")
    print(f"  Saved slim sweep ({len(corpus):,} rows) for the naive comparison")

    # ---- validation sample ----------------------------------------------
    n_each = C.VALIDATION_SAMPLE_SIZE
    parts = [final.sample(min(int(n_each * 0.6), len(final)),
                          random_state=C.SEED).assign(_side="retained")]
    dropped = corpus[~corpus.in_corpus]
    near = dropped[dropped.cond_ai ^ dropped.cond_hw]
    far = dropped[~dropped.cond_ai & ~dropped.cond_hw]
    if len(near):
        parts.append(near.sample(min(int(n_each * 0.3), len(near)),
                                 random_state=C.SEED).assign(_side="excluded_near"))
    if len(far):
        parts.append(far.sample(min(int(n_each * 0.1), len(far)),
                                random_state=C.SEED).assign(_side="excluded_far"))
    val = pd.concat(parts)
    cols = [c for c in [C.ID_COL, "Display Key", "Title", "Abstract", "tier",
                        "_side", "_source", cpc_col] if c in val.columns]
    val = val[cols].copy()
    val["is_ai_hardware"] = ""
    val["architecture"] = ""
    val["is_about_class"] = ""
    val["notes"] = ""
    val.sample(frac=1, random_state=C.SEED).to_csv(
        C.STAGE1 / "validation_sample.csv", index=False)

    (C.STAGE1 / "manifest.json").write_text(json.dumps({
        "stage": 1, "timestamp": datetime.now().isoformat(),
        "runtime_sec": (datetime.now() - t0).total_seconds(),
        "n_folder_app": meta.get("n_folder_app"),
        "n_folder_grant": meta.get("n_folder_grant"),
        "n_publications": n_pub, "n_inventions": n_inv,
        "double_counted_inventions": meta.get("n_both"),
        "corpus_mode": C.CORPUS_MODE,
        "jurisdiction_filter": C.JURISDICTION_FILTER,
        "dedup_level": C.DEDUP_LEVEL,
        "retained": len(final),
        "retention_pct_of_inventions": 100 * len(final) / n_inv,
        "tier1": int(ev_t1.sum()), "tier2": int((keep & ~ev_t1).sum()),
        "tier1_prefixes": list(t1_pref),
        "hw_cpc_prefixes": list(C.HW_CPC_PREFIXES),
        "ai_mechanism_enabled": C.AI_MECHANISM_COUNTS_AS_AI,
        "hw_weak_only_excluded": n_weak_only,
        "shared_columns": meta.get("n_shared_columns"),
    }, indent=2))

    print("\n" + "=" * 74)
    print(f"  {n_pub:,} publications -> {n_inv:,} inventions -> "
          f"{len(final):,} AI hardware")
    print(f"  Wrote -> {C.STAGE1}")
    if len(final) > 400_000:
        print("\n  NOTE: corpus large; rule may be too permissive. Inspect titles.")
    if len(final) < 30_000:
        print("\n  NOTE: corpus small; RTA may lack mass per firm.")
    print("=" * 74)


if __name__ == "__main__":
    main()