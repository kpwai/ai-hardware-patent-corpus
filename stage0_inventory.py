#!/usr/bin/env python3
"""
STAGE 0 -- Inventory the raw Lens.org export folders.

Run this before Stage 1. Lens exports drift: column sets change between the
application and grant products, and sometimes across years within a folder.
Concatenating without checking is how malformed lines and silent column
mismatches get in.

Reports:
  * files, row counts, and year coverage per folder
  * the column schema of each file, and which columns are NOT universal
  * parse failures per file, with the offending line
  * application/grant overlap on application number (the double-count estimate)

Usage:
    python stage0_inventory.py --app-dir /path/to/app --grant-dir /path/to/grant
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def read_head(path, n=2000):
    """Read a file's head robustly; return (df, n_bad, first_error)."""
    bad, err = [], None

    def on_bad(line):
        bad.append(line)
        return None

    try:
        # NOTE: low_memory is not supported with engine='python'.
        df = pd.read_csv(path, engine="python", on_bad_lines=on_bad,
                         quoting=csv.QUOTE_MINIMAL, nrows=n)
    except Exception as e:
        return None, len(bad), str(e).split("\n")[0]
    if bad:
        err = str(bad[0])[:120]
    return df, len(bad), err


def count_rows(path):
    """Row count without loading the whole frame. Uses the fast C engine;
    on_bad_lines='skip' here is fine because Stage 1 captures and logs the
    malformed lines properly."""
    try:
        total = 0
        for chunk in pd.read_csv(path, on_bad_lines="skip", chunksize=100_000,
                                 low_memory=False):
            total += len(chunk)
        return total
    except Exception:
        return None


def inventory(folder, label, deep):
    print("\n" + "=" * 74)
    print(f"{label.upper()}  --  {folder}")
    print("=" * 74)
    p = Path(folder)
    if not p.exists():
        print("  MISSING -- check the path")
        return {}

    files = sorted(p.glob("*.csv")) + sorted(p.glob("*.CSV"))
    print(f"  {len(files)} CSV files")
    if not files:
        return {}

    schemas, info, total = {}, [], 0
    for f in files:
        df, n_bad, err = read_head(f)
        if df is None:
            print(f"  {f.name:32} PARSE FAILED: {err}")
            info.append({"file": f.name, "error": err})
            continue
        cols = tuple(df.columns)
        schemas.setdefault(cols, []).append(f.name)
        n = count_rows(f) if deep else None
        if n:
            total += n
        info.append({"file": f.name, "cols": len(cols), "rows": n,
                     "bad_lines": n_bad, "first_bad": err})
        flag = f"  bad_lines={n_bad}" if n_bad else ""
        print(f"  {f.name:32} cols={len(cols):>3} "
              f"rows={n if n else '?':>10}{flag}")

    if deep:
        print(f"\n  TOTAL ROWS: {total:,}")
        # Lens caps rows per export by tier. A file sitting on exactly a round
        # number was almost certainly truncated, and the missing records will
        # concentrate in the highest-volume years -- biasing any growth curve.
        CAPS = (1000, 5000, 10_000, 20_000, 25_000, 50_000, 100_000, 250_000)
        suspect = [r for r in info
                   if r.get("rows") in CAPS or
                   (r.get("rows") and r["rows"] % 10_000 == 0 and r["rows"] > 0)]
        if suspect:
            print("\n  *** POSSIBLE EXPORT TRUNCATION ***")
            for r in suspect:
                print(f"    {r['file']:32} {r['rows']:>10,}  <- exact round count")
            print("    Files sitting on a round number were probably capped by")
            print("    the Lens export limit. Re-export those years in smaller")
            print("    slices (split by month or by jurisdiction) and confirm the")
            print("    counts change. Truncation concentrates in high-volume")
            print("    recent years, which is where your trend claims live.")
        else:
            print("  No round-number row counts -- no obvious export truncation.")

    if not schemas:
        print("\n  NO FILES PARSED SUCCESSFULLY -- see the errors above.")
        return {"files": info, "n_schemas": 0, "total_rows": 0}

    print(f"\n  Distinct column schemas: {len(schemas)}")
    if len(schemas) > 1:
        print("  >> SCHEMA DRIFT. Files do not share one column set.")
        allcols = Counter()
        for cols, names in schemas.items():
            allcols.update(cols)
        universal = {c for c, n in allcols.items() if n == len(schemas)}
        for i, (cols, names) in enumerate(schemas.items(), 1):
            print(f"\n    Schema {i} ({len(names)} files): "
                  f"{', '.join(names[:4])}{' ...' if len(names) > 4 else ''}")
            extra = set(cols) - universal
            if extra:
                print(f"      unique to this schema: {sorted(extra)}")
        print(f"\n    Columns present in EVERY schema: {len(universal)}")
        missing_somewhere = set(allcols) - universal
        if missing_somewhere:
            print(f"    Columns missing from some files: "
                  f"{sorted(missing_somewhere)}")
    else:
        cols = list(schemas.keys())[0]
        print(f"  Consistent across all files. {len(cols)} columns:")
        for i in range(0, len(cols), 3):
            print("    " + " | ".join(f"{c[:24]:24}" for c in cols[i:i+3]))

    return {"files": info, "n_schemas": len(schemas), "total_rows": total}


def overlap_check(app_dir, grant_dir, sample_files=4):
    """Estimate application/grant double counting via application number."""
    print("\n" + "=" * 74)
    print("APPLICATION / GRANT OVERLAP")
    print("=" * 74)

    def collect(folder, limit):
        out = set()
        files = sorted(Path(folder).glob("*.csv"))[:limit]
        for f in files:
            try:
                df = pd.read_csv(f, on_bad_lines="skip",
                                 usecols=lambda c: "Application Number" in c,
                                 low_memory=False)
                col = [c for c in df.columns if "Application Number" in c]
                if col:
                    out |= set(df[col[0]].dropna().astype(str)
                               .str.replace(r"\s+", "", regex=True))
            except Exception as e:
                print(f"    skipped {f.name}: {str(e)[:60]}")
        return out, files

    a, af = collect(app_dir, sample_files)
    g, gf = collect(grant_dir, sample_files)
    print(f"  Sampled {len(af)} application files -> {len(a):,} application numbers")
    print(f"  Sampled {len(gf)} grant files       -> {len(g):,} application numbers")
    if a and g:
        inter = a & g
        print(f"  Overlap: {len(inter):,} application numbers appear in BOTH")
        print(f"    = {100*len(inter)/max(len(a),1):.1f}% of sampled applications")
        print("    Each overlapping pair is ONE invention counted TWICE in a")
        print("    naive concatenation. Invention-level dedup removes this.")
    else:
        print("  Could not compare -- check that 'Application Number' exists")
        print("  in both products and that the sampled years overlap.")


def coverage_check(app_dir, grant_dir, sample_files=6):
    """Determine each folder's date window and jurisdiction mix.

    The folders are two date slicings of the same set -- one filtered on filing
    date, one on grant/publication date -- not an application/grant split. Their
    completeness therefore differs depending on which time axis you analyse on.
    """
    print("\n" + "=" * 74)
    print("DATE WINDOW AND JURISDICTION COVERAGE")
    print("=" * 74)

    for folder, label in ((app_dir, "folder_app"), (grant_dir, "folder_grant")):
        files = sorted(Path(folder).glob("*.csv"))[:sample_files]
        if not files:
            continue
        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(f, on_bad_lines="skip",
                                          low_memory=False))
            except Exception:
                pass
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)

        print(f"\n  {label}  ({len(files)} files sampled, {len(d):,} rows)")

        if "Jurisdiction" in d.columns:
            jur = d["Jurisdiction"].value_counts()
            top = ", ".join(f"{j} {100*c/len(d):.0f}%"
                            for j, c in jur.head(6).items())
            print(f"    jurisdictions ({len(jur)}): {top}")

        if "Document Type" in d.columns:
            dt = d["Document Type"].value_counts()
            print("    document types: " +
                  ", ".join(f"{k} {v:,}" for k, v in dt.items()))

        for col in ("Application Date", "Publication Date",
                    "Earliest Priority Date"):
            if col in d.columns:
                s = pd.to_datetime(d[col], dayfirst=True, errors="coerce")
                if s.notna().any():
                    yrs = s.dt.year
                    print(f"    {col:24} {int(yrs.min())}-{int(yrs.max())}  "
                          f"median {int(yrs.median())}  "
                          f"missing {100*s.isna().mean():.1f}%")

    print("\n  >> Compare the Application Date range in folder_app against the")
    print("     Publication Date range in folder_grant. The analysis window")
    print("     should be the span where BOTH are complete on your chosen axis;")
    print("     outside it, apparent trends are coverage artefacts.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--grant-dir", required=True)
    ap.add_argument("--deep", action="store_true",
                    help="Count rows in every file (slower)")
    ap.add_argument("--out", default="stage0_inventory.json")
    args = ap.parse_args()

    print("=" * 74)
    print("STAGE 0 -- RAW EXPORT INVENTORY")
    print("=" * 74)

    res = {
        "applications": inventory(args.app_dir, "applications", args.deep),
        "grants": inventory(args.grant_dir, "grants", args.deep),
    }
    overlap_check(args.app_dir, args.grant_dir)
    coverage_check(args.app_dir, args.grant_dir)

    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"\n  Wrote {args.out}")

    print("\n" + "=" * 74)
    print("WHAT TO CHECK IN THIS OUTPUT")
    print("=" * 74)
    print("  1. Schema drift. If >1 schema per folder, Stage 1 must harmonise")
    print("     on the intersection of columns, not the union.")
    print("  2. Year basis. Confirm whether the per-year split is by PUBLICATION")
    print("     year or filing year -- it changes what the file names mean.")
    print("  3. Bad lines. If concentrated in particular files, the export may")
    print("     need regenerating with proper quoting rather than repairing.")
    print("  4. Overlap. A high app/grant overlap confirms that the previous")
    print("     1.47M figure was inflated by double counting.")


if __name__ == "__main__":
    main()