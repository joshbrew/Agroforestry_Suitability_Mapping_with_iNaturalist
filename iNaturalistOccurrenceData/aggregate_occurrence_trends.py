#!/usr/bin/env python3
import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Full run:
# python aggregate_occurrence_trends.py D:/envpull_association_test/occurrences_enriched.csv --group-by matched_species_name

# Just run the new downstream steps against an existing trend_summary:
# python aggregate_occurrence_trends.py D:/envpull_association_test/occurrences_enriched.csv --group-by matched_species_name --outdir D:/envpull_association_test/trend_summary --next-steps-only

# Force recompute instead of reusing existing summaries:
# python aggregate_occurrence_trends.py D:/envpull_association_test/occurrences_enriched.csv --group-by matched_species_name --outdir D:/envpull_association_test/trend_summary --no-resume-existing

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUMBERS = [f"{i:02d}" for i in range(1, 13)]

TERRACLIMATE_MONTHLY_RX = re.compile(r"^(terraclimate_[^_]+?)(?:_(\d{4}))?_m(0[1-9]|1[0-2])$")
TERRACLIMATE_AGG_RX = re.compile(r"^(terraclimate_[^_]+?)(?:_(\d{4}))?_(mean|sum|min|max)$")

DEFAULT_ID_COLUMNS = {
    "id",
    "row_id",
    "species_node_id",
    "gbifID",
    "occurrenceID",
    "scientificName",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "country",
    "stateProvince",
    "eventDate",
    "year",
    "month",
    "day",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "basisOfRecord",
    "matched_species_name",
}

DEFAULT_SKIP_NUMERIC_PATTERNS = [
    r"(?:^|_)lon$",
    r"(?:^|_)lat$",
    r"(?:^|_)x$",
    r"(?:^|_)y$",
    r"(?:^|_)row$",
    r"(?:^|_)col$",
    r"(?:^|_)year$",
    r"(?:^|_)sample_found$",
    r"(?:^|_)hit$",
    r"(?:^|_)ok$",
    r"(?:^|_)in_bounds$",
    r"(?:^|_)is_nodata$",
    r"(?:^|_)flowdir$",
]

DEFAULT_CATEGORICAL_PATTERNS = [
    r"^glim(?:_|$)",
    r"^mcd12q1(?:_|$)",
    r"^dem_continent$",
]

DEFAULT_NUMERIC_PREFIXES = [
    "dem_",
    "twi_",
    "soilgrids_",
    "terraclimate_",
]

DEFAULT_CATEGORICAL_PREFIXES = [
    "glim_",
    "mcd12q1_",
]

DEFAULT_FAMILY_RELIABILITY_PRIORS = {
    "terraclimate": 1.00,
    "dem": 0.95,
    "twi": 0.90,
    "soilgrids": 0.75,
    "glim": 0.80,
    "mcd12q1": 0.60,
    "other": 0.80,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Aggregate enriched occurrence data into numeric summaries, categorical frequencies, envelope-overlap matrices, charts, and empirical feature weights."
    )
    ap.add_argument("csv", help="Input enriched CSV")
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to trend_summary beside the input CSV.",
    )
    ap.add_argument("--group-by", default=None, help="Optional grouping column, such as matched_species_name or species")
    ap.add_argument("--top-n", type=int, default=50, help="Top N values to keep per categorical column")
    ap.add_argument("--fig-dpi", type=int, default=160, help="DPI for saved charts")
    ap.add_argument(
        "--min-valid-months",
        type=int,
        default=1,
        help="Minimum non-null monthly columns required to include a TerraClimate variable",
    )
    ap.add_argument("--delimiter", default=",", help="CSV delimiter")
    ap.add_argument(
        "--max-category-label-len",
        type=int,
        default=64,
        help="Maximum label length to render in categorical charts",
    )
    ap.add_argument(
        "--max-compare-groups",
        type=int,
        default=24,
        help="Maximum number of groups to include in multi-group comparison charts",
    )
    ap.add_argument(
        "--max-compare-categories",
        type=int,
        default=12,
        help="Maximum number of category values to include in multi-group categorical comparison charts",
    )
    ap.add_argument(
        "--next-steps-only",
        action="store_true",
        help="Reuse existing summary CSVs in outdir when available and run the newer downstream steps without regenerating the older summary/chart outputs.",
    )
    ap.add_argument(
        "--no-resume-existing",
        action="store_true",
        help="Do not reuse existing summary CSVs in outdir; recompute them from the input CSV.",
    )
    ap.add_argument(
        "--min-valid-features-per-pair",
        type=int,
        default=8,
        help="Minimum shared features required to compute a species-pair envelope-overlap matrix.",
    )
    ap.add_argument(
        "--min-valid-values-per-feature-weight",
        type=int,
        default=20,
        help="Minimum non-null observation count required to compute a feature weight.",
    )
    ap.add_argument(
        "--family-reliability-priors",
        default="terraclimate=1.0,dem=0.95,twi=0.9,soilgrids=0.75,glim=0.8,mcd12q1=0.6,other=0.8",
        help="Comma-separated family=reliability priors used to temper feature weights and blended overlap scores.",
    )
    ap.add_argument(
        "--min-iqr-floor-frac",
        type=float,
        default=0.05,
        help="Lower bound fraction of total cross-group feature span used to stabilize IQR-based signal weighting.",
    )
    ap.add_argument(
        "--categorical-overlap-share",
        type=float,
        default=0.15,
        help="Share of the final blended overlap score reserved for categorical-family overlap when categorical summaries exist.",
    )
    return ap.parse_args()


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def safe_slug(value):
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "group"


def group_value_to_label(value):
    return "NA" if pd.isna(value) else str(value)


def matches_any_pattern(name, patterns):
    return any(re.search(pat, name) for pat in patterns)


def parse_terraclimate_monthly_column(name):
    m = TERRACLIMATE_MONTHLY_RX.match(str(name))
    if not m:
        return None
    return {
        "variable": str(m.group(1)),
        "year": int(m.group(2)) if m.group(2) else None,
        "month_num": int(m.group(3)),
        "month": str(m.group(3)),
        "canonical_column": f"{m.group(1)}_m{m.group(3)}",
    }


def parse_terraclimate_aggregate_column(name):
    m = TERRACLIMATE_AGG_RX.match(str(name))
    if not m:
        return None
    return {
        "variable": str(m.group(1)),
        "year": int(m.group(2)) if m.group(2) else None,
        "aggregate": str(m.group(3)),
        "canonical_column": f"{m.group(1)}_{m.group(3)}",
    }


def is_monthly_terraclimate_column(name):
    return parse_terraclimate_monthly_column(name) is not None


def is_year_specific_terraclimate_column(name):
    info = parse_terraclimate_monthly_column(name)
    return bool(info and info["year"] is not None)


def get_monthly_terraclimate_map(columns):
    out = {}
    for col in columns:
        info = parse_terraclimate_monthly_column(col)
        if info is None or info["year"] is not None:
            continue
        out.setdefault(info["variable"], {})[info["month"]] = col
    return {
        var: [month_map.get(m) for m in MONTH_NUMBERS]
        for var, month_map in sorted(out.items())
    }


def build_terraclimate_working_df(df):
    work = df.copy()
    rows = []
    grouped = {}
    drop_cols = set()

    for col in df.columns:
        monthly = parse_terraclimate_monthly_column(col)
        if monthly is not None and monthly["year"] is not None:
            key = (monthly["canonical_column"], monthly["variable"], "monthly", monthly["month_num"], None)
            grouped.setdefault(key, []).append((int(monthly["year"]), str(col)))
            drop_cols.add(str(col))
            continue

        aggregate = parse_terraclimate_aggregate_column(col)
        if aggregate is not None and aggregate["year"] is not None:
            key = (aggregate["canonical_column"], aggregate["variable"], "aggregate", None, aggregate["aggregate"])
            grouped.setdefault(key, []).append((int(aggregate["year"]), str(col)))
            drop_cols.add(str(col))

    for (canonical_col, variable, feature_kind, month_num, aggregate_name), entries in sorted(grouped.items()):
        if not entries:
            continue
        entries = sorted(entries, key=lambda x: (x[0], x[1]))
        source_cols = [col for _, col in entries]
        years = [int(year) for year, _ in entries]
        source_frame = work[source_cols].apply(pd.to_numeric, errors="coerce")
        work[canonical_col] = source_frame.mean(axis=1, skipna=True)
        rows.append(
            {
                "canonical_column": canonical_col,
                "variable": variable,
                "feature_kind": feature_kind,
                "aggregate": aggregate_name,
                "month_num": month_num,
                "month": MONTH_NAMES[month_num - 1] if month_num is not None else pd.NA,
                "n_years": len(years),
                "years": ",".join(str(y) for y in years),
                "source_columns": ",".join(source_cols),
            }
        )

    if drop_cols:
        work = work.drop(columns=sorted(drop_cols), errors="ignore")

    meta = pd.DataFrame(rows)
    if not meta.empty:
        sort_cols = ["variable", "feature_kind", "aggregate", "month_num", "canonical_column"]
        meta = meta.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return work, meta


def get_numeric_family(name):
    for prefix in DEFAULT_NUMERIC_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_")
    if name.startswith("mcd12q1_"):
        return "mcd12q1"
    return "other"


def get_categorical_family(name):
    if name.startswith("glim_"):
        return "glim"
    if name.startswith("mcd12q1_") or name == "mcd12q1":
        return "mcd12q1"
    if name.startswith("dem_"):
        return "dem"
    return "other"


def pretty_label(name):
    return name.replace("terraclimate_", "").replace("soilgrids_", "soil ").replace("_", " ")


def clip_label(text, max_len):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max(1, max_len - 1)] + "…"


def is_banned_numeric_column(name):
    return matches_any_pattern(str(name), DEFAULT_SKIP_NUMERIC_PATTERNS)


def is_forced_categorical_column(name):
    name = str(name)
    return (not is_banned_numeric_column(name)) and matches_any_pattern(name, DEFAULT_CATEGORICAL_PATTERNS)


def should_use_numeric_column(name):
    name = str(name)
    return (not is_banned_numeric_column(name)) and (not is_forced_categorical_column(name))


def filter_rows_by_feature_column(df, column_field="column"):
    if df is None or df.empty or column_field not in df.columns:
        return df
    mask = df[column_field].map(should_use_numeric_column)
    mask = mask.fillna(False)
    return df.loc[mask].copy()


def detect_numeric_columns(df):
    numeric_cols = []
    for col in df.columns:
        if col in DEFAULT_ID_COLUMNS:
            continue
        if not should_use_numeric_column(col):
            continue
        if is_monthly_terraclimate_column(col):
            numeric_cols.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(col.startswith(prefix) for prefix in DEFAULT_NUMERIC_PREFIXES):
            numeric_cols.append(col)
            continue
    return sorted(set(numeric_cols))


def detect_categorical_columns(df):
    categorical_cols = []
    for col in df.columns:
        if col in DEFAULT_ID_COLUMNS:
            continue
        if is_forced_categorical_column(col):
            categorical_cols.append(col)
            continue
        if any(col.startswith(prefix) for prefix in DEFAULT_CATEGORICAL_PREFIXES):
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                categorical_cols.append(col)
    return sorted(set(categorical_cols))


def summarize_numeric(df_work, numeric_cols):
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df_work[col], errors="coerce")
        valid = s.dropna()
        n_total = len(s)
        n_valid = len(valid)
        if n_valid == 0:
            rows.append(
                {
                    "column": col,
                    "family": get_numeric_family(col),
                    "n_total": n_total,
                    "n_valid": 0,
                    "n_missing": n_total,
                    "missing_fraction": 1.0,
                    "min": np.nan,
                    "q25": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "std": np.nan,
                }
            )
            continue

        q = valid.quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "column": col,
                "family": get_numeric_family(col),
                "n_total": n_total,
                "n_valid": n_valid,
                "n_missing": n_total - n_valid,
                "missing_fraction": (n_total - n_valid) / n_total if n_total else np.nan,
                "min": valid.min(),
                "q25": q.loc[0.25],
                "mean": valid.mean(),
                "median": q.loc[0.5],
                "q75": q.loc[0.75],
                "max": valid.max(),
                "iqr": q.loc[0.75] - q.loc[0.25],
                "std": valid.std(ddof=1) if n_valid > 1 else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["family", "column"]).reset_index(drop=True)
    return out


def summarize_categorical(df, categorical_cols, top_n):
    rows = []
    for col in categorical_cols:
        s = df[col].astype("string").fillna(pd.NA)
        valid = s.dropna().str.strip()
        valid = valid[valid != ""]
        n_total = len(s)
        n_valid = len(valid)

        if n_valid == 0:
            continue

        vc = valid.value_counts(dropna=False)
        vc = vc.head(top_n)
        for value, count in vc.items():
            rows.append(
                {
                    "column": col,
                    "family": get_categorical_family(col),
                    "value": value,
                    "count": int(count),
                    "fraction_of_non_null": float(count) / float(n_valid),
                    "fraction_of_all_rows": float(count) / float(n_total) if n_total else np.nan,
                    "n_total": n_total,
                    "n_valid": n_valid,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["family", "column", "count", "value"], ascending=[True, True, False, True]).reset_index(drop=True)
    return out


def summarize_monthly_terraclimate(df, monthly_map, min_valid_months):
    rows = []
    usable = {}
    for var, cols in monthly_map.items():
        valid_month_count = sum((c is not None and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0) for c in cols)
        if valid_month_count < min_valid_months:
            continue
        usable[var] = cols
        for idx, col in enumerate(cols, start=1):
            if col is None:
                rows.append(
                    {
                        "variable": var,
                        "month_num": idx,
                        "month": MONTH_NAMES[idx - 1],
                        "column": None,
                        "n_valid": 0,
                        "min": np.nan,
                        "q25": np.nan,
                        "mean": np.nan,
                        "median": np.nan,
                        "q75": np.nan,
                        "max": np.nan,
                        "iqr": np.nan,
                        "std": np.nan,
                    }
                )
                continue

            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                rows.append(
                    {
                        "variable": var,
                        "month_num": idx,
                        "month": MONTH_NAMES[idx - 1],
                        "column": col,
                        "n_valid": 0,
                        "min": np.nan,
                        "q25": np.nan,
                        "mean": np.nan,
                        "median": np.nan,
                        "q75": np.nan,
                        "max": np.nan,
                        "iqr": np.nan,
                        "std": np.nan,
                    }
                )
                continue

            q = s.quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "variable": var,
                    "month_num": idx,
                    "month": MONTH_NAMES[idx - 1],
                    "column": col,
                    "n_valid": int(s.shape[0]),
                    "min": float(s.min()),
                    "q25": float(q.loc[0.25]),
                    "mean": float(s.mean()),
                    "median": float(q.loc[0.5]),
                    "q75": float(q.loc[0.75]),
                    "max": float(s.max()),
                    "iqr": float(q.loc[0.75] - q.loc[0.25]),
                    "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["variable", "month_num"]).reset_index(drop=True)
    return out, usable


def summarize_numeric_by_group(df, group_by, numeric_cols):
    frames = []
    for group_value, df_group in df.groupby(group_by, dropna=False, sort=True):
        label = group_value_to_label(group_value)
        sub = summarize_numeric(df_group, numeric_cols)
        if sub.empty:
            continue
        sub.insert(0, "group", label)
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["family", "column", "group"]).reset_index(drop=True)


def summarize_categorical_by_group(df, group_by, categorical_cols, top_n):
    frames = []
    for group_value, df_group in df.groupby(group_by, dropna=False, sort=True):
        label = group_value_to_label(group_value)
        sub = summarize_categorical(df_group, categorical_cols, top_n=top_n)
        if sub.empty:
            continue
        sub.insert(0, "group", label)
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["family", "column", "group", "count", "value"], ascending=[True, True, True, False, True]).reset_index(drop=True)


def summarize_monthly_terraclimate_by_group(df, group_by, monthly_map, min_valid_months):
    frames = []
    usable_vars = set()
    for group_value, df_group in df.groupby(group_by, dropna=False, sort=True):
        label = group_value_to_label(group_value)
        sub, usable = summarize_monthly_terraclimate(df_group, monthly_map, min_valid_months=min_valid_months)
        if sub.empty:
            continue
        usable_vars.update(usable.keys())
        sub.insert(0, "group", label)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(), {}
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["variable", "group", "month_num"]).reset_index(drop=True)
    return out, {k: monthly_map[k] for k in sorted(usable_vars)}


def save_monthly_plots(monthly_summary, charts_dir, fig_dpi):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if monthly_summary.empty:
        return

    for var in sorted(monthly_summary["variable"].dropna().unique()):
        sub = monthly_summary[monthly_summary["variable"] == var].sort_values("month_num")
        if sub.empty:
            continue

        x = sub["month_num"].to_numpy()
        mean = sub["mean"].to_numpy(dtype=float)
        median = sub["median"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        vmin = sub["min"].to_numpy(dtype=float)
        vmax = sub["max"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.fill_between(x, q25, q75, alpha=0.25, label="IQR (Q25-Q75)")
        ax.plot(x, mean, linewidth=2.2, label="Mean")
        ax.plot(x, median, linewidth=1.8, linestyle="--", label="Median")
        ax.plot(x, vmin, linewidth=1.0, alpha=0.8, linestyle=":", label="Min")
        ax.plot(x, vmax, linewidth=1.0, alpha=0.8, linestyle=":", label="Max")

        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_NAMES)
        ax.set_xlim(1, 12)
        ax.set_title(f"{var} monthly distribution across observations")
        ax.set_xlabel("Month")
        ax.set_ylabel(var.replace("terraclimate_", ""))
        ax.grid(True, alpha=0.25)
        ax.legend()

        out_path = charts_dir / f"{safe_slug(var)}_monthly.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_group_monthly_compare_plots(monthly_summary_by_group, charts_dir, fig_dpi, max_groups):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if monthly_summary_by_group.empty:
        return

    group_sizes = (
        monthly_summary_by_group[["group", "n_valid"]]
        .groupby("group", as_index=False)["n_valid"]
        .sum()
        .sort_values(["n_valid", "group"], ascending=[False, True])
    )
    keep_groups = group_sizes["group"].head(max_groups).tolist()
    filtered = monthly_summary_by_group[monthly_summary_by_group["group"].isin(keep_groups)].copy()

    for var in sorted(filtered["variable"].dropna().unique()):
        sub = filtered[filtered["variable"] == var].copy()
        if sub.empty:
            continue

        order = (
            sub.groupby("group", as_index=False)["mean"]
            .mean()
            .sort_values(["mean", "group"], ascending=[False, True])["group"]
            .tolist()
        )

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for group in order:
            gsub = sub[sub["group"] == group].sort_values("month_num")
            if gsub.empty:
                continue
            x = gsub["month_num"].to_numpy(dtype=float)
            mean = gsub["mean"].to_numpy(dtype=float)
            ax.plot(x, mean, linewidth=1.8, label=group)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES)
        ax.set_xlim(1, 12)
        ax.set_title(f"{var} monthly mean by group")
        ax.set_xlabel("Month")
        ax.set_ylabel(var.replace("terraclimate_", ""))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        out_path = charts_dir / f"{safe_slug(var)}_monthly_group_compare.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_numeric_distribution_plots(df, numeric_summary, charts_dir, fig_dpi):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if numeric_summary.empty:
        return

    for row in numeric_summary.itertuples(index=False):
        col = row.column
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        values = s.to_numpy(dtype=float)
        n = values.shape[0]
        if n == 1:
            bins = 1
        else:
            bins = max(10, min(80, int(math.sqrt(n))))

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.hist(values, bins=bins, alpha=0.8)
        ax.axvspan(row.q25, row.q75, alpha=0.18, label="IQR")
        ax.axvline(row.mean, linewidth=2.0, label="Mean")
        ax.axvline(row.median, linewidth=1.8, linestyle="--", label="Median")
        ax.axvline(row.min, linewidth=1.0, linestyle=":", alpha=0.9, label="Min")
        ax.axvline(row.max, linewidth=1.0, linestyle=":", alpha=0.9, label="Max")
        ax.set_title(f"{col} distribution across observations")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right")

        stats_text = "\n".join(
            [
                f"n={int(row.n_valid)}",
                f"min={row.min:.6g}",
                f"q25={row.q25:.6g}",
                f"mean={row.mean:.6g}",
                f"median={row.median:.6g}",
                f"q75={row.q75:.6g}",
                f"max={row.max:.6g}",
                f"iqr={row.iqr:.6g}",
            ]
        )
        ax.text(
            0.995,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "pad": 0.35, "alpha": 0.12},
        )

        out_path = charts_dir / f"{safe_slug(col)}_distribution.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_group_numeric_compare_plots(df, group_by, numeric_cols, charts_dir, fig_dpi, max_groups):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if not numeric_cols:
        return

    group_counts = (
        df.groupby(group_by, dropna=False, sort=True)
        .size()
        .reset_index(name="group_rows")
        .assign(group_label=lambda d: d[group_by].map(group_value_to_label))
        .sort_values(["group_rows", "group_label"], ascending=[False, True])
    )
    keep_groups = group_counts["group_label"].head(max_groups).tolist()
    if not keep_groups:
        return

    group_row_map = dict(zip(group_counts["group_label"], group_counts["group_rows"]))
    df_work = df.copy()
    df_work["__group_label__"] = df_work[group_by].map(group_value_to_label)
    df_work = df_work[df_work["__group_label__"].isin(keep_groups)]

    for col in numeric_cols:
        s = pd.to_numeric(df_work[col], errors="coerce")
        temp = df_work[["__group_label__"]].copy()
        temp["__value__"] = s
        temp = temp.dropna(subset=["__value__"])
        if temp.empty:
            continue

        grouped = []
        for group, gsub in temp.groupby("__group_label__", sort=False):
            vals = gsub["__value__"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            grouped.append(
                {
                    "group": group,
                    "values": vals,
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "n_valid": int(vals.size),
                    "group_rows": int(group_row_map.get(group, vals.size)),
                }
            )

        if len(grouped) < 2:
            continue

        grouped = sorted(grouped, key=lambda d: (-d["mean"], d["group"]))
        labels = [f"{g['group']} (n={g['n_valid']})" for g in grouped]
        values = [g["values"] for g in grouped]
        positions = np.arange(1, len(values) + 1)

        fig_h = max(4.0, 0.42 * len(values) + 1.8)
        fig, ax = plt.subplots(figsize=(11.5, fig_h))
        ax.boxplot(values, vert=False, tick_labels=labels, showfliers=False, patch_artist=False)
        means = [g["mean"] for g in grouped]
        medians = [g["median"] for g in grouped]
        ax.scatter(means, positions, s=26, label="Mean")
        ax.scatter(medians, positions, s=22, marker="s", label="Median")
        ax.set_title(f"{col} by {group_by}")
        ax.set_xlabel(col)
        ax.set_ylabel(group_by)
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="best")

        out_path = charts_dir / f"{safe_slug(col)}_group_compare_boxplot.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_numeric_family_overview_plots(numeric_summary, charts_dir, fig_dpi):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if numeric_summary.empty:
        return

    for family, sub in numeric_summary.groupby("family", sort=True):
        sub = sub.dropna(subset=["min", "q25", "mean", "median", "q75", "max"]).copy()
        if sub.empty:
            continue

        sub = sub.sort_values("column").reset_index(drop=True)
        y = np.arange(len(sub), dtype=float)

        mins = sub["min"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        means = sub["mean"].to_numpy(dtype=float)
        medians = sub["median"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        maxs = sub["max"].to_numpy(dtype=float)

        span = maxs - mins
        norm_q25 = np.where(span > 0, (q25 - mins) / span, 0.5)
        norm_mean = np.where(span > 0, (means - mins) / span, 0.5)
        norm_median = np.where(span > 0, (medians - mins) / span, 0.5)
        norm_q75 = np.where(span > 0, (q75 - mins) / span, 0.5)

        fig_h = max(4.0, 0.42 * len(sub) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.hlines(y, 0, 1, linewidth=1.0, alpha=0.35)
        ax.hlines(y, norm_q25, norm_q75, linewidth=5.0, alpha=0.85)
        ax.scatter(norm_mean, y, s=24, label="Mean")
        ax.scatter(norm_median, y, s=24, marker="s", label="Median")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.75, len(sub) - 0.25)
        ax.set_xlabel("Relative position within each variable range (min to max)")
        ax.set_title(f"{family} numeric overview")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["column"].tolist())
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="lower right")

        out_path = charts_dir / f"{safe_slug(family)}_numeric_overview_scaled.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_group_numeric_family_heatmaps(numeric_summary_by_group, charts_dir, fig_dpi, max_groups):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if numeric_summary_by_group.empty:
        return

    group_order = (
        numeric_summary_by_group[["group", "n_valid"]]
        .groupby("group", as_index=False)["n_valid"]
        .sum()
        .sort_values(["n_valid", "group"], ascending=[False, True])["group"]
        .head(max_groups)
        .tolist()
    )
    filtered = numeric_summary_by_group[numeric_summary_by_group["group"].isin(group_order)].copy()

    for family, sub in filtered.groupby("family", sort=True):
        sub = sub[sub["column"].map(lambda x: not is_monthly_terraclimate_column(str(x)))].copy()
        if sub.empty:
            continue

        pivot = sub.pivot_table(index="column", columns="group", values="mean", aggfunc="first")
        pivot = pivot.reindex(columns=group_order)
        pivot = pivot.dropna(how="all")
        if pivot.empty or pivot.shape[0] < 1 or pivot.shape[1] < 2:
            continue

        arr = pivot.to_numpy(dtype=float)
        scaled = np.full_like(arr, np.nan, dtype=float)
        for i in range(arr.shape[0]):
            row = arr[i, :]
            finite = np.isfinite(row)
            if not finite.any():
                continue
            rmin = np.nanmin(row)
            rmax = np.nanmax(row)
            if rmax > rmin:
                scaled[i, finite] = (row[finite] - rmin) / (rmax - rmin)
            else:
                scaled[i, finite] = 0.5

        fig_w = max(8.5, 0.55 * pivot.shape[1] + 3.5)
        fig_h = max(4.5, 0.34 * pivot.shape[0] + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(scaled, aspect="auto")
        ax.set_title(f"{family} mean by group (scaled within variable)")
        ax.set_xlabel("Group")
        ax.set_ylabel("Variable")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.tolist())
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        out_path = charts_dir / f"{safe_slug(family)}_group_mean_heatmap_scaled.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_categorical_frequency_plots(categorical_summary, charts_dir, fig_dpi, max_label_len):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if categorical_summary.empty:
        return

    for col, sub in categorical_summary.groupby("column", sort=True):
        sub = sub.sort_values(["count", "value"], ascending=[True, True]).reset_index(drop=True)
        labels = [clip_label(v, max_label_len) for v in sub["value"].astype(str).tolist()]
        counts = sub["count"].to_numpy(dtype=float)
        fractions = sub["fraction_of_non_null"].to_numpy(dtype=float)
        y = np.arange(len(sub), dtype=float)

        fig_h = max(3.5, 0.42 * len(sub) + 1.6)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        bars = ax.barh(y, counts)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Count")
        ax.set_title(f"{col} categorical frequency")
        ax.grid(True, axis="x", alpha=0.25)

        max_count = counts.max() if len(counts) else 0.0
        text_offset = max_count * 0.01 if max_count > 0 else 0.1
        for bar, count, frac in zip(bars, counts, fractions):
            ax.text(
                count + text_offset,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(count)} ({frac:.1%})",
                va="center",
                ha="left",
                fontsize=9,
            )

        out_path = charts_dir / f"{safe_slug(col)}_frequency.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_group_categorical_compare_plots(categorical_summary_by_group, charts_dir, fig_dpi, max_label_len, max_groups, max_categories):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if categorical_summary_by_group.empty:
        return

    group_order = (
        categorical_summary_by_group[["group", "n_total"]]
        .groupby("group", as_index=False)["n_total"]
        .max()
        .sort_values(["n_total", "group"], ascending=[False, True])["group"]
        .head(max_groups)
        .tolist()
    )
    filtered = categorical_summary_by_group[categorical_summary_by_group["group"].isin(group_order)].copy()

    for col, sub in filtered.groupby("column", sort=True):
        top_values = (
            sub.groupby("value", as_index=False)["count"]
            .sum()
            .sort_values(["count", "value"], ascending=[False, True])["value"]
            .head(max_categories)
            .tolist()
        )
        csub = sub[sub["value"].isin(top_values)].copy()
        if csub.empty:
            continue

        pivot = csub.pivot_table(index="group", columns="value", values="fraction_of_non_null", aggfunc="first", fill_value=0.0)
        pivot = pivot.reindex(index=group_order).fillna(0.0)
        pivot = pivot[[c for c in top_values if c in pivot.columns]]
        if pivot.empty or pivot.shape[1] == 0:
            continue

        labels = [clip_label(v, max_label_len) for v in pivot.index.tolist()]
        fig_h = max(4.0, 0.5 * len(labels) + 1.8)
        fig, ax = plt.subplots(figsize=(11.5, fig_h))
        pivot.plot(kind="barh", stacked=True, ax=ax, width=0.85)
        ax.set_title(f"{col} categorical mix by group")
        ax.set_xlabel("Fraction of non-null rows")
        ax.set_ylabel("Group")
        ax.set_yticklabels(labels)
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="best", fontsize=8, title="Value")

        out_path = charts_dir / f"{safe_slug(col)}_group_compare_stacked.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)




def parse_csv_arg_list(raw):
    return [part.strip().lower() for part in str(raw or "").split(",") if part.strip()]


def maybe_read_csv(file_path):
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        log(f"Could not reuse existing {path}: {exc}")
        return None


def maybe_read_matrix_csv(file_path):
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        log(f"Could not reuse existing matrix {path}: {exc}")
        return None
    if df is None or df.empty:
        return df
    df.index = df.index.map(str)
    df.columns = [str(c) for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pair_table_from_matrix(matrix_df, value_name="value"):
    if matrix_df is None or matrix_df.empty:
        return pd.DataFrame(columns=["group_a", "group_b", value_name, f"abs_{value_name}", "pair"])
    labels = list(matrix_df.index)
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = str(labels[i])
            b = str(labels[j])
            if b not in matrix_df.columns:
                continue
            value = pd.to_numeric(pd.Series([matrix_df.loc[a, b]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            pair = f"{a} | {b}"
            rows.append(
                {
                    "group_a": a,
                    "group_b": b,
                    value_name: float(value),
                    f"abs_{value_name}": float(abs(value)),
                    "pair": pair,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values([value_name, "pair"], ascending=[False, True]).reset_index(drop=True)
    return out


def save_pair_barplot(pair_df, value_col, out_path, title, fig_dpi, top_n=15):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pair_df is None or pair_df.empty or value_col not in pair_df.columns:
        return

    plot_df = pair_df.sort_values([value_col, "pair"], ascending=[False, True]).head(top_n).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values([value_col, "pair"], ascending=[True, True]).reset_index(drop=True)

    y = np.arange(len(plot_df), dtype=float)
    vals = plot_df[value_col].to_numpy(dtype=float)
    labels = [clip_label(v, 96) for v in plot_df["pair"].astype(str).tolist()]

    fig_h = max(4.0, 0.42 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    bars = ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)

    xpad = (float(np.nanmax(np.abs(vals))) * 0.02) if len(vals) else 0.01
    xpad = max(xpad, 0.01)
    for bar, value in zip(bars, vals):
        ax.text(
            float(bar.get_width()) + (xpad if value >= 0 else -xpad),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)


def build_family_pair_support_table(by_family_dir, stat, method):
    by_family_dir = Path(by_family_dir)
    pattern = f"*_species_profile_{safe_slug(stat)}_{safe_slug(method)}.csv"
    rows = []
    for csv_path in sorted(by_family_dir.glob(pattern)):
        stem = csv_path.stem
        suffix = f"_species_profile_{safe_slug(stat)}_{safe_slug(method)}"
        family = stem[: -len(suffix)] if stem.endswith(suffix) else stem
        matrix_df = maybe_read_matrix_csv(csv_path)
        if matrix_df is None or matrix_df.empty:
            continue
        pair_df = pair_table_from_matrix(matrix_df, value_name="correlation")
        if pair_df.empty:
            continue
        pair_df["family"] = family
        rows.append(pair_df[["pair", "family", "correlation"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out


def save_pair_family_support_heatmap(pair_family_df, out_path, title, fig_dpi, pair_order=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pair_family_df is None or pair_family_df.empty:
        return

    pivot = pair_family_df.pivot_table(index="pair", columns="family", values="correlation", aggfunc="first")
    if pair_order:
        ordered = [pair for pair in pair_order if pair in pivot.index]
        remainder = [pair for pair in pivot.index.tolist() if pair not in ordered]
        pivot = pivot.reindex(ordered + remainder)
    pivot = pivot.sort_index(axis=1)
    pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if pivot.empty:
        return

    arr = pivot.to_numpy(dtype=float)
    fig_w = max(7.5, 0.7 * pivot.shape[1] + 3.0)
    fig_h = max(4.5, 0.42 * pivot.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Family")
    ax.set_ylabel("Species pair")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([clip_label(v, 96) for v in pivot.index.tolist()])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)


def save_feature_weight_family_plots(feature_weights, charts_dir, fig_dpi, max_label_len, top_n=12):
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    if feature_weights is None or feature_weights.empty:
        return

    for family, sub in feature_weights.groupby("family", sort=True):
        sub = sub.sort_values(["blended_weight_global", "column"], ascending=[False, True]).head(top_n).copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["blended_weight_global", "column"], ascending=[True, True]).reset_index(drop=True)

        y = np.arange(len(sub), dtype=float)
        vals = sub["blended_weight_global"].to_numpy(dtype=float)
        labels = [clip_label(v, max_label_len) for v in sub["column"].astype(str).tolist()]

        fig_h = max(3.8, 0.38 * len(sub) + 1.6)
        fig, ax = plt.subplots(figsize=(10.5, fig_h))
        bars = ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Blended empirical weight")
        ax.set_title(f"{family} top weighted features")
        ax.grid(True, axis="x", alpha=0.25)

        xpad = (float(np.nanmax(vals)) * 0.015) if len(vals) and float(np.nanmax(vals)) > 0 else 0.001
        for bar, eta2, sig in zip(bars, sub["eta2"], sub["signal_to_iqr"]):
            parts = [f"eta2={eta2:.3f}"]
            if np.isfinite(sig):
                parts.append(f"sig/IQR={sig:.3f}")
            ax.text(
                float(bar.get_width()) + xpad,
                bar.get_y() + bar.get_height() / 2.0,
                "  ".join(parts),
                va="center",
                ha="left",
                fontsize=8,
            )

        out_path = charts_dir / f"{safe_slug(family)}_top_feature_weights.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def remove_obsolete_correlation_outputs(outdir):
    outdir = Path(outdir)
    stale_dir = outdir / "correlations"
    if stale_dir.exists():
        import shutil
        shutil.rmtree(stale_dir, ignore_errors=True)
        log(f"Removed obsolete legacy correlation outputs at {stale_dir}")

    insights_dir = outdir / "insights"
    if insights_dir.exists():
        for stale_csv in insights_dir.glob("*pearson*.csv"):
            stale_csv.unlink(missing_ok=True)
        for stale_csv in insights_dir.glob("*spearman*.csv"):
            stale_csv.unlink(missing_ok=True)

    insights_charts_dir = outdir / "charts" / "insights"
    if insights_charts_dir.exists():
        for stale_png in insights_charts_dir.glob("*pearson*.png"):
            stale_png.unlink(missing_ok=True)
        for stale_png in insights_charts_dir.glob("*spearman*.png"):
            stale_png.unlink(missing_ok=True)


def generate_correlation_and_weight_insight_visuals(outdir, fig_dpi, max_label_len):
    outdir = Path(outdir)
    remove_obsolete_correlation_outputs(outdir)
    correlations_dir = outdir / "correlations"
    by_family_dir = correlations_dir / "by_family"
    insights_dir = outdir / "insights"
    insights_charts_dir = outdir / "charts" / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    insights_charts_dir.mkdir(parents=True, exist_ok=True)

    for corr_csv in sorted(correlations_dir.glob("species_profile_*_*.csv")):
        name = corr_csv.stem
        if name.endswith("_shared_feature_counts"):
            continue
        m = re.match(r"^species_profile_(.+)_(pearson|spearman)$", name)
        if not m:
            continue
        stat = m.group(1)
        method = m.group(2)

        matrix_df = maybe_read_matrix_csv(corr_csv)
        if matrix_df is None or matrix_df.empty:
            continue

        pair_df = pair_table_from_matrix(matrix_df, value_name="correlation")
        if pair_df.empty:
            continue

        pair_df.to_csv(insights_dir / f"species_profile_{safe_slug(stat)}_{safe_slug(method)}_pairwise.csv", index=False)

        top_positive = pair_df[pair_df["correlation"].notna()].sort_values(
            ["correlation", "pair"], ascending=[False, True]
        ).head(15).copy()
        if not top_positive.empty:
            top_positive.to_csv(
                insights_dir / f"species_profile_{safe_slug(stat)}_{safe_slug(method)}_top_positive_pairs.csv",
                index=False,
            )
            save_pair_barplot(
                top_positive,
                value_col="correlation",
                out_path=insights_charts_dir / f"species_profile_{safe_slug(stat)}_{safe_slug(method)}_top_positive_pairs.png",
                title=f"Top positive species-profile pairs ({method}, {stat})",
                fig_dpi=fig_dpi,
                top_n=15,
            )

        top_absolute = pair_df.sort_values(
            ["abs_correlation", "pair"], ascending=[False, True]
        ).head(15).copy()
        if not top_absolute.empty:
            top_absolute.to_csv(
                insights_dir / f"species_profile_{safe_slug(stat)}_{safe_slug(method)}_top_absolute_pairs.csv",
                index=False,
            )
            save_pair_barplot(
                top_absolute,
                value_col="abs_correlation",
                out_path=insights_charts_dir / f"species_profile_{safe_slug(stat)}_{safe_slug(method)}_top_absolute_pairs.png",
                title=f"Top absolute species-profile pairs ({method}, {stat})",
                fig_dpi=fig_dpi,
                top_n=15,
            )

        family_pair_df = build_family_pair_support_table(by_family_dir, stat, method)
        if family_pair_df is not None and not family_pair_df.empty:
            family_pair_df.to_csv(
                insights_dir / f"species_profile_{safe_slug(stat)}_{safe_slug(method)}_pair_family_support.csv",
                index=False,
            )
            pair_order = top_positive["pair"].tolist() if not top_positive.empty else pair_df["pair"].head(15).tolist()
            save_pair_family_support_heatmap(
                family_pair_df[family_pair_df["pair"].isin(pair_order)].copy(),
                out_path=insights_charts_dir / f"species_profile_{safe_slug(stat)}_{safe_slug(method)}_top_pair_family_support.png",
                title=f"Family support for top species-profile pairs ({method}, {stat})",
                fig_dpi=fig_dpi,
                pair_order=pair_order,
            )

    feature_weights = maybe_read_csv(outdir / "feature_signal_weights.csv")
    if feature_weights is not None and not feature_weights.empty:
        feature_weights = normalize_summary_frame(feature_weights)
    family_weights = maybe_read_csv(outdir / "family_signal_weights.csv")
    if family_weights is not None and not family_weights.empty:
        family_weights = normalize_summary_frame(family_weights)

    save_feature_weight_plots(
        feature_weights=feature_weights,
        family_weights=family_weights,
        charts_dir=outdir / "charts" / "feature_weights",
        fig_dpi=fig_dpi,
        max_label_len=max_label_len,
    )
    save_feature_weight_family_plots(
        feature_weights=feature_weights,
        charts_dir=outdir / "charts" / "feature_weights" / "by_family",
        fig_dpi=fig_dpi,
        max_label_len=max_label_len,
    )


def normalize_summary_frame(df, numeric_cols=None):
    if df is None or df.empty:
        return df
    out = df.copy()
    numeric_candidates = set(
        [
            "n_total",
            "n_valid",
            "n_missing",
            "missing_fraction",
            "min",
            "q25",
            "mean",
            "median",
            "q75",
            "max",
            "iqr",
            "std",
            "count",
            "fraction_of_non_null",
            "fraction_of_all_rows",
            "month_num",
        ]
    )
    if numeric_cols:
        numeric_candidates.update(numeric_cols)
    for col in out.columns:
        if col in numeric_candidates:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_group_profile_matrix(numeric_summary_by_group, stat="mean", family=None):
    if numeric_summary_by_group is None or numeric_summary_by_group.empty:
        return pd.DataFrame()

    if stat not in numeric_summary_by_group.columns:
        raise ValueError(f"Profile stat not found in by-group numeric summary: {stat}")

    sub = numeric_summary_by_group.copy()
    if family:
        sub = sub[sub["family"] == family].copy()
    sub = sub.dropna(subset=["group", "column"])
    if sub.empty:
        return pd.DataFrame()

    pivot = sub.pivot_table(index="group", columns="column", values=stat, aggfunc="first")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return pivot


def parse_family_reliability_priors(raw):
    priors = dict(DEFAULT_FAMILY_RELIABILITY_PRIORS)
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        try:
            priors[key] = float(value)
        except ValueError:
            continue
    return priors


def normalize_nonnegative_series(series):
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.clip(lower=0)
    total = float(x.sum())
    if total <= 0:
        return pd.Series(np.zeros(len(x), dtype=float), index=series.index)
    return x / total


def interval_overlap_fraction(lo_a, hi_a, lo_b, hi_b):
    vals = [lo_a, hi_a, lo_b, hi_b]
    if not all(np.isfinite(v) for v in vals):
        return np.nan
    if hi_a < lo_a or hi_b < lo_b:
        return np.nan
    union_lo = min(lo_a, lo_b)
    union_hi = max(hi_a, hi_b)
    union_span = union_hi - union_lo
    if union_span <= 0:
        return 1.0
    overlap = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    return float(overlap / union_span)


def point_proximity(a, b, span):
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    if not np.isfinite(span) or span <= 0:
        return 1.0 if a == b else 0.0
    return float(max(0.0, 1.0 - (abs(a - b) / span)))


def build_numeric_feature_overlap_table(numeric_summary_by_group):
    if numeric_summary_by_group is None or numeric_summary_by_group.empty:
        return pd.DataFrame()

    rows = []
    cols = ["group", "column", "family", "min", "q25", "median", "q75", "max", "n_valid"]
    sub_all = filter_rows_by_feature_column(numeric_summary_by_group.copy(), column_field="column")
    sub_all = sub_all.dropna(subset=["group", "column"])

    for col, sub in sub_all.groupby("column", sort=True):
        sub = sub.copy()
        family = str(sub["family"].dropna().iloc[0]) if sub["family"].notna().any() else "other"
        sub = sub.dropna(subset=["min", "q25", "median", "q75", "max"])
        if sub.shape[0] < 2:
            continue
        recs = sub[cols].to_dict("records")
        for i in range(len(recs)):
            a = recs[i]
            for j in range(i + 1, len(recs)):
                b = recs[j]
                support_lo = min(float(a["min"]), float(b["min"]))
                support_hi = max(float(a["max"]), float(b["max"]))
                support_span = support_hi - support_lo
                iqr_overlap = interval_overlap_fraction(float(a["q25"]), float(a["q75"]), float(b["q25"]), float(b["q75"]))
                range_overlap = interval_overlap_fraction(float(a["min"]), float(a["max"]), float(b["min"]), float(b["max"]))
                center_overlap = point_proximity(float(a["median"]), float(b["median"]), support_span)
                parts = [v for v in [iqr_overlap, range_overlap, center_overlap] if np.isfinite(v)]
                if not parts:
                    continue
                feature_overlap = np.average(
                    [v for v in [iqr_overlap, range_overlap, center_overlap] if np.isfinite(v)],
                    weights=[w for v, w in zip([iqr_overlap, range_overlap, center_overlap], [0.5, 0.3, 0.2]) if np.isfinite(v)],
                )
                rows.append(
                    {
                        "group_a": str(a["group"]),
                        "group_b": str(b["group"]),
                        "column": str(col),
                        "family": family,
                        "n_valid_min_pair": int(min(float(pd.to_numeric(a.get("n_valid"), errors="coerce") if a.get("n_valid") is not None else 0.0), float(pd.to_numeric(b.get("n_valid"), errors="coerce") if b.get("n_valid") is not None else 0.0))),
                        "iqr_overlap": float(iqr_overlap) if np.isfinite(iqr_overlap) else np.nan,
                        "range_overlap": float(range_overlap) if np.isfinite(range_overlap) else np.nan,
                        "center_proximity": float(center_overlap) if np.isfinite(center_overlap) else np.nan,
                        "feature_overlap": float(feature_overlap),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["family", "column", "group_a", "group_b"]).reset_index(drop=True)
    return out


def build_monthly_variable_overlap_table(monthly_summary_by_group):
    if monthly_summary_by_group is None or monthly_summary_by_group.empty:
        return pd.DataFrame()

    rows = []
    sub_all = monthly_summary_by_group.dropna(subset=["group", "variable", "month_num"]).copy()
    sub_all = sub_all.dropna(subset=["min", "q25", "median", "q75", "max"])
    if sub_all.empty:
        return pd.DataFrame()

    for (variable, month_num), sub in sub_all.groupby(["variable", "month_num"], sort=True):
        recs = sub[["group", "min", "q25", "median", "q75", "max", "n_valid"]].to_dict("records")
        if len(recs) < 2:
            continue
        for i in range(len(recs)):
            a = recs[i]
            for j in range(i + 1, len(recs)):
                b = recs[j]
                support_lo = min(float(a["min"]), float(b["min"]))
                support_hi = max(float(a["max"]), float(b["max"]))
                support_span = support_hi - support_lo
                iqr_overlap = interval_overlap_fraction(float(a["q25"]), float(a["q75"]), float(b["q25"]), float(b["q75"]))
                range_overlap = interval_overlap_fraction(float(a["min"]), float(a["max"]), float(b["min"]), float(b["max"]))
                center_overlap = point_proximity(float(a["median"]), float(b["median"]), support_span)
                feature_overlap = np.average(
                    [v for v in [iqr_overlap, range_overlap, center_overlap] if np.isfinite(v)],
                    weights=[w for v, w in zip([iqr_overlap, range_overlap, center_overlap], [0.5, 0.3, 0.2]) if np.isfinite(v)],
                )
                rows.append(
                    {
                        "group_a": str(a["group"]),
                        "group_b": str(b["group"]),
                        "variable": str(variable),
                        "month_num": int(month_num),
                        "feature_key": f"{variable}_m{int(month_num):02d}",
                        "family": "terraclimate",
                        "n_valid_min_pair": int(min(float(pd.to_numeric(a.get("n_valid"), errors="coerce") if a.get("n_valid") is not None else 0.0), float(pd.to_numeric(b.get("n_valid"), errors="coerce") if b.get("n_valid") is not None else 0.0))),
                        "iqr_overlap": float(iqr_overlap) if np.isfinite(iqr_overlap) else np.nan,
                        "range_overlap": float(range_overlap) if np.isfinite(range_overlap) else np.nan,
                        "center_proximity": float(center_overlap) if np.isfinite(center_overlap) else np.nan,
                        "feature_overlap": float(feature_overlap),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["variable", "month_num", "group_a", "group_b"]).reset_index(drop=True)
    return out


def build_categorical_overlap_table(categorical_summary_by_group):
    if categorical_summary_by_group is None or categorical_summary_by_group.empty:
        return pd.DataFrame()

    rows = []
    for col, sub in categorical_summary_by_group.groupby("column", sort=True):
        family = str(sub["family"].dropna().iloc[0]) if sub["family"].notna().any() else "other"
        group_maps = {}
        for group, gsub in sub.groupby("group", sort=True):
            freqs = {}
            total = 0.0
            for row in gsub.itertuples(index=False):
                frac = pd.to_numeric(pd.Series([row.fraction_of_non_null]), errors="coerce").iloc[0]
                if not np.isfinite(frac) or frac < 0:
                    continue
                freqs[str(row.value)] = float(frac)
                total += float(frac)
            remainder = max(0.0, 1.0 - total)
            if remainder > 1e-9:
                freqs["__other__"] = remainder
            group_maps[str(group)] = {
                "freqs": freqs,
                "n_valid": int(pd.to_numeric(gsub["n_valid"], errors="coerce").max()) if gsub["n_valid"].notna().any() else 0,
            }
        groups = sorted(group_maps.keys())
        if len(groups) < 2:
            continue
        for i in range(len(groups)):
            ga = groups[i]
            for j in range(i + 1, len(groups)):
                gb = groups[j]
                fa = group_maps[ga]["freqs"]
                fb = group_maps[gb]["freqs"]
                keys = set(fa) | set(fb)
                overlap = sum(min(float(fa.get(k, 0.0)), float(fb.get(k, 0.0))) for k in keys)
                rows.append(
                    {
                        "group_a": ga,
                        "group_b": gb,
                        "column": str(col),
                        "family": family,
                        "n_valid_min_pair": int(min(group_maps[ga]["n_valid"], group_maps[gb]["n_valid"])),
                        "categorical_overlap": float(max(0.0, min(1.0, overlap))),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["family", "column", "group_a", "group_b"]).reset_index(drop=True)
    return out


def build_pair_matrix_from_table(pair_df, value_col, weight_col=None, key_col="column", min_features=1, fill_diag=1.0):
    if pair_df is None or pair_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = sorted(set(pair_df["group_a"].astype(str)) | set(pair_df["group_b"].astype(str)))
    matrix = pd.DataFrame(np.nan, index=groups, columns=groups, dtype=float)
    counts = pd.DataFrame(0, index=groups, columns=groups, dtype=int)
    for g in groups:
        matrix.loc[g, g] = fill_diag
    grouped = pair_df.groupby(["group_a", "group_b"], sort=True)
    for (ga, gb), sub in grouped:
        vals = pd.to_numeric(sub[value_col], errors="coerce")
        valid = vals.notna()
        if not valid.any():
            continue
        sub_valid = sub.loc[valid].copy()
        vals = vals.loc[valid].to_numpy(dtype=float)
        if weight_col and weight_col in sub_valid.columns:
            weights = pd.to_numeric(sub_valid[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            if np.nansum(weights) > 0:
                score = float(np.average(vals, weights=weights))
            else:
                score = float(np.mean(vals))
        else:
            score = float(np.mean(vals))
        feature_count = int(sub_valid[key_col].astype(str).nunique()) if key_col in sub_valid.columns else int(valid.sum())
        if feature_count < max(1, int(min_features)):
            score = np.nan
        matrix.loc[str(ga), str(gb)] = score
        matrix.loc[str(gb), str(ga)] = score
        counts.loc[str(ga), str(gb)] = feature_count
        counts.loc[str(gb), str(ga)] = feature_count
    return matrix, counts


def save_matrix_heatmap(matrix_df, out_path, title, fig_dpi, vmin=None, vmax=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix_df is None or matrix_df.empty:
        return

    arr = matrix_df.to_numpy(dtype=float)
    fig_w = max(6.5, 0.6 * matrix_df.shape[1] + 2.0)
    fig_h = max(5.5, 0.45 * matrix_df.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel("Group")
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index.tolist())
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)


def compute_feature_signal_weights(
    df,
    group_by,
    numeric_cols,
    numeric_summary_by_group,
    min_valid_values,
    min_iqr_floor_frac=0.05,
    family_priors=None,
    numeric_feature_overlap=None,
):
    if not numeric_cols:
        return pd.DataFrame()

    family_priors = family_priors or dict(DEFAULT_FAMILY_RELIABILITY_PRIORS)
    group_series = df[group_by].map(group_value_to_label)
    rows = []

    pair_sep_lookup = {}
    mean_iqr_overlap_lookup = {}
    mean_range_overlap_lookup = {}
    if numeric_feature_overlap is not None and not numeric_feature_overlap.empty:
        pair_sep_lookup = (
            1.0 - numeric_feature_overlap.groupby("column", sort=False)["feature_overlap"].mean().astype(float)
        ).to_dict()
        mean_iqr_overlap_lookup = numeric_feature_overlap.groupby("column", sort=False)["iqr_overlap"].mean().astype(float).to_dict()
        mean_range_overlap_lookup = numeric_feature_overlap.groupby("column", sort=False)["range_overlap"].mean().astype(float).to_dict()

    group_summary_lookup = {}
    if numeric_summary_by_group is not None and not numeric_summary_by_group.empty:
        for col, sub in numeric_summary_by_group.groupby("column", sort=False):
            iqr_series = pd.to_numeric(sub["iqr"], errors="coerce")
            median_series = pd.to_numeric(sub["median"], errors="coerce")
            min_series = pd.to_numeric(sub["min"], errors="coerce")
            max_series = pd.to_numeric(sub["max"], errors="coerce")
            global_lo = float(min_series.min()) if min_series.notna().any() else np.nan
            global_hi = float(max_series.max()) if max_series.notna().any() else np.nan
            global_span = global_hi - global_lo if np.isfinite(global_lo) and np.isfinite(global_hi) else np.nan
            group_summary_lookup[col] = {
                "median_group_iqr": float(iqr_series.median(skipna=True)) if iqr_series.notna().any() else np.nan,
                "mean_group_iqr": float(iqr_series.mean(skipna=True)) if iqr_series.notna().any() else np.nan,
                "between_group_std": float(median_series.std(ddof=1)) if median_series.notna().sum() > 1 else 0.0,
                "global_span": global_span,
                "n_groups_with_summary": int(sub["group"].nunique()),
            }

    for col in numeric_cols:
        if not should_use_numeric_column(col):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid_mask = s.notna() & group_series.notna()
        n_valid = int(valid_mask.sum())
        if n_valid < max(2, int(min_valid_values)):
            continue

        temp = pd.DataFrame({"group": group_series.loc[valid_mask].astype(str), "value": s.loc[valid_mask].astype(float)})
        grouped = temp.groupby("group", sort=True)["value"]
        n_groups = int(grouped.ngroups)
        if n_groups < 2:
            continue

        counts = grouped.size().astype(float)
        means = grouped.mean().astype(float)
        overall_mean = float(temp["value"].mean())
        ss_between = float(((means - overall_mean) ** 2 * counts).sum())
        ss_total = float(((temp["value"] - overall_mean) ** 2).sum())
        ss_within = max(0.0, ss_total - ss_between)
        eta2 = (ss_between / ss_total) if ss_total > 0 else np.nan

        dof_between = n_groups - 1
        dof_within = n_valid - n_groups
        if dof_between > 0 and dof_within > 0 and ss_within > 0:
            ms_between = ss_between / dof_between
            ms_within = ss_within / dof_within
            f_stat = ms_between / ms_within if ms_within > 0 else np.nan
        else:
            f_stat = np.nan

        group_stats = group_summary_lookup.get(col, {})
        median_group_iqr = group_stats.get("median_group_iqr", np.nan)
        mean_group_iqr = group_stats.get("mean_group_iqr", np.nan)
        between_group_std = group_stats.get("between_group_std", float(means.std(ddof=1)) if len(means) > 1 else 0.0)
        global_span = group_stats.get("global_span", np.nan)
        iqr_floor = max(float(min_iqr_floor_frac) * global_span, 1e-9) if np.isfinite(global_span) and global_span > 0 else 1e-9
        stabilized_iqr = max(median_group_iqr, iqr_floor) if np.isfinite(median_group_iqr) else iqr_floor
        signal_to_iqr = (
            between_group_std / stabilized_iqr
            if np.isfinite(between_group_std) and np.isfinite(stabilized_iqr) and stabilized_iqr > 0
            else np.nan
        )

        family = get_numeric_family(col)
        family_prior = float(family_priors.get(family, family_priors.get("other", 0.8)))
        pairwise_separation = float(pair_sep_lookup.get(col, np.nan))
        mean_iqr_overlap = float(mean_iqr_overlap_lookup.get(col, np.nan)) if col in mean_iqr_overlap_lookup else np.nan
        mean_range_overlap = float(mean_range_overlap_lookup.get(col, np.nan)) if col in mean_range_overlap_lookup else np.nan

        rows.append(
            {
                "column": col,
                "family": family,
                "family_prior": family_prior,
                "n_valid": n_valid,
                "n_groups": n_groups,
                "overall_mean": overall_mean,
                "overall_std": float(temp["value"].std(ddof=1)) if n_valid > 1 else 0.0,
                "between_group_std": between_group_std,
                "median_group_iqr": median_group_iqr,
                "mean_group_iqr": mean_group_iqr,
                "global_span": global_span,
                "stabilized_iqr_floor": iqr_floor,
                "stabilized_iqr": stabilized_iqr,
                "signal_to_iqr": signal_to_iqr,
                "pairwise_separation": pairwise_separation,
                "mean_iqr_overlap": mean_iqr_overlap,
                "mean_range_overlap": mean_range_overlap,
                "eta2": eta2,
                "f_stat": f_stat,
                "ss_between": ss_between,
                "ss_within": ss_within,
                "ss_total": ss_total,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    signal_component = np.log1p(pd.to_numeric(out["signal_to_iqr"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0))
    eta2_component = pd.to_numeric(out["eta2"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
    separation_component = pd.to_numeric(out["pairwise_separation"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
    prior_component = pd.to_numeric(out["family_prior"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)

    out["eta2_weight_global"] = normalize_nonnegative_series(eta2_component)
    out["signal_to_iqr_weight_global"] = normalize_nonnegative_series(signal_component)
    out["pairwise_separation_weight_global"] = normalize_nonnegative_series(separation_component)
    out["family_prior_weight_global"] = normalize_nonnegative_series(prior_component)

    raw = (
        0.35 * out["eta2_weight_global"]
        + 0.45 * out["pairwise_separation_weight_global"]
        + 0.20 * out["signal_to_iqr_weight_global"]
    ) * (0.5 + 0.5 * prior_component)
    raw = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
    total_raw = float(raw.sum())
    out["blended_weight_global"] = (raw / total_raw) if total_raw > 0 else 0.0

    out = out.sort_values(
        ["blended_weight_global", "pairwise_separation", "eta2", "signal_to_iqr", "column"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return out


def summarize_family_signal_weights(feature_weights, family_priors=None):
    if feature_weights is None or feature_weights.empty:
        return pd.DataFrame()

    family_priors = family_priors or dict(DEFAULT_FAMILY_RELIABILITY_PRIORS)
    out = (
        feature_weights.groupby("family", as_index=False)
        .agg(
            n_features=("column", "count"),
            mean_eta2=("eta2", "mean"),
            median_eta2=("eta2", "median"),
            mean_signal_to_iqr=("signal_to_iqr", "mean"),
            median_signal_to_iqr=("signal_to_iqr", "median"),
            mean_pairwise_separation=("pairwise_separation", "mean"),
            mean_blended_weight=("blended_weight_global", "mean"),
            sum_blended_weight=("blended_weight_global", "sum"),
        )
        .sort_values(["sum_blended_weight", "family"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out["family_prior"] = out["family"].map(lambda x: float(family_priors.get(str(x), family_priors.get("other", 0.8))))
    out["family_weight_mean_normalized"] = normalize_nonnegative_series(out["mean_blended_weight"])
    out["family_weight_sum_normalized"] = normalize_nonnegative_series(out["sum_blended_weight"])
    return out


def save_feature_weight_plots(feature_weights, family_weights, charts_dir, fig_dpi, max_label_len):
    charts_dir.mkdir(parents=True, exist_ok=True)
    if feature_weights is not None and not feature_weights.empty:
        top = feature_weights.head(30).copy()
        top = top.sort_values(["blended_weight_global", "column"], ascending=[True, True]).reset_index(drop=True)
        y = np.arange(len(top), dtype=float)
        labels = [clip_label(v, max_label_len) for v in top["column"].astype(str).tolist()]
        vals = top["blended_weight_global"].to_numpy(dtype=float)

        fig_h = max(4.0, 0.36 * len(top) + 1.8)
        fig, ax = plt.subplots(figsize=(10.8, fig_h))
        bars = ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Blended empirical weight")
        ax.set_title("Top weighted numeric features by envelope separation")
        ax.grid(True, axis="x", alpha=0.25)

        text_offset = (vals.max() * 0.015) if len(vals) and float(np.nanmax(vals)) > 0 else 0.001
        for bar, eta2, sep, sig in zip(top["blended_weight_global"], top["eta2"], top["pairwise_separation"], top["signal_to_iqr"]):
            pass
        for bar_patch, eta2, sep, sig in zip(bars, top["eta2"], top["pairwise_separation"], top["signal_to_iqr"]):
            parts = [f"sep={sep:.3f}"] if np.isfinite(sep) else []
            if np.isfinite(eta2):
                parts.append(f"eta2={eta2:.3f}")
            if np.isfinite(sig):
                parts.append(f"sig/IQR={sig:.3f}")
            ax.text(
                float(bar_patch.get_width()) + text_offset,
                bar_patch.get_y() + bar_patch.get_height() / 2.0,
                "  ".join(parts),
                va="center",
                ha="left",
                fontsize=8,
            )

        out_path = charts_dir / "top_feature_weights.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)

    if family_weights is not None and not family_weights.empty:
        sub = family_weights.sort_values(["family_weight_sum_normalized", "family"], ascending=[True, True]).reset_index(drop=True)
        y = np.arange(len(sub), dtype=float)
        vals = sub["family_weight_sum_normalized"].to_numpy(dtype=float)

        fig_h = max(3.6, 0.45 * len(sub) + 1.6)
        fig, ax = plt.subplots(figsize=(9.8, fig_h))
        bars = ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["family"].astype(str).tolist())
        ax.set_xlabel("Normalized family weight")
        ax.set_title("Numeric family weight by envelope-separation signal")
        ax.grid(True, axis="x", alpha=0.25)

        text_offset = (vals.max() * 0.02) if len(vals) and float(np.nanmax(vals)) > 0 else 0.001
        for bar_patch, n_features, prior in zip(bars, sub["n_features"], sub["family_prior"]):
            ax.text(
                float(bar_patch.get_width()) + text_offset,
                bar_patch.get_y() + bar_patch.get_height() / 2.0,
                f"n={int(n_features)}  prior={float(prior):.2f}",
                va="center",
                ha="left",
                fontsize=8,
            )

        out_path = charts_dir / "family_signal_weights.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def save_feature_weight_family_plots(feature_weights, charts_dir, fig_dpi, max_label_len, top_n=12):
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    if feature_weights is None or feature_weights.empty:
        return

    for family, sub in feature_weights.groupby("family", sort=True):
        sub = sub.sort_values(["blended_weight_global", "column"], ascending=[False, True]).head(top_n).copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["blended_weight_global", "column"], ascending=[True, True]).reset_index(drop=True)

        y = np.arange(len(sub), dtype=float)
        vals = sub["blended_weight_global"].to_numpy(dtype=float)
        labels = [clip_label(v, max_label_len) for v in sub["column"].astype(str).tolist()]

        fig_h = max(3.8, 0.38 * len(sub) + 1.6)
        fig, ax = plt.subplots(figsize=(10.8, fig_h))
        bars = ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Blended empirical weight")
        ax.set_title(f"{family} top weighted features")
        ax.grid(True, axis="x", alpha=0.25)

        xpad = (float(np.nanmax(vals)) * 0.015) if len(vals) and float(np.nanmax(vals)) > 0 else 0.001
        for bar_patch, eta2, sep, sig in zip(bars, sub["eta2"], sub["pairwise_separation"], sub["signal_to_iqr"]):
            parts = []
            if np.isfinite(sep):
                parts.append(f"sep={sep:.3f}")
            if np.isfinite(eta2):
                parts.append(f"eta2={eta2:.3f}")
            if np.isfinite(sig):
                parts.append(f"sig/IQR={sig:.3f}")
            ax.text(
                float(bar_patch.get_width()) + xpad,
                bar_patch.get_y() + bar_patch.get_height() / 2.0,
                "  ".join(parts),
                va="center",
                ha="left",
                fontsize=8,
            )

        out_path = charts_dir / f"{safe_slug(family)}_top_feature_weights.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def pair_table_from_matrix(matrix_df, value_name="value"):
    if matrix_df is None or matrix_df.empty:
        return pd.DataFrame(columns=["group_a", "group_b", value_name, f"abs_{value_name}", "pair"])
    labels = list(matrix_df.index)
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = str(labels[i])
            b = str(labels[j])
            if b not in matrix_df.columns:
                continue
            value = pd.to_numeric(pd.Series([matrix_df.loc[a, b]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            pair = f"{a} | {b}"
            rows.append({"group_a": a, "group_b": b, value_name: float(value), f"abs_{value_name}": float(abs(value)), "pair": pair})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values([value_name, "pair"], ascending=[False, True]).reset_index(drop=True)
    return out


def save_pair_barplot(pair_df, value_col, out_path, title, fig_dpi, top_n=15):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pair_df is None or pair_df.empty or value_col not in pair_df.columns:
        return

    plot_df = pair_df.sort_values([value_col, "pair"], ascending=[False, True]).head(top_n).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values([value_col, "pair"], ascending=[True, True]).reset_index(drop=True)

    y = np.arange(len(plot_df), dtype=float)
    vals = plot_df[value_col].to_numpy(dtype=float)
    labels = [clip_label(v, 96) for v in plot_df["pair"].astype(str).tolist()]

    fig_h = max(4.0, 0.42 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    bars = ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)

    xpad = (float(np.nanmax(np.abs(vals))) * 0.02) if len(vals) else 0.01
    xpad = max(xpad, 0.01)
    for bar, value in zip(bars, vals):
        ax.text(
            float(bar.get_width()) + (xpad if value >= 0 else -xpad),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)


def build_family_pair_support_table(by_family_dir, suffix):
    by_family_dir = Path(by_family_dir)
    rows = []
    for csv_path in sorted(by_family_dir.glob(f"*_{suffix}.csv")):
        stem = csv_path.stem
        family = stem[: -(len(suffix) + 1)] if stem.endswith("_" + suffix) else stem
        matrix_df = maybe_read_matrix_csv(csv_path)
        if matrix_df is None or matrix_df.empty:
            continue
        pair_df = pair_table_from_matrix(matrix_df, value_name="overlap")
        if pair_df.empty:
            continue
        pair_df["family"] = family
        rows.append(pair_df[["pair", "family", "overlap"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def save_pair_family_support_heatmap(pair_family_df, out_path, title, fig_dpi, pair_order=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pair_family_df is None or pair_family_df.empty:
        return

    pivot = pair_family_df.pivot_table(index="pair", columns="family", values="overlap", aggfunc="first")
    if pair_order:
        ordered = [pair for pair in pair_order if pair in pivot.index]
        remainder = [pair for pair in pivot.index.tolist() if pair not in ordered]
        pivot = pivot.reindex(ordered + remainder)
    pivot = pivot.sort_index(axis=1)
    pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if pivot.empty:
        return

    arr = pivot.to_numpy(dtype=float)
    fig_w = max(7.5, 0.7 * pivot.shape[1] + 3.0)
    fig_h = max(4.5, 0.42 * pivot.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Family")
    ax.set_ylabel("Species pair")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([clip_label(v, 96) for v in pivot.index.tolist()])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)


def generate_correlation_and_weight_insight_visuals(outdir, fig_dpi, max_label_len):
    outdir = Path(outdir)
    remove_obsolete_correlation_outputs(outdir)
    overlaps_dir = outdir / "overlaps"
    by_family_dir = overlaps_dir / "by_family"
    insights_dir = outdir / "insights"
    insights_charts_dir = outdir / "charts" / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    insights_charts_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [
        "species_profile_numeric_overlap_weighted",
        "species_profile_numeric_overlap_unweighted",
        "species_profile_categorical_overlap",
        "species_profile_blended_overlap",
        "species_profile_terraclimate_variable_overlap",
    ]
    for suffix in suffixes:
        csv_path = overlaps_dir / f"{suffix}.csv"
        matrix_df = maybe_read_matrix_csv(csv_path)
        if matrix_df is None or matrix_df.empty:
            continue
        pair_df = pair_table_from_matrix(matrix_df, value_name="overlap")
        if pair_df.empty:
            continue
        pair_df.to_csv(insights_dir / f"{suffix}_pairwise.csv", index=False)
        top_pairs = pair_df.sort_values(["overlap", "pair"], ascending=[False, True]).head(15).copy()
        if not top_pairs.empty:
            top_pairs.to_csv(insights_dir / f"{suffix}_top_pairs.csv", index=False)
            save_pair_barplot(
                top_pairs,
                value_col="overlap",
                out_path=insights_charts_dir / f"{suffix}_top_pairs.png",
                title=suffix.replace("_", " "),
                fig_dpi=fig_dpi,
                top_n=15,
            )

    family_pair_df = build_family_pair_support_table(by_family_dir, "species_profile_numeric_overlap_weighted")
    main_pair_df = maybe_read_csv(insights_dir / "species_profile_blended_overlap_pairwise.csv")
    if family_pair_df is not None and not family_pair_df.empty:
        family_pair_df.to_csv(insights_dir / "species_profile_numeric_overlap_weighted_pair_family_support.csv", index=False)
        pair_order = main_pair_df["pair"].head(15).tolist() if main_pair_df is not None and not main_pair_df.empty else family_pair_df["pair"].drop_duplicates().head(15).tolist()
        save_pair_family_support_heatmap(
            family_pair_df[family_pair_df["pair"].isin(pair_order)].copy(),
            out_path=insights_charts_dir / "species_profile_numeric_overlap_weighted_top_pair_family_support.png",
            title="Family support for top numeric-overlap species pairs",
            fig_dpi=fig_dpi,
            pair_order=pair_order,
        )

    feature_weights = maybe_read_csv(outdir / "feature_signal_weights.csv")
    if feature_weights is not None and not feature_weights.empty:
        feature_weights = normalize_summary_frame(feature_weights)
    family_weights = maybe_read_csv(outdir / "family_signal_weights.csv")
    if family_weights is not None and not family_weights.empty:
        family_weights = normalize_summary_frame(family_weights)

    save_feature_weight_plots(
        feature_weights=feature_weights,
        family_weights=family_weights,
        charts_dir=outdir / "charts" / "feature_weights",
        fig_dpi=fig_dpi,
        max_label_len=max_label_len,
    )
    save_feature_weight_family_plots(
        feature_weights=feature_weights,
        charts_dir=outdir / "charts" / "feature_weights" / "by_family",
        fig_dpi=fig_dpi,
        max_label_len=max_label_len,
    )


def run_group_correlation_and_weight_steps(
    df,
    outdir,
    args,
    numeric_cols,
    numeric_by_group,
    categorical_by_group=None,
    monthly_by_group=None,
):
    if numeric_by_group is None or numeric_by_group.empty:
        generate_correlation_and_weight_insight_visuals(
            outdir=outdir,
            fig_dpi=args.fig_dpi,
            max_label_len=args.max_category_label_len,
        )
        return

    reuse_existing = not args.no_resume_existing
    family_priors = parse_family_reliability_priors(args.family_reliability_priors)
    remove_obsolete_correlation_outputs(outdir)

    overlaps_dir = outdir / "overlaps"
    overlaps_heatmaps_dir = outdir / "charts" / "overlaps"
    by_family_dir = overlaps_dir / "by_family"
    by_family_heatmaps_dir = overlaps_heatmaps_dir / "by_family"
    overlaps_dir.mkdir(parents=True, exist_ok=True)
    overlaps_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    by_family_dir.mkdir(parents=True, exist_ok=True)
    by_family_heatmaps_dir.mkdir(parents=True, exist_ok=True)

    numeric_feature_overlap_path = overlaps_dir / "numeric_feature_overlap_pairs.csv"
    numeric_feature_overlap = maybe_read_csv(numeric_feature_overlap_path) if reuse_existing else None
    if numeric_feature_overlap is not None and not numeric_feature_overlap.empty:
        numeric_feature_overlap = filter_rows_by_feature_column(normalize_summary_frame(numeric_feature_overlap), column_field="column")
        log(f"Reusing {numeric_feature_overlap_path}")
        numeric_feature_overlap.to_csv(numeric_feature_overlap_path, index=False)
    else:
        numeric_feature_overlap = filter_rows_by_feature_column(build_numeric_feature_overlap_table(numeric_by_group), column_field="column")
        if numeric_feature_overlap is not None and not numeric_feature_overlap.empty:
            numeric_feature_overlap.to_csv(numeric_feature_overlap_path, index=False)

    monthly_variable_overlap_path = overlaps_dir / "terraclimate_monthly_variable_overlap_pairs.csv"
    monthly_variable_overlap = maybe_read_csv(monthly_variable_overlap_path) if reuse_existing else None
    if monthly_variable_overlap is not None and not monthly_variable_overlap.empty:
        monthly_variable_overlap = normalize_summary_frame(monthly_variable_overlap)
        log(f"Reusing {monthly_variable_overlap_path}")
    else:
        monthly_variable_overlap = build_monthly_variable_overlap_table(monthly_by_group)
        if monthly_variable_overlap is not None and not monthly_variable_overlap.empty:
            monthly_variable_overlap.to_csv(monthly_variable_overlap_path, index=False)

    categorical_overlap_path = overlaps_dir / "categorical_overlap_pairs.csv"
    categorical_overlap = maybe_read_csv(categorical_overlap_path) if reuse_existing else None
    if categorical_overlap is not None and not categorical_overlap.empty:
        categorical_overlap = normalize_summary_frame(categorical_overlap)
        log(f"Reusing {categorical_overlap_path}")
    else:
        categorical_overlap = build_categorical_overlap_table(categorical_by_group)
        if categorical_overlap is not None and not categorical_overlap.empty:
            categorical_overlap.to_csv(categorical_overlap_path, index=False)

    feature_weights_path = outdir / "feature_signal_weights.csv"
    feature_weights = maybe_read_csv(feature_weights_path) if reuse_existing else None
    if feature_weights is not None and not feature_weights.empty:
        feature_weights = filter_rows_by_feature_column(normalize_summary_frame(feature_weights), column_field="column")
        log(f"Reusing {feature_weights_path}")
        feature_weights.to_csv(feature_weights_path, index=False)
    else:
        feature_weights = filter_rows_by_feature_column(
            compute_feature_signal_weights(
                df=df,
                group_by=args.group_by,
                numeric_cols=[c for c in numeric_cols if should_use_numeric_column(c)],
                numeric_summary_by_group=numeric_by_group,
                min_valid_values=args.min_valid_values_per_feature_weight,
                min_iqr_floor_frac=args.min_iqr_floor_frac,
                family_priors=family_priors,
                numeric_feature_overlap=numeric_feature_overlap,
            ),
            column_field="column",
        )
        if feature_weights is not None and not feature_weights.empty:
            feature_weights.to_csv(feature_weights_path, index=False)

    family_weights_path = outdir / "family_signal_weights.csv"
    family_weights = maybe_read_csv(family_weights_path) if reuse_existing else None
    if family_weights is not None and not family_weights.empty:
        family_weights = normalize_summary_frame(family_weights)
        log(f"Reusing {family_weights_path}")
    else:
        family_weights = summarize_family_signal_weights(feature_weights, family_priors=family_priors)
        if family_weights is not None and not family_weights.empty:
            family_weights.to_csv(family_weights_path, index=False)

    weight_map = {}
    if feature_weights is not None and not feature_weights.empty:
        weight_map = dict(zip(feature_weights["column"].astype(str), pd.to_numeric(feature_weights["blended_weight_global"], errors="coerce").fillna(0.0)))

    numeric_weighted = numeric_feature_overlap.copy() if numeric_feature_overlap is not None else pd.DataFrame()
    if numeric_weighted is not None and not numeric_weighted.empty:
        numeric_weighted["feature_weight"] = numeric_weighted["column"].map(lambda x: float(weight_map.get(str(x), 0.0)))
    numeric_weighted_matrix, numeric_weighted_counts = build_pair_matrix_from_table(
        numeric_weighted,
        value_col="feature_overlap",
        weight_col="feature_weight",
        key_col="column",
        min_features=args.min_valid_features_per_pair,
    )
    if numeric_weighted_matrix is not None and not numeric_weighted_matrix.empty:
        numeric_weighted_matrix.to_csv(overlaps_dir / "species_profile_numeric_overlap_weighted.csv")
        numeric_weighted_counts.to_csv(overlaps_dir / "species_profile_numeric_overlap_weighted_feature_counts.csv")
        save_matrix_heatmap(
            numeric_weighted_matrix,
            overlaps_heatmaps_dir / "species_profile_numeric_overlap_weighted.png",
            title="Species profile numeric envelope overlap (weighted)",
            fig_dpi=args.fig_dpi,
            vmin=0.0,
            vmax=1.0,
        )

    numeric_unweighted_matrix, numeric_unweighted_counts = build_pair_matrix_from_table(
        numeric_feature_overlap,
        value_col="feature_overlap",
        weight_col=None,
        key_col="column",
        min_features=args.min_valid_features_per_pair,
    )
    if numeric_unweighted_matrix is not None and not numeric_unweighted_matrix.empty:
        numeric_unweighted_matrix.to_csv(overlaps_dir / "species_profile_numeric_overlap_unweighted.csv")
        numeric_unweighted_counts.to_csv(overlaps_dir / "species_profile_numeric_overlap_unweighted_feature_counts.csv")
        save_matrix_heatmap(
            numeric_unweighted_matrix,
            overlaps_heatmaps_dir / "species_profile_numeric_overlap_unweighted.png",
            title="Species profile numeric envelope overlap (unweighted)",
            fig_dpi=args.fig_dpi,
            vmin=0.0,
            vmax=1.0,
        )

    terraclimate_variable_matrix, terraclimate_variable_counts = build_pair_matrix_from_table(
        monthly_variable_overlap,
        value_col="feature_overlap",
        weight_col=None,
        key_col="variable",
        min_features=3,
    )
    if terraclimate_variable_matrix is not None and not terraclimate_variable_matrix.empty:
        terraclimate_variable_matrix.to_csv(overlaps_dir / "species_profile_terraclimate_variable_overlap.csv")
        terraclimate_variable_counts.to_csv(overlaps_dir / "species_profile_terraclimate_variable_overlap_feature_counts.csv")
        save_matrix_heatmap(
            terraclimate_variable_matrix,
            overlaps_heatmaps_dir / "species_profile_terraclimate_variable_overlap.png",
            title="Species profile monthly-climate envelope overlap",
            fig_dpi=args.fig_dpi,
            vmin=0.0,
            vmax=1.0,
        )

    available_families = [family for family in sorted(numeric_by_group["family"].dropna().astype(str).unique()) if family and family != "mcd12q1"]
    for family in available_families:
        fsub = numeric_weighted[numeric_weighted["family"].astype(str) == family].copy() if numeric_weighted is not None else pd.DataFrame()
        fam_matrix, fam_counts = build_pair_matrix_from_table(
            fsub,
            value_col="feature_overlap",
            weight_col="feature_weight",
            key_col="column",
            min_features=1,
        )
        if fam_matrix is None or fam_matrix.empty:
            continue
        fam_matrix.to_csv(by_family_dir / f"{safe_slug(family)}_species_profile_numeric_overlap_weighted.csv")
        fam_counts.to_csv(by_family_dir / f"{safe_slug(family)}_species_profile_numeric_overlap_weighted_feature_counts.csv")
        save_matrix_heatmap(
            fam_matrix,
            by_family_heatmaps_dir / f"{safe_slug(family)}_species_profile_numeric_overlap_weighted.png",
            title=f"{family} species profile envelope overlap (weighted)",
            fig_dpi=args.fig_dpi,
            vmin=0.0,
            vmax=1.0,
        )

    categorical_matrix, categorical_counts = build_pair_matrix_from_table(
        categorical_overlap,
        value_col="categorical_overlap",
        weight_col=None,
        key_col="column",
        min_features=1,
    )
    if categorical_matrix is not None and not categorical_matrix.empty:
        categorical_matrix.to_csv(overlaps_dir / "species_profile_categorical_overlap.csv")
        categorical_counts.to_csv(overlaps_dir / "species_profile_categorical_overlap_feature_counts.csv")
        save_matrix_heatmap(
            categorical_matrix,
            overlaps_heatmaps_dir / "species_profile_categorical_overlap.png",
            title="Species profile categorical overlap",
            fig_dpi=args.fig_dpi,
            vmin=0.0,
            vmax=1.0,
        )

    if categorical_overlap is not None and not categorical_overlap.empty:
        for family in sorted(categorical_overlap["family"].dropna().astype(str).unique()):
            csub = categorical_overlap[categorical_overlap["family"].astype(str) == family].copy()
            fam_matrix, fam_counts = build_pair_matrix_from_table(
                csub,
                value_col="categorical_overlap",
                weight_col=None,
                key_col="column",
                min_features=1,
            )
            if fam_matrix is None or fam_matrix.empty:
                continue
            fam_matrix.to_csv(by_family_dir / f"{safe_slug(family)}_species_profile_categorical_overlap.csv")
            fam_counts.to_csv(by_family_dir / f"{safe_slug(family)}_species_profile_categorical_overlap_feature_counts.csv")
            save_matrix_heatmap(
                fam_matrix,
                by_family_heatmaps_dir / f"{safe_slug(family)}_species_profile_categorical_overlap.png",
                title=f"{family} categorical overlap",
                fig_dpi=args.fig_dpi,
                vmin=0.0,
                vmax=1.0,
            )

    blended_components = []
    numeric_share = 1.0 - max(0.0, min(1.0, float(args.categorical_overlap_share)))
    categorical_share = max(0.0, min(1.0, float(args.categorical_overlap_share)))
    if numeric_weighted_matrix is not None and not numeric_weighted_matrix.empty:
        blended_components.append((numeric_weighted_matrix, numeric_share))
    if categorical_matrix is not None and not categorical_matrix.empty:
        blended_components.append((categorical_matrix, categorical_share if numeric_weighted_matrix is not None and not numeric_weighted_matrix.empty else 1.0))
    if blended_components:
        groups = sorted(set().union(*[set(m.index) for m, _ in blended_components]))
        num = pd.DataFrame(0.0, index=groups, columns=groups)
        den = pd.DataFrame(0.0, index=groups, columns=groups)
        for matrix, share in blended_components:
            share = float(share)
            m = matrix.reindex(index=groups, columns=groups)
            mask = m.notna().astype(float)
            num = num + m.fillna(0.0) * share
            den = den + mask * share
        blended = num.divide(den.where(den > 0), fill_value=np.nan)
        for g in groups:
            blended.loc[g, g] = 1.0
        blended.to_csv(overlaps_dir / "species_profile_blended_overlap.csv")
        save_matrix_heatmap(
            blended,
            overlaps_heatmaps_dir / "species_profile_blended_overlap.png",
            title="Species profile blended overlap",
            fig_dpi=args.fig_dpi,
            vmin=0.0,
            vmax=1.0,
        )

    save_feature_weight_plots(
        feature_weights=feature_weights,
        family_weights=family_weights,
        charts_dir=outdir / "charts" / "feature_weights",
        fig_dpi=args.fig_dpi,
        max_label_len=args.max_category_label_len,
    )
    save_feature_weight_family_plots(
        feature_weights=feature_weights,
        charts_dir=outdir / "charts" / "feature_weights" / "by_family",
        fig_dpi=args.fig_dpi,
        max_label_len=args.max_category_label_len,
    )
    generate_correlation_and_weight_insight_visuals(
        outdir=outdir,
        fig_dpi=args.fig_dpi,
        max_label_len=args.max_category_label_len,
    )


def write_readme(
    outdir,
    input_csv,
    group_by,
    n_rows,
    numeric_cols,
    categorical_cols,
    monthly_vars,
    has_group_compare=False,
    terraclimate_multiyear_meta=None,
):
    lines = [
        f"input_csv: {input_csv}",
        f"group_by: {group_by or 'none'}",
        f"rows: {n_rows}",
        "",
        "files:",
        "  numeric_summary.csv",
        "  categorical_frequencies.csv",
        "  terraclimate_monthly_summary.csv",
        "  terraclimate_multiyear_columns.csv",
        "  charts/terraclimate_monthly/*.png",
        "  charts/numeric/*.png",
        "  charts/numeric_overview/*.png",
        "  charts/categorical/*.png",
    ]
    if has_group_compare:
        lines.extend(
            [
                "  by_group_numeric_summary.csv",
                "  by_group_categorical_frequencies.csv",
                "  by_group_terraclimate_monthly_summary.csv",
                "  overlaps/numeric_feature_overlap_pairs.csv",
                "  overlaps/terraclimate_monthly_variable_overlap_pairs.csv",
                "  overlaps/categorical_overlap_pairs.csv",
                "  overlaps/species_profile_numeric_overlap_weighted.csv",
                "  overlaps/species_profile_numeric_overlap_unweighted.csv",
                "  overlaps/species_profile_terraclimate_variable_overlap.csv",
                "  overlaps/species_profile_categorical_overlap.csv",
                "  overlaps/species_profile_blended_overlap.csv",
                "  overlaps/by_family/<family>_species_profile_numeric_overlap_weighted.csv",
                "  feature_signal_weights.csv",
                "  family_signal_weights.csv",
                "  charts/group_compare/numeric/*.png",
                "  charts/group_compare/numeric_overview/*.png",
                "  charts/group_compare/categorical/*.png",
                "  charts/group_compare/terraclimate_monthly/*.png",
                "  charts/overlaps/*.png",
                "  charts/overlaps/by_family/*.png",
                "  charts/feature_weights/*.png",
                "  charts/feature_weights/by_family/*.png",
                "  insights/*.csv",
                "  charts/insights/*.png",
                "  groups/<group_name>/*",
            ]
        )
    lines.extend(
        [
            "",
            f"numeric_columns_detected: {len(numeric_cols)}",
            f"categorical_columns_detected: {len(categorical_cols)}",
            f"monthly_terraclimate_variables_detected: {len(monthly_vars)}",
            f"multiyear_terraclimate_features_detected: {0 if terraclimate_multiyear_meta is None else len(terraclimate_multiyear_meta)}",
        ]
    )
    if terraclimate_multiyear_meta is not None and not terraclimate_multiyear_meta.empty:
        lines.extend(
            [
                "",
                "multiyear_terraclimate_years:",
            ]
        )
        for row in terraclimate_multiyear_meta.itertuples(index=False):
            label = row.canonical_column
            if pd.notna(row.month):
                label = f"{label} ({row.month})"
            lines.append(f"  {label}: years={row.years}")
    (outdir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_group(df_group, outdir, args, input_csv_value=None, group_by_value=None):
    outdir.mkdir(parents=True, exist_ok=True)

    df_work, terraclimate_multiyear_meta = build_terraclimate_working_df(df_group)
    if terraclimate_multiyear_meta is not None and not terraclimate_multiyear_meta.empty:
        terraclimate_multiyear_meta.to_csv(outdir / "terraclimate_multiyear_columns.csv", index=False)

    numeric_cols = detect_numeric_columns(df_work)
    categorical_cols = detect_categorical_columns(df_work)
    monthly_map = get_monthly_terraclimate_map(df_work.columns)

    numeric_summary = filter_rows_by_feature_column(summarize_numeric(df_work, numeric_cols), column_field="column")
    categorical_summary = summarize_categorical(df_work, categorical_cols, args.top_n)
    monthly_summary, usable_monthly = summarize_monthly_terraclimate(
        df_work, monthly_map, min_valid_months=args.min_valid_months
    )

    numeric_summary.to_csv(outdir / "numeric_summary.csv", index=False)
    categorical_summary.to_csv(outdir / "categorical_frequencies.csv", index=False)
    monthly_summary.to_csv(outdir / "terraclimate_monthly_summary.csv", index=False)

    numeric_plot_summary = numeric_summary[~numeric_summary["column"].map(is_monthly_terraclimate_column)].copy()

    save_numeric_distribution_plots(df_work, numeric_plot_summary, outdir / "charts" / "numeric", fig_dpi=args.fig_dpi)
    save_numeric_family_overview_plots(numeric_plot_summary, outdir / "charts" / "numeric_overview", fig_dpi=args.fig_dpi)
    save_categorical_frequency_plots(
        categorical_summary,
        outdir / "charts" / "categorical",
        fig_dpi=args.fig_dpi,
        max_label_len=args.max_category_label_len,
    )

    if not monthly_summary.empty:
        save_monthly_plots(monthly_summary, outdir / "charts" / "terraclimate_monthly", fig_dpi=args.fig_dpi)

    write_readme(
        outdir=outdir,
        input_csv=input_csv_value or args.csv,
        group_by=group_by_value,
        n_rows=len(df_group),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        monthly_vars=sorted(usable_monthly.keys()),
        has_group_compare=False,
        terraclimate_multiyear_meta=terraclimate_multiyear_meta,
    )



def process_with_group_compare(df, outdir, args):
    outdir.mkdir(parents=True, exist_ok=True)

    df_work, terraclimate_multiyear_meta = build_terraclimate_working_df(df)
    if terraclimate_multiyear_meta is not None and not terraclimate_multiyear_meta.empty:
        terraclimate_multiyear_meta.to_csv(outdir / "terraclimate_multiyear_columns.csv", index=False)
        years = sorted({int(y) for value in terraclimate_multiyear_meta["years"].astype(str) for y in value.split(",") if y.strip()})
        log(f"Using multiyear TerraClimate aggregation across years: {', '.join(str(y) for y in years)}")

    reuse_existing = not args.no_resume_existing
    numeric_cols = detect_numeric_columns(df_work)
    categorical_cols = detect_categorical_columns(df_work)
    monthly_map = get_monthly_terraclimate_map(df_work.columns)

    overall_numeric_path = outdir / "numeric_summary.csv"
    overall_categorical_path = outdir / "categorical_frequencies.csv"
    overall_monthly_path = outdir / "terraclimate_monthly_summary.csv"
    by_group_numeric_path = outdir / "by_group_numeric_summary.csv"
    by_group_categorical_path = outdir / "by_group_categorical_frequencies.csv"
    by_group_monthly_path = outdir / "by_group_terraclimate_monthly_summary.csv"

    overall_numeric = maybe_read_csv(overall_numeric_path) if reuse_existing else None
    if overall_numeric is not None:
        overall_numeric = filter_rows_by_feature_column(normalize_summary_frame(overall_numeric), column_field="column")
        log(f"Reusing {overall_numeric_path}")
        overall_numeric.to_csv(overall_numeric_path, index=False)
    else:
        overall_numeric = filter_rows_by_feature_column(summarize_numeric(df_work, numeric_cols), column_field="column")
        overall_numeric.to_csv(overall_numeric_path, index=False)

    overall_categorical = maybe_read_csv(overall_categorical_path) if reuse_existing else None
    if overall_categorical is not None:
        overall_categorical = normalize_summary_frame(overall_categorical)
        log(f"Reusing {overall_categorical_path}")
    else:
        overall_categorical = summarize_categorical(df_work, categorical_cols, args.top_n)
        overall_categorical.to_csv(overall_categorical_path, index=False)

    overall_monthly = maybe_read_csv(overall_monthly_path) if reuse_existing else None
    if overall_monthly is not None:
        overall_monthly = normalize_summary_frame(overall_monthly)
        log(f"Reusing {overall_monthly_path}")
        usable_monthly = {var: monthly_map[var] for var in sorted(monthly_map.keys()) if var in set(overall_monthly["variable"].dropna().astype(str))}
    else:
        overall_monthly, usable_monthly = summarize_monthly_terraclimate(
            df_work, monthly_map, min_valid_months=args.min_valid_months
        )
        overall_monthly.to_csv(overall_monthly_path, index=False)

    numeric_by_group = maybe_read_csv(by_group_numeric_path) if reuse_existing else None
    if numeric_by_group is not None:
        numeric_by_group = filter_rows_by_feature_column(normalize_summary_frame(numeric_by_group), column_field="column")
        log(f"Reusing {by_group_numeric_path}")
        numeric_by_group.to_csv(by_group_numeric_path, index=False)
    else:
        numeric_by_group = filter_rows_by_feature_column(summarize_numeric_by_group(df_work, args.group_by, numeric_cols), column_field="column")
        numeric_by_group.to_csv(by_group_numeric_path, index=False)

    categorical_by_group = maybe_read_csv(by_group_categorical_path) if reuse_existing else None
    if categorical_by_group is not None:
        categorical_by_group = normalize_summary_frame(categorical_by_group)
        log(f"Reusing {by_group_categorical_path}")
    else:
        categorical_by_group = summarize_categorical_by_group(df_work, args.group_by, categorical_cols, args.top_n)
        categorical_by_group.to_csv(by_group_categorical_path, index=False)

    monthly_by_group = maybe_read_csv(by_group_monthly_path) if reuse_existing else None
    if monthly_by_group is not None:
        monthly_by_group = normalize_summary_frame(monthly_by_group)
        log(f"Reusing {by_group_monthly_path}")
        usable_monthly_by_group = {
            var: monthly_map[var]
            for var in sorted(monthly_map.keys())
            if var in set(monthly_by_group["variable"].dropna().astype(str))
        }
    else:
        monthly_by_group, usable_monthly_by_group = summarize_monthly_terraclimate_by_group(
            df_work, args.group_by, monthly_map, min_valid_months=args.min_valid_months
        )
        monthly_by_group.to_csv(by_group_monthly_path, index=False)

    if not args.next_steps_only:
        overall_numeric_plot_summary = overall_numeric[
            ~overall_numeric["column"].map(is_monthly_terraclimate_column)
        ].copy()

        save_numeric_distribution_plots(
            df_work,
            overall_numeric_plot_summary,
            outdir / "charts" / "numeric",
            fig_dpi=args.fig_dpi,
        )
        save_numeric_family_overview_plots(
            overall_numeric_plot_summary,
            outdir / "charts" / "numeric_overview",
            fig_dpi=args.fig_dpi,
        )
        save_categorical_frequency_plots(
            overall_categorical,
            outdir / "charts" / "categorical",
            fig_dpi=args.fig_dpi,
            max_label_len=args.max_category_label_len,
        )
        if overall_monthly is not None and not overall_monthly.empty:
            save_monthly_plots(
                overall_monthly,
                outdir / "charts" / "terraclimate_monthly",
                fig_dpi=args.fig_dpi,
            )

        non_monthly_numeric_cols = [c for c in numeric_cols if not is_monthly_terraclimate_column(c)]
        save_group_numeric_compare_plots(
            df_work,
            args.group_by,
            non_monthly_numeric_cols,
            outdir / "charts" / "group_compare" / "numeric",
            fig_dpi=args.fig_dpi,
            max_groups=args.max_compare_groups,
        )
        save_group_numeric_family_heatmaps(
            numeric_by_group[~numeric_by_group["column"].map(is_monthly_terraclimate_column)].copy()
            if numeric_by_group is not None and not numeric_by_group.empty
            else numeric_by_group,
            outdir / "charts" / "group_compare" / "numeric_overview",
            fig_dpi=args.fig_dpi,
            max_groups=args.max_compare_groups,
        )
        save_group_categorical_compare_plots(
            categorical_by_group,
            outdir / "charts" / "group_compare" / "categorical",
            fig_dpi=args.fig_dpi,
            max_label_len=args.max_category_label_len,
            max_groups=args.max_compare_groups,
            max_categories=args.max_compare_categories,
        )
        if monthly_by_group is not None and not monthly_by_group.empty:
            save_group_monthly_compare_plots(
                monthly_by_group,
                outdir / "charts" / "group_compare" / "terraclimate_monthly",
                fig_dpi=args.fig_dpi,
                max_groups=args.max_compare_groups,
            )

        for group_value, df_group in df_work.groupby(args.group_by, dropna=False, sort=True):
            group_name = group_value_to_label(group_value)
            group_outdir = outdir / "groups" / safe_slug(group_name)
            log(f"Processing group {group_name!r} with {len(df_group):,} rows")
            process_group(
                df_group.copy(),
                group_outdir,
                args,
                input_csv_value=args.csv,
                group_by_value=args.group_by,
            )
    else:
        log("Skipping existing base summary/chart generation and per-group folders because --next-steps-only was requested")

    run_group_correlation_and_weight_steps(
        df=df,
        outdir=outdir,
        args=args,
        numeric_cols=numeric_cols,
        numeric_by_group=numeric_by_group,
        categorical_by_group=categorical_by_group,
        monthly_by_group=monthly_by_group,
    )

    write_readme(
        outdir=outdir,
        input_csv=args.csv,
        group_by=args.group_by,
        n_rows=len(df),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        monthly_vars=sorted(set(usable_monthly.keys()) | set(usable_monthly_by_group.keys())),
        has_group_compare=True,
        terraclimate_multiyear_meta=terraclimate_multiyear_meta,
    )


def main():
    args = parse_args()
    input_path = Path(args.csv)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    outdir = Path(args.outdir) if args.outdir else input_path.parent / "trend_summary"

    log(f"Reading {input_path} ...")
    df = pd.read_csv(input_path, sep=args.delimiter, low_memory=False)
    log(f"Loaded {len(df):,} rows and {len(df.columns):,} columns")

    if args.group_by:
        if args.group_by not in df.columns:
            raise SystemExit(f"--group-by column not found: {args.group_by}")
        process_with_group_compare(df, outdir, args)
    else:
        process_group(df, outdir, args)

    log(f"Done. Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
