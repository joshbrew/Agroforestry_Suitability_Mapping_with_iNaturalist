#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Example:
# py build_suitability_maps.py D:/envpull_association_test/occurrences_enriched.csv D:/Oregon_Suitability/oregon_grid_1000m_env/grid_with_env.csv --trend-summary D:/envpull_association_test/trend_summary --outdir D:/Oregon_Suitability/oregon_grid_1000m_env/suitability_maps_stressier2 --group-by matched_species_name --stress-strength 4 --stress-power 4 --stress-grace-frac 0

# Isolated species from larger dataset
#  py build_suitability_maps.py D:/envpull_association_test/occurrences_enriched.csv D:/Oregon_Suitability/oregon_grid_1000m_env/grid_with_env.csv --trend-summary D:/envpull_association_test/trend_summary --outdir D:/Oregon_Suitability/oregon_grid_1000m_env/suitability_maps_madrone --group-by matched_species_name --stress-strength 4 --stress-power 4 --stress-grace-frac 0 --include-taxa "species:Arbutus menziesii,species:Kopsiopsis strobilacea"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUMBERS = [f"{i:02d}" for i in range(1, 13)]
TERRACLIMATE_MONTHLY_RX = re.compile(r"^(terraclimate_[^_]+?)(?:_(\d{4}))?_m(0[1-9]|1[0-2])$")
TERRACLIMATE_AGG_RX = re.compile(r"^(terraclimate_[^_]+?)(?:_(\d{4}))?_(mean|sum|min|max)$")

DEFAULT_NUMERIC_PREFIXES = [
    "dem_",
    "twi_",
    "soilgrids_",
    "terraclimate_",
]
DEFAULT_CATEGORICAL_PATTERNS = [
    r"^glim_.*(?:IDENTITY_|Litho|xx)$",
    r"^mcd12q1_.*(?:label|class|type|name|desc)$",
    r"^mcd12q1$",
    r"^dem_continent$",
]
DEFAULT_CATEGORICAL_PREFIXES = [
    "glim_",
    "mcd12q1_",
]
DEFAULT_PREVIEW_SUITABILITY_VMAX = 0.65
DEFAULT_FAMILY_RELIABILITY_PRIORS = {
    "terraclimate": 1.00,
    "dem": 0.95,
    "twi": 0.90,
    "soilgrids": 0.75,
    "glim": 0.80,
    "mcd12q1": 0.60,
    "other": 0.80,
}
TAXON_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
STRESS_RULES: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"^terraclimate_vpd(?:_|$)"), "vpd", "high"),
    (re.compile(r"^terraclimate_pet(?:_|$)"), "atmospheric_demand", "high"),
    (re.compile(r"^terraclimate_tmax(?:_|$)"), "heat", "high"),
    (re.compile(r"^terraclimate_tmean(?:_|$)"), "heat", "high"),
    (re.compile(r"^terraclimate_tmmx(?:_|$)"), "heat", "high"),
    (re.compile(r"^terraclimate_ppt(?:_|$)"), "moisture_supply", "low"),
    (re.compile(r"^terraclimate_aet(?:_|$)"), "moisture_supply", "low"),
    (re.compile(r"^terraclimate_pdsi(?:_|$)"), "moisture_balance", "low"),
]


@dataclass
class TaxonSelector:
    raw: str
    rank: Optional[str]
    name: str
    name_norm: str


@dataclass
class NumericFeatureModel:
    column: str
    family: str
    weight: float
    min_v: float
    q25: float
    median: float
    q75: float
    max_v: float
    n_valid: int


@dataclass
class CategoricalFeatureModel:
    column: str
    family: str
    weight: float
    values_to_score: Dict[str, float]
    n_valid: int


@dataclass
class StressFeatureModel:
    column: str
    family: str
    axis: str
    direction: str
    weight: float
    min_v: float
    q25: float
    median: float
    q75: float
    max_v: float
    n_valid: int


@dataclass
class GroupModel:
    group: str
    numeric_features: List[NumericFeatureModel]
    categorical_features: List[CategoricalFeatureModel]
    stress_features: List[StressFeatureModel]
    occurrence_count: int
    taxonomy: Dict[str, str]
    redundancy_weight: float = 1.0


class CsvAppender:
    def __init__(self, path: Path, header: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = list(header)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.header)

    def write_rows(self, rows: Iterable[Sequence[object]]) -> None:
        for row in rows:
            self._writer.writerow(row)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class MultiAppender:
    def __init__(self):
        self._writers: Dict[str, CsvAppender] = {}

    def add(self, key: str, path: Path, header: Sequence[str]) -> CsvAppender:
        writer = CsvAppender(path, header)
        self._writers[key] = writer
        return writer

    def get(self, key: str) -> CsvAppender:
        return self._writers[key]

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "group"


def parse_family_reliability_priors(raw: str) -> Dict[str, float]:
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


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        text = ""
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_selector_list(raw_values: Optional[Sequence[str]]) -> List[TaxonSelector]:
    selectors: List[TaxonSelector] = []
    for raw in raw_values or []:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            rank = None
            name = part
            if ":" in part:
                maybe_rank, maybe_name = part.split(":", 1)
                mr = normalize_text(maybe_rank)
                if mr in TAXON_RANKS or mr in ("matched_species_name", "group"):
                    rank = mr
                    name = maybe_name.strip()
            selectors.append(
                TaxonSelector(
                    raw=part,
                    rank=rank,
                    name=name,
                    name_norm=normalize_text(name),
                )
            )
    return selectors


def maybe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_summary_frame(df: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
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
            "eta2_weight_global",
            "signal_to_iqr_weight_global",
            "pairwise_separation_weight_global",
            "family_prior_weight_global",
            "blended_weight_global",
            "family_weight_mean_normalized",
            "family_weight_sum_normalized",
            "sum_blended_weight",
            "mean_blended_weight",
            "family_prior",
            "pairwise_separation",
            "eta2",
            "signal_to_iqr",
        ]
    )
    if numeric_cols:
        numeric_candidates.update(numeric_cols)
    for col in out.columns:
        if col in numeric_candidates:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_terraclimate_monthly_column(name: object) -> Optional[Dict[str, object]]:
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


def parse_terraclimate_aggregate_column(name: object) -> Optional[Dict[str, object]]:
    m = TERRACLIMATE_AGG_RX.match(str(name))
    if not m:
        return None
    return {
        "variable": str(m.group(1)),
        "year": int(m.group(2)) if m.group(2) else None,
        "aggregate": str(m.group(3)),
        "canonical_column": f"{m.group(1)}_{m.group(3)}",
    }


def build_terraclimate_working_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    rows = []
    grouped: Dict[Tuple[str, str, str, Optional[int], Optional[str]], List[Tuple[int, str]]] = {}
    drop_cols = set()

    for col in df.columns:
        monthly = parse_terraclimate_monthly_column(col)
        if monthly is not None and monthly["year"] is not None:
            key = (
                str(monthly["canonical_column"]),
                str(monthly["variable"]),
                "monthly",
                int(monthly["month_num"]),
                None,
            )
            grouped.setdefault(key, []).append((int(monthly["year"]), str(col)))
            drop_cols.add(str(col))
            continue

        aggregate = parse_terraclimate_aggregate_column(col)
        if aggregate is not None and aggregate["year"] is not None:
            key = (
                str(aggregate["canonical_column"]),
                str(aggregate["variable"]),
                "aggregate",
                None,
                str(aggregate["aggregate"]),
            )
            grouped.setdefault(key, []).append((int(aggregate["year"]), str(col)))
            drop_cols.add(str(col))

    for (canonical_col, variable, feature_kind, month_num, aggregate_name), entries in sorted(grouped.items()):
        if not entries:
            continue
        entries = sorted(entries, key=lambda x: (x[0], x[1]))
        source_cols = [col for _, col in entries if col in work.columns]
        years = [int(year) for year, _ in entries]
        if not source_cols:
            continue
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
        meta = meta.sort_values(["variable", "feature_kind", "aggregate", "month_num", "canonical_column"], na_position="last").reset_index(drop=True)
    return work, meta


def get_numeric_family(name: str) -> str:
    for prefix in DEFAULT_NUMERIC_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_")
    if name.startswith("mcd12q1_"):
        return "mcd12q1"
    return "other"


def get_categorical_family(name: str) -> str:
    if name.startswith("glim_"):
        return "glim"
    if name.startswith("mcd12q1_") or name == "mcd12q1":
        return "mcd12q1"
    if name.startswith("dem_"):
        return "dem"
    return "other"


def matches_any_pattern(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pat, name) for pat in patterns)


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        if matches_any_pattern(str(col), DEFAULT_CATEGORICAL_PATTERNS):
            out.append(str(col))
            continue
        if any(str(col).startswith(prefix) for prefix in DEFAULT_CATEGORICAL_PREFIXES):
            out.append(str(col))
    return sorted(set(out))


def first_non_null(series: pd.Series) -> object:
    for value in series:
        if pd.notna(value) and str(value).strip() != "":
            return value
    return pd.NA


def infer_trend_summary_dir(occ_csv: Path, explicit_outdir: Optional[Path]) -> Path:
    if explicit_outdir is not None:
        return explicit_outdir
    return occ_csv.resolve().parent / "trend_summary"


def resolve_group_meta(df_occ: pd.DataFrame, group_by: str) -> pd.DataFrame:
    needed = []
    for col in [group_by, "matched_species_name", *TAXON_RANKS]:
        if col in df_occ.columns and col not in needed:
            needed.append(col)
    sub = df_occ[needed].copy()
    sub[group_by] = sub[group_by].astype("string").fillna(pd.NA).map(lambda x: "NA" if pd.isna(x) else str(x))
    grouped = sub.groupby(group_by, dropna=False, sort=True)
    rows = []
    for group, gdf in grouped:
        row = {"group": str(group), "occurrence_count": int(gdf.shape[0])}
        for col in needed:
            if col == group_by:
                continue
            row[col] = first_non_null(gdf[col]) if col in gdf.columns else pd.NA
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["group", "occurrence_count", *TAXON_RANKS, "matched_species_name"])
    for col in ["group", "matched_species_name", *TAXON_RANKS]:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def selector_matches_row(selector: TaxonSelector, row: pd.Series, group_by: str) -> bool:
    fields = {}
    for col in [group_by, "matched_species_name", *TAXON_RANKS]:
        if col in row.index:
            fields[col] = normalize_text(row[col])
    if selector.rank is not None:
        return selector.name_norm == fields.get(selector.rank, "")
    return any(selector.name_norm == v for v in fields.values() if v)


def apply_taxon_filters(group_meta: pd.DataFrame, include_selectors: Sequence[TaxonSelector], exclude_selectors: Sequence[TaxonSelector], group_by: str) -> pd.DataFrame:
    if group_meta.empty:
        return group_meta.copy()
    mask = pd.Series(True, index=group_meta.index)
    if include_selectors:
        include_hits = []
        for _, row in group_meta.iterrows():
            include_hits.append(any(selector_matches_row(sel, row, group_by) for sel in include_selectors))
        mask &= pd.Series(include_hits, index=group_meta.index)
    if exclude_selectors:
        exclude_hits = []
        for _, row in group_meta.iterrows():
            exclude_hits.append(any(selector_matches_row(sel, row, group_by) for sel in exclude_selectors))
        mask &= ~pd.Series(exclude_hits, index=group_meta.index)
    return group_meta.loc[mask].copy().reset_index(drop=True)


def load_summary_inputs(occ_csv: Path, trend_summary_dir: Path) -> Dict[str, pd.DataFrame]:
    by_group_numeric = normalize_summary_frame(maybe_read_csv(trend_summary_dir / "by_group_numeric_summary.csv"))
    by_group_categorical_path = trend_summary_dir / "by_group_categorical_frequencies.csv"
    by_group_categorical = normalize_summary_frame(maybe_read_csv(by_group_categorical_path)) if by_group_categorical_path.exists() else pd.DataFrame()
    feature_weights_path = trend_summary_dir / "feature_signal_weights.csv"
    family_weights_path = trend_summary_dir / "family_signal_weights.csv"
    feature_weights = normalize_summary_frame(maybe_read_csv(feature_weights_path)) if feature_weights_path.exists() else pd.DataFrame()
    family_weights = normalize_summary_frame(maybe_read_csv(family_weights_path)) if family_weights_path.exists() else pd.DataFrame()

    occ_df = maybe_read_csv(occ_csv)
    occ_work, terraclimate_meta = build_terraclimate_working_df(occ_df)
    return {
        "occ_df": occ_df,
        "occ_work": occ_work,
        "terraclimate_meta": terraclimate_meta,
        "by_group_numeric": by_group_numeric,
        "by_group_categorical": by_group_categorical,
        "feature_weights": feature_weights,
        "family_weights": family_weights,
    }


def build_family_weight_lookup(family_weights: pd.DataFrame, priors: Dict[str, float]) -> Dict[str, float]:
    lookup = dict((k, float(v)) for k, v in priors.items())
    if family_weights is not None and not family_weights.empty:
        for row in family_weights.itertuples(index=False):
            fam = str(getattr(row, "family", "other"))
            v = getattr(row, "family_weight_sum_normalized", np.nan)
            if pd.notna(v):
                lookup[fam] = float(v)
    return lookup


def build_feature_weight_lookup(feature_weights: pd.DataFrame, family_lookup: Dict[str, float]) -> Dict[str, float]:
    if feature_weights is None or feature_weights.empty:
        return {}
    out: Dict[str, float] = {}
    for row in feature_weights.itertuples(index=False):
        col = str(getattr(row, "column", ""))
        if not col:
            continue
        w = pd.to_numeric(getattr(row, "blended_weight_global", np.nan), errors="coerce")
        fam = str(getattr(row, "family", get_numeric_family(col)))
        if pd.notna(w) and float(w) > 0:
            out[col] = float(w)
        else:
            out[col] = float(family_lookup.get(fam, family_lookup.get("other", 0.8)))
    return out


def renormalize_numeric_weights(models: List[NumericFeatureModel]) -> List[NumericFeatureModel]:
    total = float(sum(max(0.0, m.weight) for m in models))
    if total <= 0:
        return models
    out = []
    for m in models:
        out.append(
            NumericFeatureModel(
                column=m.column,
                family=m.family,
                weight=float(m.weight) / total,
                min_v=m.min_v,
                q25=m.q25,
                median=m.median,
                q75=m.q75,
                max_v=m.max_v,
                n_valid=m.n_valid,
            )
        )
    return out


def renormalize_cat_weights(models: List[CategoricalFeatureModel]) -> List[CategoricalFeatureModel]:
    total = float(sum(max(0.0, m.weight) for m in models))
    if total <= 0:
        return models
    out = []
    for m in models:
        out.append(
            CategoricalFeatureModel(
                column=m.column,
                family=m.family,
                weight=float(m.weight) / total,
                values_to_score=m.values_to_score,
                n_valid=m.n_valid,
            )
        )
    return out


def renormalize_stress_weights(models: List[StressFeatureModel]) -> List[StressFeatureModel]:
    total = float(sum(max(0.0, m.weight) for m in models))
    if total <= 0:
        return models
    out = []
    for m in models:
        out.append(
            StressFeatureModel(
                column=m.column,
                family=m.family,
                axis=m.axis,
                direction=m.direction,
                weight=float(m.weight) / total,
                min_v=m.min_v,
                q25=m.q25,
                median=m.median,
                q75=m.q75,
                max_v=m.max_v,
                n_valid=m.n_valid,
            )
        )
    return out


def load_overlap_matrix(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return None
    if df.empty:
        return None
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)
    return df


def compute_redundancy_weights(groups: Sequence[str], overlap_matrix: Optional[pd.DataFrame], top_k: int) -> Dict[str, float]:
    if overlap_matrix is None or overlap_matrix.empty:
        return {g: 1.0 for g in groups}
    selected = [g for g in groups if g in overlap_matrix.index and g in overlap_matrix.columns]
    if not selected:
        return {g: 1.0 for g in groups}
    out: Dict[str, float] = {}
    for g in groups:
        if g not in selected:
            out[g] = 1.0
            continue
        vals = pd.to_numeric(overlap_matrix.loc[g, selected], errors="coerce")
        vals = vals.drop(labels=[g], errors="ignore")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            out[g] = 1.0
            continue
        vals = vals.sort_values(ascending=False)
        if top_k > 0:
            vals = vals.head(top_k)
        mean_overlap = float(vals.mean()) if not vals.empty else 0.0
        out[g] = 1.0 / max(1e-9, 1.0 + mean_overlap)
    series = pd.Series(out, dtype=float)
    mean_val = float(series.mean()) if len(series) else 1.0
    if mean_val > 0:
        series = series / mean_val
    return {str(k): float(v) for k, v in series.items()}


def classify_stress_feature(column: str) -> Optional[Tuple[str, str]]:
    for rx, axis, direction in STRESS_RULES:
        if rx.search(str(column)):
            return axis, direction
    return None


def build_group_models(
    selected_group_meta: pd.DataFrame,
    numeric_by_group: pd.DataFrame,
    categorical_by_group: pd.DataFrame,
    feature_weights: pd.DataFrame,
    family_weights: pd.DataFrame,
    grid_columns: Sequence[str],
    min_feature_n_valid: int,
    min_features_per_group: int,
    min_stress_features_per_group: int,
    categorical_share: float,
    family_priors: Dict[str, float],
    overlap_matrix: Optional[pd.DataFrame],
    redundancy_top_k: int,
) -> List[GroupModel]:
    grid_cols = set(str(c) for c in grid_columns)
    family_lookup = build_family_weight_lookup(family_weights, family_priors)
    feature_lookup = build_feature_weight_lookup(feature_weights, family_lookup)

    selected_groups = set(selected_group_meta["group"].astype(str))
    numeric_sub = numeric_by_group[numeric_by_group["group"].astype(str).isin(selected_groups)].copy()
    cat_sub = categorical_by_group[categorical_by_group["group"].astype(str).isin(selected_groups)].copy() if categorical_by_group is not None and not categorical_by_group.empty else pd.DataFrame()

    redundancy_lookup = compute_redundancy_weights(sorted(selected_groups), overlap_matrix, redundancy_top_k)
    models: List[GroupModel] = []

    for row in selected_group_meta.itertuples(index=False):
        group = str(getattr(row, "group"))
        occ_count = int(pd.to_numeric(getattr(row, "occurrence_count", 0), errors="coerce") or 0)
        taxonomy = {rank: ("" if pd.isna(getattr(row, rank, pd.NA)) else str(getattr(row, rank))) for rank in TAXON_RANKS}
        if hasattr(row, "matched_species_name"):
            taxonomy["matched_species_name"] = "" if pd.isna(getattr(row, "matched_species_name")) else str(getattr(row, "matched_species_name"))

        gnum = numeric_sub[numeric_sub["group"].astype(str) == group].copy()
        if not gnum.empty:
            gnum = gnum[gnum["column"].astype(str).isin(grid_cols)].copy()
            gnum = gnum[pd.to_numeric(gnum["n_valid"], errors="coerce").fillna(0) >= max(1, int(min_feature_n_valid))].copy()
        numeric_models: List[NumericFeatureModel] = []
        stress_models: List[StressFeatureModel] = []
        for r in gnum.itertuples(index=False):
            col = str(getattr(r, "column"))
            vals = [
                pd.to_numeric(getattr(r, "min", np.nan), errors="coerce"),
                pd.to_numeric(getattr(r, "q25", np.nan), errors="coerce"),
                pd.to_numeric(getattr(r, "median", np.nan), errors="coerce"),
                pd.to_numeric(getattr(r, "q75", np.nan), errors="coerce"),
                pd.to_numeric(getattr(r, "max", np.nan), errors="coerce"),
            ]
            if not all(pd.notna(v) for v in vals):
                continue
            weight = feature_lookup.get(col, float(family_lookup.get(get_numeric_family(col), family_lookup.get("other", 0.8))))
            fam = str(getattr(r, "family", get_numeric_family(col)))
            numeric_models.append(
                NumericFeatureModel(
                    column=col,
                    family=fam,
                    weight=float(weight),
                    min_v=float(vals[0]),
                    q25=float(vals[1]),
                    median=float(vals[2]),
                    q75=float(vals[3]),
                    max_v=float(vals[4]),
                    n_valid=int(pd.to_numeric(getattr(r, "n_valid", 0), errors="coerce") or 0),
                )
            )
            stress_info = classify_stress_feature(col)
            if stress_info is not None:
                axis, direction = stress_info
                stress_models.append(
                    StressFeatureModel(
                        column=col,
                        family=fam,
                        axis=axis,
                        direction=direction,
                        weight=float(weight),
                        min_v=float(vals[0]),
                        q25=float(vals[1]),
                        median=float(vals[2]),
                        q75=float(vals[3]),
                        max_v=float(vals[4]),
                        n_valid=int(pd.to_numeric(getattr(r, "n_valid", 0), errors="coerce") or 0),
                    )
                )
        numeric_models = renormalize_numeric_weights(numeric_models)
        stress_models = renormalize_stress_weights(stress_models)

        cat_models: List[CategoricalFeatureModel] = []
        if categorical_share > 0 and cat_sub is not None and not cat_sub.empty:
            gcat = cat_sub[cat_sub["group"].astype(str) == group].copy()
            if not gcat.empty:
                gcat = gcat[gcat["column"].astype(str).isin(grid_cols)].copy()
                gcat = gcat[pd.to_numeric(gcat["n_valid"], errors="coerce").fillna(0) >= max(1, int(min_feature_n_valid))].copy()
            if not gcat.empty:
                per_family_cols: Dict[str, List[str]] = defaultdict(list)
                for col in sorted(gcat["column"].dropna().astype(str).unique()):
                    fam = get_categorical_family(col)
                    per_family_cols[fam].append(col)
                for col, sub in gcat.groupby("column", sort=True):
                    fam = get_categorical_family(str(col))
                    fam_weight = float(family_lookup.get(fam, family_lookup.get("other", 0.8)))
                    denom = max(1, len(per_family_cols.get(fam, [])))
                    col_weight = fam_weight / float(denom)
                    value_map: Dict[str, float] = {}
                    for rr in sub.itertuples(index=False):
                        vv = normalize_text(getattr(rr, "value", ""))
                        if not vv:
                            continue
                        frac = pd.to_numeric(getattr(rr, "fraction_of_non_null", np.nan), errors="coerce")
                        if pd.notna(frac) and float(frac) > 0:
                            value_map[vv] = float(frac)
                    if not value_map:
                        continue
                    cat_models.append(
                        CategoricalFeatureModel(
                            column=str(col),
                            family=fam,
                            weight=float(col_weight),
                            values_to_score=value_map,
                            n_valid=int(pd.to_numeric(sub["n_valid"], errors="coerce").max() or 0),
                        )
                    )
                cat_models = renormalize_cat_weights(cat_models)

        total_features = len(numeric_models) + len(cat_models)
        if total_features < max(1, int(min_features_per_group)):
            continue
        if min_stress_features_per_group > 0 and len(stress_models) < int(min_stress_features_per_group):
            stress_models = []

        models.append(
            GroupModel(
                group=group,
                numeric_features=numeric_models,
                categorical_features=cat_models,
                stress_features=stress_models,
                occurrence_count=occ_count,
                taxonomy=taxonomy,
                redundancy_weight=float(redundancy_lookup.get(group, 1.0)),
            )
        )

    models.sort(key=lambda m: (-m.occurrence_count, m.group))
    return models


def trapezoid_score(values: np.ndarray, lo: float, q25: float, q75: float, hi: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros(x.shape, dtype=np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return out

    lo2 = float(lo)
    q25_2 = float(q25)
    q75_2 = float(q75)
    hi2 = float(hi)

    if not all(np.isfinite(v) for v in [lo2, q25_2, q75_2, hi2]):
        return out
    if hi2 < lo2:
        lo2, hi2 = hi2, lo2
    if q25_2 < lo2:
        q25_2 = lo2
    if q75_2 > hi2:
        q75_2 = hi2
    if q25_2 > q75_2:
        mid = 0.5 * (q25_2 + q75_2)
        q25_2 = mid
        q75_2 = mid

    xv = x[valid]
    score = np.zeros_like(xv, dtype=np.float32)

    if hi2 == lo2:
        score[:] = np.where(np.isclose(xv, hi2), 1.0, 0.0)
        out[valid] = score
        return out

    core_mask = (xv >= q25_2) & (xv <= q75_2)
    score[core_mask] = 1.0

    left_mask = (xv >= lo2) & (xv < q25_2)
    if np.any(left_mask):
        denom = max(q25_2 - lo2, 1e-12)
        score[left_mask] = ((xv[left_mask] - lo2) / denom).astype(np.float32)

    right_mask = (xv > q75_2) & (xv <= hi2)
    if np.any(right_mask):
        denom = max(hi2 - q75_2, 1e-12)
        score[right_mask] = ((hi2 - xv[right_mask]) / denom).astype(np.float32)

    score = np.clip(score, 0.0, 1.0)
    out[valid] = score
    return out


def reliability_numeric_score(values: np.ndarray, lo: float, q25: float, q75: float, hi: float, novelty_scale: float = 1.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return out

    lo2 = float(min(lo, hi))
    hi2 = float(max(lo, hi))
    span = max(hi2 - lo2, 1e-12)
    iqr = max(float(q75) - float(q25), 0.0)
    scale = max(iqr, 0.10 * span, 1e-12) * max(0.05, float(novelty_scale))

    xv = x[valid]
    dist = np.zeros_like(xv, dtype=float)
    below = xv < lo2
    above = xv > hi2
    dist[below] = lo2 - xv[below]
    dist[above] = xv[above] - hi2
    rel = 1.0 / (1.0 + (dist / scale))
    rel = np.clip(rel, 0.0, 1.0)
    out[valid] = rel.astype(np.float32)
    return out


def one_sided_stress_modifier(
    values: np.ndarray,
    direction: str,
    lo: float,
    q25: float,
    q75: float,
    hi: float,
    stress_power: float = 1.0,
    stress_strength: float = 0.45,
    stress_grace_frac: float = 0.15,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return out

    lo2 = float(min(lo, hi))
    hi2 = float(max(lo, hi))
    q25_2 = float(np.clip(q25, lo2, hi2))
    q75_2 = float(np.clip(q75, lo2, hi2))
    if q25_2 > q75_2:
        mid = 0.5 * (q25_2 + q75_2)
        q25_2 = mid
        q75_2 = mid

    xv = x[valid]
    score = np.ones_like(xv, dtype=np.float32)
    span = max(hi2 - lo2, 1e-12)
    grace = max(0.0, float(stress_grace_frac)) * span
    strength = float(np.clip(stress_strength, 0.0, 1.0))

    if direction == "high":
        start = min(hi2, q75_2 + grace)
        stressed = xv > start
        if np.any(stressed):
            denom = max(hi2 - start, max(0.10 * span, 1e-12))
            frac = (xv[stressed] - start) / denom
            frac = np.clip(frac, 0.0, 1.0)
            raw_penalty = np.power(1.0 - frac, max(0.05, float(stress_power))).astype(np.float32)
            score[stressed] = (1.0 - strength * (1.0 - raw_penalty)).astype(np.float32)
        score[xv >= hi2] = np.float32(max(0.0, 1.0 - strength))
    else:
        start = max(lo2, q25_2 - grace)
        stressed = xv < start
        if np.any(stressed):
            denom = max(start - lo2, max(0.10 * span, 1e-12))
            frac = (start - xv[stressed]) / denom
            frac = np.clip(frac, 0.0, 1.0)
            raw_penalty = np.power(1.0 - frac, max(0.05, float(stress_power))).astype(np.float32)
            score[stressed] = (1.0 - strength * (1.0 - raw_penalty)).astype(np.float32)
        score[xv <= lo2] = np.float32(max(0.0, 1.0 - strength))
    out[valid] = np.clip(score, max(0.0, 1.0 - strength), 1.0)
    return out


def categorical_match_score(series: pd.Series, value_scores: Dict[str, float]) -> np.ndarray:
    values = series.astype("string").fillna(pd.NA)
    norm = values.map(lambda x: normalize_text(x) if pd.notna(x) else "")
    out = norm.map(lambda x: float(value_scores.get(x, 0.0)) if x else np.nan).to_numpy(dtype=float)
    return out.astype(np.float32, copy=False)


def categorical_reliability_score(series: pd.Series, value_scores: Dict[str, float]) -> np.ndarray:
    values = series.astype("string").fillna(pd.NA)
    norm = values.map(lambda x: normalize_text(x) if pd.notna(x) else "")
    out = norm.map(lambda x: 1.0 if x and x in value_scores else (np.nan if not x else 0.0)).to_numpy(dtype=float)
    return out.astype(np.float32, copy=False)


def transform_similarity_scores(scores: np.ndarray, sharpness: float = 1.0, floor: float = 0.0) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    power = max(0.05, float(sharpness))
    floor2 = float(np.clip(floor, 0.0, 0.95))
    work = np.clip(arr[valid], 0.0, 1.0)
    if power != 1.0:
        work = np.power(work, power)
    if floor2 > 0.0:
        work = floor2 + (1.0 - floor2) * work
    out[valid] = np.clip(work, 0.0, 1.0).astype(np.float32)
    return out


def weighted_geometric_finalize(log_num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(log_num.shape, np.nan, dtype=np.float32)
    valid = den > 0
    if not np.any(valid):
        return out
    out[valid] = np.exp(log_num[valid] / den[valid]).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def reliability_to_adjustment_factor(reliability: np.ndarray, power: float = 0.35, floor: float = 0.75) -> np.ndarray:
    rel = np.asarray(reliability, dtype=float)
    out = np.ones(rel.shape, dtype=np.float32)
    valid = np.isfinite(rel)
    if not np.any(valid):
        return out
    power2 = max(0.05, float(power))
    floor2 = float(np.clip(floor, 0.0, 1.0))
    work = np.clip(rel[valid], 0.0, 1.0)
    if power2 != 1.0:
        work = np.power(work, power2)
    if floor2 > 0.0:
        work = floor2 + (1.0 - floor2) * work
    out[valid] = np.clip(work, 0.0, 1.0).astype(np.float32)
    return out


def combine_numeric_and_categorical(
    numeric_score: np.ndarray,
    cat_score: np.ndarray,
    numeric_weight_present: np.ndarray,
    cat_weight_present: np.ndarray,
    categorical_share: float,
    combine_floor: float = 0.0,
) -> np.ndarray:
    result = np.full(numeric_score.shape, np.nan, dtype=np.float32)
    has_num = numeric_weight_present > 0
    has_cat = cat_weight_present > 0

    only_num = has_num & ~has_cat
    only_cat = has_cat & ~has_num
    both = has_num & has_cat

    result[only_num] = numeric_score[only_num]
    result[only_cat] = cat_score[only_cat]
    if np.any(both):
        mix = float(max(0.0, min(1.0, categorical_share)))
        floor2 = float(np.clip(combine_floor, 0.0, 0.95))
        eps = max(1e-12, floor2 if floor2 > 0.0 else 1e-12)
        num = np.clip(numeric_score[both], eps, 1.0)
        cat = np.clip(cat_score[both], eps, 1.0)
        result[both] = np.exp((1.0 - mix) * np.log(num) + mix * np.log(cat)).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def score_group_components_on_chunk(
    chunk: pd.DataFrame,
    model: GroupModel,
    categorical_share: float,
    stress_power: float,
    stress_strength: float,
    stress_grace_frac: float,
    novelty_scale: float,
    score_sharpness: float,
    score_floor: float,
    reliability_power: float,
    reliability_floor: float,
    use_stress_adjustment: bool,
    use_reliability_adjustment: bool,
) -> Dict[str, np.ndarray]:
    n = len(chunk)
    numeric_log_num = np.zeros(n, dtype=np.float32)
    numeric_den = np.zeros(n, dtype=np.float32)
    cat_log_num = np.zeros(n, dtype=np.float32)
    cat_den = np.zeros(n, dtype=np.float32)
    rel_num = np.zeros(n, dtype=np.float32)
    rel_den = np.zeros(n, dtype=np.float32)
    stress_num = np.zeros(n, dtype=np.float32)
    stress_den = np.zeros(n, dtype=np.float32)

    for feat in model.numeric_features:
        if feat.column not in chunk.columns:
            continue
        vals = pd.to_numeric(chunk[feat.column], errors="coerce").to_numpy(dtype=float, copy=False)
        valid = np.isfinite(vals)
        w = float(feat.weight)
        if w <= 0:
            continue
        core_scores = trapezoid_score(vals, feat.min_v, feat.q25, feat.q75, feat.max_v)
        core_scores = transform_similarity_scores(core_scores, score_sharpness, score_floor)
        rel_scores = reliability_numeric_score(vals, feat.min_v, feat.q25, feat.q75, feat.max_v, novelty_scale)
        numeric_log_num[valid] += (np.log(np.clip(core_scores[valid], 1e-12, 1.0)) * w).astype(np.float32)
        numeric_den[valid] += w
        rel_valid = np.isfinite(rel_scores)
        rel_num[rel_valid] += rel_scores[rel_valid] * w
        rel_den[rel_valid] += w

    for feat in model.categorical_features:
        if feat.column not in chunk.columns:
            continue
        w = float(feat.weight)
        if w <= 0:
            continue
        cat_scores = categorical_match_score(chunk[feat.column], feat.values_to_score)
        cat_scores = transform_similarity_scores(cat_scores, score_sharpness, score_floor)
        cat_valid = np.isfinite(cat_scores)
        cat_log_num[cat_valid] += (np.log(np.clip(cat_scores[cat_valid], 1e-12, 1.0)) * w).astype(np.float32)
        cat_den[cat_valid] += w
        rel_scores = categorical_reliability_score(chunk[feat.column], feat.values_to_score)
        rel_valid = np.isfinite(rel_scores)
        rel_num[rel_valid] += rel_scores[rel_valid] * w
        rel_den[rel_valid] += w

    if use_stress_adjustment and model.stress_features:
        for feat in model.stress_features:
            if feat.column not in chunk.columns:
                continue
            vals = pd.to_numeric(chunk[feat.column], errors="coerce").to_numpy(dtype=float, copy=False)
            scores = one_sided_stress_modifier(vals, feat.direction, feat.min_v, feat.q25, feat.q75, feat.max_v, stress_power, stress_strength, stress_grace_frac)
            valid = np.isfinite(scores)
            w = float(feat.weight)
            if w <= 0:
                continue
            stress_num[valid] += scores[valid] * w
            stress_den[valid] += w

    numeric_score = weighted_geometric_finalize(numeric_log_num, numeric_den)
    cat_score = weighted_geometric_finalize(cat_log_num, cat_den)
    reliability = np.full(n, np.nan, dtype=np.float32)
    reliability_factor = np.ones(n, dtype=np.float32)
    stress_modifier = np.full(n, 1.0, dtype=np.float32)

    rel_valid = rel_den > 0
    stress_valid = stress_den > 0

    reliability[rel_valid] = rel_num[rel_valid] / rel_den[rel_valid]
    if np.any(stress_valid):
        stress_modifier[stress_valid] = stress_num[stress_valid] / stress_den[stress_valid]

    core_score = combine_numeric_and_categorical(numeric_score, cat_score, numeric_den, cat_den, categorical_share, combine_floor=score_floor)

    if not use_stress_adjustment:
        stress_modifier[:] = 1.0
    else:
        stress_modifier = np.where(np.isfinite(stress_modifier), np.clip(stress_modifier, 0.0, 1.0), 1.0).astype(np.float32)

    if not use_reliability_adjustment:
        reliability_factor[:] = 1.0
    else:
        reliability = np.where(np.isfinite(reliability), np.clip(reliability, 0.0, 1.0), np.nan).astype(np.float32)
        reliability_factor = reliability_to_adjustment_factor(reliability, reliability_power, reliability_floor).astype(np.float32, copy=False)

    adjusted = np.full(n, np.nan, dtype=np.float32)
    core_valid = np.isfinite(core_score)
    adjusted[core_valid] = core_score[core_valid] * stress_modifier[core_valid] * reliability_factor[core_valid]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    return {
        "core": core_score.astype(np.float32, copy=False),
        "stress_modifier": stress_modifier.astype(np.float32, copy=False),
        "reliability": reliability.astype(np.float32, copy=False),
        "reliability_factor": reliability_factor.astype(np.float32, copy=False),
        "adjusted": adjusted.astype(np.float32, copy=False),
    }


def _estimate_regular_step(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    diffs = np.diff(np.sort(np.unique(values)))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def _grid_edges(values: np.ndarray, step: float) -> np.ndarray:
    vals = np.sort(np.unique(values))
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if vals.size == 1 or not np.isfinite(step) or step <= 0:
        half = 0.5
        return np.array([vals[0] - half, vals[0] + half], dtype=float)
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = (vals[:-1] + vals[1:]) * 0.5
    edges[0] = vals[0] - step * 0.5
    edges[-1] = vals[-1] + step * 0.5
    return edges


def _try_regular_grid(sub: pd.DataFrame, x_col: str, y_col: str, value_col: str):
    x_vals = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
    z_vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
    if not np.any(ok):
        return None
    x_vals = x_vals[ok]
    y_vals = y_vals[ok]
    z_vals = z_vals[ok]

    n = x_vals.size
    if n < 4:
        return None

    max_cells = min(max(n * 4, 4096), 4_000_000)
    decimals_to_try = (8, 7, 6, 5, 4, 3)

    for decimals in decimals_to_try:
        xr = np.round(x_vals, decimals)
        yr = np.round(y_vals, decimals)

        xs = np.sort(np.unique(xr))
        ys = np.sort(np.unique(yr))
        if xs.size < 2 or ys.size < 2:
            continue

        cell_count = int(xs.size) * int(ys.size)
        if cell_count <= 0 or cell_count > max_cells:
            continue

        dx = _estimate_regular_step(xs)
        dy = _estimate_regular_step(ys)
        if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
            continue

        x_index = {float(v): i for i, v in enumerate(xs)}
        y_index = {float(v): i for i, v in enumerate(ys)}
        grid = np.full((ys.size, xs.size), np.nan, dtype=np.float32)
        counts = np.zeros((ys.size, xs.size), dtype=np.uint16)

        for xv, yv, zv in zip(xr, yr, z_vals):
            ix = x_index.get(float(xv))
            iy = y_index.get(float(yv))
            if ix is None or iy is None:
                continue
            if np.isnan(grid[iy, ix]):
                grid[iy, ix] = np.float32(zv)
                counts[iy, ix] = 1
            else:
                c = int(counts[iy, ix])
                grid[iy, ix] = np.float32((float(grid[iy, ix]) * c + float(zv)) / float(c + 1))
                counts[iy, ix] = min(c + 1, np.iinfo(np.uint16).max)

        filled_fraction = float(np.isfinite(grid).mean())
        if filled_fraction < 0.25:
            continue

        x_edges = _grid_edges(xs, dx)
        y_edges = _grid_edges(ys, dy)
        return xs, ys, grid, x_edges, y_edges

    return None


def _coarsen_regular_grid(grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, factor: int):
    factor = max(1, int(factor))
    if factor <= 1:
        return grid, x_edges, y_edges
    h, w = grid.shape
    out_h = int(math.ceil(h / factor))
    out_w = int(math.ceil(w / factor))
    coarse = np.full((out_h, out_w), np.nan, dtype=np.float32)
    for oy in range(out_h):
        y0 = oy * factor
        y1 = min(h, (oy + 1) * factor)
        for ox in range(out_w):
            x0 = ox * factor
            x1 = min(w, (ox + 1) * factor)
            block = grid[y0:y1, x0:x1]
            if np.isfinite(block).any():
                coarse[oy, ox] = np.float32(np.nanmean(block))
    x_edges2 = x_edges[::factor].copy()
    if x_edges2[-1] != x_edges[-1]:
        x_edges2 = np.concatenate([x_edges2, [x_edges[-1]]])
    y_edges2 = y_edges[::factor].copy()
    if y_edges2[-1] != y_edges[-1]:
        y_edges2 = np.concatenate([y_edges2, [y_edges[-1]]])
    if coarse.shape[1] + 1 != len(x_edges2):
        x_edges2 = np.linspace(x_edges[0], x_edges[-1], coarse.shape[1] + 1)
    if coarse.shape[0] + 1 != len(y_edges2):
        y_edges2 = np.linspace(y_edges[0], y_edges[-1], coarse.shape[0] + 1)
    return coarse, x_edges2, y_edges2


def preview_color_limits(value_col: str, title: str, suitability_vmax: Optional[float] = DEFAULT_PREVIEW_SUITABILITY_VMAX) -> Tuple[Optional[float], Optional[float]]:
    value_key = str(value_col).strip().lower()
    title_key = str(title).strip().lower()
    joined = f"{value_key} {title_key}"
    if "richness" in joined:
        return None, None
    if "reliability" in joined:
        return 0.0, 1.0
    suitability_tokens = (
        "suitability",
        "_score",
        "score_",
        "core",
        "adjusted",
        "standardized",
    )
    if any(token in joined for token in suitability_tokens):
        vmax = pd.to_numeric(suitability_vmax, errors="coerce")
        if pd.notna(vmax) and float(vmax) > 0:
            return 0.0, float(vmax)
        return None, None
    return None, None


def lower_tail_weighted_mean(values: np.ndarray, valid: np.ndarray, tail_fraction: float = 0.25, rank_power: float = 1.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("values must be 2D with shape [n_groups, n_rows]")

    n_groups, n_rows = arr.shape
    out = np.full(n_rows, np.nan, dtype=np.float32)
    if n_groups == 0 or n_rows == 0:
        return out

    k = max(1, int(math.ceil(n_groups * max(0.0, min(1.0, float(tail_fraction))))))
    k = min(k, n_groups)

    safe = np.where(ok, arr, np.inf).astype(np.float32, copy=False)
    tail = np.partition(safe, kth=k - 1, axis=0)[:k]
    tail.sort(axis=0)

    finite_tail = np.isfinite(tail)
    if not np.any(finite_tail):
        return out

    ranks = np.arange(1, k + 1, dtype=np.float32)[:, None]
    weights = 1.0 / np.power(ranks, max(0.05, float(rank_power)))
    weights = np.where(finite_tail, weights, 0.0)

    denom = weights.sum(axis=0)
    numer = np.where(finite_tail, tail * weights, 0.0).sum(axis=0)
    good = denom > 0
    out[good] = (numer[good] / denom[good]).astype(np.float32)
    return out


def joint_support_score(values: np.ndarray, valid: np.ndarray, min_share: float = 0.7, tail_fraction: float = 0.25, rank_power: float = 1.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool)

    safe = np.where(ok, arr, np.inf)
    joint_min = np.min(safe, axis=0)
    joint_min = np.where(np.isfinite(joint_min), joint_min, np.nan).astype(np.float32)

    tail_mean = lower_tail_weighted_mean(arr, ok, tail_fraction=tail_fraction, rank_power=rank_power)

    alpha = float(np.clip(min_share, 0.0, 1.0))
    out = np.full(joint_min.shape, np.nan, dtype=np.float32)
    good = np.isfinite(joint_min) & np.isfinite(tail_mean)
    out[good] = (alpha * joint_min[good] + (1.0 - alpha) * tail_mean[good]).astype(np.float32)

    only_min = np.isfinite(joint_min) & ~np.isfinite(tail_mean)
    out[only_min] = joint_min[only_min]
    only_tail = ~np.isfinite(joint_min) & np.isfinite(tail_mean)
    out[only_tail] = tail_mean[only_tail]

    return np.clip(out, 0.0, 1.0)


def finite_min_score(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool)
    safe = np.where(ok, arr, np.inf)
    out = np.min(safe, axis=0)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def preview_point_map(df: pd.DataFrame, value_col: str, title: str, out_path: Path, x_col: str, y_col: str, point_size: float = 4.0, point_alpha: float = 0.35, preview_coarsen: int = 4, suitability_vmax: Optional[float] = DEFAULT_PREVIEW_SUITABILITY_VMAX) -> None:
    if plt is None:
        return
    if value_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return
    sub = df[[x_col, y_col, value_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col, value_col])
    if sub.empty:
        return

    vmin, vmax = preview_color_limits(value_col, title, suitability_vmax=suitability_vmax)
    colorbar_extend = "neither"
    if vmax is not None and np.nanmax(sub[value_col].to_numpy(dtype=np.float32, copy=False)) > float(vmax):
        colorbar_extend = "max"

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    grid_payload = _try_regular_grid(sub, x_col, y_col, value_col)
    if grid_payload is not None:
        _, _, grid, x_edges, y_edges = grid_payload
        grid, x_edges, y_edges = _coarsen_regular_grid(grid, x_edges, y_edges, preview_coarsen)
        masked = np.ma.masked_invalid(grid)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            masked,
            shading="flat",
            cmap="viridis",
            antialiased=False,
            linewidth=0,
            vmin=vmin,
            vmax=vmax,
        )
        color_obj = mesh
    else:
        cmap = plt.get_cmap("viridis")
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ax.scatter(
            sub[x_col],
            sub[y_col],
            c=sub[value_col],
            s=point_size,
            alpha=float(np.clip(point_alpha, 0.02, 1.0)),
            linewidths=0,
            rasterized=True,
            cmap=cmap,
            norm=norm,
        )
        color_obj = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        color_obj.set_array([])

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.colorbar(color_obj, ax=ax, shrink=0.78, extend=colorbar_extend)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_manifest(out_path: Path, payload: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_group_order(models: Sequence[GroupModel], per_species_top_n: int) -> List[GroupModel]:
    out = list(models)
    if per_species_top_n > 0 and len(out) > per_species_top_n:
        out = out[:per_species_top_n]
    return out


def score_to_hist_bin(values: np.ndarray, bins: int) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    idx = np.floor(clipped * bins).astype(np.int32)
    idx[idx >= bins] = bins - 1
    idx[idx < 0] = 0
    return idx


def histogram_cdf_lookup(hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hist = np.asarray(hist, dtype=np.int64)
    total = int(hist.sum())
    if total <= 0:
        centers = (np.arange(hist.shape[0], dtype=np.float32) + 0.5) / float(hist.shape[0])
        return centers, centers
    csum = np.cumsum(hist, dtype=np.int64)
    cdf = csum.astype(np.float64) / float(total)
    centers = (np.arange(hist.shape[0], dtype=np.float32) + 0.5) / float(hist.shape[0])
    return centers, cdf.astype(np.float32)


def standardize_from_histogram(values: np.ndarray, centers: np.ndarray, cdf: np.ndarray, bins: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        return out
    idx = score_to_hist_bin(values[valid], bins)
    out[valid] = cdf[idx]
    return out


def append_histogram(hist: np.ndarray, values: np.ndarray, bins: int) -> None:
    valid = np.isfinite(values)
    if not np.any(valid):
        return
    idx = score_to_hist_bin(values[valid], bins)
    counts = np.bincount(idx, minlength=bins)
    hist[: len(counts)] += counts.astype(hist.dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build generalized envelope-based suitability maps with core fit, stress adjustment, reliability, and standardized stacked outputs.")
    ap.add_argument("occurrences_csv", help="Cleaned enriched occurrence CSV, usually occurrences_enriched.cleaned.csv")
    ap.add_argument("grid_csv", help="Cleaned enriched grid CSV, usually grid_with_env.csv")
    ap.add_argument("--trend-summary", default=None, help="Directory with aggregate_occurrence_trends.py outputs. Defaults to <occurrences_csv_dir>/trend_summary")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--group-by", default="matched_species_name", help="Grouping column used in the trend summary, usually matched_species_name")
    ap.add_argument("--include-taxa", nargs="*", default=None, help="Optional rank selectors such as family:Rosaceae genus:Ribes species:Daucus pusillus")
    ap.add_argument("--exclude-taxa", nargs="*", default=None, help="Optional rank selectors to exclude")
    ap.add_argument("--aggregate-ranks", nargs="*", default=["family", "genus"], help="Optional aggregate taxon ranks to emit, such as family genus")
    ap.add_argument("--id-col", default="id", help="Grid id column")
    ap.add_argument("--x-col", default="lon", help="Grid x column")
    ap.add_argument("--y-col", default="lat", help="Grid y column")
    ap.add_argument("--grid-chunk-size", type=int, default=100000, help="Rows per grid chunk")
    ap.add_argument("--min-feature-n-valid", type=int, default=10, help="Minimum per-group non-null values required to keep a feature")
    ap.add_argument("--min-features-per-group", type=int, default=8, help="Minimum active features required to keep a group model")
    ap.add_argument("--min-stress-features-per-group", type=int, default=0, help="Optional minimum number of curated stress features required before applying stress adjustment to a group")
    ap.add_argument("--min-occurrences-per-group", type=int, default=10, help="Minimum occurrence rows required to keep a group")
    ap.add_argument("--max-groups", type=int, default=0, help="Optional maximum number of selected groups to keep after filtering; 0 keeps all")
    ap.add_argument("--categorical-share", type=float, default=0.15, help="Final core score share reserved for categorical features")
    ap.add_argument("--richness-threshold", type=float, default=0.75, help="Threshold used for richness counting and for interpreting cells as likely enough to count")
    ap.add_argument("--family-reliability-priors", default="terraclimate=1.0,dem=0.95,twi=0.9,soilgrids=0.75,glim=0.8,mcd12q1=0.6,other=0.8")
    ap.add_argument("--redundancy-top-k", type=int, default=5, help="Top K overlap neighbors used for redundancy penalty")
    ap.add_argument("--hist-bins", type=int, default=512, help="Histogram bins used for within-species score standardization")
    ap.add_argument("--stress-power", type=float, default=1.0, help="Exponent controlling how quickly one-sided stress penalties decline")
    ap.add_argument("--stress-strength", type=float, default=0.45, help="Maximum share of suitability removed by the generalized stress modifier at the most extreme observed edge")
    ap.add_argument("--stress-grace-frac", type=float, default=0.15, help="Fraction of the observed feature span added beyond q25 or q75 before stress penalties begin")
    ap.add_argument("--novelty-scale", type=float, default=1.0, help="Scale factor controlling how quickly reliability declines outside observed support")
    ap.add_argument("--score-sharpness", type=float, default=2.0, help="Exponent applied to per-feature similarity before geometric aggregation; values above 1 make middling fits fall faster")
    ap.add_argument("--score-floor", type=float, default=0.02, help="Minimum transformed per-feature similarity retained during geometric aggregation so one weak feature penalizes strongly without zeroing the whole score")
    ap.add_argument("--reliability-power", type=float, default=0.35, help="Exponent applied to raw reliability before converting it to a mild adjustment factor")
    ap.add_argument("--reliability-floor", type=float, default=0.85, help="Minimum multiplicative influence of reliability on adjusted suitability so reliability stays a garnish instead of the main driver")
    ap.add_argument("--preview-coarsen", type=int, default=2, help="Display-only raster coarsening factor for preview PNGs; 2 makes cells about twice as large")
    ap.add_argument("--preview-point-alpha", type=float, default=0.35, help="Scatter-point alpha for preview PNGs when they are rendered as points instead of a regular grid")
    ap.add_argument("--preview-suitability-vmax", type=float, default=DEFAULT_PREVIEW_SUITABILITY_VMAX, help="Upper color limit used for suitability-style preview maps; set <= 0 to disable fixed scaling for those previews")
    ap.add_argument("--joint-min-share", type=float, default=0.7, help="Share of the joint aggregate anchored to the per-cell minimum across contributing groups")
    ap.add_argument("--joint-tail-fraction", type=float, default=0.25, help="Fraction of the lowest-scoring groups blended into the joint aggregate at each cell")
    ap.add_argument("--joint-rank-power", type=float, default=1.5, help="Rank-decay exponent used when averaging the lower tail for joint aggregate outputs")
    ap.add_argument("--no-stress-adjustment", action="store_true", help="Disable curated climate stress adjustment")
    ap.add_argument("--no-reliability-adjustment", action="store_true", help="Disable novelty/reliability adjustment")
    ap.add_argument("--no-per-species", action="store_true", help="Skip writing one CSV per selected group")
    ap.add_argument("--per-species-top-n", type=int, default=0, help="Write per-species outputs only for the top N selected groups; 0 keeps all")
    ap.add_argument("--preview-top-n", type=int, default=8, help="Generate quick preview PNGs for the top N per-species maps by mean adjusted suitability; 0 disables")
    args = ap.parse_args()

    occurrences_csv = Path(args.occurrences_csv).resolve()
    grid_csv = Path(args.grid_csv).resolve()
    trend_summary_dir = infer_trend_summary_dir(occurrences_csv, Path(args.trend_summary).resolve() if args.trend_summary else None)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not occurrences_csv.exists():
        raise FileNotFoundError(f"Missing occurrences_csv: {occurrences_csv}")
    if not grid_csv.exists():
        raise FileNotFoundError(f"Missing grid_csv: {grid_csv}")
    if not trend_summary_dir.exists():
        raise FileNotFoundError(f"Missing trend summary directory: {trend_summary_dir}")

    include_selectors = parse_selector_list(args.include_taxa)
    exclude_selectors = parse_selector_list(args.exclude_taxa)
    family_priors = parse_family_reliability_priors(args.family_reliability_priors)

    log(f"[load] occurrences={occurrences_csv}")
    log(f"[load] grid={grid_csv}")
    log(f"[load] trend_summary={trend_summary_dir}")
    inputs = load_summary_inputs(occurrences_csv, trend_summary_dir)

    occ_df = inputs["occ_df"]
    numeric_by_group = inputs["by_group_numeric"]
    categorical_by_group = inputs["by_group_categorical"]
    feature_weights = inputs["feature_weights"]
    family_weights = inputs["family_weights"]

    if args.group_by not in occ_df.columns:
        raise KeyError(f"group-by column not found in occurrences CSV: {args.group_by}")
    if numeric_by_group.empty:
        raise RuntimeError(f"Missing or empty by_group_numeric_summary.csv under {trend_summary_dir}")

    group_meta = resolve_group_meta(occ_df, args.group_by)
    group_meta = apply_taxon_filters(group_meta, include_selectors, exclude_selectors, args.group_by)
    if args.min_occurrences_per_group > 0:
        group_meta = group_meta[pd.to_numeric(group_meta["occurrence_count"], errors="coerce").fillna(0) >= int(args.min_occurrences_per_group)].copy()
    group_meta = group_meta[group_meta["group"].astype(str).isin(set(numeric_by_group["group"].dropna().astype(str)))].copy()
    group_meta = group_meta.sort_values(["occurrence_count", "group"], ascending=[False, True]).reset_index(drop=True)
    if args.max_groups > 0 and len(group_meta) > args.max_groups:
        group_meta = group_meta.head(args.max_groups).copy()

    if group_meta.empty:
        raise RuntimeError("No groups remain after taxon filtering and occurrence thresholds")

    grid_head = pd.read_csv(grid_csv, nrows=2000, low_memory=False)
    grid_work_head, grid_tc_meta = build_terraclimate_working_df(grid_head)
    grid_columns = list(grid_work_head.columns)

    overlap_matrix = load_overlap_matrix(trend_summary_dir / "overlaps" / "species_profile_blended_overlap.csv")
    models = build_group_models(
        selected_group_meta=group_meta,
        numeric_by_group=numeric_by_group,
        categorical_by_group=categorical_by_group,
        feature_weights=feature_weights,
        family_weights=family_weights,
        grid_columns=grid_columns,
        min_feature_n_valid=args.min_feature_n_valid,
        min_features_per_group=args.min_features_per_group,
        min_stress_features_per_group=args.min_stress_features_per_group,
        categorical_share=args.categorical_share,
        family_priors=family_priors,
        overlap_matrix=overlap_matrix,
        redundancy_top_k=args.redundancy_top_k,
    )
    if not models:
        raise RuntimeError("No valid group models could be built after intersecting summary features with the grid columns")

    selected_groups_df = pd.DataFrame(
        [
            {
                "group": m.group,
                "occurrence_count": m.occurrence_count,
                "numeric_feature_count": len(m.numeric_features),
                "categorical_feature_count": len(m.categorical_features),
                "stress_feature_count": len(m.stress_features),
                "redundancy_weight": m.redundancy_weight,
                **m.taxonomy,
            }
            for m in models
        ]
    )
    selected_groups_df.to_csv(outdir / "selected_groups.csv", index=False)

    numeric_model_rows = []
    categorical_model_rows = []
    stress_model_rows = []
    for m in models:
        for feat in m.numeric_features:
            numeric_model_rows.append(
                {
                    "group": m.group,
                    "column": feat.column,
                    "family": feat.family,
                    "weight": feat.weight,
                    "n_valid": feat.n_valid,
                    "min": feat.min_v,
                    "q25": feat.q25,
                    "median": feat.median,
                    "q75": feat.q75,
                    "max": feat.max_v,
                }
            )
        for feat in m.categorical_features:
            for value, value_score in sorted(feat.values_to_score.items()):
                categorical_model_rows.append(
                    {
                        "group": m.group,
                        "column": feat.column,
                        "family": feat.family,
                        "weight": feat.weight,
                        "n_valid": feat.n_valid,
                        "value": value,
                        "value_score": value_score,
                    }
                )
        for feat in m.stress_features:
            stress_model_rows.append(
                {
                    "group": m.group,
                    "column": feat.column,
                    "family": feat.family,
                    "axis": feat.axis,
                    "direction": feat.direction,
                    "weight": feat.weight,
                    "n_valid": feat.n_valid,
                    "min": feat.min_v,
                    "q25": feat.q25,
                    "median": feat.median,
                    "q75": feat.q75,
                    "max": feat.max_v,
                }
            )
    pd.DataFrame(numeric_model_rows).to_csv(outdir / "numeric_envelopes_selected.csv", index=False)
    pd.DataFrame(categorical_model_rows).to_csv(outdir / "categorical_envelopes_selected.csv", index=False)
    pd.DataFrame(stress_model_rows).to_csv(outdir / "stress_envelopes_selected.csv", index=False)

    per_species_models = resolve_group_order(models, args.per_species_top_n)

    group_to_rank_value: Dict[str, Dict[str, str]] = {}
    for m in models:
        group_to_rank_value[m.group] = {}
        for rank in args.aggregate_ranks or []:
            rank_key = str(rank).strip().lower()
            if rank_key not in TAXON_RANKS and rank_key not in ("matched_species_name",):
                continue
            value = str(m.taxonomy.get(rank_key, "") or "").strip()
            if value:
                group_to_rank_value[m.group][rank_key] = value

    rank_members_raw: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for m in models:
        for rank_key, value in group_to_rank_value.get(m.group, {}).items():
            rank_members_raw[rank_key][value].append(m.group)

    rank_members: Dict[str, Dict[str, List[str]]] = {}
    for rank_key, member_map in rank_members_raw.items():
        kept = {value: groups for value, groups in member_map.items() if len(groups) > 1}
        if kept:
            rank_members[rank_key] = kept

    species_index = {m.group: i for i, m in enumerate(models)}
    group_names = [m.group for m in models]
    hist_bins = max(32, int(args.hist_bins))
    adjusted_hist = np.zeros((len(models), hist_bins), dtype=np.int64)
    summary_rows: List[Dict[str, object]] = []
    species_sums = defaultdict(float)
    species_counts = defaultdict(int)
    species_core_sums = defaultdict(float)
    species_rel_sums = defaultdict(float)
    species_stress_sums = defaultdict(float)
    species_likely_counts = defaultdict(int)
    species_top_adjusted = defaultdict(float)
    preview_cache_species: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    pass1_writers = MultiAppender()
    if not args.no_per_species:
        for m in per_species_models:
            pass1_writers.add(
                f"species::{m.group}",
                outdir / "by_species" / f"{safe_slug(m.group)}.csv",
                [args.id_col, args.x_col, args.y_col, "core_score", "stress_modifier", "reliability", "reliability_factor", "adjusted_score", "likely_adjusted"],
            )

    log(f"[pass1] selected_groups={len(models)} per_species_outputs={0 if args.no_per_species else len(per_species_models)}")
    chunk_iter = pd.read_csv(grid_csv, chunksize=max(1, int(args.grid_chunk_size)), low_memory=False)
    total_rows = 0
    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        chunk_work, _ = build_terraclimate_working_df(chunk)
        for required_col in [args.id_col, args.x_col, args.y_col]:
            if required_col not in chunk_work.columns:
                chunk_work[required_col] = chunk[required_col] if required_col in chunk.columns else pd.NA

        n_rows = len(chunk_work)
        total_rows += n_rows
        id_vals = chunk_work[args.id_col].tolist() if args.id_col in chunk_work.columns else list(range(total_rows - n_rows, total_rows))
        x_vals = chunk_work[args.x_col].tolist() if args.x_col in chunk_work.columns else [pd.NA] * n_rows
        y_vals = chunk_work[args.y_col].tolist() if args.y_col in chunk_work.columns else [pd.NA] * n_rows

        for model in models:
            comps = score_group_components_on_chunk(
                chunk_work,
                model,
                categorical_share=args.categorical_share,
                stress_power=args.stress_power,
                stress_strength=args.stress_strength,
                stress_grace_frac=args.stress_grace_frac,
                novelty_scale=args.novelty_scale,
                score_sharpness=args.score_sharpness,
                score_floor=args.score_floor,
                reliability_power=args.reliability_power,
                reliability_floor=args.reliability_floor,
                use_stress_adjustment=not args.no_stress_adjustment,
                use_reliability_adjustment=not args.no_reliability_adjustment,
            )
            g = model.group
            adjusted = comps["adjusted"]
            core = comps["core"]
            rel = comps["reliability"]
            rel_factor = comps["reliability_factor"]
            stress = comps["stress_modifier"]
            append_histogram(adjusted_hist[species_index[g]], adjusted, hist_bins)

            finite_adj = np.isfinite(adjusted)
            if np.any(finite_adj):
                species_sums[g] += float(np.nansum(adjusted[finite_adj]))
                species_counts[g] += int(np.sum(finite_adj))
                species_top_adjusted[g] = max(species_top_adjusted[g], float(np.nanmax(adjusted[finite_adj])))
                species_likely_counts[g] += int(np.sum(adjusted[finite_adj] >= float(args.richness_threshold)))
            finite_core = np.isfinite(core)
            if np.any(finite_core):
                species_core_sums[g] += float(np.nansum(core[finite_core]))
            finite_rel = np.isfinite(rel)
            if np.any(finite_rel):
                species_rel_sums[g] += float(np.nansum(rel[finite_rel]))
            finite_stress = np.isfinite(stress)
            if np.any(finite_stress):
                species_stress_sums[g] += float(np.nansum(stress[finite_stress]))

            if not args.no_per_species and g in {m.group for m in per_species_models}:
                pass1_writers.get(f"species::{g}").write_rows(zip(id_vals, x_vals, y_vals, core.tolist(), stress.tolist(), rel.tolist(), rel_factor.tolist(), adjusted.tolist(), (adjusted >= float(args.richness_threshold)).tolist()))
                if chunk_idx <= 4 and args.preview_top_n > 0:
                    preview_cache_species[g].append(
                        pd.DataFrame(
                            {
                                args.id_col: id_vals,
                                args.x_col: x_vals,
                                args.y_col: y_vals,
                                "adjusted_score": adjusted,
                                "core_score": core,
                                "reliability": rel,
                                "reliability_factor": rel_factor,
                            }
                        )
                    )

        log(f"[pass1] chunk={chunk_idx} rows={n_rows:,} total_rows={total_rows:,}")

    pass1_writers.close()

    cdf_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for m in models:
        cdf_lookup[m.group] = histogram_cdf_lookup(adjusted_hist[species_index[m.group]])
        count = species_counts.get(m.group, 0)
        mean_adj = species_sums[g] / count if (g := m.group) and count > 0 else np.nan
        summary_rows.append(
            {
                "group": m.group,
                "occurrence_count": m.occurrence_count,
                "numeric_feature_count": len(m.numeric_features),
                "categorical_feature_count": len(m.categorical_features),
                "stress_feature_count": len(m.stress_features),
                "redundancy_weight": m.redundancy_weight,
                "grid_cells_scored": int(count),
                "grid_mean_core": (species_core_sums[m.group] / count) if count > 0 else np.nan,
                "grid_mean_adjusted": mean_adj,
                "grid_mean_reliability": (species_rel_sums[m.group] / count) if count > 0 else np.nan,
                "grid_mean_stress_modifier": (species_stress_sums[m.group] / count) if count > 0 else np.nan,
                "grid_cells_at_or_above_threshold": int(species_likely_counts.get(m.group, 0)),
                "grid_fraction_at_or_above_threshold": (species_likely_counts.get(m.group, 0) / count) if count > 0 else np.nan,
                "grid_max_adjusted": species_top_adjusted.get(m.group, np.nan),
                **m.taxonomy,
            }
        )
    species_summary_df = pd.DataFrame(summary_rows).sort_values(["grid_mean_adjusted", "grid_max_adjusted", "group"], ascending=[False, False, True])
    species_summary_df.to_csv(outdir / "species_score_summary.csv", index=False)

    overall_writer = CsvAppender(
        outdir / "overall_suitability.csv",
        [
            args.id_col,
            args.x_col,
            args.y_col,
            "overall_core",
            "overall_core_min",
            "overall_adjusted",
            "overall_adjusted_min",
            "overall_adjusted_joint",
            "overall_reliability_mean",
            "richness_core",
            "richness_adjusted",
            "top_group_adjusted",
            "top_adjusted_score",
        ],
    )
    rank_writers: Dict[Tuple[str, str], CsvAppender] = {}
    for rank_key, member_map in rank_members.items():
        for value in sorted(member_map.keys()):
            rank_writers[(rank_key, value)] = CsvAppender(
                outdir / f"by_{rank_key}" / f"{safe_slug(value)}.csv",
                [args.id_col, args.x_col, args.y_col, "min_core", "joint_core", "min_adjusted", "joint_adjusted", "mean_reliability"],
            )

    preview_cache_overall: List[pd.DataFrame] = []
    preview_cache_rank: Dict[Tuple[str, str], List[pd.DataFrame]] = defaultdict(list)

    log(f"[pass2] selected_groups={len(models)}")
    chunk_iter = pd.read_csv(grid_csv, chunksize=max(1, int(args.grid_chunk_size)), low_memory=False)
    total_rows_second = 0
    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        chunk_work, _ = build_terraclimate_working_df(chunk)
        for required_col in [args.id_col, args.x_col, args.y_col]:
            if required_col not in chunk_work.columns:
                chunk_work[required_col] = chunk[required_col] if required_col in chunk.columns else pd.NA

        n_rows = len(chunk_work)
        total_rows_second += n_rows
        id_vals = chunk_work[args.id_col].tolist() if args.id_col in chunk_work.columns else list(range(total_rows_second - n_rows, total_rows_second))
        x_vals = chunk_work[args.x_col].tolist() if args.x_col in chunk_work.columns else [pd.NA] * n_rows
        y_vals = chunk_work[args.y_col].tolist() if args.y_col in chunk_work.columns else [pd.NA] * n_rows

        adjusted_stack = []
        core_stack = []
        rel_stack = []
        std_stack = []
        for model in models:
            comps = score_group_components_on_chunk(
                chunk_work,
                model,
                categorical_share=args.categorical_share,
                stress_power=args.stress_power,
                stress_strength=args.stress_strength,
                stress_grace_frac=args.stress_grace_frac,
                novelty_scale=args.novelty_scale,
                score_sharpness=args.score_sharpness,
                score_floor=args.score_floor,
                reliability_power=args.reliability_power,
                reliability_floor=args.reliability_floor,
                use_stress_adjustment=not args.no_stress_adjustment,
                use_reliability_adjustment=not args.no_reliability_adjustment,
            )
            adjusted = comps["adjusted"]
            core = comps["core"]
            reliability = comps["reliability"]
            centers, cdf = cdf_lookup[model.group]
            standardized = standardize_from_histogram(adjusted, centers, cdf, hist_bins)
            adjusted_stack.append(adjusted)
            core_stack.append(core)
            rel_stack.append(reliability)
            std_stack.append(standardized)

        adjusted_arr = np.vstack(adjusted_stack).astype(np.float32)
        core_arr = np.vstack(core_stack).astype(np.float32)
        rel_arr = np.vstack(rel_stack).astype(np.float32)
        std_arr = np.vstack(std_stack).astype(np.float32)
        adj_valid = np.isfinite(adjusted_arr)
        core_valid = np.isfinite(core_arr)
        rel_valid = np.isfinite(rel_arr)
        std_valid = np.isfinite(std_arr)

        safe_adj = np.where(adj_valid, adjusted_arr, -np.inf)
        adj_max_idx = np.argmax(safe_adj, axis=0)
        top_adjusted_score = safe_adj[adj_max_idx, np.arange(n_rows)]
        top_group_adjusted = np.array([group_names[i] for i in adj_max_idx], dtype=object)
        no_adj = ~np.isfinite(top_adjusted_score) | (top_adjusted_score == -np.inf)
        top_adjusted_score = np.where(no_adj, np.nan, top_adjusted_score)
        top_group_adjusted = np.where(no_adj, "", top_group_adjusted)

        safe_core_cov = np.where(core_valid, core_arr, -np.inf)
        overall_core = np.max(safe_core_cov, axis=0)
        overall_core = np.where(np.isfinite(overall_core) & (overall_core > -np.inf), overall_core, np.nan).astype(np.float32)

        safe_core_min = np.where(core_valid, core_arr, np.inf)
        overall_core_min = np.min(safe_core_min, axis=0)
        overall_core_min = np.where(np.isfinite(overall_core_min) & (overall_core_min < np.inf), overall_core_min, np.nan).astype(np.float32)

        safe_adj_cov = np.where(adj_valid, adjusted_arr, -np.inf)
        overall_adjusted = np.max(safe_adj_cov, axis=0)
        overall_adjusted = np.where(np.isfinite(overall_adjusted) & (overall_adjusted > -np.inf), overall_adjusted, np.nan).astype(np.float32)

        safe_adj_min = np.where(adj_valid, adjusted_arr, np.inf)
        overall_adjusted_min = np.min(safe_adj_min, axis=0)
        overall_adjusted_min = np.where(np.isfinite(overall_adjusted_min) & (overall_adjusted_min < np.inf), overall_adjusted_min, np.nan).astype(np.float32)

        overall_adjusted_joint = joint_support_score(
            adjusted_arr,
            adj_valid,
            min_share=args.joint_min_share,
            tail_fraction=args.joint_tail_fraction,
            rank_power=args.joint_rank_power,
        )

        rel_num = np.where(rel_valid, rel_arr, 0.0)
        rel_den = rel_valid.sum(axis=0)
        overall_reliability_mean = np.where(rel_den > 0, rel_num.sum(axis=0) / rel_den, np.nan).astype(np.float32)

        richness_core = np.sum(np.where(core_valid, core_arr >= float(args.richness_threshold), False), axis=0).astype(np.int32)
        richness_adjusted = np.sum(np.where(adj_valid, adjusted_arr >= float(args.richness_threshold), False), axis=0).astype(np.int32)

        overall_writer.write_rows(
            zip(
                id_vals,
                x_vals,
                y_vals,
                overall_core.tolist(),
                overall_core_min.tolist(),
                overall_adjusted.tolist(),
                overall_adjusted_min.tolist(),
                overall_adjusted_joint.tolist(),
                overall_reliability_mean.tolist(),
                richness_core.tolist(),
                richness_adjusted.tolist(),
                top_group_adjusted.tolist(),
                top_adjusted_score.tolist(),
            )
        )

        if chunk_idx <= 8:
            preview_cache_overall.append(
                pd.DataFrame(
                    {
                        args.id_col: id_vals,
                        args.x_col: x_vals,
                        args.y_col: y_vals,
                        "overall_core": overall_core,
                        "overall_core_min": overall_core_min,
                        "overall_adjusted": overall_adjusted,
                        "overall_adjusted_min": overall_adjusted_min,
                        "overall_adjusted_joint": overall_adjusted_joint,
                        "overall_reliability_mean": overall_reliability_mean,
                        "richness_adjusted": richness_adjusted,
                    }
                )
            )

        for rank_key, member_map in rank_members.items():
            for value, member_groups in member_map.items():
                idxs = [species_index[g] for g in member_groups]
                r_core = core_arr[idxs]
                r_adj = adjusted_arr[idxs]
                r_rel = rel_arr[idxs]
                r_std = std_arr[idxs]
                r_core_valid = np.isfinite(r_core)
                r_adj_valid = np.isfinite(r_adj)
                r_rel_valid = np.isfinite(r_rel)
                r_std_valid = np.isfinite(r_std)
                min_core = finite_min_score(r_core, r_core_valid)
                min_adjusted = finite_min_score(r_adj, r_adj_valid)
                joint_core = joint_support_score(
                    r_core,
                    r_core_valid,
                    min_share=args.joint_min_share,
                    tail_fraction=args.joint_tail_fraction,
                    rank_power=args.joint_rank_power,
                )
                joint_adjusted = joint_support_score(
                    r_adj,
                    r_adj_valid,
                    min_share=args.joint_min_share,
                    tail_fraction=args.joint_tail_fraction,
                    rank_power=args.joint_rank_power,
                )
                mean_reliability_den = r_rel_valid.sum(axis=0)
                mean_reliability = np.where(mean_reliability_den > 0, np.where(r_rel_valid, r_rel, 0.0).sum(axis=0) / mean_reliability_den, np.nan).astype(np.float32)
                rank_writers[(rank_key, value)].write_rows(zip(id_vals, x_vals, y_vals, min_core.tolist(), joint_core.tolist(), min_adjusted.tolist(), joint_adjusted.tolist(), mean_reliability.tolist()))
                if chunk_idx <= 4 and len(preview_cache_rank) < 24:
                    preview_cache_rank[(rank_key, value)].append(
                        pd.DataFrame(
                            {
                                args.id_col: id_vals,
                                args.x_col: x_vals,
                                args.y_col: y_vals,
                                "min_adjusted": min_adjusted,
                                "joint_adjusted": joint_adjusted,
                            }
                        )
                    )

        log(f"[pass2] chunk={chunk_idx} rows={n_rows:,} total_rows={total_rows_second:,}")

    overall_writer.close()
    for writer in rank_writers.values():
        writer.close()

    overall_preview_df = pd.concat(preview_cache_overall, ignore_index=True) if preview_cache_overall else pd.DataFrame()
    if not overall_preview_df.empty:
        preview_point_map(overall_preview_df, "overall_core", "overall core suitability", outdir / "previews" / "overall_core.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
        preview_point_map(overall_preview_df, "overall_adjusted", "overall adjusted suitability", outdir / "previews" / "overall_adjusted.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
        preview_point_map(overall_preview_df, "overall_core_min", "overall core minimum overlap", outdir / "previews" / "overall_core_min.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
        preview_point_map(overall_preview_df, "overall_adjusted_min", "overall adjusted minimum overlap", outdir / "previews" / "overall_adjusted_min.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
        preview_point_map(overall_preview_df, "overall_adjusted_joint", "overall joint adjusted suitability", outdir / "previews" / "overall_adjusted_joint.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
        preview_point_map(overall_preview_df, "overall_reliability_mean", "overall reliability mean", outdir / "previews" / "overall_reliability_mean.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
        preview_point_map(overall_preview_df, "richness_adjusted", "richness above threshold (adjusted)", outdir / "previews" / "richness_adjusted.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)

    if args.preview_top_n > 0 and preview_cache_species:
        species_preview_order = species_summary_df.head(int(args.preview_top_n)) if not species_summary_df.empty else pd.DataFrame()
        if not species_preview_order.empty:
            species_preview_order.to_csv(outdir / "species_preview_ranking.csv", index=False)
            for row in species_preview_order.itertuples(index=False):
                group = str(row.group)
                cached = preview_cache_species.get(group)
                if not cached:
                    continue
                dfp = pd.concat(cached, ignore_index=True)
                preview_point_map(dfp, "adjusted_score", f"{group} stress-adjusted suitability", outdir / "previews" / "by_species" / f"{safe_slug(group)}_adjusted.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
                preview_point_map(dfp, "core_score", f"{group} core suitability", outdir / "previews" / "by_species" / f"{safe_slug(group)}_core.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
                preview_point_map(dfp, "reliability", f"{group} reliability", outdir / "previews" / "by_species" / f"{safe_slug(group)}_reliability.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)

    for (rank_key, value), cached_frames in list(preview_cache_rank.items())[:24]:
        if not cached_frames:
            continue
        dfp = pd.concat(cached_frames, ignore_index=True)
        preview_point_map(dfp, "min_adjusted", f"{rank_key} {value} min adjusted suitability", outdir / "previews" / f"by_{rank_key}" / f"{safe_slug(value)}_min_adjusted.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)
        preview_point_map(dfp, "joint_adjusted", f"{rank_key} {value} joint adjusted suitability", outdir / "previews" / f"by_{rank_key}" / f"{safe_slug(value)}_joint_adjusted.png", args.x_col, args.y_col, point_alpha=args.preview_point_alpha, preview_coarsen=args.preview_coarsen, suitability_vmax=args.preview_suitability_vmax)

    manifest = {
        "occurrences_csv": str(occurrences_csv),
        "grid_csv": str(grid_csv),
        "trend_summary": str(trend_summary_dir),
        "outdir": str(outdir),
        "group_by": args.group_by,
        "include_taxa": [s.raw for s in include_selectors],
        "exclude_taxa": [s.raw for s in exclude_selectors],
        "selected_group_count": len(models),
        "selected_groups": [m.group for m in models],
        "aggregate_ranks": [str(x).lower() for x in (args.aggregate_ranks or [])],
        "categorical_share": float(args.categorical_share),
        "richness_threshold": float(args.richness_threshold),
        "grid_chunk_size": int(args.grid_chunk_size),
        "min_feature_n_valid": int(args.min_feature_n_valid),
        "min_features_per_group": int(args.min_features_per_group),
        "min_stress_features_per_group": int(args.min_stress_features_per_group),
        "min_occurrences_per_group": int(args.min_occurrences_per_group),
        "redundancy_top_k": int(args.redundancy_top_k),
        "hist_bins": int(hist_bins),
        "stress_power": float(args.stress_power),
        "novelty_scale": float(args.novelty_scale),
        "score_sharpness": float(args.score_sharpness),
        "score_floor": float(args.score_floor),
        "reliability_power": float(args.reliability_power),
        "reliability_floor": float(args.reliability_floor),
        "preview_point_alpha": float(args.preview_point_alpha),
        "preview_suitability_vmax": float(args.preview_suitability_vmax),
        "joint_min_share": float(args.joint_min_share),
        "joint_tail_fraction": float(args.joint_tail_fraction),
        "joint_rank_power": float(args.joint_rank_power),
        "stress_adjustment_enabled": not bool(args.no_stress_adjustment),
        "reliability_adjustment_enabled": not bool(args.no_reliability_adjustment),
        "per_species_output_count": 0 if args.no_per_species else len(per_species_models),
        "terraclimate_grid_multiyear_columns_detected": 0 if grid_tc_meta is None else int(len(grid_tc_meta)),
        "stacking_method": "two_pass_adjusted_score_standardization_via_target_grid_histograms",
        "stress_axes": sorted({feat.axis for m in models for feat in m.stress_features}),
    }
    write_manifest(outdir / "manifest.json", manifest)
    log(f"[done] rows_scored={total_rows_second:,} selected_groups={len(models)} outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
