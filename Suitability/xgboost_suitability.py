#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception:
    HAVE_XGBOOST = False


# py xgboost_suitability.py D:/envpull_association_test/occurrences_enriched.csv D:/Oregon_Suitability/oregon_grid_1000m_env/grid_with_env.csv --trend-summary D:/envpull_association_test/trend_summary --outdir D:/Oregon_Suitability/oregon_grid_1000m_env/ml_association_test_conservative_final --group-by matched_species_name --background-source mixed --background-sampling regional --cv-block-size 1.0 --cv-buffer-blocks 1
# py xgboost_suitability.py D:/envpull_association_test/occurrences_enriched.csv D:/Oregon_Suitability/oregon_grid_1000m_env/grid_with_env.csv --trend-summary D:/envpull_association_test/trend_summary --outdir D:/Oregon_Suitability/oregon_grid_1000m_env/ml_association_test_conservative_final_all_but_excluded --group-by matched_species_name --background-source mixed --background-sampling regional --cv-block-size 1.0 --cv-buffer-blocks 1 --exclude-taxa "species:Acer rubrum" "species:Pseudotsuga menziesii" "species:Epifagus virginiana" "species:Fagus grandifolia" "species:Alnus rubra" "species:Kopsiopsis strobilacea" "species:Arbutus menziesii"
# py xgboost_suitability.py --reuse-models-from D:/Suitability/alaska_1000m/ml_association_full D:/Suitability/fnsb/grid_with_env.csv --outdir D:/Suitability/fnsb/ml_from_full_alaska --group-by matched_species_name --include-taxa "species:Populus balsamifera" "species:Picea glauca"
# py xgboost_suitability.py --reuse-models-from D:/Suitability/alaska_1000m/ml_association_full --deploy-grid-csv D:/Suitability/fnsb/grid_with_env.csv --outdir D:/Suitability/fnsb/ml_from_full_alaska --group-by matched_species_name

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUMBERS = [f"{i:02d}" for i in range(1, 13)]
TERRACLIMATE_MONTHLY_RX = re.compile(r"^(terraclimate_[^_]+?)(?:_(\d{4}))?_m(0[1-9]|1[0-2])$")
TERRACLIMATE_AGG_RX = re.compile(r"^(terraclimate_[^_]+?)(?:_(\d{4}))?_(mean|sum|min|max)$")

DEFAULT_ID_COLUMNS = {
    "id", "row_id", "species_node_id", "gbifID", "occurrenceID", "scientificName",
    "kingdom", "phylum", "class", "order", "family", "genus", "species", "country",
    "stateProvince", "eventDate", "year", "month", "day", "decimalLatitude",
    "decimalLongitude", "coordinateUncertaintyInMeters", "basisOfRecord", "matched_species_name",
}
DEFAULT_SKIP_NUMERIC_PATTERNS = [
    r"(?:^|_)lon$", r"(?:^|_)lat$", r"(?:^|_)x$", r"(?:^|_)y$", r"(?:^|_)row$", r"(?:^|_)col$",
    r"(?:^|_)year$", r"(?:^|_)sample_found$", r"(?:^|_)hit$", r"(?:^|_)ok$", r"(?:^|_)in_bounds$",
    r"(?:^|_)is_nodata$", r"(?:^|_)flowdir$",
]
DEFAULT_NUMERIC_PREFIXES = ["dem_", "twi_", "soilgrids_", "terraclimate_"]
DEFAULT_CATEGORICAL_PATTERNS = [r"^glim(?:_|$)", r"^mcd12q1(?:_|$)", r"^dem_continent$"]
DEFAULT_CATEGORICAL_PREFIXES = ["glim_", "mcd12q1_"]
DEFAULT_PREVIEW_VMAX = 1.0
TAXON_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


@dataclass
class TaxonSelector:
    raw: str
    rank: Optional[str]
    name: str
    name_norm: str


@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]
    categorical_vocab: Dict[str, List[str]]
    encoded_columns: List[str]
    feature_priority: Dict[str, float]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "group"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def parse_selector_list(raw_values: Optional[Sequence[str]]) -> List[TaxonSelector]:
    selectors: List[TaxonSelector] = []
    for raw in raw_values or []:
        for part in str(raw).split(','):
            part = part.strip()
            if not part:
                continue
            rank = None
            name = part
            if ':' in part:
                maybe_rank, maybe_name = part.split(':', 1)
                mr = normalize_text(maybe_rank)
                if mr in TAXON_RANKS or mr in ('matched_species_name', 'group'):
                    rank = mr
                    name = maybe_name.strip()
            selectors.append(TaxonSelector(raw=part, rank=rank, name=name, name_norm=normalize_text(name)))
    return selectors


def first_non_null(series: pd.Series) -> object:
    for value in series:
        if pd.notna(value) and str(value).strip() != "":
            return value
    return pd.NA


def resolve_group_meta(df_occ: pd.DataFrame, group_by: str) -> pd.DataFrame:
    needed = []
    for col in [group_by, 'matched_species_name', *TAXON_RANKS]:
        if col in df_occ.columns and col not in needed:
            needed.append(col)
    sub = df_occ[needed].copy()
    sub[group_by] = sub[group_by].astype('string').fillna(pd.NA).map(lambda x: 'NA' if pd.isna(x) else str(x))
    rows = []
    for group, gdf in sub.groupby(group_by, dropna=False, sort=True):
        row = {'group': str(group), 'occurrence_count': int(gdf.shape[0])}
        for col in needed:
            if col == group_by:
                continue
            row[col] = first_non_null(gdf[col]) if col in gdf.columns else pd.NA
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=['group', 'occurrence_count', *TAXON_RANKS, 'matched_species_name'])
    for col in ['group', 'matched_species_name', *TAXON_RANKS]:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def selector_matches_row(selector: TaxonSelector, row: pd.Series, group_by: str) -> bool:
    fields = {}
    for col in [group_by, 'matched_species_name', *TAXON_RANKS]:
        if col in row.index:
            fields[col] = normalize_text(row[col])
    if selector.rank is not None:
        return selector.name_norm == fields.get(selector.rank, '')
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


def parse_terraclimate_monthly_column(name: object) -> Optional[Dict[str, object]]:
    m = TERRACLIMATE_MONTHLY_RX.match(str(name))
    if not m:
        return None
    return {
        'variable': str(m.group(1)),
        'year': int(m.group(2)) if m.group(2) else None,
        'month_num': int(m.group(3)),
        'month': str(m.group(3)),
        'canonical_column': f"{m.group(1)}_m{m.group(3)}",
    }


def parse_terraclimate_aggregate_column(name: object) -> Optional[Dict[str, object]]:
    m = TERRACLIMATE_AGG_RX.match(str(name))
    if not m:
        return None
    return {
        'variable': str(m.group(1)),
        'year': int(m.group(2)) if m.group(2) else None,
        'aggregate': str(m.group(3)),
        'canonical_column': f"{m.group(1)}_{m.group(3)}",
    }


def build_terraclimate_working_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    rows = []
    grouped: Dict[Tuple[str, str, str, Optional[int], Optional[str]], List[Tuple[int, str]]] = {}
    drop_cols = set()
    for col in df.columns:
        monthly = parse_terraclimate_monthly_column(col)
        if monthly is not None and monthly['year'] is not None:
            key = (str(monthly['canonical_column']), str(monthly['variable']), 'monthly', int(monthly['month_num']), None)
            grouped.setdefault(key, []).append((int(monthly['year']), str(col)))
            drop_cols.add(str(col))
            continue
        aggregate = parse_terraclimate_aggregate_column(col)
        if aggregate is not None and aggregate['year'] is not None:
            key = (str(aggregate['canonical_column']), str(aggregate['variable']), 'aggregate', None, str(aggregate['aggregate']))
            grouped.setdefault(key, []).append((int(aggregate['year']), str(col)))
            drop_cols.add(str(col))
    for (canonical_col, variable, feature_kind, month_num, aggregate_name), entries in sorted(grouped.items()):
        entries = sorted(entries, key=lambda x: (x[0], x[1]))
        source_cols = [col for _, col in entries if col in work.columns]
        years = [int(year) for year, _ in entries]
        if not source_cols:
            continue
        source_frame = work[source_cols].apply(pd.to_numeric, errors='coerce')
        work[canonical_col] = source_frame.mean(axis=1, skipna=True)
        rows.append({
            'canonical_column': canonical_col,
            'variable': variable,
            'feature_kind': feature_kind,
            'aggregate': aggregate_name,
            'month_num': month_num,
            'month': MONTH_NAMES[month_num - 1] if month_num is not None else pd.NA,
            'n_years': len(years),
            'years': ','.join(str(y) for y in years),
            'source_columns': ','.join(source_cols),
        })
    if drop_cols:
        work = work.drop(columns=sorted(drop_cols), errors='ignore')
    meta = pd.DataFrame(rows)
    if not meta.empty:
        meta = meta.sort_values(['variable', 'feature_kind', 'aggregate', 'month_num', 'canonical_column'], na_position='last').reset_index(drop=True)
    return work, meta


def matches_any_pattern(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pat, name) for pat in patterns)


def is_banned_numeric_column(name: str) -> bool:
    return matches_any_pattern(str(name), DEFAULT_SKIP_NUMERIC_PATTERNS)


def is_forced_categorical_column(name: str) -> bool:
    name = str(name)
    return (not is_banned_numeric_column(name)) and matches_any_pattern(name, DEFAULT_CATEGORICAL_PATTERNS)


def should_use_numeric_column(name: str) -> bool:
    name = str(name)
    return (not is_banned_numeric_column(name)) and (not is_forced_categorical_column(name))


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        if col in DEFAULT_ID_COLUMNS:
            continue
        if not should_use_numeric_column(col):
            continue
        if parse_terraclimate_monthly_column(col) is not None:
            numeric_cols.append(str(col))
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(str(col).startswith(prefix) for prefix in DEFAULT_NUMERIC_PREFIXES):
            numeric_cols.append(str(col))
    return sorted(set(numeric_cols))


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        if col in DEFAULT_ID_COLUMNS:
            continue
        if matches_any_pattern(str(col), DEFAULT_CATEGORICAL_PATTERNS):
            out.append(str(col))
            continue
        if any(str(col).startswith(prefix) for prefix in DEFAULT_CATEGORICAL_PREFIXES):
            out.append(str(col))
    return sorted(set(out))


def infer_trend_summary_dir(occ_csv: Path, explicit_outdir: Optional[Path]) -> Path:
    if explicit_outdir is not None:
        return explicit_outdir
    return occ_csv.resolve().parent / 'trend_summary'


def read_minimal_csv(path: Path, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if usecols:
        return pd.read_csv(path, usecols=list(dict.fromkeys(usecols)), low_memory=False)
    return pd.read_csv(path, low_memory=False)


def resolve_required_raw_columns(raw_head: pd.DataFrame, tc_meta: pd.DataFrame, selected_working_cols: Sequence[str], extra_raw_cols: Sequence[str]) -> List[str]:
    raw_cols = set(map(str, raw_head.columns))
    wanted = set(str(c) for c in extra_raw_cols if str(c) in raw_cols)
    meta_lookup: Dict[str, List[str]] = {}
    if tc_meta is not None and not tc_meta.empty:
        for row in tc_meta.itertuples(index=False):
            canonical = str(getattr(row, 'canonical_column', '') or '')
            source_columns = str(getattr(row, 'source_columns', '') or '')
            if canonical and source_columns:
                meta_lookup[canonical] = [s for s in source_columns.split(',') if s]
    for col in selected_working_cols:
        key = str(col)
        if key in raw_cols:
            wanted.add(key)
            continue
        for src in meta_lookup.get(key, []):
            if src in raw_cols:
                wanted.add(src)
    return sorted(wanted)


def load_priority_tables(trend_summary_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_weights_path = trend_summary_dir / 'feature_signal_weights.csv'
    family_weights_path = trend_summary_dir / 'family_signal_weights.csv'
    by_group_cat_path = trend_summary_dir / 'by_group_categorical_frequencies.csv'
    feature_weights = pd.read_csv(feature_weights_path, low_memory=False) if feature_weights_path.exists() else pd.DataFrame()
    family_weights = pd.read_csv(family_weights_path, low_memory=False) if family_weights_path.exists() else pd.DataFrame()
    by_group_categorical = pd.read_csv(by_group_cat_path, low_memory=False) if by_group_cat_path.exists() else pd.DataFrame()
    return feature_weights, family_weights, by_group_categorical


def build_feature_priority(feature_weights: pd.DataFrame, family_weights: pd.DataFrame) -> Dict[str, float]:
    family_lookup: Dict[str, float] = {}
    if family_weights is not None and not family_weights.empty:
        for row in family_weights.itertuples(index=False):
            fam = str(getattr(row, 'family', 'other'))
            val = pd.to_numeric(getattr(row, 'family_weight_sum_normalized', np.nan), errors='coerce')
            if pd.notna(val):
                family_lookup[fam] = float(val)
    priority: Dict[str, float] = {}
    if feature_weights is not None and not feature_weights.empty:
        for row in feature_weights.itertuples(index=False):
            col = str(getattr(row, 'column', '') or '')
            if not col:
                continue
            val = pd.to_numeric(getattr(row, 'blended_weight_global', np.nan), errors='coerce')
            fam = str(getattr(row, 'family', 'other'))
            if pd.notna(val):
                priority[col] = float(val)
            elif fam in family_lookup:
                priority[col] = float(family_lookup[fam])
    return priority


def choose_features(occ_head_work: pd.DataFrame, grid_head_work: pd.DataFrame, by_group_categorical: pd.DataFrame, feature_priority: Dict[str, float], include_coordinates: bool, top_numeric_features: int, max_categorical_columns: int, max_categories_per_column: int) -> FeatureSpec:
    occ_numeric = set(detect_numeric_columns(occ_head_work))
    grid_numeric = set(detect_numeric_columns(grid_head_work))
    numeric_cols = sorted(occ_numeric & grid_numeric, key=lambda c: (-float(feature_priority.get(c, 0.0)), c))
    if not include_coordinates:
        numeric_cols = [c for c in numeric_cols if normalize_text(c) not in {'decimallongitude', 'decimallatitude', 'lon', 'lat', 'x', 'y'}]
    if top_numeric_features > 0:
        numeric_cols = numeric_cols[:top_numeric_features]

    occ_cat = set(detect_categorical_columns(occ_head_work))
    grid_cat = set(detect_categorical_columns(grid_head_work))
    categorical_cols = sorted(occ_cat & grid_cat, key=lambda c: (-float(feature_priority.get(c, 0.0)), c))
    if max_categorical_columns > 0:
        categorical_cols = categorical_cols[:max_categorical_columns]

    categorical_vocab: Dict[str, List[str]] = {}
    if categorical_cols and by_group_categorical is not None and not by_group_categorical.empty:
        cat_freq = by_group_categorical.copy()
        for col in categorical_cols:
            sub = cat_freq[cat_freq['column'].astype(str) == str(col)].copy()
            if sub.empty:
                continue
            value_col = 'fraction_of_non_null' if 'fraction_of_non_null' in sub.columns else ('count' if 'count' in sub.columns else None)
            if value_col is None:
                continue
            sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
            sub = sub.sort_values([value_col, 'value'], ascending=[False, True])
            vals = []
            for value in sub['value'].dropna().astype(str):
                nv = normalize_text(value)
                if nv and nv not in vals:
                    vals.append(nv)
                if len(vals) >= max_categories_per_column:
                    break
            if vals:
                categorical_vocab[col] = vals

    encoded_columns = list(numeric_cols)
    for col in categorical_cols:
        for value in categorical_vocab.get(col, []):
            encoded_columns.append(f"{col}__{safe_slug(value)}")
    return FeatureSpec(numeric_cols=numeric_cols, categorical_cols=categorical_cols, categorical_vocab=categorical_vocab, encoded_columns=encoded_columns, feature_priority=feature_priority)


def transform_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    data = {}
    n = len(df)

    for col in spec.numeric_cols:
        if col in df.columns:
            data[col] = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float32, copy=False)
        else:
            data[col] = np.full(n, np.nan, dtype=np.float32)

    for col in spec.categorical_cols:
        values = spec.categorical_vocab.get(col, [])
        if not values:
            continue
        if col in df.columns:
            s = df[col].astype('string').fillna(pd.NA).map(lambda x: normalize_text(x) if pd.notna(x) else '')
        else:
            s = pd.Series([''] * n, index=df.index, dtype='string')
        for value in values:
            data[f"{col}__{safe_slug(value)}"] = (s == value).to_numpy(dtype=np.int8, copy=False)

    return pd.DataFrame(data, index=df.index)


def variance_filter(df: pd.DataFrame, threshold: float = 1e-10) -> List[str]:
    keep = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=np.float64)
        if np.nanvar(vals) > threshold:
            keep.append(c)
    return keep


def correlation_prune(df: pd.DataFrame, columns: Sequence[str], threshold: float = 0.98) -> List[str]:
    if len(columns) <= 1:
        return list(columns)
    corr = df[list(columns)].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [c for c in columns if c not in drop]


def stratified_cap_rows(labels: np.ndarray, max_rows: Optional[int], random_state: int) -> np.ndarray:
    n = len(labels)
    if max_rows is None or max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(random_state))
    y = np.asarray(labels)
    out = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        keep_n = max(1, int(round(max_rows * (len(idx) / max(1, n)))))
        keep_n = min(len(idx), keep_n)
        out.append(rng.choice(idx, size=keep_n, replace=False))
    merged = np.concatenate(out).astype(np.int64)
    merged.sort()
    if len(merged) > max_rows:
        merged = rng.choice(merged, size=max_rows, replace=False)
        merged.sort()
    return merged


def pack_spatial_blocks(block_x: np.ndarray, block_y: np.ndarray) -> np.ndarray:
    bx = np.asarray(block_x, dtype=np.int64)
    by = np.asarray(block_y, dtype=np.int64)
    return (bx << 32) ^ (by & 0xFFFFFFFF)


def compute_spatial_block_indices(x: np.ndarray, y: np.ndarray, block_size: float, origin_x: Optional[float] = None, origin_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    step = max(float(block_size), 1e-9)
    valid = np.isfinite(xv) & np.isfinite(yv)
    if origin_x is None:
        origin_x = float(np.nanmin(xv[valid])) if np.any(valid) else 0.0
    if origin_y is None:
        origin_y = float(np.nanmin(yv[valid])) if np.any(valid) else 0.0
    bx = np.full(xv.shape, np.iinfo(np.int32).min, dtype=np.int32)
    by = np.full(yv.shape, np.iinfo(np.int32).min, dtype=np.int32)
    if np.any(valid):
        bx[valid] = np.floor((xv[valid] - float(origin_x)) / step).astype(np.int32)
        by[valid] = np.floor((yv[valid] - float(origin_y)) / step).astype(np.int32)
    return bx, by, float(origin_x), float(origin_y)


def build_spatial_groups(x: np.ndarray, y: np.ndarray, block_size: float, origin_x: Optional[float] = None, origin_y: Optional[float] = None) -> np.ndarray:
    bx, by, _, _ = compute_spatial_block_indices(x, y, block_size, origin_x=origin_x, origin_y=origin_y)
    return pack_spatial_blocks(bx, by)


def buffered_train_indices(train_idx: np.ndarray, test_idx: np.ndarray, block_x: np.ndarray, block_y: np.ndarray, buffer_n: int) -> np.ndarray:
    if int(buffer_n) <= 0 or len(train_idx) == 0 or len(test_idx) == 0:
        return np.asarray(train_idx, dtype=np.int64)
    tx = np.asarray(block_x[test_idx], dtype=np.int64)
    ty = np.asarray(block_y[test_idx], dtype=np.int64)
    valid_test = (tx != np.iinfo(np.int32).min) & (ty != np.iinfo(np.int32).min)
    if not np.any(valid_test):
        return np.asarray(train_idx, dtype=np.int64)
    unique_pairs = np.unique(np.column_stack([tx[valid_test], ty[valid_test]]), axis=0)
    blocked = set()
    for bx, by in unique_pairs:
        for dx in range(-int(buffer_n), int(buffer_n) + 1):
            for dy in range(-int(buffer_n), int(buffer_n) + 1):
                blocked.add((int(bx + dx), int(by + dy)))
    train_x = np.asarray(block_x[train_idx], dtype=np.int64)
    train_y = np.asarray(block_y[train_idx], dtype=np.int64)
    keep = np.array([(int(bx), int(by)) not in blocked for bx, by in zip(train_x, train_y)], dtype=bool)
    return np.asarray(train_idx, dtype=np.int64)[keep]


def build_cv_splitter(y: np.ndarray, coords_x: np.ndarray, coords_y: np.ndarray, desired_folds: int, block_size: float, random_state: int):
    y_arr = np.asarray(y, dtype=np.int8)
    class_counts = np.bincount(y_arr, minlength=2)
    min_class = int(class_counts.min()) if class_counts.size >= 2 else 0
    finite_coords = np.isfinite(coords_x).all() and np.isfinite(coords_y).all()
    if finite_coords:
        groups = build_spatial_groups(coords_x, coords_y, block_size)
        unique_groups = np.unique(groups)
        n_splits = min(int(desired_folds), int(unique_groups.size))
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits), groups, 'groupkfold'
    n_splits = min(int(desired_folds), max(0, min_class))
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)), None, 'stratified'
    return None, None, 'none'


def choose_calibration_split(y: np.ndarray, coords_x: np.ndarray, coords_y: np.ndarray, desired_fraction: float, block_size: float, random_state: int, buffer_blocks: int = 0):
    frac = float(np.clip(desired_fraction, 0.05, 0.5))
    desired_splits = max(3, int(round(1.0 / frac)))
    splitter, groups, scheme = build_cv_splitter(y, coords_x, coords_y, desired_splits, block_size, random_state)
    if splitter is None:
        return None, None, 'none'
    best = None
    target = frac
    bx = by = None
    if groups is not None and int(buffer_blocks) > 0:
        bx, by, _, _ = compute_spatial_block_indices(coords_x, coords_y, float(block_size))
    split_iter = splitter.split(np.zeros(len(y), dtype=np.int8), y, groups=groups) if groups is not None else splitter.split(np.zeros(len(y), dtype=np.int8), y)
    for train_idx, calib_idx in split_iter:
        train_idx = np.asarray(train_idx, dtype=np.int64)
        calib_idx = np.asarray(calib_idx, dtype=np.int64)
        if bx is not None and by is not None and int(buffer_blocks) > 0:
            train_idx = buffered_train_indices(train_idx, calib_idx, bx, by, int(buffer_blocks))
        if len(train_idx) == 0 or len(calib_idx) == 0:
            continue
        if np.unique(y[train_idx]).size < 2 or np.unique(y[calib_idx]).size < 2:
            continue
        score = abs((len(calib_idx) / max(1, len(y))) - target)
        if best is None or score < best[0]:
            best = (score, train_idx, calib_idx)
    if best is None:
        return None, None, 'none'
    return best[1], best[2], scheme


def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(proba, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return 0.5
    y = y[valid]
    p = p[valid]
    thresholds = np.linspace(0.05, 0.95, 37)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = p >= t
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if f1 > best_score:
            best_score = f1
            best_t = float(t)
    return best_t


def estimate_training_threshold(estimator, X: pd.DataFrame, y: np.ndarray, coords_x: np.ndarray, coords_y: np.ndarray, args) -> Tuple[float, str, int]:
    train_idx, calib_idx, scheme = choose_calibration_split(
        y,
        coords_x,
        coords_y,
        desired_fraction=float(args.threshold_calibration_fraction),
        block_size=float(args.cv_block_size),
        random_state=int(args.random_state),
        buffer_blocks=int(args.cv_buffer_blocks),
    )
    if train_idx is None or calib_idx is None or len(calib_idx) == 0:
        return 0.5, 'default_0.5', 0
    est = clone(estimator)
    est.fit(X.iloc[train_idx], y[train_idx])
    proba = est.predict_proba(X.iloc[calib_idx])[:, 1]
    return float(pick_threshold(y[calib_idx], proba)), scheme, int(len(calib_idx))


def metric_or_nan(fn, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    try:
        return float(fn(y_true, y_score, **kwargs))
    except Exception:
        return float('nan')


def sample_background_indices(grid_x: np.ndarray, grid_y: np.ndarray, grid_block_keys: np.ndarray, block_origin_x: float, block_origin_y: float, pos_x: np.ndarray, pos_y: np.ndarray, bg_target: int, args, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    valid_grid = np.isfinite(grid_x) & np.isfinite(grid_y)
    pos_valid = np.isfinite(pos_x) & np.isfinite(pos_y)
    if not np.any(valid_grid):
        return np.array([], dtype=np.int64), {'mode': str(args.background_sampling), 'candidate_global': 0, 'candidate_local': 0, 'blocked_cells': 0}
    if not np.any(pos_valid):
        pool = np.flatnonzero(valid_grid)
        take = min(int(bg_target), len(pool))
        chosen = np.sort(rng.choice(pool, size=take, replace=False)) if take > 0 else np.array([], dtype=np.int64)
        return chosen, {'mode': str(args.background_sampling), 'candidate_global': int(len(pool)), 'candidate_local': int(len(pool)), 'blocked_cells': 0}

    px = np.asarray(pos_x[pos_valid], dtype=float)
    py = np.asarray(pos_y[pos_valid], dtype=float)
    x_min = float(np.nanmin(px))
    x_max = float(np.nanmax(px))
    y_min = float(np.nanmin(py))
    y_max = float(np.nanmax(py))
    span_x = max(0.0, x_max - x_min)
    span_y = max(0.0, y_max - y_min)
    pad_x = max(float(args.background_bbox_pad_min), span_x * float(args.background_bbox_pad_fraction))
    pad_y = max(float(args.background_bbox_pad_min), span_y * float(args.background_bbox_pad_fraction))
    local_mask = valid_grid & (grid_x >= (x_min - pad_x)) & (grid_x <= (x_max + pad_x)) & (grid_y >= (y_min - pad_y)) & (grid_y <= (y_max + pad_y))

    pos_bx, pos_by, _, _ = compute_spatial_block_indices(px, py, float(args.background_block_size), origin_x=block_origin_x, origin_y=block_origin_y)
    pos_keys = np.unique(pack_spatial_blocks(pos_bx, pos_by))
    buffer_n = max(0, int(args.background_block_buffer))
    if pos_keys.size and buffer_n > 0:
        blocked_keys = []
        unique_pairs = np.unique(np.column_stack([pos_bx.astype(np.int64), pos_by.astype(np.int64)]), axis=0)
        for bx, by in unique_pairs:
            for dx in range(-buffer_n, buffer_n + 1):
                for dy in range(-buffer_n, buffer_n + 1):
                    blocked_keys.append(pack_spatial_blocks(np.array([bx + dx], dtype=np.int64), np.array([by + dy], dtype=np.int64))[0])
        blocked_keys = np.unique(np.asarray(blocked_keys, dtype=np.int64))
    else:
        blocked_keys = pos_keys.astype(np.int64, copy=False)
    blocked_mask = np.isin(grid_block_keys, blocked_keys) if blocked_keys.size else np.zeros(len(grid_x), dtype=bool)

    candidate_global = valid_grid & ~blocked_mask
    candidate_local = candidate_global & local_mask
    global_idx = np.flatnonzero(candidate_global)
    local_idx = np.flatnonzero(candidate_local)

    mode = str(args.background_sampling).strip().lower()
    chosen = np.array([], dtype=np.int64)
    if mode == 'global':
        pool = global_idx
        take = min(int(bg_target), len(pool))
        chosen = np.sort(rng.choice(pool, size=take, replace=False)) if take > 0 else chosen
    elif mode == 'regional':
        pool = local_idx if len(local_idx) >= max(1, int(bg_target * float(args.background_min_local_fraction))) else global_idx
        take = min(int(bg_target), len(pool))
        chosen = np.sort(rng.choice(pool, size=take, replace=False)) if take > 0 else chosen
    else:
        local_take = min(len(local_idx), int(round(int(bg_target) * float(args.background_local_share))))
        local_sel = np.sort(rng.choice(local_idx, size=local_take, replace=False)) if local_take > 0 else np.array([], dtype=np.int64)
        remaining_pool = np.setdiff1d(global_idx, local_sel, assume_unique=False) if len(local_sel) > 0 else global_idx
        remaining_take = min(max(0, int(bg_target) - len(local_sel)), len(remaining_pool))
        global_sel = np.sort(rng.choice(remaining_pool, size=remaining_take, replace=False)) if remaining_take > 0 else np.array([], dtype=np.int64)
        chosen = np.sort(np.concatenate([local_sel, global_sel]).astype(np.int64)) if (len(local_sel) + len(global_sel)) > 0 else np.array([], dtype=np.int64)
        if len(chosen) < int(bg_target) and len(global_idx) > len(chosen):
            refill_pool = np.setdiff1d(global_idx, chosen, assume_unique=False)
            refill_take = min(int(bg_target) - len(chosen), len(refill_pool))
            if refill_take > 0:
                refill = np.sort(rng.choice(refill_pool, size=refill_take, replace=False))
                chosen = np.sort(np.concatenate([chosen, refill]).astype(np.int64))

    return chosen.astype(np.int64, copy=False), {
        'mode': mode,
        'candidate_global': int(len(global_idx)),
        'candidate_local': int(len(local_idx)),
        'blocked_cells': int(np.sum(blocked_mask)),
    }


def sample_background_from_source(candidate_x: np.ndarray, candidate_y: np.ndarray, candidate_block_keys: np.ndarray, block_origin_x: float, block_origin_y: float, pos_x: np.ndarray, pos_y: np.ndarray, bg_target: int, args, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    return sample_background_indices(
        np.asarray(candidate_x, dtype=float),
        np.asarray(candidate_y, dtype=float),
        np.asarray(candidate_block_keys, dtype=np.int64),
        float(block_origin_x),
        float(block_origin_y),
        np.asarray(pos_x, dtype=float),
        np.asarray(pos_y, dtype=float),
        int(bg_target),
        args,
        rng,
    )


def build_estimators(args, scale_pos_weight: float):
    models = {}
    models['extratrees'] = ExtraTreesClassifier(
        n_estimators=int(args.et_n_estimators),
        max_depth=None if int(args.et_max_depth) <= 0 else int(args.et_max_depth),
        min_samples_leaf=int(args.et_min_samples_leaf),
        max_features=args.et_max_features,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=int(args.random_state),
    )
    if HAVE_XGBOOST:
        models['xgboost'] = XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            n_estimators=int(args.xgb_n_estimators),
            max_depth=int(args.xgb_max_depth),
            learning_rate=float(args.xgb_learning_rate),
            subsample=float(args.xgb_subsample),
            colsample_bytree=float(args.xgb_colsample_bytree),
            min_child_weight=float(args.xgb_min_child_weight),
            reg_alpha=float(args.xgb_reg_alpha),
            reg_lambda=float(args.xgb_reg_lambda),
            gamma=float(args.xgb_gamma),
            max_bin=int(args.xgb_max_bin),
            n_jobs=-1,
            random_state=int(args.random_state),
            eval_metric='logloss',
            verbosity=0,
            missing=np.nan,
            scale_pos_weight=float(scale_pos_weight),
        )
    return models


def evaluate_candidate_models(X: pd.DataFrame, y: np.ndarray, coords_x: np.ndarray, coords_y: np.ndarray, args, estimator_map: Dict[str, object]):
    splitter, groups, cv_scheme = build_cv_splitter(y, coords_x, coords_y, int(args.cv_folds), float(args.cv_block_size), int(args.random_state))
    if splitter is None:
        raise RuntimeError('No valid CV folds were produced')

    metrics_by_model: Dict[str, List[Dict[str, float]]] = {name: [] for name in estimator_map}
    thresholds_by_model: Dict[str, List[float]] = {name: [] for name in estimator_map}
    oof_pred_by_model: Dict[str, np.ndarray] = {name: np.full(len(y), np.nan, dtype=np.float32) for name in estimator_map}
    fold_rows: List[Dict[str, float]] = []
    bx = by = None
    if groups is not None and int(args.cv_buffer_blocks) > 0:
        bx, by, _, _ = compute_spatial_block_indices(coords_x, coords_y, float(args.cv_block_size))

    split_iter = splitter.split(X, y, groups=groups) if groups is not None else splitter.split(X, y)
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        original_train_rows = int(len(train_idx))
        if bx is not None and by is not None and int(args.cv_buffer_blocks) > 0:
            train_idx = buffered_train_indices(train_idx, test_idx, bx, by, int(args.cv_buffer_blocks))
        x_train = X.iloc[train_idx]
        y_train = y[train_idx]
        x_test = X.iloc[test_idx]
        y_test = y[test_idx]
        coords_x_train = coords_x[train_idx]
        coords_y_train = coords_y[train_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        for model_name, estimator in estimator_map.items():
            est = clone(estimator)
            threshold, threshold_scheme, calibration_rows = estimate_training_threshold(estimator, x_train, y_train, coords_x_train, coords_y_train, args)
            fit_t0 = time.perf_counter()
            est.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_t0
            pred_t0 = time.perf_counter()
            proba = est.predict_proba(x_test)[:, 1]
            predict_seconds = time.perf_counter() - pred_t0
            pred = (proba >= threshold).astype(np.int8)
            oof_pred_by_model[model_name][test_idx] = proba.astype(np.float32)
            row = {
                'fold': int(fold_idx),
                'model': model_name,
                'cv_scheme': cv_scheme,
                'train_rows': int(len(train_idx)),
                'train_rows_before_buffer': int(original_train_rows),
                'test_rows': int(len(test_idx)),
                'train_positive_rows': int(np.sum(y_train == 1)),
                'train_negative_rows': int(np.sum(y_train == 0)),
                'test_positive_rows': int(np.sum(y_test == 1)),
                'test_negative_rows': int(np.sum(y_test == 0)),
                'threshold': float(threshold),
                'threshold_scheme': threshold_scheme,
                'calibration_rows': int(calibration_rows),
                'roc_auc': metric_or_nan(roc_auc_score, y_test, proba),
                'pr_auc': metric_or_nan(average_precision_score, y_test, proba),
                'logloss': metric_or_nan(log_loss, y_test, proba, labels=[0, 1]),
                'brier': metric_or_nan(brier_score_loss, y_test, proba),
                'accuracy': metric_or_nan(accuracy_score, y_test, pred),
                'balanced_accuracy': metric_or_nan(balanced_accuracy_score, y_test, pred),
                'f1': metric_or_nan(f1_score, y_test, pred, zero_division=0),
                'precision': metric_or_nan(precision_score, y_test, pred, zero_division=0),
                'recall': metric_or_nan(recall_score, y_test, pred, zero_division=0),
                'fit_seconds': float(fit_seconds),
                'predict_seconds': float(predict_seconds),
            }
            metrics_by_model[model_name].append(row)
            thresholds_by_model[model_name].append(float(threshold))
            fold_rows.append(row)

    summary_rows = []
    for model_name, rows in metrics_by_model.items():
        oof_pred = oof_pred_by_model[model_name]
        valid_oof = np.isfinite(oof_pred)
        median_threshold = float(np.nanmedian(thresholds_by_model[model_name])) if thresholds_by_model[model_name] else 0.5
        y_oof = y[valid_oof] if np.any(valid_oof) else np.array([], dtype=np.int8)
        oof_pred_labels = (oof_pred[valid_oof] >= median_threshold).astype(np.int8) if np.any(valid_oof) else np.array([], dtype=np.int8)
        summary_rows.append({
            'model': model_name,
            'cv_scheme': cv_scheme,
            'cv_roc_auc_mean': float(np.nanmean([r['roc_auc'] for r in rows])) if rows else float('nan'),
            'cv_pr_auc_mean': float(np.nanmean([r['pr_auc'] for r in rows])) if rows else float('nan'),
            'cv_logloss_mean': float(np.nanmean([r['logloss'] for r in rows])) if rows else float('nan'),
            'cv_brier_mean': float(np.nanmean([r['brier'] for r in rows])) if rows else float('nan'),
            'cv_accuracy_mean': float(np.nanmean([r['accuracy'] for r in rows])) if rows else float('nan'),
            'cv_balanced_accuracy_mean': float(np.nanmean([r['balanced_accuracy'] for r in rows])) if rows else float('nan'),
            'cv_f1_mean': float(np.nanmean([r['f1'] for r in rows])) if rows else float('nan'),
            'cv_precision_mean': float(np.nanmean([r['precision'] for r in rows])) if rows else float('nan'),
            'cv_recall_mean': float(np.nanmean([r['recall'] for r in rows])) if rows else float('nan'),
            'cv_fit_seconds_mean': float(np.nanmean([r['fit_seconds'] for r in rows])) if rows else float('nan'),
            'cv_predict_seconds_mean': float(np.nanmean([r['predict_seconds'] for r in rows])) if rows else float('nan'),
            'cv_fit_seconds_total': float(np.nansum([r['fit_seconds'] for r in rows])) if rows else float('nan'),
            'cv_predict_seconds_total': float(np.nansum([r['predict_seconds'] for r in rows])) if rows else float('nan'),
            'cv_threshold_median': float(np.nanmedian(thresholds_by_model[model_name])) if thresholds_by_model[model_name] else 0.5,
            'cv_threshold_std': float(np.nanstd(thresholds_by_model[model_name])) if thresholds_by_model[model_name] else 0.0,
            'oof_threshold': float(median_threshold),
            'oof_roc_auc': metric_or_nan(roc_auc_score, y_oof, oof_pred[valid_oof]) if np.any(valid_oof) else float('nan'),
            'oof_pr_auc': metric_or_nan(average_precision_score, y_oof, oof_pred[valid_oof]) if np.any(valid_oof) else float('nan'),
            'oof_logloss': metric_or_nan(log_loss, y_oof, oof_pred[valid_oof], labels=[0, 1]) if np.any(valid_oof) else float('nan'),
            'oof_brier': metric_or_nan(brier_score_loss, y_oof, oof_pred[valid_oof]) if np.any(valid_oof) else float('nan'),
            'oof_accuracy': metric_or_nan(accuracy_score, y_oof, oof_pred_labels) if np.any(valid_oof) else float('nan'),
            'oof_balanced_accuracy': metric_or_nan(balanced_accuracy_score, y_oof, oof_pred_labels) if np.any(valid_oof) else float('nan'),
            'oof_f1': metric_or_nan(f1_score, y_oof, oof_pred_labels, zero_division=0) if np.any(valid_oof) else float('nan'),
            'oof_precision': metric_or_nan(precision_score, y_oof, oof_pred_labels, zero_division=0) if np.any(valid_oof) else float('nan'),
            'oof_recall': metric_or_nan(recall_score, y_oof, oof_pred_labels, zero_division=0) if np.any(valid_oof) else float('nan'),
            'folds_used': int(len(rows)),
        })
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError('No valid CV folds were produced')
    summary_df = summary_df.sort_values(
        ['cv_pr_auc_mean', 'oof_pr_auc', 'cv_logloss_mean', 'cv_brier_mean', 'cv_balanced_accuracy_mean', 'model'],
        ascending=[False, False, True, True, False, True],
    ).reset_index(drop=True)
    best_name = str(summary_df.iloc[0]['model'])
    fold_df = pd.DataFrame(fold_rows).sort_values(['model', 'fold']).reset_index(drop=True) if fold_rows else pd.DataFrame()
    return best_name, summary_df, fold_df


def fit_final_model(X: pd.DataFrame, y: np.ndarray, estimator_map: Dict[str, object], best_name: str):
    estimator = estimator_map[best_name]
    final_est = estimator.__class__(**estimator.get_params())
    final_est.fit(X, y)
    return final_est


def lower_tail_weighted_mean(values: np.ndarray, valid: np.ndarray, tail_fraction: float = 0.25, rank_power: float = 1.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool)
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
    out[np.isfinite(joint_min) & ~np.isfinite(tail_mean)] = joint_min[np.isfinite(joint_min) & ~np.isfinite(tail_mean)]
    out[~np.isfinite(joint_min) & np.isfinite(tail_mean)] = tail_mean[~np.isfinite(joint_min) & np.isfinite(tail_mean)]
    return np.clip(out, 0.0, 1.0)


def finite_min_score(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool)
    safe = np.where(ok, arr, np.inf)
    out = np.min(safe, axis=0)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def _estimate_regular_step(values: np.ndarray) -> float:
    if values.size < 2:
        return float('nan')
    diffs = np.diff(np.sort(np.unique(values)))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else float('nan')


def _grid_edges(values: np.ndarray, step: float) -> np.ndarray:
    vals = np.sort(np.unique(values))
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if vals.size == 1 or not np.isfinite(step) or step <= 0:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = (vals[:-1] + vals[1:]) * 0.5
    edges[0] = vals[0] - step * 0.5
    edges[-1] = vals[-1] + step * 0.5
    return edges


def _try_regular_grid(sub: pd.DataFrame, x_col: str, y_col: str, value_col: str):
    x_vals = pd.to_numeric(sub[x_col], errors='coerce').to_numpy(dtype=float)
    y_vals = pd.to_numeric(sub[y_col], errors='coerce').to_numpy(dtype=float)
    z_vals = pd.to_numeric(sub[value_col], errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
    if not np.any(ok):
        return None
    x_vals = x_vals[ok]
    y_vals = y_vals[ok]
    z_vals = z_vals[ok]
    if x_vals.size < 4:
        return None
    max_cells = min(max(x_vals.size * 4, 4096), 4_000_000)
    for decimals in (8, 7, 6, 5, 4, 3):
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
        if float(np.isfinite(grid).mean()) < 0.25:
            continue
        return xs, ys, grid, _grid_edges(xs, dx), _grid_edges(ys, dy)
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


def preview_point_map(df: pd.DataFrame, value_col: str, title: str, out_path: Path, x_col: str, y_col: str, point_alpha: float, preview_coarsen: int, vmax: Optional[float], minimum_score_threshold: Optional[float] = None) -> None:
    if plt is None or value_col not in df.columns:
        return
    sub = df[[x_col, y_col, value_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors='coerce')
    sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
    sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
    thr = pd.to_numeric(minimum_score_threshold, errors='coerce')
    if not pd.isna(thr) and float(thr) > 0:
        mask = np.isfinite(sub[value_col].to_numpy(dtype=np.float32, copy=False)) & (sub[value_col].to_numpy(dtype=np.float32, copy=False) < float(thr))
        if np.any(mask):
            sub.loc[mask, value_col] = 0.0
    sub = sub.dropna(subset=[x_col, y_col, value_col])
    if sub.empty:
        return
    vmin = 0.0
    vmax2 = None if vmax is None or float(vmax) <= 0 else float(vmax)
    colorbar_extend = 'neither'
    if vmax2 is not None and np.nanmax(sub[value_col].to_numpy(dtype=np.float32, copy=False)) > vmax2:
        colorbar_extend = 'max'
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    payload = _try_regular_grid(sub, x_col, y_col, value_col)
    if payload is not None:
        _, _, grid, x_edges, y_edges = payload
        grid, x_edges, y_edges = _coarsen_regular_grid(grid, x_edges, y_edges, preview_coarsen)
        masked = np.ma.masked_invalid(grid)
        color_obj = ax.pcolormesh(x_edges, y_edges, masked, shading='flat', cmap='viridis', antialiased=False, linewidth=0, vmin=vmin, vmax=vmax2)
    else:
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax2)
        ax.scatter(sub[x_col], sub[y_col], c=sub[value_col], s=4.0, alpha=float(np.clip(point_alpha, 0.02, 1.0)), linewidths=0, rasterized=True, cmap=cmap, norm=norm)
        color_obj = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        color_obj.set_array([])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2)
    fig.colorbar(color_obj, ax=ax, shrink=0.78, extend=colorbar_extend)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)



def generate_previews_from_existing_outputs(outdir: Path, args) -> None:
    if plt is None:
        log('[previews] matplotlib not available; skipping previews')
        return
    outdir = Path(outdir)
    previews_root = outdir / 'previews'
    previews_root.mkdir(parents=True, exist_ok=True)

    overall_path = outdir / 'overall_suitability.csv'
    if overall_path.exists():
        try:
            overall_df = pd.read_csv(
                overall_path,
                usecols=[
                    args.x_col,
                    args.y_col,
                    'overall_ml',
                    'overall_ml_min',
                    'overall_ml_joint',
                    'richness_ml',
                ],
                low_memory=False,
            )
        except Exception:
            overall_df = pd.DataFrame()
        if not overall_df.empty:
            preview_point_map(overall_df, 'overall_ml', 'overall ML suitability', previews_root / 'overall_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(overall_df, 'overall_ml_min', 'overall ML minimum overlap', previews_root / 'overall_ml_min.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(overall_df, 'overall_ml_joint', 'overall ML joint suitability', previews_root / 'overall_ml_joint.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(overall_df, 'richness_ml', 'richness above species thresholds', previews_root / 'richness_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)

    community_overall_path = outdir / 'community_model' / 'community_overall.csv'
    if community_overall_path.exists():
        try:
            community_df = pd.read_csv(
                community_overall_path,
                usecols=[
                    args.x_col,
                    args.y_col,
                    'community_top_score',
                    'community_top_gap',
                    'community_effective_richness',
                    'community_richness_above_threshold',
                ],
                low_memory=False,
            )
        except Exception:
            community_df = pd.DataFrame()
        if not community_df.empty:
            community_previews = previews_root / 'community'
            community_previews.mkdir(parents=True, exist_ok=True)
            preview_point_map(community_df, 'community_top_score', 'community top-species share', community_previews / 'community_top_score.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(community_df, 'community_top_gap', 'community top-vs-second gap', community_previews / 'community_top_gap.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(community_df, 'community_effective_richness', 'community effective richness', community_previews / 'community_effective_richness.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)
            preview_point_map(community_df, 'community_richness_above_threshold', 'community richness above class thresholds', community_previews / 'community_richness_above_threshold.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)

    if int(args.preview_top_n) <= 0:
        return

    by_species_dir = outdir / 'by_species'
    if not by_species_dir.exists():
        return

    ranking_path = outdir / 'species_score_summary.csv'
    groups = []
    if ranking_path.exists():
        try:
            ranking = pd.read_csv(ranking_path, low_memory=False)
            if 'group' in ranking.columns:
                groups = ranking['group'].astype(str).tolist()
        except Exception:
            groups = []
    if not groups:
        groups = sorted([p.stem for p in by_species_dir.glob('*.csv')])

    limit = int(args.preview_top_n)
    groups = groups[:limit] if limit > 0 else groups
    if groups:
        pd.DataFrame({'group': groups}).to_csv(outdir / 'species_preview_ranking.csv', index=False)

    out_base = previews_root / 'by_species'
    out_base.mkdir(parents=True, exist_ok=True)

    for group in groups:
        src = by_species_dir / f"{safe_slug(group)}.csv"
        if not src.exists():
            alt = by_species_dir / f"{group}.csv"
            if alt.exists():
                src = alt
            else:
                continue
        value_col = None
        for candidate in ('ml_probability', 'ml_suitability'):
            try:
                dfp = pd.read_csv(src, usecols=[args.x_col, args.y_col, candidate], low_memory=False)
                value_col = candidate
                break
            except Exception:
                dfp = pd.DataFrame()
        if value_col is None or dfp.empty:
            continue
        preview_point_map(dfp, value_col, f'{group} ML suitability', out_base / f"{safe_slug(group)}_ml.png", args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)

def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')



def build_feature_spec_from_saved_payload(payload: Dict[str, object]) -> FeatureSpec:
    numeric_cols = [str(col) for col in payload.get('numeric_cols', []) if str(col)]
    categorical_cols = [str(col) for col in payload.get('categorical_cols', []) if str(col)]
    raw_feature_columns = [str(col) for col in payload.get('raw_feature_columns', []) if str(col)]
    categorical_vocab: Dict[str, List[str]] = {}
    for key, values in (payload.get('categorical_vocab') or {}).items():
        norm_values = []
        for value in values or []:
            nv = normalize_text(value)
            if nv and nv not in norm_values:
                norm_values.append(nv)
        categorical_vocab[str(key)] = norm_values
    encoded_columns = list(raw_feature_columns)
    if not encoded_columns:
        encoded_columns = list(numeric_cols)
        for col in categorical_cols:
            for value in categorical_vocab.get(col, []):
                encoded_columns.append(f"{col}__{safe_slug(value)}")
    return FeatureSpec(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        categorical_vocab=categorical_vocab,
        encoded_columns=encoded_columns,
        feature_priority={},
    )



def align_transformed_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    aligned = {}
    n = len(df)
    for col in columns:
        key = str(col)
        if key in df.columns:
            aligned[key] = df[key].to_numpy(copy=False)
        elif '__' in key:
            aligned[key] = np.zeros(n, dtype=np.float32)
        else:
            aligned[key] = np.full(n, np.nan, dtype=np.float32)
    return pd.DataFrame(aligned, index=df.index)



def load_saved_group_metadata(reuse_dir: Path) -> pd.DataFrame:
    selected_groups_path = reuse_dir / 'selected_groups.csv'
    species_summary_path = reuse_dir / 'species_score_summary.csv'
    selected_groups = pd.read_csv(selected_groups_path, low_memory=False) if selected_groups_path.exists() else pd.DataFrame()
    species_summary = pd.read_csv(species_summary_path, low_memory=False) if species_summary_path.exists() else pd.DataFrame()

    if not species_summary.empty:
        meta = species_summary.copy()
        if not selected_groups.empty:
            extra_cols = [col for col in selected_groups.columns if col not in meta.columns or col == 'group']
            selected_extra = selected_groups[extra_cols].copy()
            meta = meta.merge(selected_extra, on='group', how='left', suffixes=('', '_selected'))
            for col in ['matched_species_name', *TAXON_RANKS, 'occurrence_count']:
                alt = f'{col}_selected'
                if alt in meta.columns:
                    if col not in meta.columns:
                        meta[col] = meta[alt]
                    else:
                        meta[col] = meta[col].where(meta[col].notna(), meta[alt])
                    meta = meta.drop(columns=[alt])
    elif not selected_groups.empty:
        meta = selected_groups.copy()
    else:
        raise FileNotFoundError(f'Could not find selected_groups.csv or species_score_summary.csv under {reuse_dir}')

    if 'group' not in meta.columns:
        raise KeyError(f'Saved metadata under {reuse_dir} is missing the group column')

    for col in ['group', 'matched_species_name', *TAXON_RANKS, 'occurrence_count', 'model_name', 'model_path']:
        if col not in meta.columns:
            meta[col] = pd.NA
    return meta



def resolve_saved_model_path(reuse_dir: Path, row: pd.Series) -> Path:
    slug = safe_slug(row.get('group', 'group'))
    candidate = reuse_dir / 'models' / f'{slug}.joblib'
    if candidate.exists():
        return candidate.resolve()
    model_path = row.get('model_path', pd.NA)
    if pd.notna(model_path) and str(model_path).strip():
        candidate = Path(str(model_path)).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return (reuse_dir / 'models' / f'{slug}.joblib').resolve()



def run_saved_model_deployment(args, outdir: Path) -> int:
    reuse_dir = Path(args.reuse_models_from).expanduser().resolve()
    if not reuse_dir.exists() or not reuse_dir.is_dir():
        raise FileNotFoundError(f'Missing reuse-models directory: {reuse_dir}')

    deploy_grid_value = args.deploy_grid_csv or args.grid_csv or args.occurrences_csv
    if not deploy_grid_value:
        raise ValueError('A grid CSV is required in reuse mode. Pass it positionally or via --deploy-grid-csv.')
    grid_csv = Path(deploy_grid_value).expanduser().resolve()
    if not grid_csv.exists():
        raise FileNotFoundError(f'Missing deploy grid CSV: {grid_csv}')

    include_selectors = parse_selector_list(args.include_taxa)
    exclude_selectors = parse_selector_list(args.exclude_taxa)

    saved_meta = load_saved_group_metadata(reuse_dir)
    saved_meta = apply_taxon_filters(saved_meta, include_selectors, exclude_selectors, args.group_by)
    if args.max_groups > 0 and len(saved_meta) > int(args.max_groups):
        occ_counts = pd.to_numeric(saved_meta.get('occurrence_count', pd.Series(index=saved_meta.index, dtype=float)), errors='coerce').fillna(0)
        saved_meta = saved_meta.assign(__occurrence_count__=occ_counts)
        saved_meta = saved_meta.sort_values(['__occurrence_count__', 'group'], ascending=[False, True]).head(int(args.max_groups)).drop(columns=['__occurrence_count__'])
    saved_meta = saved_meta.reset_index(drop=True)
    if saved_meta.empty:
        raise RuntimeError('No saved models remain after taxon filtering')

    log(f'[reuse] source_models={reuse_dir}')
    log(f'[load] grid={grid_csv}')

    model_infos = []
    union_numeric_cols = set()
    union_categorical_cols = set()
    feature_rows = []
    for row in saved_meta.itertuples(index=False):
        row_dict = dict(row._asdict())
        group = str(row_dict.get('group', ''))
        model_path = resolve_saved_model_path(reuse_dir, pd.Series(row_dict))
        if not model_path.exists():
            log(f'[skip] group={group} missing_model={model_path}')
            continue
        payload = joblib.load(model_path)
        spec = build_feature_spec_from_saved_payload(payload)
        final_feature_columns = [str(col) for col in payload.get('final_feature_columns', []) if str(col)]
        raw_feature_columns = [str(col) for col in payload.get('raw_feature_columns', []) if str(col)] or list(spec.encoded_columns)
        model_name = str(payload.get('model_name', row_dict.get('model_name', 'saved_model')))
        model_infos.append({
            'group': group,
            'row': row_dict,
            'model_path': model_path,
            'model_name': model_name,
            'numeric_cols': list(spec.numeric_cols),
            'categorical_cols': list(spec.categorical_cols),
            'raw_feature_columns': raw_feature_columns,
            'final_feature_columns': final_feature_columns,
        })
        union_numeric_cols.update(spec.numeric_cols)
        union_categorical_cols.update(spec.categorical_cols)
        numeric_set = set(spec.numeric_cols)
        categorical_set = set(spec.categorical_cols)
        final_set = set(final_feature_columns)
        for feature in raw_feature_columns:
            base_feature = str(feature).split('__', 1)[0]
            if feature in numeric_set and '__' not in str(feature):
                kind = 'numeric'
            elif base_feature in categorical_set:
                kind = 'categorical'
            else:
                kind = 'encoded'
            feature_rows.append({
                'group': group,
                'feature': str(feature),
                'source_feature': base_feature,
                'kind': kind,
                'selected_for_model': int(str(feature) in final_set),
                'model_name': model_name,
            })

    if not model_infos:
        raise RuntimeError('No saved models could be loaded from the reuse directory')

    pd.DataFrame([info['row'] for info in model_infos]).to_csv(outdir / 'selected_groups.csv', index=False)

    grid_head_raw = pd.read_csv(grid_csv, nrows=2000, low_memory=False)
    grid_head_work, grid_tc_meta = build_terraclimate_working_df(grid_head_raw)
    if args.id_col not in grid_head_raw.columns:
        raise KeyError(f'Grid id column not found in deploy grid CSV: {args.id_col}')
    if args.x_col not in grid_head_raw.columns or args.y_col not in grid_head_raw.columns:
        raise KeyError(f'Grid coordinate columns not found in deploy grid CSV: {args.x_col}, {args.y_col}')

    grid_usecols = resolve_required_raw_columns(
        grid_head_raw,
        grid_tc_meta,
        sorted(union_numeric_cols) + sorted(union_categorical_cols),
        [args.id_col, args.x_col, args.y_col],
    )
    grid_raw = read_minimal_csv(grid_csv, usecols=grid_usecols)
    grid_work, _ = build_terraclimate_working_df(grid_raw)
    grid_work[args.id_col] = grid_raw[args.id_col]
    grid_work[args.x_col] = pd.to_numeric(grid_raw[args.x_col], errors='coerce')
    grid_work[args.y_col] = pd.to_numeric(grid_raw[args.y_col], errors='coerce')
    grid_work = grid_work.dropna(subset=[args.x_col, args.y_col]).reset_index(drop=True)

    if grid_work.empty:
        raise RuntimeError('No valid grid rows remain after dropping missing coordinates')

    pd.DataFrame(feature_rows).to_csv(outdir / 'selected_features.csv', index=False)

    pred_stack = []
    pred_raw_stack = []
    thresholds = []
    artifacts = []
    species_rows = []
    by_species_dir = outdir / 'by_species'
    by_species_dir.mkdir(parents=True, exist_ok=True)

    for info in model_infos:
        group = str(info['group'])
        slug = safe_slug(group)
        payload = joblib.load(info['model_path'])
        spec = build_feature_spec_from_saved_payload(payload)
        raw_feature_columns = [str(col) for col in payload.get('raw_feature_columns', []) if str(col)] or list(spec.encoded_columns)
        final_feature_columns = [str(col) for col in payload.get('final_feature_columns', []) if str(col)]
        if not final_feature_columns:
            raise RuntimeError(f'Saved model for {group} is missing final_feature_columns')
        grid_X_all = transform_features(grid_work, spec)
        grid_X_aligned = align_transformed_features(grid_X_all, raw_feature_columns)
        imputer = payload.get('imputer')
        final_model = payload.get('model')
        if imputer is None or final_model is None:
            raise RuntimeError(f'Saved model payload is incomplete for {group}: {info["model_path"]}')
        grid_X_imp = pd.DataFrame(imputer.transform(grid_X_aligned), columns=raw_feature_columns, index=grid_X_aligned.index)
        grid_X_final = grid_X_imp[final_feature_columns]
        raw_proba = predict_in_chunks(final_model, grid_X_final, int(args.predict_chunk_size))

        conservative_spec = payload.get('conservative_suitability_spec')
        if isinstance(conservative_spec, dict) and conservative_spec:
            conservative_suitability, center_weight, novelty_penalty, adjustment_factor = apply_conservative_suitability_spec(grid_X_final, raw_proba, conservative_spec)
        else:
            conservative_suitability = raw_proba.astype(np.float32, copy=False)
            center_weight = np.ones(len(raw_proba), dtype=np.float32)
            novelty_penalty = np.ones(len(raw_proba), dtype=np.float32)
            adjustment_factor = np.ones(len(raw_proba), dtype=np.float32)
        deployment_mode = str(payload.get('deployment_score_mode', 'raw'))
        deployment_threshold = float(payload.get('deployment_threshold', payload.get('threshold_raw_cv', 0.5)))
        deployed_score = conservative_suitability if deployment_mode == 'conservative' and isinstance(conservative_spec, dict) and conservative_spec else raw_proba
        model_name = str(payload.get('model_name', info.get('model_name', 'saved_model')))

        species_out = pd.DataFrame({
            args.id_col: grid_work[args.id_col].tolist(),
            args.x_col: grid_work[args.x_col].tolist(),
            args.y_col: grid_work[args.y_col].tolist(),
            'ml_probability_raw': raw_proba,
            'ml_center_weight': center_weight,
            'ml_novelty_penalty': novelty_penalty,
            'ml_adjustment_factor': adjustment_factor,
            'ml_suitability': conservative_suitability,
            'ml_probability': deployed_score,
            'ml_likely': (deployed_score >= deployment_threshold).astype(np.int8),
            'model_name': model_name,
        })
        species_out.to_csv(by_species_dir / f'{slug}.csv', index=False)

        row_meta = dict(info['row'])
        species_row = {k: (None if pd.isna(v) else v) for k, v in row_meta.items()}
        species_row.update({
            'group': group,
            'model_name': model_name,
            'feature_count': int(len(final_feature_columns)),
            'deployment_feature_count': int(len((conservative_spec or {}).get('features', []))) if isinstance(conservative_spec, dict) else 0,
            'deployment_score_mode': deployment_mode,
            'deployment_threshold': deployment_threshold,
            'mean_grid_probability_raw': float(np.nanmean(raw_proba)),
            'max_grid_probability_raw': float(np.nanmax(raw_proba)),
            'mean_grid_suitability': float(np.nanmean(conservative_suitability)),
            'max_grid_suitability': float(np.nanmax(conservative_suitability)),
            'mean_grid_probability': float(np.nanmean(deployed_score)),
            'max_grid_probability': float(np.nanmax(deployed_score)),
            'model_path': str(info['model_path']),
            'source_model_run': str(reuse_dir),
        })
        species_rows.append(species_row)

        pred_stack.append(deployed_score)
        pred_raw_stack.append(raw_proba)
        thresholds.append(deployment_threshold)
        artifacts.append({'group': group, 'slug': slug, 'model_name': model_name, 'threshold': deployment_threshold, 'model_path': str(info['model_path'])})
        log(f'[reuse] group={group} model={model_name} deploy_thr={deployment_threshold:.3f} features={len(final_feature_columns):,}')

    if not pred_stack:
        raise RuntimeError('No saved models were successfully scored on the deploy grid')

    pred_arr = np.vstack(pred_stack).astype(np.float32)
    pred_raw_arr = np.vstack(pred_raw_stack).astype(np.float32)
    pred_valid = np.isfinite(pred_arr)
    pred_raw_valid = np.isfinite(pred_raw_arr)
    selected_group_names = [a['group'] for a in artifacts]
    n_rows = pred_arr.shape[1]
    safe_pred = np.where(pred_valid, pred_arr, -np.inf)
    safe_pred_raw = np.where(pred_raw_valid, pred_raw_arr, -np.inf)
    top_idx = np.argmax(safe_pred, axis=0)
    top_score = safe_pred[top_idx, np.arange(n_rows)]
    top_group = np.array([selected_group_names[i] for i in top_idx], dtype=object)
    no_score = ~np.isfinite(top_score) | (top_score == -np.inf)
    top_score = np.where(no_score, np.nan, top_score)
    top_group = np.where(no_score, '', top_group)

    overall_ml = np.max(safe_pred, axis=0)
    overall_ml = np.where(np.isfinite(overall_ml) & (overall_ml > -np.inf), overall_ml, np.nan).astype(np.float32)
    overall_ml_raw = np.max(safe_pred_raw, axis=0)
    overall_ml_raw = np.where(np.isfinite(overall_ml_raw) & (overall_ml_raw > -np.inf), overall_ml_raw, np.nan).astype(np.float32)
    overall_ml_min = finite_min_score(pred_arr, pred_valid)
    overall_ml_joint = joint_support_score(pred_arr, pred_valid, min_share=args.joint_min_share, tail_fraction=args.joint_tail_fraction, rank_power=args.joint_rank_power)
    threshold_arr = np.asarray(thresholds, dtype=np.float32)[:, None]
    richness_ml = np.sum(np.where(pred_valid, pred_arr >= threshold_arr, False), axis=0).astype(np.int32)

    overall_df = pd.DataFrame({
        args.id_col: grid_work[args.id_col].tolist(),
        args.x_col: grid_work[args.x_col].tolist(),
        args.y_col: grid_work[args.y_col].tolist(),
        'overall_ml_raw': overall_ml_raw,
        'overall_ml_suitability': overall_ml,
        'overall_ml': overall_ml,
        'overall_ml_min': overall_ml_min,
        'overall_ml_joint': overall_ml_joint,
        'richness_ml': richness_ml,
        'top_group_ml': top_group,
        'top_ml_score': top_score,
    })
    overall_df.to_csv(outdir / 'overall_suitability.csv', index=False)

    species_summary_df = pd.DataFrame(species_rows)
    sort_cols = [col for col in ['mean_grid_probability', 'max_grid_probability', 'group'] if col in species_summary_df.columns]
    ascending = [{'mean_grid_probability': False, 'max_grid_probability': False, 'group': True}[col] for col in sort_cols]
    if sort_cols:
        species_summary_df = species_summary_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    species_summary_df.to_csv(outdir / 'species_score_summary.csv', index=False)

    if int(args.preview_top_n) > 0:
        preview_dir = outdir / 'previews'
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_point_map(overall_df, 'overall_ml', 'overall ML suitability', preview_dir / 'overall_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
        preview_point_map(overall_df, 'overall_ml_min', 'overall ML minimum overlap', preview_dir / 'overall_ml_min.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
        preview_point_map(overall_df, 'overall_ml_joint', 'overall ML joint suitability', preview_dir / 'overall_ml_joint.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
        preview_point_map(overall_df, 'richness_ml', 'richness above species thresholds', preview_dir / 'richness_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)
        for row in species_summary_df.head(int(args.preview_top_n)).itertuples(index=False):
            dfp = pd.read_csv(outdir / 'by_species' / f"{safe_slug(row.group)}.csv", usecols=[args.x_col, args.y_col, 'ml_probability'], low_memory=False)
            preview_point_map(dfp, 'ml_probability', f'{row.group} ML suitability', preview_dir / 'by_species' / f"{safe_slug(row.group)}_ml.png", args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)

    source_manifest_path = reuse_dir / 'manifest.json'
    source_manifest = {}
    if source_manifest_path.exists():
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
        except Exception:
            source_manifest = {}

    manifest = {
        'mode': 'reuse_models',
        'reuse_models_from': str(reuse_dir),
        'source_manifest': source_manifest,
        'deploy_grid_csv': str(grid_csv),
        'outdir': str(outdir),
        'group_by': args.group_by,
        'include_taxa': [s.raw for s in include_selectors],
        'exclude_taxa': [s.raw for s in exclude_selectors],
        'selected_group_count': int(len(selected_group_names)),
        'selected_groups': selected_group_names,
        'predict_chunk_size': int(args.predict_chunk_size),
        'joint_min_share': float(args.joint_min_share),
        'joint_tail_fraction': float(args.joint_tail_fraction),
        'joint_rank_power': float(args.joint_rank_power),
        'score_interpretation': 'Presence-background relative suitability score deployed from previously trained saved models. ml_probability_raw is the raw tree score. ml_suitability is the conservative center-weighted score when the saved model payload includes it. This is not a true occurrence probability unless true absences and prevalence assumptions are provided.',
    }
    write_manifest(outdir / 'manifest.json', manifest)
    log(f'[done] groups={len(artifacts)} grid_rows={len(grid_work):,} outdir={outdir}')
    return 0


def predict_in_chunks(model, X: pd.DataFrame, chunk_size: int) -> np.ndarray:
    chunk_size = max(1, int(chunk_size))
    out = np.full(len(X), np.nan, dtype=np.float32)
    for start in range(0, len(X), chunk_size):
        end = min(len(X), start + chunk_size)
        out[start:end] = model.predict_proba(X.iloc[start:end])[:, 1].astype(np.float32)
    return out


def build_feature_importance_lookup(final_model, perm_df: pd.DataFrame, feature_priority: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if perm_df is not None and not perm_df.empty:
        for row in perm_df.itertuples(index=False):
            feature = str(getattr(row, 'feature', '') or '')
            if not feature:
                continue
            value = pd.to_numeric(getattr(row, 'perm_importance_mean', np.nan), errors='coerce')
            if pd.notna(value):
                out[feature] = max(0.0, float(value))
    model_importances = getattr(final_model, 'feature_importances_', None)
    if model_importances is not None and perm_df is not None and not perm_df.empty:
        for feature, value in zip(perm_df['feature'].astype(str).tolist(), np.asarray(model_importances, dtype=float)):
            out[feature] = max(float(out.get(feature, 0.0)), max(0.0, float(value)))
    for feature, value in (feature_priority or {}).items():
        feature = str(feature)
        out.setdefault(feature, max(0.0, float(value)))
    return out


def build_conservative_suitability_spec(pos_features: pd.DataFrame, final_feature_columns: Sequence[str], numeric_feature_names: Sequence[str], feature_importance: Dict[str, float], args) -> Dict[str, object]:
    numeric_set = {str(c) for c in numeric_feature_names}
    ranked = []
    for col in final_feature_columns:
        col = str(col)
        if col not in numeric_set:
            continue
        base = col.split('__', 1)[0]
        weight = max(0.0, float(feature_importance.get(col, feature_importance.get(base, 0.0))))
        ranked.append((weight, col))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    max_features = max(0, int(getattr(args, 'deployment_center_top_features', 24)))
    if max_features > 0:
        ranked = ranked[:max_features]

    q_low = float(np.clip(float(getattr(args, 'deployment_center_low_quantile', 0.10)), 0.0, 0.49))
    q_high = float(np.clip(float(getattr(args, 'deployment_center_high_quantile', 0.90)), 0.51, 1.0))
    if q_high <= q_low:
        q_low, q_high = 0.10, 0.90

    features = []
    for weight, col in ranked:
        vals = pd.to_numeric(pos_features[col], errors='coerce').to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 8:
            continue
        ql, q25, q50, q75, qh = np.nanquantile(vals, [q_low, 0.25, 0.50, 0.75, q_high])
        spread = max(float(qh - ql), float(q75 - q25), float(np.nanstd(vals)), 1e-6)
        center_scale = max(float(q75 - q25) * float(getattr(args, 'deployment_center_iqr_multiplier', 1.5)), spread * 0.35, 1e-6)
        support_pad = spread * float(getattr(args, 'deployment_support_pad_fraction', 0.15))
        features.append({
            'column': col,
            'weight': max(weight, 1e-8),
            'center': float(q50),
            'center_scale': float(center_scale),
            'support_low': float(ql - support_pad),
            'support_high': float(qh + support_pad),
            'support_scale': float(max(spread, 1e-6)),
            'q_low': float(ql),
            'q_high': float(qh),
        })
    if not features:
        return {
            'features': [],
            'raw_power': float(getattr(args, 'deployment_raw_power', 1.1)),
            'center_floor': float(getattr(args, 'deployment_center_floor', 0.4)),
            'center_power': float(getattr(args, 'deployment_center_power', 1.25)),
            'center_distance_power': float(getattr(args, 'deployment_center_distance_power', 1.6)),
            'novelty_strength': float(getattr(args, 'deployment_novelty_strength', 1.75)),
            'novelty_floor': float(getattr(args, 'deployment_novelty_floor', 0.08)),
        }
    weights = np.asarray([max(float(item['weight']), 0.0) for item in features], dtype=float)
    if not np.any(weights > 0):
        weights = np.ones(len(features), dtype=float)
    weights = weights / np.sum(weights)
    for item, weight in zip(features, weights):
        item['weight_norm'] = float(weight)
    return {
        'features': features,
        'raw_power': float(getattr(args, 'deployment_raw_power', 1.1)),
        'center_floor': float(np.clip(float(getattr(args, 'deployment_center_floor', 0.4)), 0.0, 1.0)),
        'center_power': float(max(0.1, float(getattr(args, 'deployment_center_power', 1.25)))),
        'center_distance_power': float(max(0.1, float(getattr(args, 'deployment_center_distance_power', 1.6)))),
        'novelty_strength': float(max(0.0, float(getattr(args, 'deployment_novelty_strength', 1.75)))),
        'novelty_floor': float(np.clip(float(getattr(args, 'deployment_novelty_floor', 0.08)), 0.0, 1.0)),
    }


def apply_conservative_suitability_spec(X_features: pd.DataFrame, raw_scores: np.ndarray, conservative_spec: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_scores, dtype=np.float32)
    n = len(raw)
    raw_valid = np.isfinite(raw)
    if not conservative_spec or not conservative_spec.get('features'):
        ones = np.where(raw_valid, 1.0, np.nan).astype(np.float32)
        return raw.copy(), ones, ones, ones

    center_acc = np.zeros(n, dtype=np.float64)
    novelty_acc = np.zeros(n, dtype=np.float64)
    weight_acc = np.zeros(n, dtype=np.float64)
    center_distance_power = float(conservative_spec.get('center_distance_power', 1.6))
    for feat in conservative_spec['features']:
        col = str(feat['column'])
        if col not in X_features.columns:
            continue
        vals = pd.to_numeric(X_features[col], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if not np.any(valid):
            continue
        weight = float(feat.get('weight_norm', 0.0))
        if weight <= 0:
            continue
        dist = np.zeros(n, dtype=np.float64)
        dist[valid] = np.abs(vals[valid] - float(feat['center'])) / max(float(feat['center_scale']), 1e-6)
        center_score = np.zeros(n, dtype=np.float64)
        center_score[valid] = 1.0 / (1.0 + np.power(dist[valid], center_distance_power))

        outside = np.zeros(n, dtype=np.float64)
        low = float(feat['support_low'])
        high = float(feat['support_high'])
        support_scale = max(float(feat['support_scale']), 1e-6)
        low_mask = valid & (vals < low)
        high_mask = valid & (vals > high)
        outside[low_mask] = (low - vals[low_mask]) / support_scale
        outside[high_mask] = (vals[high_mask] - high) / support_scale
        outside = np.clip(outside, 0.0, 6.0)

        center_acc[valid] += weight * center_score[valid]
        novelty_acc[valid] += weight * outside[valid]
        weight_acc[valid] += weight

    center_core = np.full(n, np.nan, dtype=np.float64)
    novelty_core = np.full(n, np.nan, dtype=np.float64)
    good = weight_acc > 0
    center_core[good] = np.clip(center_acc[good] / weight_acc[good], 0.0, 1.0)
    novelty_core[good] = np.exp(-float(conservative_spec.get('novelty_strength', 1.75)) * (novelty_acc[good] / weight_acc[good]))

    center_floor = float(conservative_spec.get('center_floor', 0.4))
    center_power = float(conservative_spec.get('center_power', 1.25))
    novelty_floor = float(conservative_spec.get('novelty_floor', 0.08))
    center_weight = np.full(n, np.nan, dtype=np.float64)
    novelty_penalty = np.full(n, np.nan, dtype=np.float64)
    center_weight[good] = center_floor + (1.0 - center_floor) * np.power(center_core[good], center_power)
    novelty_penalty[good] = novelty_floor + (1.0 - novelty_floor) * np.clip(novelty_core[good], 0.0, 1.0)

    conservative = np.full(n, np.nan, dtype=np.float64)
    adjustment_factor = np.full(n, np.nan, dtype=np.float64)
    adjustment_factor[good] = np.clip(center_weight[good] * novelty_penalty[good], 0.0, 1.0)
    conservative[good & raw_valid] = np.power(np.clip(raw[good & raw_valid], 0.0, 1.0), float(conservative_spec.get('raw_power', 1.1))) * adjustment_factor[good & raw_valid]
    conservative = np.clip(conservative, 0.0, 1.0)
    conservative[~raw_valid] = np.nan
    center_weight[~raw_valid] = np.nan
    novelty_penalty[~raw_valid] = np.nan
    adjustment_factor[~raw_valid] = np.nan
    return conservative.astype(np.float32), center_weight.astype(np.float32), novelty_penalty.astype(np.float32), adjustment_factor.astype(np.float32)



def build_multiclass_cv_splitter(y: np.ndarray, coords_x: np.ndarray, coords_y: np.ndarray, desired_folds: int, block_size: float, random_state: int):
    y_arr = np.asarray(y, dtype=np.int32)
    _, class_counts = np.unique(y_arr, return_counts=True)
    min_class = int(class_counts.min()) if class_counts.size else 0
    finite_coords = np.isfinite(coords_x).all() and np.isfinite(coords_y).all()
    if finite_coords:
        groups = build_spatial_groups(coords_x, coords_y, block_size)
        unique_groups = np.unique(groups)
        n_splits = min(int(desired_folds), int(unique_groups.size))
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits), groups, 'groupkfold'
    n_splits = min(int(desired_folds), max(0, min_class))
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)), None, 'stratified'
    return None, None, 'none'


def multiclass_topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(proba, dtype=float)
    if y.size == 0 or p.ndim != 2 or p.shape[0] != y.size:
        return float('nan')
    valid = np.isfinite(p).all(axis=1)
    if not np.any(valid):
        return float('nan')
    y = y[valid]
    p = p[valid]
    kk = max(1, min(int(k), int(p.shape[1])))
    top_idx = np.argpartition(-p, kth=kk - 1, axis=1)[:, :kk]
    hits = np.any(top_idx == y[:, None], axis=1)
    return float(np.mean(hits)) if hits.size else float('nan')


def compute_multiclass_class_weights(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.int32)
    if y_arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    classes, counts = np.unique(y_arr, return_counts=True)
    total = float(y_arr.size)
    n_classes = max(1, len(classes))
    lookup = {int(cls): total / (n_classes * max(1, int(cnt))) for cls, cnt in zip(classes, counts)}
    return np.asarray([lookup.get(int(cls), 1.0) for cls in y_arr], dtype=np.float32)


def build_community_estimator(args, n_classes: int):
    if HAVE_XGBOOST:
        return 'xgboost_multiclass', XGBClassifier(
            objective='multi:softprob',
            num_class=int(n_classes),
            tree_method='hist',
            n_estimators=int(args.xgb_n_estimators),
            max_depth=int(args.xgb_max_depth),
            learning_rate=float(args.xgb_learning_rate),
            subsample=float(args.xgb_subsample),
            colsample_bytree=float(args.xgb_colsample_bytree),
            min_child_weight=float(args.xgb_min_child_weight),
            reg_alpha=float(args.xgb_reg_alpha),
            reg_lambda=float(args.xgb_reg_lambda),
            gamma=float(args.xgb_gamma),
            max_bin=int(args.xgb_max_bin),
            n_jobs=-1,
            random_state=int(args.random_state),
            eval_metric='mlogloss',
            verbosity=0,
            missing=np.nan,
        )
    return 'extratrees_multiclass', ExtraTreesClassifier(
        n_estimators=int(args.et_n_estimators),
        max_depth=None if int(args.et_max_depth) <= 0 else int(args.et_max_depth),
        min_samples_leaf=int(args.et_min_samples_leaf),
        max_features=args.et_max_features,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=int(args.random_state),
    )


def fit_estimator_with_optional_weights(estimator, X, y, sample_weight: Optional[np.ndarray] = None):
    if sample_weight is None:
        estimator.fit(X, y)
    else:
        try:
            estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            estimator.fit(X, y)
    return estimator


def remap_multiclass_labels_contiguous(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.int32)
    present_classes = np.unique(y_arr).astype(np.int32)
    class_to_local = {int(cls): idx for idx, cls in enumerate(present_classes.tolist())}
    y_local = np.asarray([class_to_local[int(cls)] for cls in y_arr], dtype=np.int32)
    return y_local, present_classes


def expand_multiclass_fold_probabilities(
    proba_local: np.ndarray,
    present_classes: np.ndarray,
    estimator_classes: Optional[np.ndarray],
    n_classes: int,
    eps: float = 1e-6,
) -> np.ndarray:
    local = np.asarray(proba_local, dtype=np.float32)
    if local.ndim == 1:
        local = local[:, None]
    present = np.asarray(present_classes, dtype=np.int32)
    if estimator_classes is None:
        estimator_local_classes = np.arange(local.shape[1], dtype=np.int32)
    else:
        estimator_local_classes = np.asarray(estimator_classes, dtype=np.int32)
    if local.shape[1] != len(estimator_local_classes):
        raise ValueError('predict_proba column count does not match estimator classes for community CV fold')
    if estimator_local_classes.size == 0:
        raise ValueError('Community CV fold estimator returned no classes')
    if np.any(estimator_local_classes < 0) or np.any(estimator_local_classes >= len(present)):
        raise ValueError('Community CV fold estimator classes are out of range for remapped labels')

    full_class_indices = present[estimator_local_classes]
    expanded = np.full((local.shape[0], int(n_classes)), float(eps), dtype=np.float32)
    expanded[:, full_class_indices] = np.maximum(local, float(eps))
    row_sums = expanded.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    expanded = expanded / row_sums
    return expanded.astype(np.float32, copy=False)


def evaluate_multiclass_model_cv(X: pd.DataFrame, y: np.ndarray, coords_x: np.ndarray, coords_y: np.ndarray, args, estimator, class_names: Sequence[str]):
    splitter, groups, cv_scheme = build_multiclass_cv_splitter(y, coords_x, coords_y, int(args.cv_folds), float(args.cv_block_size), int(args.random_state))
    if splitter is None:
        raise RuntimeError('No valid community-model CV folds were produced')

    y_arr = np.asarray(y, dtype=np.int32)
    n_classes = int(len(class_names))
    oof_pred = np.full((len(y_arr), n_classes), np.nan, dtype=np.float32)
    fold_rows: List[Dict[str, float]] = []
    bx = by = None
    if groups is not None and int(args.cv_buffer_blocks) > 0:
        bx, by, _, _ = compute_spatial_block_indices(coords_x, coords_y, float(args.cv_block_size))

    split_iter = splitter.split(X, y_arr, groups=groups) if groups is not None else splitter.split(X, y_arr)
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        original_train_rows = int(len(train_idx))
        if bx is not None and by is not None and int(args.cv_buffer_blocks) > 0:
            train_idx = buffered_train_indices(train_idx, test_idx, bx, by, int(args.cv_buffer_blocks))
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        x_train = X.iloc[train_idx]
        y_train = y_arr[train_idx]
        x_test = X.iloc[test_idx]
        y_test = y_arr[test_idx]
        train_classes = np.unique(y_train).astype(np.int32)
        test_classes = np.unique(y_test).astype(np.int32)
        if train_classes.size < 2 or test_classes.size < 2:
            continue
        y_train_local, present_train_classes = remap_multiclass_labels_contiguous(y_train)
        est = clone(estimator)
        weights_train = compute_multiclass_class_weights(y_train_local)
        fit_t0 = time.perf_counter()
        fit_estimator_with_optional_weights(est, x_train, y_train_local, weights_train)
        fit_seconds = time.perf_counter() - fit_t0
        pred_t0 = time.perf_counter()
        proba_local = np.asarray(est.predict_proba(x_test), dtype=np.float32)
        predict_seconds = time.perf_counter() - pred_t0
        estimator_classes = getattr(est, 'classes_', None)
        proba = expand_multiclass_fold_probabilities(proba_local, present_train_classes, estimator_classes, n_classes)
        pred = np.argmax(proba, axis=1).astype(np.int32)
        oof_pred[test_idx, :] = proba
        fold_rows.append({
            'fold': int(fold_idx),
            'cv_scheme': cv_scheme,
            'train_rows': int(len(train_idx)),
            'train_rows_before_buffer': int(original_train_rows),
            'test_rows': int(len(test_idx)),
            'train_class_count': int(train_classes.size),
            'test_class_count': int(test_classes.size),
            'missing_train_class_count': int(n_classes - train_classes.size),
            'missing_test_class_count': int(n_classes - test_classes.size),
            'accuracy': metric_or_nan(accuracy_score, y_test, pred),
            'balanced_accuracy': metric_or_nan(balanced_accuracy_score, y_test, pred),
            'macro_f1': metric_or_nan(f1_score, y_test, pred, average='macro', zero_division=0),
            'macro_precision': metric_or_nan(precision_score, y_test, pred, average='macro', zero_division=0),
            'macro_recall': metric_or_nan(recall_score, y_test, pred, average='macro', zero_division=0),
            'top3_accuracy': multiclass_topk_accuracy(y_test, proba, 3),
            'logloss': metric_or_nan(log_loss, y_test, proba, labels=list(range(n_classes))),
            'fit_seconds': float(fit_seconds),
            'predict_seconds': float(predict_seconds),
        })

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        raise RuntimeError('Community-model CV produced no valid folds')

    valid = np.isfinite(oof_pred).all(axis=1)
    y_valid = y_arr[valid]
    oof_valid = oof_pred[valid]
    oof_pred_labels = np.argmax(oof_valid, axis=1).astype(np.int32)
    summary = {
        'model': getattr(estimator, '__class__', type(estimator)).__name__,
        'cv_scheme': str(fold_df['cv_scheme'].iloc[0]),
        'cv_accuracy_mean': float(pd.to_numeric(fold_df['accuracy'], errors='coerce').mean()),
        'cv_balanced_accuracy_mean': float(pd.to_numeric(fold_df['balanced_accuracy'], errors='coerce').mean()),
        'cv_macro_f1_mean': float(pd.to_numeric(fold_df['macro_f1'], errors='coerce').mean()),
        'cv_macro_precision_mean': float(pd.to_numeric(fold_df['macro_precision'], errors='coerce').mean()),
        'cv_macro_recall_mean': float(pd.to_numeric(fold_df['macro_recall'], errors='coerce').mean()),
        'cv_top3_accuracy_mean': float(pd.to_numeric(fold_df['top3_accuracy'], errors='coerce').mean()),
        'cv_logloss_mean': float(pd.to_numeric(fold_df['logloss'], errors='coerce').mean()),
        'cv_fit_seconds_total': float(pd.to_numeric(fold_df['fit_seconds'], errors='coerce').sum()),
        'cv_predict_seconds_total': float(pd.to_numeric(fold_df['predict_seconds'], errors='coerce').sum()),
        'oof_accuracy': metric_or_nan(accuracy_score, y_valid, oof_pred_labels),
        'oof_balanced_accuracy': metric_or_nan(balanced_accuracy_score, y_valid, oof_pred_labels),
        'oof_macro_f1': metric_or_nan(f1_score, y_valid, oof_pred_labels, average='macro', zero_division=0),
        'oof_macro_precision': metric_or_nan(precision_score, y_valid, oof_pred_labels, average='macro', zero_division=0),
        'oof_macro_recall': metric_or_nan(recall_score, y_valid, oof_pred_labels, average='macro', zero_division=0),
        'oof_top3_accuracy': multiclass_topk_accuracy(y_valid, oof_valid, 3),
        'oof_logloss': metric_or_nan(log_loss, y_valid, oof_valid, labels=list(range(n_classes))),
    }

    class_rows: List[Dict[str, float]] = []
    class_thresholds = np.full(n_classes, np.nan, dtype=np.float32)
    for class_idx, class_name in enumerate(class_names):
        y_bin = (y_valid == class_idx).astype(np.int8)
        scores = oof_valid[:, class_idx]
        threshold = float(pick_threshold(y_bin, scores)) if np.any(y_bin == 1) else float('nan')
        if not np.isfinite(threshold):
            threshold = max(0.02, min(0.5, 1.0 / max(2, n_classes)))
        class_thresholds[class_idx] = np.float32(threshold)
        pred_bin = (scores >= threshold).astype(np.int8)
        pred_label_bin = (oof_pred_labels == class_idx).astype(np.int8)
        class_rows.append({
            'group': str(class_name),
            'class_index': int(class_idx),
            'occurrence_rows_used': int(np.sum(y_arr == class_idx)),
            'oof_rows_scored': int(np.sum(y_valid == class_idx)),
            'mean_true_class_probability': float(np.nanmean(scores[y_bin == 1])) if np.any(y_bin == 1) else float('nan'),
            'oof_threshold': float(threshold),
            'oof_precision_thresholded': metric_or_nan(precision_score, y_bin, pred_bin, zero_division=0),
            'oof_recall_thresholded': metric_or_nan(recall_score, y_bin, pred_bin, zero_division=0),
            'oof_f1_thresholded': metric_or_nan(f1_score, y_bin, pred_bin, zero_division=0),
            'oof_precision_argmax': metric_or_nan(precision_score, y_bin, pred_label_bin, zero_division=0),
            'oof_recall_argmax': metric_or_nan(recall_score, y_bin, pred_label_bin, zero_division=0),
            'oof_f1_argmax': metric_or_nan(f1_score, y_bin, pred_label_bin, zero_division=0),
        })
    class_metrics_df = pd.DataFrame(class_rows).sort_values(['occurrence_rows_used', 'group'], ascending=[False, True]).reset_index(drop=True)
    return summary, fold_df, class_metrics_df, class_thresholds


def train_and_write_community_model(outdir: Path, args, spec: FeatureSpec, group_meta: pd.DataFrame, occ_work: pd.DataFrame, grid_work: pd.DataFrame) -> Dict[str, object]:
    community_dir = outdir / 'community_model'
    community_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = [str(g) for g in group_meta['group'].astype(str).tolist()]
    log(f'[community] preparing shared model candidate_groups={len(selected_groups)}')
    if not selected_groups:
        return {'enabled': True, 'status': 'no_groups'}

    community_occ = occ_work[occ_work['__group__'].astype(str).isin(set(selected_groups))].copy()
    community_occ = community_occ.dropna(subset=[args.occ_x_col, args.occ_y_col]).reset_index(drop=True)
    if bool(args.dedupe_positive_by_coords):
        community_occ['__coord_key__'] = community_occ[args.occ_x_col].round(6).astype(str) + '|' + community_occ[args.occ_y_col].round(6).astype(str)
        community_occ = community_occ.drop_duplicates(subset=['__group__', '__coord_key__']).drop(columns=['__coord_key__'])

    min_occ = max(2, int(getattr(args, 'community_min_occurrences_per_group', 20)))
    group_counts = community_occ['__group__'].astype(str).value_counts(dropna=False)
    keep_groups = [g for g in selected_groups if int(group_counts.get(g, 0)) >= min_occ]
    community_occ = community_occ[community_occ['__group__'].astype(str).isin(set(keep_groups))].copy().reset_index(drop=True)
    if community_occ.empty or len(keep_groups) < 2:
        log('[community] skipped not enough groups with occurrence rows after cleanup')
        return {'enabled': True, 'status': 'skipped_insufficient_groups'}

    max_per_group = int(getattr(args, 'community_max_occurrences_per_group', 50000))
    rng = np.random.default_rng(int(args.random_state) + 1001)
    if max_per_group > 0:
        sampled_parts = []
        for group in keep_groups:
            gdf = community_occ[community_occ['__group__'].astype(str) == group]
            if len(gdf) > max_per_group:
                choose = np.sort(rng.choice(np.arange(len(gdf), dtype=np.int64), size=max_per_group, replace=False))
                gdf = gdf.iloc[choose]
            sampled_parts.append(gdf)
        community_occ = pd.concat(sampled_parts, ignore_index=True)

    class_names = sorted(community_occ['__group__'].astype(str).unique().tolist())
    log(f'[community] training groups={len(class_names)} occurrence_rows={len(community_occ):,}')
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y_multi = community_occ['__group__'].astype(str).map(class_to_idx).to_numpy(dtype=np.int32)
    coords_x = pd.to_numeric(community_occ[args.occ_x_col], errors='coerce').to_numpy(dtype=float)
    coords_y = pd.to_numeric(community_occ[args.occ_y_col], errors='coerce').to_numpy(dtype=float)

    X_all = transform_features(community_occ, spec)
    subset_idx = stratified_cap_rows(y_multi, int(getattr(args, 'community_feature_select_max_rows', args.feature_select_max_rows)), int(args.random_state) + 7)
    X_select = X_all.iloc[subset_idx].copy()
    keep = variance_filter(X_select, threshold=float(args.variance_threshold))
    if not keep:
        log('[community] skipped no features survived variance filter')
        return {'enabled': True, 'status': 'skipped_no_features_after_variance'}
    X_all = X_all[keep].copy()
    X_select = X_select[keep].copy()
    corr_keep = correlation_prune(X_select, X_select.columns, threshold=float(args.corr_threshold))
    corr_keep = sorted(corr_keep, key=lambda c: (-float(spec.feature_priority.get(c.split('__', 1)[0], 0.0)), c))
    max_after_corr = int(getattr(args, 'community_max_features_after_corr', args.max_features_after_corr))
    if max_after_corr > 0:
        corr_keep = corr_keep[:max_after_corr]
    if not corr_keep:
        log('[community] skipped no features survived correlation prune')
        return {'enabled': True, 'status': 'skipped_no_features_after_correlation'}
    X_final = X_all[corr_keep].copy()

    estimator_name, estimator = build_community_estimator(args, len(class_names))
    use_imputer = estimator_name.startswith('extratrees')
    imputer = None
    if use_imputer:
        imp_idx = stratified_cap_rows(y_multi, min(len(y_multi), int(getattr(args, 'community_imputer_fit_max_rows', 200000))), int(args.random_state) + 17)
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_final.iloc[imp_idx])
        X_final = pd.DataFrame(imputer.transform(X_final), columns=corr_keep, index=X_final.index)

    cv_summary, cv_folds_df, class_metrics_df, class_thresholds = evaluate_multiclass_model_cv(X_final, y_multi, coords_x, coords_y, args, estimator, class_names)
    final_model = clone(estimator)
    weights_full = compute_multiclass_class_weights(y_multi)
    fit_estimator_with_optional_weights(final_model, X_final, y_multi, weights_full)

    model_payload = {
        'model_name': estimator_name,
        'model': final_model,
        'class_names': class_names,
        'class_thresholds': class_thresholds.tolist(),
        'raw_feature_columns': list(corr_keep),
        'final_feature_columns': list(corr_keep),
        'categorical_vocab': spec.categorical_vocab,
        'numeric_cols': spec.numeric_cols,
        'categorical_cols': spec.categorical_cols,
    }
    if imputer is not None:
        model_payload['imputer'] = imputer
    joblib.dump(model_payload, community_dir / 'community_model.joblib')

    cv_folds_df.to_csv(community_dir / 'community_cv_folds.csv', index=False)

    grid_X = transform_features(grid_work, spec)[corr_keep].copy()
    if imputer is not None:
        grid_X = pd.DataFrame(imputer.transform(grid_X), columns=corr_keep, index=grid_X.index)

    n_rows = len(grid_work)
    n_classes = len(class_names)
    top_idx_arr = np.full(n_rows, -1, dtype=np.int32)
    top_score_arr = np.full(n_rows, np.nan, dtype=np.float32)
    second_score_arr = np.full(n_rows, np.nan, dtype=np.float32)
    top_gap_arr = np.full(n_rows, np.nan, dtype=np.float32)
    effective_richness_arr = np.full(n_rows, np.nan, dtype=np.float32)
    richness_above_arr = np.full(n_rows, 0, dtype=np.int32)
    sum_prob = np.zeros(n_classes, dtype=np.float64)
    max_prob = np.zeros(n_classes, dtype=np.float32)
    top_wins = np.zeros(n_classes, dtype=np.int64)
    above_threshold_cells = np.zeros(n_classes, dtype=np.int64)

    chunk_size = max(1, int(getattr(args, 'predict_chunk_size', 50000)))
    threshold_override = pd.to_numeric(getattr(args, 'community_share_threshold', -1.0), errors='coerce')
    if pd.notna(threshold_override) and float(threshold_override) > 0:
        thresholds_vec = np.full(n_classes, float(threshold_override), dtype=np.float32)
        threshold_mode = 'fixed'
    else:
        thresholds_vec = np.asarray(class_thresholds, dtype=np.float32)
        fill_val = max(0.02, min(0.5, 1.0 / max(2, n_classes)))
        thresholds_vec[~np.isfinite(thresholds_vec)] = fill_val
        threshold_mode = 'oof_per_class'

    for start in range(0, n_rows, chunk_size):
        end = min(n_rows, start + chunk_size)
        proba = np.asarray(final_model.predict_proba(grid_X.iloc[start:end]), dtype=np.float32)
        top_idx = np.argmax(proba, axis=1).astype(np.int32)
        top_score = proba[np.arange(len(proba)), top_idx]
        second_score = np.partition(proba, proba.shape[1] - 2, axis=1)[:, -2] if proba.shape[1] >= 2 else np.zeros(len(proba), dtype=np.float32)
        eff_rich = 1.0 / np.maximum(np.sum(np.square(proba, dtype=np.float32), axis=1), 1e-6)
        above = np.sum(proba >= thresholds_vec[None, :], axis=1).astype(np.int32)

        top_idx_arr[start:end] = top_idx
        top_score_arr[start:end] = top_score.astype(np.float32)
        second_score_arr[start:end] = second_score.astype(np.float32)
        top_gap_arr[start:end] = (top_score - second_score).astype(np.float32)
        effective_richness_arr[start:end] = eff_rich.astype(np.float32)
        richness_above_arr[start:end] = above

        sum_prob += np.sum(proba, axis=0, dtype=np.float64)
        max_prob = np.maximum(max_prob, np.max(proba, axis=0))
        top_wins += np.bincount(top_idx, minlength=n_classes).astype(np.int64)
        above_threshold_cells += np.sum(proba >= thresholds_vec[None, :], axis=0).astype(np.int64)

    top_group = np.array([class_names[i] if i >= 0 else '' for i in top_idx_arr], dtype=object)
    community_overall = pd.DataFrame({
        args.id_col: grid_work[args.id_col].tolist(),
        args.x_col: grid_work[args.x_col].tolist(),
        args.y_col: grid_work[args.y_col].tolist(),
        'community_top_group': top_group,
        'community_top_score': top_score_arr,
        'community_second_score': second_score_arr,
        'community_top_gap': top_gap_arr,
        'community_effective_richness': effective_richness_arr,
        'community_richness_above_threshold': richness_above_arr,
    })
    community_overall.to_csv(community_dir / 'community_overall.csv', index=False)

    grid_rows = []
    class_metric_lookup = class_metrics_df.set_index('group', drop=False) if not class_metrics_df.empty else pd.DataFrame()
    for class_idx, class_name in enumerate(class_names):
        row = {
            'group': class_name,
            'class_index': int(class_idx),
            'occurrence_rows_used': int(np.sum(y_multi == class_idx)),
            'mean_grid_share': float(sum_prob[class_idx] / max(1, n_rows)),
            'max_grid_share': float(max_prob[class_idx]),
            'grid_top_wins': int(top_wins[class_idx]),
            'grid_cells_above_threshold': int(above_threshold_cells[class_idx]),
            'community_threshold': float(thresholds_vec[class_idx]),
            'community_threshold_mode': threshold_mode,
        }
        if not class_metric_lookup.empty and class_name in class_metric_lookup.index:
            metric_row = class_metric_lookup.loc[class_name]
            if isinstance(metric_row, pd.DataFrame):
                metric_row = metric_row.iloc[0]
            for key in ['mean_true_class_probability', 'oof_threshold', 'oof_precision_thresholded', 'oof_recall_thresholded', 'oof_f1_thresholded', 'oof_precision_argmax', 'oof_recall_argmax', 'oof_f1_argmax']:
                val = pd.to_numeric(metric_row.get(key, np.nan), errors='coerce')
                row[key] = float(val) if pd.notna(val) else float('nan')
        grid_rows.append(row)
    community_species_summary = pd.DataFrame(grid_rows).sort_values(['mean_grid_share', 'max_grid_share', 'group'], ascending=[False, False, True]).reset_index(drop=True)
    community_species_summary.to_csv(community_dir / 'community_species_summary.csv', index=False)
    pd.DataFrame([cv_summary]).to_csv(community_dir / 'community_model_summary.csv', index=False)

    if int(args.preview_top_n) > 0:
        preview_dir = outdir / 'previews' / 'community'
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_point_map(community_overall, 'community_top_score', 'community top-species share', preview_dir / 'community_top_score.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
        preview_point_map(community_overall, 'community_top_gap', 'community top-vs-second gap', preview_dir / 'community_top_gap.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
        preview_point_map(community_overall, 'community_effective_richness', 'community effective richness', preview_dir / 'community_effective_richness.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)
        preview_point_map(community_overall, 'community_richness_above_threshold', 'community richness above class thresholds', preview_dir / 'community_richness_above_threshold.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)

    log(f'[community] done groups={len(class_names)} grid_rows={n_rows:,} outdir={community_dir}')
    return {
        'enabled': True,
        'status': 'ok',
        'model_name': estimator_name,
        'group_count': int(len(class_names)),
        'occurrence_rows_used': int(len(community_occ)),
        'feature_count': int(len(corr_keep)),
        'threshold_mode': threshold_mode,
        'cv_accuracy_mean': float(cv_summary.get('cv_accuracy_mean', float('nan'))),
        'cv_macro_f1_mean': float(cv_summary.get('cv_macro_f1_mean', float('nan'))),
        'oof_accuracy': float(cv_summary.get('oof_accuracy', float('nan'))),
        'oof_macro_f1': float(cv_summary.get('oof_macro_f1', float('nan'))),
        'oof_top3_accuracy': float(cv_summary.get('oof_top3_accuracy', float('nan'))),
        'oof_logloss': float(cv_summary.get('oof_logloss', float('nan'))),
    }

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Train classical ML suitability models after aggregate_occurrence_trends.py, compare ExtraTrees vs XGBoost, and keep the better model per species.')
    ap.add_argument('occurrences_csv', nargs='?', default=None, help='Enriched occurrence CSV, usually occurrences_enriched.cleaned.csv')
    ap.add_argument('grid_csv', nargs='?', default=None, help='Grid CSV with matching environmental predictors, usually grid_with_env.csv')
    ap.add_argument('--trend-summary', default=None, help='Directory with aggregate_occurrence_trends.py outputs. Defaults to <occurrences_csv_dir>/trend_summary')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--reuse-models-from', default=None, help='Existing xgboost_suitability.py output directory containing models/*.joblib to score on a new grid without retraining')
    ap.add_argument('--deploy-grid-csv', default=None, help='Grid CSV to score when --reuse-models-from is set. You can also pass the grid CSV as the first positional argument in reuse mode.')
    ap.add_argument('--make-previews', action='store_true', help='Generate preview PNGs from existing outputs under --outdir without retraining the ML models')
    ap.add_argument('--group-by', default='matched_species_name', help='Grouping column, usually matched_species_name')
    ap.add_argument('--include-taxa', nargs='*', default=None, help='Optional rank selectors such as family:Rosaceae genus:Ribes species:Daucus pusillus')
    ap.add_argument('--exclude-taxa', nargs='*', default=None, help='Optional rank selectors to exclude')
    ap.add_argument('--id-col', default='id', help='Grid id column')
    ap.add_argument('--x-col', default='lon', help='Grid x column')
    ap.add_argument('--y-col', default='lat', help='Grid y column')
    ap.add_argument('--occ-x-col', default='decimalLongitude', help='Occurrence x column')
    ap.add_argument('--occ-y-col', default='decimalLatitude', help='Occurrence y column')
    ap.add_argument('--min-occurrences-per-group', type=int, default=10)
    ap.add_argument('--max-groups', type=int, default=0)
    ap.add_argument('--include-coordinates', action='store_true')
    ap.add_argument('--top-numeric-features', type=int, default=96)
    ap.add_argument('--max-categorical-columns', type=int, default=4)
    ap.add_argument('--max-categories-per-column', type=int, default=12)
    ap.add_argument('--feature-select-max-rows', type=int, default=30000)
    ap.add_argument('--variance-threshold', type=float, default=1e-10)
    ap.add_argument('--corr-threshold', type=float, default=0.98)
    ap.add_argument('--max-features-after-corr', type=int, default=96)
    ap.add_argument('--background-multiplier', type=float, default=3.0)
    ap.add_argument('--background-source', choices=['target_group', 'grid', 'mixed'], default='mixed', help='Use other occurrence records as target-group background, grid background, or a mixture of both')
    ap.add_argument('--target-group-share', type=float, default=0.4, help='When background-source=mixed, share of background rows drawn from target-group occurrences')
    ap.add_argument('--background-sampling', choices=['mixed', 'regional', 'global'], default='regional')
    ap.add_argument('--background-local-share', type=float, default=0.7)
    ap.add_argument('--background-min-local-fraction', type=float, default=0.5)
    ap.add_argument('--background-bbox-pad-fraction', type=float, default=0.12)
    ap.add_argument('--background-bbox-pad-min', type=float, default=0.15)
    ap.add_argument('--background-block-size', type=float, default=0.25)
    ap.add_argument('--background-block-buffer', type=int, default=2)
    ap.add_argument('--min-background', type=int, default=2000)
    ap.add_argument('--max-background', type=int, default=30000)
    ap.add_argument('--dedupe-positive-by-coords', dest='dedupe_positive_by_coords', action='store_true')
    ap.add_argument('--no-dedupe-positive-by-coords', dest='dedupe_positive_by_coords', action='store_false')
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--cv-block-size', type=float, default=1.0)
    ap.add_argument('--cv-buffer-blocks', type=int, default=1, help='Exclude training samples in neighboring spatial CV blocks around each test block')
    ap.add_argument('--threshold-calibration-fraction', type=float, default=0.2)
    ap.add_argument('--predict-chunk-size', type=int, default=50000)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--probability-threshold', type=float, default=-1.0)
    ap.add_argument('--joint-min-share', type=float, default=0.7)
    ap.add_argument('--joint-tail-fraction', type=float, default=0.25)
    ap.add_argument('--joint-rank-power', type=float, default=1.5)
    ap.add_argument('--preview-top-n', type=int, default=500000)
    ap.add_argument('--preview-coarsen', type=int, default=2)
    ap.add_argument('--preview-point-alpha', type=float, default=0.35)
    ap.add_argument('--preview-vmax', type=float, default=DEFAULT_PREVIEW_VMAX)
    ap.add_argument('--preview-minimum-score-threshold', type=float, default=0.0, help='Set preview-only suitability values below this threshold to 0 before plotting; does not modify saved CSVs')
    ap.add_argument('--et-n-estimators', dest='et_n_estimators', type=int, default=500)
    ap.add_argument('--et-max-depth', dest='et_max_depth', type=int, default=0)
    ap.add_argument('--et-min-samples-leaf', dest='et_min_samples_leaf', type=int, default=1)
    ap.add_argument('--et-max-features', dest='et_max_features', default='sqrt')
    ap.add_argument('--xgb-n-estimators', dest='xgb_n_estimators', type=int, default=350)
    ap.add_argument('--xgb-max-depth', dest='xgb_max_depth', type=int, default=3)
    ap.add_argument('--xgb-learning-rate', dest='xgb_learning_rate', type=float, default=0.05)
    ap.add_argument('--xgb-subsample', dest='xgb_subsample', type=float, default=0.7)
    ap.add_argument('--xgb-colsample-bytree', dest='xgb_colsample_bytree', type=float, default=0.7)
    ap.add_argument('--xgb-min-child-weight', dest='xgb_min_child_weight', type=float, default=6.0)
    ap.add_argument('--xgb-reg-alpha', dest='xgb_reg_alpha', type=float, default=0.5)
    ap.add_argument('--xgb-reg-lambda', dest='xgb_reg_lambda', type=float, default=5.0)
    ap.add_argument('--xgb-gamma', dest='xgb_gamma', type=float, default=1.0)
    ap.add_argument('--xgb-max-bin', dest='xgb_max_bin', type=int, default=256)
    ap.add_argument('--deployment-score-mode', choices=['conservative', 'raw'], default='conservative', help='Use the conservative center-weighted score for ml_probability, or expose the raw tree score directly')
    ap.add_argument('--deployment-center-top-features', type=int, default=24, help='Maximum number of top numeric predictors used to build the conservative center-of-niche adjustment')
    ap.add_argument('--deployment-center-low-quantile', type=float, default=0.10, help='Lower positive-training quantile used as the broad central support envelope for conservative ML suitability')
    ap.add_argument('--deployment-center-high-quantile', type=float, default=0.90, help='Upper positive-training quantile used as the broad central support envelope for conservative ML suitability')
    ap.add_argument('--deployment-center-iqr-multiplier', type=float, default=1.5, help='Widen the smooth center-weight decay relative to the positive IQR')
    ap.add_argument('--deployment-support-pad-fraction', type=float, default=0.15, help='Expand the observed positive support range before novelty penalties begin')
    ap.add_argument('--deployment-raw-power', type=float, default=1.1, help='Shrink raw tree probabilities slightly before the conservative center and novelty adjustments')
    ap.add_argument('--deployment-center-floor', type=float, default=0.4, help='Minimum centrality factor retained at the margins before novelty penalties are applied')
    ap.add_argument('--deployment-center-power', type=float, default=1.25, help='Additional emphasis placed on the niche center after averaging centrality scores across features')
    ap.add_argument('--deployment-center-distance-power', type=float, default=1.6, help='How fast the centrality score falls as a grid cell moves away from the positive median')
    ap.add_argument('--deployment-novelty-strength', type=float, default=1.75, help='Strength of the extrapolation penalty once a cell moves beyond the broadened observed support range')
    ap.add_argument('--deployment-novelty-floor', type=float, default=0.08, help='Minimum retained extrapolation factor for the conservative ML suitability score')
    ap.add_argument('--train-community-model', action='store_true', help='Train an additional shared community-composition multiclass model alongside the per-species models')
    ap.add_argument('--community-only', action='store_true', help='Skip per-species model training and train only the shared community-composition model')
    ap.add_argument('--community-min-occurrences-per-group', type=int, default=20, help='Minimum cleaned occurrence rows required for a group to participate in the shared community model')
    ap.add_argument('--community-max-occurrences-per-group', type=int, default=50000, help='Cap occurrence rows per group for the shared community model to control memory and imbalance; <= 0 keeps all')
    ap.add_argument('--community-feature-select-max-rows', type=int, default=60000, help='Maximum stratified rows used while pruning features for the shared community model')
    ap.add_argument('--community-max-features-after-corr', type=int, default=96, help='Maximum retained encoded predictors after correlation pruning for the shared community model')
    ap.add_argument('--community-imputer-fit-max-rows', type=int, default=200000, help='Maximum rows used when fitting the fallback community-model imputer if XGBoost is unavailable')
    ap.add_argument('--community-share-threshold', type=float, default=-1.0, help='Override threshold used for the community richness preview. Use <= 0 to derive one threshold per species from out-of-fold community predictions')
    ap.set_defaults(dedupe_positive_by_coords=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if bool(getattr(args, 'community_only', False)):
        args.train_community_model = True
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if bool(args.make_previews):
        generate_previews_from_existing_outputs(outdir, args)
        log(f'[previews] regenerated previews under {outdir / "previews"}')
        return 0

    if args.reuse_models_from:
        if bool(getattr(args, 'train_community_model', False)):
            log('[reuse] train-community-model is ignored in reuse mode')
        return run_saved_model_deployment(args, outdir)

    if not args.occurrences_csv or not args.grid_csv:
        raise ValueError('occurrences_csv and grid_csv are required unless --make-previews or --reuse-models-from is set')

    occ_csv = Path(args.occurrences_csv).resolve()
    grid_csv = Path(args.grid_csv).resolve()
    trend_summary_dir = infer_trend_summary_dir(occ_csv, Path(args.trend_summary).resolve() if args.trend_summary else None)

    if not occ_csv.exists():
        raise FileNotFoundError(f'Missing occurrences_csv: {occ_csv}')
    if not grid_csv.exists():
        raise FileNotFoundError(f'Missing grid_csv: {grid_csv}')
    if not trend_summary_dir.exists():
        raise FileNotFoundError(f'Missing trend summary directory: {trend_summary_dir}')

    include_selectors = parse_selector_list(args.include_taxa)
    exclude_selectors = parse_selector_list(args.exclude_taxa)

    log(f'[load] occurrences={occ_csv}')
    log(f'[load] grid={grid_csv}')
    log(f'[load] trend_summary={trend_summary_dir}')

    feature_weights, family_weights, by_group_categorical = load_priority_tables(trend_summary_dir)
    feature_priority = build_feature_priority(feature_weights, family_weights)

    occ_head_raw = pd.read_csv(occ_csv, nrows=2000, low_memory=False)
    grid_head_raw = pd.read_csv(grid_csv, nrows=2000, low_memory=False)
    occ_head_work, occ_tc_meta = build_terraclimate_working_df(occ_head_raw)
    grid_head_work, grid_tc_meta = build_terraclimate_working_df(grid_head_raw)

    if args.group_by not in occ_head_raw.columns and args.group_by not in occ_head_work.columns:
        raise KeyError(f'group-by column not found in occurrences CSV: {args.group_by}')

    spec = choose_features(
        occ_head_work=occ_head_work,
        grid_head_work=grid_head_work,
        by_group_categorical=by_group_categorical,
        feature_priority=feature_priority,
        include_coordinates=bool(args.include_coordinates),
        top_numeric_features=int(args.top_numeric_features),
        max_categorical_columns=int(args.max_categorical_columns),
        max_categories_per_column=int(args.max_categories_per_column),
    )
    if not spec.encoded_columns:
        raise RuntimeError('No shared features remain after intersecting occurrences, grid, and trend summary priorities')

    occ_extra_cols = [args.group_by, args.occ_x_col, args.occ_y_col, 'matched_species_name', *TAXON_RANKS]
    grid_extra_cols = [args.id_col, args.x_col, args.y_col]
    occ_usecols = resolve_required_raw_columns(occ_head_raw, occ_tc_meta, spec.numeric_cols + spec.categorical_cols, occ_extra_cols)
    grid_usecols = resolve_required_raw_columns(grid_head_raw, grid_tc_meta, spec.numeric_cols + spec.categorical_cols, grid_extra_cols)

    occ_raw = read_minimal_csv(occ_csv, usecols=occ_usecols)
    grid_raw = read_minimal_csv(grid_csv, usecols=grid_usecols)
    occ_work, _ = build_terraclimate_working_df(occ_raw)
    grid_work, _ = build_terraclimate_working_df(grid_raw)

    group_meta = resolve_group_meta(occ_raw, args.group_by)
    group_meta = apply_taxon_filters(group_meta, include_selectors, exclude_selectors, args.group_by)
    if args.min_occurrences_per_group > 0:
        group_meta = group_meta[pd.to_numeric(group_meta['occurrence_count'], errors='coerce').fillna(0) >= int(args.min_occurrences_per_group)].copy()
    group_meta = group_meta.sort_values(['occurrence_count', 'group'], ascending=[False, True]).reset_index(drop=True)
    if args.max_groups > 0 and len(group_meta) > args.max_groups:
        group_meta = group_meta.head(args.max_groups).copy()
    if group_meta.empty:
        raise RuntimeError('No groups remain after taxon filtering and occurrence thresholds')

    occ_group_values = occ_raw[args.group_by].astype('string').fillna(pd.NA).map(lambda x: 'NA' if pd.isna(x) else str(x))
    occ_work['__group__'] = occ_group_values
    occ_work[args.occ_x_col] = pd.to_numeric(occ_raw[args.occ_x_col], errors='coerce')
    occ_work[args.occ_y_col] = pd.to_numeric(occ_raw[args.occ_y_col], errors='coerce')
    grid_work[args.id_col] = grid_raw[args.id_col]
    grid_work[args.x_col] = pd.to_numeric(grid_raw[args.x_col], errors='coerce')
    grid_work[args.y_col] = pd.to_numeric(grid_raw[args.y_col], errors='coerce')
    grid_work = grid_work.dropna(subset=[args.x_col, args.y_col]).reset_index(drop=True)
    grid_x_all = pd.to_numeric(grid_work[args.x_col], errors='coerce').to_numpy(dtype=float)
    grid_y_all = pd.to_numeric(grid_work[args.y_col], errors='coerce').to_numpy(dtype=float)
    _, _, bg_origin_x, bg_origin_y = compute_spatial_block_indices(grid_x_all, grid_y_all, float(args.background_block_size))
    grid_bg_groups = build_spatial_groups(grid_x_all, grid_y_all, float(args.background_block_size), origin_x=bg_origin_x, origin_y=bg_origin_y)

    occ_bg_work = occ_work.dropna(subset=[args.occ_x_col, args.occ_y_col]).copy().reset_index(drop=True)
    occ_bg_work['__group__'] = occ_bg_work['__group__'].astype(str)
    if bool(args.dedupe_positive_by_coords):
        occ_bg_work['__coord_key__'] = occ_bg_work[args.occ_x_col].round(6).astype(str) + '|' + occ_bg_work[args.occ_y_col].round(6).astype(str)
        occ_bg_work = occ_bg_work.drop_duplicates(subset=['__group__', '__coord_key__']).drop(columns=['__coord_key__'])
    occ_bg_groups_arr = occ_bg_work['__group__'].astype(str).to_numpy(dtype=object)
    occ_bg_x_all = pd.to_numeric(occ_bg_work[args.occ_x_col], errors='coerce').to_numpy(dtype=float)
    occ_bg_y_all = pd.to_numeric(occ_bg_work[args.occ_y_col], errors='coerce').to_numpy(dtype=float)
    occ_bg_block_keys = build_spatial_groups(occ_bg_x_all, occ_bg_y_all, float(args.background_block_size), origin_x=bg_origin_x, origin_y=bg_origin_y)

    group_meta.to_csv(outdir / 'selected_groups.csv', index=False)

    run_species_models = not bool(getattr(args, 'community_only', False))
    artifacts = []
    species_rows = []
    pred_stack = []
    pred_raw_stack = []
    thresholds = []
    selected_group_names = [str(g) for g in group_meta['group'].astype(str).tolist()]
    group_names = []

    model_comp_dir = outdir / 'model_comparison'
    if run_species_models:
        model_comp_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'feature': spec.encoded_columns,
        'kind': ['numeric' if f in spec.numeric_cols else 'categorical' for f in spec.encoded_columns],
        'priority': [float(spec.feature_priority.get(f.split('__', 1)[0], 0.0)) for f in spec.encoded_columns],
    }).to_csv(outdir / 'selected_features.csv', index=False)

    rng = np.random.default_rng(int(args.random_state))
    if run_species_models:
        for row in group_meta.itertuples(index=False):
            group = str(getattr(row, 'group'))
            pos_df = occ_work[occ_work['__group__'].astype(str) == group].copy()
            pos_df = pos_df.dropna(subset=[args.occ_x_col, args.occ_y_col]).reset_index(drop=True)
            if bool(args.dedupe_positive_by_coords):
                pos_df['__coord_key__'] = pos_df[args.occ_x_col].round(6).astype(str) + '|' + pos_df[args.occ_y_col].round(6).astype(str)
                pos_df = pos_df.drop_duplicates(subset=['__coord_key__']).drop(columns=['__coord_key__'])
            if len(pos_df) < max(2, int(args.min_occurrences_per_group)):
                log(f'[skip] group={group} positives_after_cleanup={len(pos_df)}')
                continue

            bg_target = int(round(max(float(args.min_background), float(args.background_multiplier) * float(len(pos_df)))))
            bg_target = max(int(args.min_background), min(int(args.max_background), bg_target))
            pos_x_vals = pd.to_numeric(pos_df[args.occ_x_col], errors='coerce').to_numpy(dtype=float)
            pos_y_vals = pd.to_numeric(pos_df[args.occ_y_col], errors='coerce').to_numpy(dtype=float)

            bg_source = str(args.background_source)
            bg_parts = []
            bg_stats_parts = []
            if bg_source in {'target_group', 'mixed'}:
                target_pool_mask = occ_bg_groups_arr != group
                target_idx_all = np.flatnonzero(target_pool_mask)
                if len(target_idx_all) > 0:
                    target_target = int(bg_target) if bg_source == 'target_group' else int(round(bg_target * float(args.target_group_share)))
                    target_target = min(target_target, len(target_idx_all))
                    target_rel_idx, target_stats = sample_background_from_source(
                        occ_bg_x_all[target_idx_all],
                        occ_bg_y_all[target_idx_all],
                        occ_bg_block_keys[target_idx_all],
                        bg_origin_x,
                        bg_origin_y,
                        pos_x_vals,
                        pos_y_vals,
                        target_target,
                        args,
                        rng,
                    )
                    if len(target_rel_idx) > 0:
                        target_bg = occ_bg_work.iloc[target_idx_all[target_rel_idx]].copy().reset_index(drop=True)
                        target_bg[args.x_col] = pd.to_numeric(target_bg[args.occ_x_col], errors='coerce')
                        target_bg[args.y_col] = pd.to_numeric(target_bg[args.occ_y_col], errors='coerce')
                        bg_parts.append(target_bg)
                    target_stats['source'] = 'target_group'
                    target_stats['rows_selected'] = int(len(target_rel_idx))
                    bg_stats_parts.append(target_stats)
            if bg_source in {'grid', 'mixed'}:
                grid_target = int(bg_target) if bg_source == 'grid' else max(0, int(bg_target) - sum(len(df) for df in bg_parts))
                grid_target = min(grid_target, len(grid_work))
                grid_idx, grid_stats = sample_background_indices(
                    grid_x_all,
                    grid_y_all,
                    grid_bg_groups,
                    bg_origin_x,
                    bg_origin_y,
                    pos_x_vals,
                    pos_y_vals,
                    grid_target,
                    args,
                    rng,
                )
                if len(grid_idx) > 0:
                    bg_parts.append(grid_work.iloc[grid_idx].copy().reset_index(drop=True))
                grid_stats['source'] = 'grid'
                grid_stats['rows_selected'] = int(len(grid_idx))
                bg_stats_parts.append(grid_stats)
            if not bg_parts:
                log(f'[skip] group={group} insufficient background candidates after spatial filtering background_rows=0')
                continue
            bg_df = pd.concat(bg_parts, ignore_index=True)
            if len(bg_df) > int(bg_target):
                choose_idx = np.sort(rng.choice(np.arange(len(bg_df), dtype=np.int64), size=int(bg_target), replace=False))
                bg_df = bg_df.iloc[choose_idx].reset_index(drop=True)
            if len(bg_df) < max(2, int(args.min_background * 0.5)):
                log(f'[skip] group={group} insufficient background candidates after spatial filtering background_rows={len(bg_df)}')
                continue

            pos_x = transform_features(pos_df, spec)
            bg_x = transform_features(bg_df, spec)
            y = np.concatenate([np.ones(len(pos_x), dtype=np.int8), np.zeros(len(bg_x), dtype=np.int8)])
            X = pd.concat([pos_x, bg_x], axis=0, ignore_index=True)
            coords_x = np.concatenate([pd.to_numeric(pos_df[args.occ_x_col], errors='coerce').to_numpy(dtype=float), pd.to_numeric(bg_df[args.x_col], errors='coerce').to_numpy(dtype=float)])
            coords_y = np.concatenate([pd.to_numeric(pos_df[args.occ_y_col], errors='coerce').to_numpy(dtype=float), pd.to_numeric(bg_df[args.y_col], errors='coerce').to_numpy(dtype=float)])

            subset_idx = stratified_cap_rows(y, int(args.feature_select_max_rows), int(args.random_state))
            X_select = X.iloc[subset_idx].copy()
            keep = variance_filter(X_select, threshold=float(args.variance_threshold))
            if not keep:
                log(f'[skip] group={group} no features survived variance filter')
                continue
            X = X[keep].copy()
            X_select = X_select[keep].copy()

            imputer = SimpleImputer(strategy='median')
            X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            X_select_imp = X_imp.iloc[subset_idx].reset_index(drop=True)
            corr_keep = correlation_prune(X_select_imp, X_select_imp.columns, threshold=float(args.corr_threshold))
            corr_keep = sorted(corr_keep, key=lambda c: (-float(spec.feature_priority.get(c.split('__', 1)[0], 0.0)), c))
            if int(args.max_features_after_corr) > 0:
                corr_keep = corr_keep[:int(args.max_features_after_corr)]
            if not corr_keep:
                log(f'[skip] group={group} no features survived correlation prune')
                continue
            X_final = X_imp[corr_keep].copy()

            scale_pos_weight = float(max(1.0, len(bg_x) / max(1, len(pos_x))))
            estimator_map = build_estimators(args, scale_pos_weight)
            if not estimator_map:
                raise RuntimeError('No estimators available')
            best_name, cv_summary, cv_folds_df = evaluate_candidate_models(X_final, y, coords_x, coords_y, args, estimator_map)
            final_model = fit_final_model(X_final, y, estimator_map, best_name)
            threshold = float(args.probability_threshold) if float(args.probability_threshold) >= 0 else float(cv_summary.loc[cv_summary['model'] == best_name, 'cv_threshold_median'].iloc[0])

            perm_idx = stratified_cap_rows(y, min(len(y), 12000), int(args.random_state))
            perm_X = X_final.iloc[perm_idx].reset_index(drop=True)
            perm_y = y[perm_idx]
            perm = permutation_importance(final_model, perm_X, perm_y, n_repeats=5, random_state=int(args.random_state), scoring='average_precision', n_jobs=-1)
            perm_df = pd.DataFrame({'feature': corr_keep, 'perm_importance_mean': perm.importances_mean, 'perm_importance_std': perm.importances_std})
            perm_df = perm_df.sort_values(['perm_importance_mean', 'feature'], ascending=[False, True]).reset_index(drop=True)

            model_dir = outdir / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            imp_dir = outdir / 'feature_importance'
            imp_dir.mkdir(parents=True, exist_ok=True)
            slug = safe_slug(group)
            model_path = model_dir / f'{slug}.joblib'
            feature_path = imp_dir / f'{slug}.csv'
            perm_df.to_csv(feature_path, index=False)

            grid_X_all = transform_features(grid_work, spec)
            grid_X_imp = pd.DataFrame(imputer.transform(grid_X_all[X.columns]), columns=X.columns)
            grid_X_final = grid_X_imp[corr_keep]
            raw_proba = predict_in_chunks(final_model, grid_X_final, int(args.predict_chunk_size))

            importance_lookup = build_feature_importance_lookup(final_model, perm_df, spec.feature_priority)
            pos_X_final = X_final.iloc[:len(pos_x)].reset_index(drop=True)
            conservative_spec = build_conservative_suitability_spec(pos_X_final, corr_keep, spec.numeric_cols, importance_lookup, args)
            conservative_suitability, center_weight, novelty_penalty, adjustment_factor = apply_conservative_suitability_spec(grid_X_final, raw_proba, conservative_spec)
            train_raw_proba = predict_in_chunks(final_model, X_final, int(args.predict_chunk_size))
            train_conservative_suitability, _, _, _ = apply_conservative_suitability_spec(X_final, train_raw_proba, conservative_spec)
            if str(args.deployment_score_mode) == 'conservative':
                deployment_threshold = float(args.probability_threshold) if float(args.probability_threshold) >= 0 else float(pick_threshold(y, train_conservative_suitability))
                deployed_score = conservative_suitability
            else:
                deployment_threshold = float(args.probability_threshold) if float(args.probability_threshold) >= 0 else float(threshold)
                deployed_score = raw_proba

            joblib.dump({
                'model_name': best_name,
                'model': final_model,
                'imputer': imputer,
                'raw_feature_columns': list(X.columns),
                'final_feature_columns': corr_keep,
                'categorical_vocab': spec.categorical_vocab,
                'numeric_cols': spec.numeric_cols,
                'categorical_cols': spec.categorical_cols,
                'threshold_raw_cv': threshold,
                'deployment_threshold': deployment_threshold,
                'deployment_score_mode': str(args.deployment_score_mode),
                'conservative_suitability_spec': conservative_spec,
            }, model_path)

            pred_stack.append(deployed_score)
            pred_raw_stack.append(raw_proba)
            thresholds.append(deployment_threshold)
            artifacts.append({'group': group, 'slug': slug, 'model_name': best_name, 'threshold': deployment_threshold})

            species_out = pd.DataFrame({
                args.id_col: grid_work[args.id_col].tolist(),
                args.x_col: grid_work[args.x_col].tolist(),
                args.y_col: grid_work[args.y_col].tolist(),
                'ml_probability_raw': raw_proba,
                'ml_center_weight': center_weight,
                'ml_novelty_penalty': novelty_penalty,
                'ml_adjustment_factor': adjustment_factor,
                'ml_suitability': conservative_suitability,
                'ml_probability': deployed_score,
                'ml_likely': (deployed_score >= deployment_threshold).astype(np.int8),
                'model_name': best_name,
            })
            species_dir = outdir / 'by_species'
            species_dir.mkdir(parents=True, exist_ok=True)
            species_out.to_csv(species_dir / f'{slug}.csv', index=False)

            best_row = cv_summary[cv_summary['model'] == best_name].iloc[0]
            species_rows.append({
                'group': group,
                'occurrence_count': int(getattr(row, 'occurrence_count', len(pos_df))),
                'positive_rows_used': int(len(pos_df)),
                'background_rows_used': int(len(bg_df)),
                'background_source': bg_source,
                'background_sampling_mode': ';'.join(str(part.get('mode', 'mixed')) for part in bg_stats_parts),
                'background_source_rows': ';'.join(f"{part.get('source', 'unknown')}={int(part.get('rows_selected', 0))}" for part in bg_stats_parts),
                'background_candidate_global': int(sum(int(part.get('candidate_global', 0)) for part in bg_stats_parts)),
                'background_candidate_local': int(sum(int(part.get('candidate_local', 0)) for part in bg_stats_parts)),
                'background_blocked_cells': int(sum(int(part.get('blocked_cells', 0)) for part in bg_stats_parts)),
                'selected_model': best_name,
                'threshold_raw_cv': threshold,
                'deployment_threshold': deployment_threshold,
                'cv_scheme': str(best_row['cv_scheme']),
                'cv_roc_auc_mean': float(best_row['cv_roc_auc_mean']),
                'cv_pr_auc_mean': float(best_row['cv_pr_auc_mean']),
                'cv_logloss_mean': float(best_row['cv_logloss_mean']),
                'cv_brier_mean': float(best_row['cv_brier_mean']),
                'cv_accuracy_mean': float(best_row['cv_accuracy_mean']),
                'cv_balanced_accuracy_mean': float(best_row['cv_balanced_accuracy_mean']),
                'cv_f1_mean': float(best_row['cv_f1_mean']),
                'cv_precision_mean': float(best_row['cv_precision_mean']),
                'cv_recall_mean': float(best_row['cv_recall_mean']),
                'oof_roc_auc': float(best_row['oof_roc_auc']),
                'oof_pr_auc': float(best_row['oof_pr_auc']),
                'oof_logloss': float(best_row['oof_logloss']),
                'oof_brier': float(best_row['oof_brier']),
                'oof_accuracy': float(best_row['oof_accuracy']),
                'oof_balanced_accuracy': float(best_row['oof_balanced_accuracy']),
                'oof_f1': float(best_row['oof_f1']),
                'oof_precision': float(best_row['oof_precision']),
                'oof_recall': float(best_row['oof_recall']),
                'cv_fit_seconds_mean': float(best_row['cv_fit_seconds_mean']),
                'cv_predict_seconds_mean': float(best_row['cv_predict_seconds_mean']),
                'cv_fit_seconds_total': float(best_row['cv_fit_seconds_total']),
                'cv_predict_seconds_total': float(best_row['cv_predict_seconds_total']),
                'feature_count': int(len(corr_keep)),
                'deployment_feature_count': int(len(conservative_spec.get('features', []))),
                'deployment_score_mode': str(args.deployment_score_mode),
                'mean_grid_probability_raw': float(np.nanmean(raw_proba)),
                'max_grid_probability_raw': float(np.nanmax(raw_proba)),
                'mean_grid_suitability': float(np.nanmean(conservative_suitability)),
                'max_grid_suitability': float(np.nanmax(conservative_suitability)),
                'mean_grid_probability': float(np.nanmean(deployed_score)),
                'max_grid_probability': float(np.nanmax(deployed_score)),
                'model_path': str(model_path),
                'feature_importance_csv': str(feature_path),
                **{rank: getattr(row, rank, pd.NA) for rank in ['matched_species_name', *TAXON_RANKS] if hasattr(row, rank)},
            })
            cv_summary.to_csv(model_comp_dir / f'{slug}.csv', index=False)
            if not cv_folds_df.empty:
                cv_folds_df.to_csv(model_comp_dir / f'{slug}_folds.csv', index=False)
            cv_pr_auc_mean = float(best_row["cv_pr_auc_mean"])
            cv_f1_mean = float(best_row["cv_f1_mean"])
            oof_pr_auc = float(best_row["oof_pr_auc"])
            oof_f1 = float(best_row["oof_f1"])
            threshold_std = float(best_row["cv_threshold_std"])
            cv_fit_seconds_total = float(best_row["cv_fit_seconds_total"])
            cv_predict_seconds_total = float(best_row["cv_predict_seconds_total"])
            log(
                f"[fit] group={group} model={best_name} positives={len(pos_df):,} background={len(bg_df):,} features={len(corr_keep):,} "
                f"cv_pr_auc={cv_pr_auc_mean:.3f} cv_f1={cv_f1_mean:.3f} oof_pr_auc={oof_pr_auc:.3f} oof_f1={oof_f1:.3f} "
                f"raw_thr={threshold:.3f} deploy_thr={deployment_threshold:.3f} thr_sd={threshold_std:.3f} fit_s={cv_fit_seconds_total:.2f} pred_s={cv_predict_seconds_total:.2f}"
            )

        if not pred_stack:
            raise RuntimeError('No models were fitted. Check taxon filters and occurrence counts.')

        pred_arr = np.vstack(pred_stack).astype(np.float32)
        pred_raw_arr = np.vstack(pred_raw_stack).astype(np.float32)
        pred_valid = np.isfinite(pred_arr)
        pred_raw_valid = np.isfinite(pred_raw_arr)
        selected_group_names = [a['group'] for a in artifacts]
        n_rows = pred_arr.shape[1]
        safe_pred = np.where(pred_valid, pred_arr, -np.inf)
        safe_pred_raw = np.where(pred_raw_valid, pred_raw_arr, -np.inf)
        top_idx = np.argmax(safe_pred, axis=0)
        top_score = safe_pred[top_idx, np.arange(n_rows)]
        top_group = np.array([selected_group_names[i] for i in top_idx], dtype=object)
        no_score = ~np.isfinite(top_score) | (top_score == -np.inf)
        top_score = np.where(no_score, np.nan, top_score)
        top_group = np.where(no_score, '', top_group)

        overall_ml = np.max(safe_pred, axis=0)
        overall_ml = np.where(np.isfinite(overall_ml) & (overall_ml > -np.inf), overall_ml, np.nan).astype(np.float32)
        overall_ml_raw = np.max(safe_pred_raw, axis=0)
        overall_ml_raw = np.where(np.isfinite(overall_ml_raw) & (overall_ml_raw > -np.inf), overall_ml_raw, np.nan).astype(np.float32)
        overall_ml_min = finite_min_score(pred_arr, pred_valid)
        overall_ml_joint = joint_support_score(pred_arr, pred_valid, min_share=args.joint_min_share, tail_fraction=args.joint_tail_fraction, rank_power=args.joint_rank_power)
        threshold_arr = np.asarray(thresholds, dtype=np.float32)[:, None]
        richness_ml = np.sum(np.where(pred_valid, pred_arr >= threshold_arr, False), axis=0).astype(np.int32)

        overall_df = pd.DataFrame({
            args.id_col: grid_work[args.id_col].tolist(),
            args.x_col: grid_work[args.x_col].tolist(),
            args.y_col: grid_work[args.y_col].tolist(),
            'overall_ml_raw': overall_ml_raw,
            'overall_ml_suitability': overall_ml,
            'overall_ml': overall_ml,
            'overall_ml_min': overall_ml_min,
            'overall_ml_joint': overall_ml_joint,
            'richness_ml': richness_ml,
            'top_group_ml': top_group,
            'top_ml_score': top_score,
        })
        overall_df.to_csv(outdir / 'overall_suitability.csv', index=False)

        species_summary_df = pd.DataFrame(species_rows).sort_values(['mean_grid_probability', 'max_grid_probability', 'group'], ascending=[False, False, True]).reset_index(drop=True)
        species_summary_df.to_csv(outdir / 'species_score_summary.csv', index=False)

        if int(args.preview_top_n) > 0:
            preview_dir = outdir / 'previews'
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_point_map(overall_df, 'overall_ml', 'overall ML suitability', preview_dir / 'overall_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(overall_df, 'overall_ml_min', 'overall ML minimum overlap', preview_dir / 'overall_ml_min.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(overall_df, 'overall_ml_joint', 'overall ML joint suitability', preview_dir / 'overall_ml_joint.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)
            preview_point_map(overall_df, 'richness_ml', 'richness above species thresholds', preview_dir / 'richness_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)
            for row in species_summary_df.head(int(args.preview_top_n)).itertuples(index=False):
                dfp = pd.read_csv(outdir / 'by_species' / f"{safe_slug(row.group)}.csv", usecols=[args.x_col, args.y_col, 'ml_probability'], low_memory=False)
                preview_point_map(dfp, 'ml_probability', f'{row.group} ML suitability', preview_dir / 'by_species' / f"{safe_slug(row.group)}_ml.png", args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax, args.preview_minimum_score_threshold)

    community_manifest = {'enabled': False, 'status': 'not_requested'}
    if bool(getattr(args, 'train_community_model', False)):
        community_manifest = train_and_write_community_model(outdir, args, spec, group_meta, occ_work, grid_work)

    manifest = {
        'occurrences_csv': str(occ_csv),
        'grid_csv': str(grid_csv),
        'trend_summary': str(trend_summary_dir),
        'outdir': str(outdir),
        'group_by': args.group_by,
        'include_taxa': [s.raw for s in include_selectors],
        'exclude_taxa': [s.raw for s in exclude_selectors],
        'selected_group_count': int(len(selected_group_names)),
        'selected_groups': selected_group_names,
        'community_only': bool(getattr(args, 'community_only', False)),
        'numeric_feature_count': int(len(spec.numeric_cols)),
        'categorical_feature_count': int(len(spec.categorical_cols)),
        'encoded_feature_count': int(len(spec.encoded_columns)),
        'feature_select_max_rows': int(args.feature_select_max_rows),
        'variance_threshold': float(args.variance_threshold),
        'corr_threshold': float(args.corr_threshold),
        'max_features_after_corr': int(args.max_features_after_corr),
        'background_multiplier': float(args.background_multiplier),
        'background_source': str(args.background_source),
        'target_group_share': float(args.target_group_share),
        'background_sampling': str(args.background_sampling),
        'background_local_share': float(args.background_local_share),
        'background_min_local_fraction': float(args.background_min_local_fraction),
        'background_bbox_pad_fraction': float(args.background_bbox_pad_fraction),
        'background_bbox_pad_min': float(args.background_bbox_pad_min),
        'background_block_size': float(args.background_block_size),
        'background_block_buffer': int(args.background_block_buffer),
        'min_background': int(args.min_background),
        'max_background': int(args.max_background),
        'dedupe_positive_by_coords': bool(args.dedupe_positive_by_coords),
        'cv_folds': int(args.cv_folds),
        'cv_block_size': float(args.cv_block_size),
        'cv_buffer_blocks': int(args.cv_buffer_blocks),
        'threshold_calibration_fraction': float(args.threshold_calibration_fraction),
        'predict_chunk_size': int(args.predict_chunk_size),
        'joint_min_share': float(args.joint_min_share),
        'joint_tail_fraction': float(args.joint_tail_fraction),
        'joint_rank_power': float(args.joint_rank_power),
        'available_models': sorted(list(build_estimators(args, 1.0).keys())),
        'community_model': community_manifest,
        'score_interpretation': 'Presence-background relative suitability score. ml_probability_raw is the raw tree score. ml_suitability is the conservative center-weighted score intended for deployment and blending. This is not a true occurrence probability unless true absences and prevalence assumptions are provided.',
    }
    write_manifest(outdir / 'manifest.json', manifest)
    log(f'[done] groups={len(artifacts)} grid_rows={len(grid_work):,} outdir={outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
