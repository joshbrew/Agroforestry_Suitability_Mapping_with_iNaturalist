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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception:
    HAVE_XGBOOST = False


# py xgboost_suitability.py D:/envpull_association_test/occurrences_enriched.csv D:/Oregon_Suitability/oregon_grid_1000m_env/grid_with_env.csv --trend-summary D:/envpull_association_test/trend_summary --outdir D:/Oregon_Suitability/oregon_grid_1000m_env/ml_association_test --group-by matched_species_name

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


def build_spatial_groups(x: np.ndarray, y: np.ndarray, block_size: float) -> np.ndarray:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    bx = np.floor((xv - np.nanmin(xv)) / max(block_size, 1e-9)).astype(np.int64)
    by = np.floor((yv - np.nanmin(yv)) / max(block_size, 1e-9)).astype(np.int64)
    return (bx.astype(np.int64) << 32) ^ (by.astype(np.int64) & 0xFFFFFFFF)


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


def metric_or_nan(fn, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    try:
        return float(fn(y_true, y_score, **kwargs))
    except Exception:
        return float('nan')


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
    groups = None
    cv_scheme = 'stratified'
    if np.isfinite(coords_x).all() and np.isfinite(coords_y).all():
        spatial_groups = build_spatial_groups(coords_x, coords_y, float(args.cv_block_size))
        if np.unique(spatial_groups).size >= max(3, min(int(args.cv_folds), 5)):
            groups = spatial_groups
            cv_scheme = 'groupkfold'
    metrics_by_model: Dict[str, List[Dict[str, float]]] = {name: [] for name in estimator_map}
    thresholds_by_model: Dict[str, List[float]] = {name: [] for name in estimator_map}
    fold_rows: List[Dict[str, float]] = []

    if groups is not None:
        splitter = GroupKFold(n_splits=min(int(args.cv_folds), np.unique(groups).size))
        split_iter = splitter.split(X, y, groups=groups)
    else:
        n_splits = min(int(args.cv_folds), max(2, int(np.bincount(y).min())))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(args.random_state))
        split_iter = splitter.split(X, y)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        x_train = X.iloc[train_idx]
        y_train = y[train_idx]
        x_test = X.iloc[test_idx]
        y_test = y[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        for model_name, estimator in estimator_map.items():
            est = estimator.__class__(**estimator.get_params())
            fit_t0 = time.perf_counter()
            est.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_t0
            pred_t0 = time.perf_counter()
            proba = est.predict_proba(x_test)[:, 1]
            predict_seconds = time.perf_counter() - pred_t0
            threshold = pick_threshold(y_test, proba)
            pred = (proba >= threshold).astype(np.int8)
            row = {
                'fold': int(fold_idx),
                'model': model_name,
                'cv_scheme': cv_scheme,
                'train_rows': int(len(train_idx)),
                'test_rows': int(len(test_idx)),
                'train_positive_rows': int(np.sum(y_train == 1)),
                'train_negative_rows': int(np.sum(y_train == 0)),
                'test_positive_rows': int(np.sum(y_test == 1)),
                'test_negative_rows': int(np.sum(y_test == 0)),
                'threshold': float(threshold),
                'roc_auc': metric_or_nan(roc_auc_score, y_test, proba),
                'pr_auc': metric_or_nan(average_precision_score, y_test, proba),
                'logloss': metric_or_nan(log_loss, y_test, proba, labels=[0, 1]),
                'brier': metric_or_nan(brier_score_loss, y_test, proba),
                'accuracy': metric_or_nan(accuracy_score, y_test, pred),
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
        summary_rows.append({
            'model': model_name,
            'cv_scheme': cv_scheme,
            'cv_roc_auc_mean': float(np.nanmean([r['roc_auc'] for r in rows])) if rows else float('nan'),
            'cv_pr_auc_mean': float(np.nanmean([r['pr_auc'] for r in rows])) if rows else float('nan'),
            'cv_logloss_mean': float(np.nanmean([r['logloss'] for r in rows])) if rows else float('nan'),
            'cv_brier_mean': float(np.nanmean([r['brier'] for r in rows])) if rows else float('nan'),
            'cv_accuracy_mean': float(np.nanmean([r['accuracy'] for r in rows])) if rows else float('nan'),
            'cv_f1_mean': float(np.nanmean([r['f1'] for r in rows])) if rows else float('nan'),
            'cv_precision_mean': float(np.nanmean([r['precision'] for r in rows])) if rows else float('nan'),
            'cv_recall_mean': float(np.nanmean([r['recall'] for r in rows])) if rows else float('nan'),
            'cv_fit_seconds_mean': float(np.nanmean([r['fit_seconds'] for r in rows])) if rows else float('nan'),
            'cv_predict_seconds_mean': float(np.nanmean([r['predict_seconds'] for r in rows])) if rows else float('nan'),
            'cv_fit_seconds_total': float(np.nansum([r['fit_seconds'] for r in rows])) if rows else float('nan'),
            'cv_predict_seconds_total': float(np.nansum([r['predict_seconds'] for r in rows])) if rows else float('nan'),
            'threshold': float(np.nanmedian(thresholds_by_model[model_name])) if thresholds_by_model[model_name] else 0.5,
            'folds_used': int(len(rows)),
        })
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError('No valid CV folds were produced')
    summary_df = summary_df.sort_values(['cv_pr_auc_mean', 'cv_f1_mean', 'cv_roc_auc_mean', 'cv_logloss_mean', 'model'], ascending=[False, False, False, True, True]).reset_index(drop=True)
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


def preview_point_map(df: pd.DataFrame, value_col: str, title: str, out_path: Path, x_col: str, y_col: str, point_alpha: float, preview_coarsen: int, vmax: Optional[float]) -> None:
    if plt is None or value_col not in df.columns:
        return
    sub = df[[x_col, y_col, value_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors='coerce')
    sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
    sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
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


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def predict_in_chunks(model, X: pd.DataFrame, chunk_size: int) -> np.ndarray:
    chunk_size = max(1, int(chunk_size))
    out = np.full(len(X), np.nan, dtype=np.float32)
    for start in range(0, len(X), chunk_size):
        end = min(len(X), start + chunk_size)
        out[start:end] = model.predict_proba(X.iloc[start:end])[:, 1].astype(np.float32)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Train classical ML suitability models after aggregate_occurrence_trends.py, compare ExtraTrees vs XGBoost, and keep the better model per species.')
    ap.add_argument('occurrences_csv', help='Enriched occurrence CSV, usually occurrences_enriched.cleaned.csv')
    ap.add_argument('grid_csv', help='Grid CSV with matching environmental predictors, usually grid_with_env.csv')
    ap.add_argument('--trend-summary', default=None, help='Directory with aggregate_occurrence_trends.py outputs. Defaults to <occurrences_csv_dir>/trend_summary')
    ap.add_argument('--outdir', required=True, help='Output directory')
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
    ap.add_argument('--min-background', type=int, default=2000)
    ap.add_argument('--max-background', type=int, default=30000)
    ap.add_argument('--dedupe-positive-by-coords', action='store_true')
    ap.add_argument('--cv-folds', type=int, default=4)
    ap.add_argument('--cv-block-size', type=float, default=0.35)
    ap.add_argument('--predict-chunk-size', type=int, default=50000)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--probability-threshold', type=float, default=-1.0)
    ap.add_argument('--joint-min-share', type=float, default=0.7)
    ap.add_argument('--joint-tail-fraction', type=float, default=0.25)
    ap.add_argument('--joint-rank-power', type=float, default=1.5)
    ap.add_argument('--preview-top-n', type=int, default=8)
    ap.add_argument('--preview-coarsen', type=int, default=2)
    ap.add_argument('--preview-point-alpha', type=float, default=0.35)
    ap.add_argument('--preview-vmax', type=float, default=DEFAULT_PREVIEW_VMAX)
    ap.add_argument('--et-n-estimators', dest='et_n_estimators', type=int, default=500)
    ap.add_argument('--et-max-depth', dest='et_max_depth', type=int, default=0)
    ap.add_argument('--et-min-samples-leaf', dest='et_min_samples_leaf', type=int, default=1)
    ap.add_argument('--et-max-features', dest='et_max_features', default='sqrt')
    ap.add_argument('--xgb-n-estimators', dest='xgb_n_estimators', type=int, default=350)
    ap.add_argument('--xgb-max-depth', dest='xgb_max_depth', type=int, default=5)
    ap.add_argument('--xgb-learning-rate', dest='xgb_learning_rate', type=float, default=0.05)
    ap.add_argument('--xgb-subsample', dest='xgb_subsample', type=float, default=0.8)
    ap.add_argument('--xgb-colsample-bytree', dest='xgb_colsample_bytree', type=float, default=0.8)
    ap.add_argument('--xgb-min-child-weight', dest='xgb_min_child_weight', type=float, default=1.0)
    ap.add_argument('--xgb-reg-alpha', dest='xgb_reg_alpha', type=float, default=0.0)
    ap.add_argument('--xgb-reg-lambda', dest='xgb_reg_lambda', type=float, default=1.0)
    ap.add_argument('--xgb-gamma', dest='xgb_gamma', type=float, default=0.0)
    ap.add_argument('--xgb-max-bin', dest='xgb_max_bin', type=int, default=256)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    occ_csv = Path(args.occurrences_csv).resolve()
    grid_csv = Path(args.grid_csv).resolve()
    trend_summary_dir = infer_trend_summary_dir(occ_csv, Path(args.trend_summary).resolve() if args.trend_summary else None)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

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

    group_meta.to_csv(outdir / 'selected_groups.csv', index=False)

    artifacts = []
    species_rows = []
    pred_stack = []
    thresholds = []

    model_comp_dir = outdir / 'model_comparison'
    model_comp_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'feature': spec.encoded_columns,
        'kind': ['numeric' if f in spec.numeric_cols else 'categorical' for f in spec.encoded_columns],
        'priority': [float(spec.feature_priority.get(f.split('__', 1)[0], 0.0)) for f in spec.encoded_columns],
    }).to_csv(outdir / 'selected_features.csv', index=False)

    rng = np.random.default_rng(int(args.random_state))
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
        bg_target = min(bg_target, len(grid_work))
        bg_indices = rng.choice(grid_work.index.to_numpy(), size=bg_target, replace=False)
        bg_df = grid_work.loc[bg_indices].copy().reset_index(drop=True)

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
        threshold = float(args.probability_threshold) if float(args.probability_threshold) >= 0 else float(cv_summary.loc[cv_summary['model'] == best_name, 'threshold'].iloc[0])

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
        joblib.dump({
            'model_name': best_name,
            'model': final_model,
            'imputer': imputer,
            'raw_feature_columns': list(X.columns),
            'final_feature_columns': corr_keep,
            'categorical_vocab': spec.categorical_vocab,
            'numeric_cols': spec.numeric_cols,
            'categorical_cols': spec.categorical_cols,
            'threshold': threshold,
        }, model_path)
        perm_df.to_csv(feature_path, index=False)

        grid_X_all = transform_features(grid_work, spec)
        grid_X_imp = pd.DataFrame(imputer.transform(grid_X_all[X.columns]), columns=X.columns)
        grid_X_final = grid_X_imp[corr_keep]
        proba = predict_in_chunks(final_model, grid_X_final, int(args.predict_chunk_size))

        pred_stack.append(proba)
        thresholds.append(threshold)
        artifacts.append({'group': group, 'slug': slug, 'model_name': best_name, 'threshold': threshold})

        species_out = pd.DataFrame({
            args.id_col: grid_work[args.id_col].tolist(),
            args.x_col: grid_work[args.x_col].tolist(),
            args.y_col: grid_work[args.y_col].tolist(),
            'ml_probability': proba,
            'ml_likely': (proba >= threshold).astype(np.int8),
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
            'selected_model': best_name,
            'threshold': threshold,
            'cv_scheme': str(best_row['cv_scheme']),
            'cv_roc_auc_mean': float(best_row['cv_roc_auc_mean']),
            'cv_pr_auc_mean': float(best_row['cv_pr_auc_mean']),
            'cv_logloss_mean': float(best_row['cv_logloss_mean']),
            'cv_brier_mean': float(best_row['cv_brier_mean']),
            'cv_accuracy_mean': float(best_row['cv_accuracy_mean']),
            'cv_f1_mean': float(best_row['cv_f1_mean']),
            'cv_precision_mean': float(best_row['cv_precision_mean']),
            'cv_recall_mean': float(best_row['cv_recall_mean']),
            'cv_fit_seconds_mean': float(best_row['cv_fit_seconds_mean']),
            'cv_predict_seconds_mean': float(best_row['cv_predict_seconds_mean']),
            'cv_fit_seconds_total': float(best_row['cv_fit_seconds_total']),
            'cv_predict_seconds_total': float(best_row['cv_predict_seconds_total']),
            'feature_count': int(len(corr_keep)),
            'mean_grid_probability': float(np.nanmean(proba)),
            'max_grid_probability': float(np.nanmax(proba)),
            'model_path': str(model_path),
            'feature_importance_csv': str(feature_path),
            **{rank: getattr(row, rank, pd.NA) for rank in ['matched_species_name', *TAXON_RANKS] if hasattr(row, rank)},
        })
        cv_summary.to_csv(model_comp_dir / f'{slug}.csv', index=False)
        if not cv_folds_df.empty:
            cv_folds_df.to_csv(model_comp_dir / f'{slug}_folds.csv', index=False)
        cv_f1_mean = float(best_row["cv_f1_mean"])
        cv_accuracy_mean = float(best_row["cv_accuracy_mean"])
        cv_fit_seconds_total = float(best_row["cv_fit_seconds_total"])
        cv_predict_seconds_total = float(best_row["cv_predict_seconds_total"])
        log(
            f"[fit] group={group} model={best_name} positives={len(pos_df):,} background={len(bg_df):,} features={len(corr_keep):,} "
            f"cv_f1={cv_f1_mean:.3f} cv_acc={cv_accuracy_mean:.3f} fit_s={cv_fit_seconds_total:.2f} pred_s={cv_predict_seconds_total:.2f}"
        )

    if not pred_stack:
        raise RuntimeError('No models were fitted. Check taxon filters and occurrence counts.')

    pred_arr = np.vstack(pred_stack).astype(np.float32)
    pred_valid = np.isfinite(pred_arr)
    group_names = [a['group'] for a in artifacts]
    n_rows = pred_arr.shape[1]
    safe_pred = np.where(pred_valid, pred_arr, -np.inf)
    top_idx = np.argmax(safe_pred, axis=0)
    top_score = safe_pred[top_idx, np.arange(n_rows)]
    top_group = np.array([group_names[i] for i in top_idx], dtype=object)
    no_score = ~np.isfinite(top_score) | (top_score == -np.inf)
    top_score = np.where(no_score, np.nan, top_score)
    top_group = np.where(no_score, '', top_group)

    overall_ml = np.max(safe_pred, axis=0)
    overall_ml = np.where(np.isfinite(overall_ml) & (overall_ml > -np.inf), overall_ml, np.nan).astype(np.float32)
    overall_ml_min = finite_min_score(pred_arr, pred_valid)
    overall_ml_joint = joint_support_score(pred_arr, pred_valid, min_share=args.joint_min_share, tail_fraction=args.joint_tail_fraction, rank_power=args.joint_rank_power)
    threshold_arr = np.asarray(thresholds, dtype=np.float32)[:, None]
    richness_ml = np.sum(np.where(pred_valid, pred_arr >= threshold_arr, False), axis=0).astype(np.int32)

    overall_df = pd.DataFrame({
        args.id_col: grid_work[args.id_col].tolist(),
        args.x_col: grid_work[args.x_col].tolist(),
        args.y_col: grid_work[args.y_col].tolist(),
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
        preview_point_map(overall_df, 'overall_ml', 'overall ML suitability', preview_dir / 'overall_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax)
        preview_point_map(overall_df, 'overall_ml_min', 'overall ML minimum overlap', preview_dir / 'overall_ml_min.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax)
        preview_point_map(overall_df, 'overall_ml_joint', 'overall ML joint suitability', preview_dir / 'overall_ml_joint.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax)
        preview_point_map(overall_df, 'richness_ml', 'richness above species thresholds', preview_dir / 'richness_ml.png', args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, None)
        for row in species_summary_df.head(int(args.preview_top_n)).itertuples(index=False):
            dfp = pd.read_csv(outdir / 'by_species' / f"{safe_slug(row.group)}.csv", usecols=[args.x_col, args.y_col, 'ml_probability'], low_memory=False)
            preview_point_map(dfp, 'ml_probability', f'{row.group} ML suitability', preview_dir / 'by_species' / f"{safe_slug(row.group)}_ml.png", args.x_col, args.y_col, args.preview_point_alpha, args.preview_coarsen, args.preview_vmax)

    manifest = {
        'occurrences_csv': str(occ_csv),
        'grid_csv': str(grid_csv),
        'trend_summary': str(trend_summary_dir),
        'outdir': str(outdir),
        'group_by': args.group_by,
        'include_taxa': [s.raw for s in include_selectors],
        'exclude_taxa': [s.raw for s in exclude_selectors],
        'selected_group_count': int(len(artifacts)),
        'selected_groups': group_names,
        'numeric_feature_count': int(len(spec.numeric_cols)),
        'categorical_feature_count': int(len(spec.categorical_cols)),
        'encoded_feature_count': int(len(spec.encoded_columns)),
        'feature_select_max_rows': int(args.feature_select_max_rows),
        'variance_threshold': float(args.variance_threshold),
        'corr_threshold': float(args.corr_threshold),
        'max_features_after_corr': int(args.max_features_after_corr),
        'background_multiplier': float(args.background_multiplier),
        'min_background': int(args.min_background),
        'max_background': int(args.max_background),
        'cv_folds': int(args.cv_folds),
        'cv_block_size': float(args.cv_block_size),
        'predict_chunk_size': int(args.predict_chunk_size),
        'joint_min_share': float(args.joint_min_share),
        'joint_tail_fraction': float(args.joint_tail_fraction),
        'joint_rank_power': float(args.joint_rank_power),
        'available_models': sorted(list(build_estimators(args, 1.0).keys())),
    }
    write_manifest(outdir / 'manifest.json', manifest)
    log(f'[done] groups={len(artifacts)} grid_rows={len(grid_work):,} outdir={outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
