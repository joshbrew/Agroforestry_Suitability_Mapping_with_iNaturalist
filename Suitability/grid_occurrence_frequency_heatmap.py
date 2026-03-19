#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
except Exception:
    plt = None

TAXON_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


@dataclass
class TaxonSelector:
    raw: str
    rank: Optional[str]
    name: str
    name_norm: str


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
        if pd.notna(value) and str(value).strip() != '':
            return value
    return pd.NA


def resolve_group_meta(df_occ: pd.DataFrame, group_by: str) -> pd.DataFrame:
    needed = []
    for col in [group_by, 'matched_species_name', *TAXON_RANKS]:
        if col in df_occ.columns and col not in needed:
            needed.append(col)
    sub = df_occ[needed].copy()
    sub[group_by] = sub[group_by].astype('string').fillna(pd.NA).map(lambda x: 'NA' if pd.isna(x) else str(x))
    grouped = sub.groupby(group_by, dropna=False, sort=True)
    rows = []
    for group, gdf in grouped:
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


def estimate_step(values: np.ndarray) -> float:
    vals = np.sort(np.unique(values[np.isfinite(values)]))
    if vals.size < 2:
        return float('nan')
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float('nan')
    return float(np.median(diffs))


def grid_edges(values: np.ndarray, step: float) -> np.ndarray:
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


def infer_grid_layout(grid_df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray, float, float, float, pd.DataFrame, cKDTree]:
    layout = grid_df.copy()
    layout[x_col] = pd.to_numeric(layout[x_col], errors='coerce')
    layout[y_col] = pd.to_numeric(layout[y_col], errors='coerce')
    layout = layout.dropna(subset=[x_col, y_col]).copy().reset_index(drop=True)
    if layout.empty:
        raise RuntimeError('Grid CSV has no valid coordinate rows')

    xs = np.sort(layout[x_col].dropna().unique())
    ys = np.sort(layout[y_col].dropna().unique())
    dx = estimate_step(xs)
    dy = estimate_step(ys)

    coords = layout[[x_col, y_col]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    if len(layout) > 1:
        sample_n = min(len(layout), 20000)
        sample_idx = np.linspace(0, len(layout) - 1, sample_n, dtype=np.int64)
        dists, _ = tree.query(coords[sample_idx], k=2)
        nn = dists[:, 1]
        nn = nn[np.isfinite(nn) & (nn > 0)]
        cell_radius = float(np.median(nn) * 0.55) if nn.size else float('nan')
    else:
        cell_radius = float('nan')

    if not np.isfinite(cell_radius) or cell_radius <= 0:
        if np.isfinite(dx) and dx > 0 and np.isfinite(dy) and dy > 0:
            cell_radius = 0.55 * float((dx ** 2 + dy ** 2) ** 0.5)
        else:
            cell_radius = 1e-4

    layout['ix'] = np.arange(len(layout), dtype=np.int64)
    layout['iy'] = 0
    layout['cell_key'] = layout.index.astype(str)
    return xs, ys, dx, dy, cell_radius, layout, tree


def assign_occurrences_to_cells(
    occ_df: pd.DataFrame,
    grid_layout: pd.DataFrame,
    occ_x_col: str,
    occ_y_col: str,
    tree: cKDTree,
    tolerance_radius: float,
    tolerance_frac: float,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    occ = occ_df.copy()
    occ[occ_x_col] = pd.to_numeric(occ[occ_x_col], errors='coerce')
    occ[occ_y_col] = pd.to_numeric(occ[occ_y_col], errors='coerce')
    occ = occ.dropna(subset=[occ_x_col, occ_y_col]).copy()
    if occ.empty:
        return occ.assign(ix=pd.Series(dtype='int64'), iy=pd.Series(dtype='int64'), cell_key=pd.Series(dtype='string'))

    occ_coords = occ[[occ_x_col, occ_y_col]].to_numpy(dtype=float)
    dists, idx = tree.query(occ_coords, k=1)
    tol = max(float(tolerance_radius) * float(tolerance_frac), 1e-12)
    keep = np.isfinite(dists) & (dists <= tol)
    occ = occ.loc[keep].copy()
    if occ.empty:
        return occ.assign(ix=pd.Series(dtype='int64'), iy=pd.Series(dtype='int64'), cell_key=pd.Series(dtype='string'))

    matched = grid_layout.iloc[idx[keep]][['ix', 'iy', 'cell_key', x_col, y_col]].reset_index(drop=True)
    occ = occ.reset_index(drop=True)
    occ[['ix', 'iy', 'cell_key']] = matched[['ix', 'iy', 'cell_key']]
    occ['_matched_grid_x'] = matched[x_col].to_numpy(dtype=float)
    occ['_matched_grid_y'] = matched[y_col].to_numpy(dtype=float)
    occ['_match_distance'] = dists[keep].astype(float)
    return occ


def count_to_grid(grid_layout: pd.DataFrame, counts: pd.DataFrame, id_col: str, x_col: str, y_col: str, count_col: str = 'occurrence_count') -> pd.DataFrame:
    base = grid_layout[[id_col, x_col, y_col, 'ix', 'iy', 'cell_key']].copy()
    out = base.merge(counts[['cell_key', count_col]], on='cell_key', how='left')
    out[count_col] = pd.to_numeric(out[count_col], errors='coerce').fillna(0).astype(np.int32)
    return out.sort_values(['iy', 'ix']).reset_index(drop=True)



def _estimate_regular_step(values: np.ndarray) -> float:
    if values.size < 2:
        return float('nan')
    diffs = np.diff(np.sort(np.unique(values)))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float('nan')
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
    x_vals = pd.to_numeric(sub[x_col], errors='coerce').to_numpy(dtype=float)
    y_vals = pd.to_numeric(sub[y_col], errors='coerce').to_numpy(dtype=float)
    z_vals = pd.to_numeric(sub[value_col], errors='coerce').to_numpy(dtype=float)

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

def render_heatmap(df: pd.DataFrame, frame_df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str, out_path: Path, log1p_scale: bool = False, preview_coarsen: int = 2) -> None:
    if plt is None:
        return
    if value_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return

    frame = frame_df[[x_col, y_col]].copy()
    frame[x_col] = pd.to_numeric(frame[x_col], errors='coerce')
    frame[y_col] = pd.to_numeric(frame[y_col], errors='coerce')
    frame = frame.dropna(subset=[x_col, y_col])
    if frame.empty:
        return

    sub = df[[x_col, y_col, value_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors='coerce')
    sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
    sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
    sub = sub.dropna(subset=[x_col, y_col, value_col]).copy()
    if sub.empty:
        return

    if log1p_scale:
        sub[value_col] = np.log1p(np.clip(sub[value_col].to_numpy(dtype=float), 0.0, None)).astype(np.float32)

    sub = sub[sub[value_col] > 0].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    payload = _try_regular_grid(sub, x_col, y_col, value_col)
    if payload is not None:
        _, _, grid, x_edges, y_edges = payload
        grid, x_edges, y_edges = _coarsen_regular_grid(grid, x_edges, y_edges, preview_coarsen)
        grid_vis = grid.copy()
        grid_vis[~np.isfinite(grid_vis)] = np.nan
        grid_vis[grid_vis <= 0] = np.nan
        masked = np.ma.masked_invalid(grid_vis)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        finite_vals = grid_vis[np.isfinite(grid_vis)]
        vmin = float(np.nanpercentile(finite_vals, 5)) if finite_vals.size else 0.0
        vmax = float(np.nanpercentile(finite_vals, 99)) if finite_vals.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            masked,
            shading='flat',
            cmap=cmap,
            antialiased=False,
            linewidth=0,
            vmin=vmin,
            vmax=vmax,
        )
        color_obj = mesh
    else:
        vals = sub[value_col].to_numpy(dtype=float)
        vmin = float(np.nanpercentile(vals, 5)) if vals.size else 0.0
        vmax = float(np.nanpercentile(vals, 99)) if vals.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('viridis')
        ax.scatter(
            sub[x_col],
            sub[y_col],
            c=sub[value_col],
            s=4.0,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            cmap=cmap,
            norm=norm,
        )
        color_obj = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        color_obj.set_array([])

    x_all = frame[x_col].to_numpy(dtype=float)
    y_all = frame[y_col].to_numpy(dtype=float)
    ax.set_xlim(float(np.nanmin(x_all)), float(np.nanmax(x_all)))
    ax.set_ylim(float(np.nanmin(y_all)), float(np.nanmax(y_all)))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.15)
    cbar = fig.colorbar(color_obj, ax=ax, shrink=0.78)
    cbar.set_label(f'log1p({value_col})' if log1p_scale else value_col)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)

def main() -> int:

    ap = argparse.ArgumentParser(description='Count filtered occurrence observations that fall within available grid cells and render frequency heatmaps.')
    ap.add_argument('occurrences_csv', help='Occurrence CSV, usually occurrences_enriched.csv or occurrences_enriched.cleaned.csv')
    ap.add_argument('grid_csv', help='Grid CSV, usually grid_with_env.csv')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--group-by', default='matched_species_name', help='Grouping column used for the filtered taxa, usually matched_species_name')
    ap.add_argument('--include-taxa', nargs='*', default=None, help='Optional rank selectors such as family:Rosaceae genus:Ribes species:Daucus pusillus')
    ap.add_argument('--exclude-taxa', nargs='*', default=None, help='Optional rank selectors to exclude')
    ap.add_argument('--id-col', default='id', help='Grid id column')
    ap.add_argument('--x-col', default='lon', help='Grid x column')
    ap.add_argument('--y-col', default='lat', help='Grid y column')
    ap.add_argument('--occ-x-col', default='decimalLongitude', help='Occurrence x column')
    ap.add_argument('--occ-y-col', default='decimalLatitude', help='Occurrence y column')
    ap.add_argument('--min-occurrences-per-group', type=int, default=1, help='Minimum occurrence rows required to keep a group after filtering')
    ap.add_argument('--per-group-top-n', type=int, default=0, help='Write per-group outputs only for the top N selected groups by mapped frequency; 0 keeps all')
    ap.add_argument('--tolerance-frac', type=float, default=1.0, help='Multiplier applied to the inferred centroid-to-cell radius when matching an occurrence to the nearest available grid cell')
    ap.add_argument('--log1p-preview', action='store_true', help='Render preview heatmaps using log1p(count) instead of raw counts')
    args = ap.parse_args()

    occurrences_csv = Path(args.occurrences_csv).resolve()
    grid_csv = Path(args.grid_csv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not occurrences_csv.exists():
        raise FileNotFoundError(f'Missing occurrences_csv: {occurrences_csv}')
    if not grid_csv.exists():
        raise FileNotFoundError(f'Missing grid_csv: {grid_csv}')

    include_selectors = parse_selector_list(args.include_taxa)
    exclude_selectors = parse_selector_list(args.exclude_taxa)

    log(f'[load] occurrences={occurrences_csv}')
    log(f'[load] grid={grid_csv}')

    occ_df = pd.read_csv(occurrences_csv, low_memory=False)
    grid_df = pd.read_csv(grid_csv, usecols=[args.id_col, args.x_col, args.y_col], low_memory=False)

    if args.group_by not in occ_df.columns:
        raise KeyError(f'group-by column not found in occurrences CSV: {args.group_by}')
    for col in [args.occ_x_col, args.occ_y_col]:
        if col not in occ_df.columns:
            raise KeyError(f'occurrence coordinate column not found: {col}')
    for col in [args.id_col, args.x_col, args.y_col]:
        if col not in grid_df.columns:
            raise KeyError(f'grid column not found: {col}')

    group_meta = resolve_group_meta(occ_df, args.group_by)
    group_meta = apply_taxon_filters(group_meta, include_selectors, exclude_selectors, args.group_by)
    if args.min_occurrences_per_group > 0:
        group_meta = group_meta[pd.to_numeric(group_meta['occurrence_count'], errors='coerce').fillna(0) >= int(args.min_occurrences_per_group)].copy()
    group_meta = group_meta.sort_values(['occurrence_count', 'group'], ascending=[False, True]).reset_index(drop=True)
    if group_meta.empty:
        raise RuntimeError('No groups remain after taxon filtering and occurrence thresholds')

    selected_groups = set(group_meta['group'].astype(str))
    occ_df = occ_df.copy()
    occ_df[args.group_by] = occ_df[args.group_by].astype('string').fillna(pd.NA).map(lambda x: 'NA' if pd.isna(x) else str(x))
    occ_selected = occ_df[occ_df[args.group_by].astype(str).isin(selected_groups)].copy()
    if occ_selected.empty:
        raise RuntimeError('No occurrence rows remain after applying the selected groups')

    xs, ys, dx, dy, cell_radius, grid_layout, tree = infer_grid_layout(grid_df, args.x_col, args.y_col)
    log(f'[grid] cells={len(grid_layout):,} unique_x={len(xs):,} unique_y={len(ys):,} dx={dx:.8f} dy={dy:.8f} cell_radius={cell_radius:.8f}')

    occ_mapped = assign_occurrences_to_cells(
        occ_df=occ_selected,
        grid_layout=grid_layout,
        occ_x_col=args.occ_x_col,
        occ_y_col=args.occ_y_col,
        tree=tree,
        tolerance_radius=cell_radius,
        tolerance_frac=float(args.tolerance_frac),
        x_col=args.x_col,
        y_col=args.y_col,
    )

    mapped_summary = (
        occ_selected.groupby(args.group_by, dropna=False).size().rename('selected_occurrences').reset_index().rename(columns={args.group_by: 'group'})
        .merge(occ_mapped.groupby(args.group_by, dropna=False).size().rename('mapped_occurrences').reset_index().rename(columns={args.group_by: 'group'}), on='group', how='left')
    )
    mapped_summary['mapped_occurrences'] = mapped_summary['mapped_occurrences'].fillna(0).astype(int)
    mapped_summary['unmapped_occurrences'] = mapped_summary['selected_occurrences'] - mapped_summary['mapped_occurrences']
    mapped_summary['mapped_fraction'] = np.where(mapped_summary['selected_occurrences'] > 0, mapped_summary['mapped_occurrences'] / mapped_summary['selected_occurrences'], np.nan)
    mapped_summary = mapped_summary.merge(group_meta, on='group', how='left')
    mapped_summary = mapped_summary.sort_values(['mapped_occurrences', 'group'], ascending=[False, True]).reset_index(drop=True)
    mapped_summary.to_csv(outdir / 'group_mapping_summary.csv', index=False)

    log(f'[map] selected_occurrences={len(occ_selected):,} mapped_occurrences={len(occ_mapped):,} unmapped_occurrences={len(occ_selected) - len(occ_mapped):,}')

    overall_counts = occ_mapped.groupby('cell_key', dropna=False).size().rename('occurrence_count').reset_index()
    overall_grid = count_to_grid(grid_layout, overall_counts, args.id_col, args.x_col, args.y_col, count_col='occurrence_count')
    overall_grid.to_csv(outdir / 'overall_occurrence_frequency.csv', index=False)
    render_heatmap(overall_grid, grid_layout, args.x_col, args.y_col, 'occurrence_count', 'Occurrence frequency within available grid cells', outdir / 'previews' / 'overall_occurrence_frequency.png', log1p_scale=bool(args.log1p_preview), preview_coarsen=2)

    groups_to_write = mapped_summary['group'].astype(str).tolist()
    if args.per_group_top_n > 0:
        groups_to_write = groups_to_write[: int(args.per_group_top_n)]

    (outdir / 'by_group').mkdir(parents=True, exist_ok=True)
    (outdir / 'previews' / 'by_group').mkdir(parents=True, exist_ok=True)

    for i, group in enumerate(groups_to_write, start=1):
        sub = occ_mapped[occ_mapped[args.group_by].astype(str) == str(group)].copy()
        counts = sub.groupby('cell_key', dropna=False).size().rename('occurrence_count').reset_index()
        grid_counts = count_to_grid(grid_layout, counts, args.id_col, args.x_col, args.y_col, count_col='occurrence_count')
        slug = safe_slug(group)
        grid_counts.to_csv(outdir / 'by_group' / f'{slug}.csv', index=False)
        render_heatmap(grid_counts, grid_layout, args.x_col, args.y_col, 'occurrence_count', f'{group} occurrence frequency within available grid cells', outdir / 'previews' / 'by_group' / f'{slug}.png', log1p_scale=bool(args.log1p_preview), preview_coarsen=2)
        if i <= 20 or i == len(groups_to_write):
            log(f'[group] {i}/{len(groups_to_write)} wrote {group}')

    manifest = {
        'occurrences_csv': str(occurrences_csv),
        'grid_csv': str(grid_csv),
        'outdir': str(outdir),
        'group_by': args.group_by,
        'include_taxa': [s.raw for s in include_selectors],
        'exclude_taxa': [s.raw for s in exclude_selectors],
        'selected_group_count': int(len(group_meta)),
        'selected_groups': group_meta['group'].astype(str).tolist(),
        'grid_cell_count': int(len(grid_layout)),
        'grid_dx': float(dx),
        'grid_dy': float(dy),
        'grid_cell_radius': float(cell_radius),
        'occurrence_x_col': args.occ_x_col,
        'occurrence_y_col': args.occ_y_col,
        'tolerance_frac': float(args.tolerance_frac),
        'selected_occurrences': int(len(occ_selected)),
        'mapped_occurrences': int(len(occ_mapped)),
        'unmapped_occurrences': int(len(occ_selected) - len(occ_mapped)),
        'per_group_written': int(len(groups_to_write)),
        'log1p_preview': bool(args.log1p_preview),
    }
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log(f'[done] outdir={outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
