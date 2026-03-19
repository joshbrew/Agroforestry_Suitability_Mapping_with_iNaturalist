#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ex after running both build_suitability_maps.py and xgboost_suitability.py
#  D:\Oregon_Suitability> py blend_suitability_with_ml.py D:/Oregon_Suitability/oregon_grid_1000m_env/suitability_maps D:/Oregon_Suitability/oregon_grid_1000m_env/ml_association_test --outdir D:/Oregon_Suitability/oregon_grid_1000m_env/blended_suitability_maps4 --ml-floor 0.3 --ml-prob-power 0.25 --mix-alpha 1 --smooth-radius 8 --sharpen-amount 0.5
def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "group"


def detect_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the expected columns were found: {candidates}")


def clamp01(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def ml_gate(ml: np.ndarray, floor: float, power: float) -> np.ndarray:
    floor2 = float(np.clip(floor, 0.0, 1.0))
    power2 = max(0.05, float(power))
    work = np.power(np.clip(ml, 0.0, 1.0), power2)
    return floor2 + (1.0 - floor2) * work


def blend_gate(base: np.ndarray, ml: np.ndarray, floor: float, power: float) -> np.ndarray:
    gate = ml_gate(ml, floor=floor, power=power)
    out = np.where(np.isfinite(base) & np.isfinite(ml), base * gate, np.nan)
    return clamp01(out)


def blend_mix(base: np.ndarray, ml: np.ndarray, alpha: float) -> np.ndarray:
    alpha2 = float(np.clip(alpha, 0.0, 1.0))
    out = np.where(np.isfinite(base) & np.isfinite(ml), (1.0 - alpha2) * base + alpha2 * ml, np.nan)
    return clamp01(out)


def blend_geo(base: np.ndarray, ml: np.ndarray, floor: float) -> np.ndarray:
    floor2 = float(np.clip(floor, 1e-6, 1.0))
    ml_safe = np.maximum(np.clip(ml, 0.0, 1.0), floor2)
    out = np.where(np.isfinite(base) & np.isfinite(ml), np.sqrt(np.clip(base, 0.0, 1.0) * ml_safe), np.nan)
    return clamp01(out)


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


def try_regular_grid(df: pd.DataFrame, x_col: str, y_col: str, value_col: str):
    x_vals = pd.to_numeric(df[x_col], errors='coerce').to_numpy(dtype=float)
    y_vals = pd.to_numeric(df[y_col], errors='coerce').to_numpy(dtype=float)
    z_vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
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


def nan_box_filter_2d(grid: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return grid.astype(np.float32, copy=True)
    work = np.asarray(grid, dtype=np.float32)
    valid = np.isfinite(work)
    vals = np.where(valid, work, 0.0).astype(np.float32)
    counts = valid.astype(np.float32)
    pad = radius
    vals_pad = np.pad(vals, ((pad, pad), (pad, pad)), mode='constant', constant_values=0.0)
    counts_pad = np.pad(counts, ((pad, pad), (pad, pad)), mode='constant', constant_values=0.0)
    H, W = vals.shape
    out_vals = np.zeros((H, W), dtype=np.float32)
    out_counts = np.zeros((H, W), dtype=np.float32)
    k = 2 * radius + 1
    for dy in range(k):
        ys = dy
        ye = dy + H
        for dx in range(k):
            xs = dx
            xe = dx + W
            out_vals += vals_pad[ys:ye, xs:xe]
            out_counts += counts_pad[ys:ye, xs:xe]
    out = np.full((H, W), np.nan, dtype=np.float32)
    ok = out_counts > 0
    out[ok] = out_vals[ok] / out_counts[ok]
    return out


def unsharp_mask(grid: np.ndarray, radius: int, amount: float) -> np.ndarray:
    amount2 = max(0.0, float(amount))
    if radius <= 0 or amount2 <= 0:
        return clamp01(grid)
    smooth = nan_box_filter_2d(grid, radius=radius)
    out = np.where(np.isfinite(grid) & np.isfinite(smooth), grid + amount2 * (grid - smooth), np.where(np.isfinite(grid), grid, np.nan))
    return clamp01(out)


def apply_grid_filter(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, smooth_radius: int, sharpen_amount: float) -> pd.DataFrame:
    payload = try_regular_grid(df, x_col=x_col, y_col=y_col, value_col=value_col)
    if payload is None:
        out = df.copy()
        out[f'{value_col}_smooth'] = out[value_col]
        out[f'{value_col}_sharpen'] = out[value_col]
        return out
    xs, ys, grid, _, _ = payload
    smooth = nan_box_filter_2d(grid, radius=smooth_radius)
    sharp = unsharp_mask(smooth if smooth_radius > 0 else grid, radius=max(1, smooth_radius), amount=sharpen_amount)
    x_map = {float(v): i for i, v in enumerate(xs)}
    y_map = {float(v): i for i, v in enumerate(ys)}
    out = df.copy()
    sx = []
    sh = []
    xr = pd.to_numeric(out[x_col], errors='coerce').to_numpy(dtype=float)
    yr = pd.to_numeric(out[y_col], errors='coerce').to_numpy(dtype=float)
    vals = pd.to_numeric(out[value_col], errors='coerce').to_numpy(dtype=float)
    decimals = 6
    # align to available keys by nearest rounded precision that matches the payload
    xr_round = np.round(xr, decimals=6)
    yr_round = np.round(yr, decimals=6)
    for xv, yv, zv in zip(xr_round, yr_round, vals):
        ix = x_map.get(float(xv))
        iy = y_map.get(float(yv))
        if ix is None or iy is None or not np.isfinite(zv):
            sx.append(np.nan)
            sh.append(np.nan)
        else:
            sx.append(float(smooth[iy, ix]))
            sh.append(float(sharp[iy, ix]))
    out[f'{value_col}_smooth'] = np.asarray(sx, dtype=np.float32)
    out[f'{value_col}_sharpen'] = np.asarray(sh, dtype=np.float32)
    return out


def preview_point_map(df: pd.DataFrame, value_col: str, title: str, out_path: Path, x_col: str, y_col: str, vmax: Optional[float] = 1.0) -> None:
    if plt is None:
        return
    if value_col not in df.columns:
        return
    sub = df[[x_col, y_col, value_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors='coerce')
    sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
    sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
    sub = sub.dropna(subset=[x_col, y_col, value_col])
    if sub.empty:
        return
    payload = try_regular_grid(sub, x_col=x_col, y_col=y_col, value_col=value_col)
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    if payload is not None:
        _, _, grid, x_edges, y_edges = payload
        masked = np.ma.masked_invalid(grid)
        mesh = ax.pcolormesh(x_edges, y_edges, masked, shading='flat', cmap='viridis', antialiased=False, linewidth=0, vmin=0.0, vmax=vmax)
        color_obj = mesh
    else:
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
        ax.scatter(sub[x_col], sub[y_col], c=sub[value_col], s=4.0, alpha=0.35, linewidths=0, rasterized=True, cmap='viridis', norm=norm)
        color_obj = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
        color_obj.set_array([])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2)
    fig.colorbar(color_obj, ax=ax, shrink=0.78)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def blend_overall(base_df: pd.DataFrame, ml_df: pd.DataFrame, id_col: str, x_col: str, y_col: str, ml_prefix: str, floor: float, power: float, alpha: float, smooth_radius: int, sharpen_amount: float) -> pd.DataFrame:
    merged = base_df.merge(ml_df, on=[id_col, x_col, y_col], how='inner', suffixes=('_base', '_ml'))
    base_adj = pd.to_numeric(merged['overall_adjusted'], errors='coerce').to_numpy(dtype=np.float32)
    base_min = pd.to_numeric(merged['overall_adjusted_min'], errors='coerce').to_numpy(dtype=np.float32)
    base_joint = pd.to_numeric(merged['overall_adjusted_joint'], errors='coerce').to_numpy(dtype=np.float32)
    ml_adj = pd.to_numeric(merged[f'{ml_prefix}'], errors='coerce').to_numpy(dtype=np.float32)
    ml_min = pd.to_numeric(merged[f'{ml_prefix}_min'], errors='coerce').to_numpy(dtype=np.float32)
    ml_joint = pd.to_numeric(merged[f'{ml_prefix}_joint'], errors='coerce').to_numpy(dtype=np.float32)

    merged['blend_gate_adjusted'] = blend_gate(base_adj, ml_adj, floor=floor, power=power)
    merged['blend_mix_adjusted'] = blend_mix(base_adj, ml_adj, alpha=alpha)
    merged['blend_geo_adjusted'] = blend_geo(base_adj, ml_adj, floor=floor)

    merged['blend_gate_adjusted_min'] = blend_gate(base_min, ml_min, floor=floor, power=power)
    merged['blend_mix_adjusted_min'] = blend_mix(base_min, ml_min, alpha=alpha)
    merged['blend_geo_adjusted_min'] = blend_geo(base_min, ml_min, floor=floor)

    merged['blend_gate_adjusted_joint'] = blend_gate(base_joint, ml_joint, floor=floor, power=power)
    merged['blend_mix_adjusted_joint'] = blend_mix(base_joint, ml_joint, alpha=alpha)
    merged['blend_geo_adjusted_joint'] = blend_geo(base_joint, ml_joint, floor=floor)

    for col in ['blend_gate_adjusted', 'blend_gate_adjusted_min', 'blend_gate_adjusted_joint']:
        filtered = apply_grid_filter(merged[[id_col, x_col, y_col, col]].copy(), x_col=x_col, y_col=y_col, value_col=col, smooth_radius=smooth_radius, sharpen_amount=sharpen_amount)
        merged[col] = filtered[col]
    return merged


def blend_species(base_path: Path, ml_path: Path, id_col: str, x_col: str, y_col: str, floor: float, power: float, alpha: float, smooth_radius: int, sharpen_amount: float) -> Optional[pd.DataFrame]:
    base_df = pd.read_csv(base_path, low_memory=False)
    ml_df = pd.read_csv(ml_path, low_memory=False)
    if 'adjusted_score' not in base_df.columns:
        return None
    ml_col = 'ml_probability' if 'ml_probability' in ml_df.columns else ('xgb_probability' if 'xgb_probability' in ml_df.columns else None)
    if ml_col is None:
        return None
    merged = base_df.merge(ml_df[[id_col, x_col, y_col, ml_col]], on=[id_col, x_col, y_col], how='inner')
    base_adj = pd.to_numeric(merged['adjusted_score'], errors='coerce').to_numpy(dtype=np.float32)
    ml = pd.to_numeric(merged[ml_col], errors='coerce').to_numpy(dtype=np.float32)
    merged['blend_gate_adjusted'] = blend_gate(base_adj, ml, floor=floor, power=power)
    merged['blend_mix_adjusted'] = blend_mix(base_adj, ml, alpha=alpha)
    merged['blend_geo_adjusted'] = blend_geo(base_adj, ml, floor=floor)
    filtered = apply_grid_filter(merged[[id_col, x_col, y_col, 'blend_gate_adjusted']].copy(), x_col=x_col, y_col=y_col, value_col='blend_gate_adjusted', smooth_radius=smooth_radius, sharpen_amount=sharpen_amount)
    merged['blend_gate_adjusted'] = filtered['blend_gate_adjusted']
    return merged


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Blend empirical suitability_maps outputs with ML outputs and optionally bake a light grid filter directly into the main blend columns.')
    ap.add_argument('base_dir', help='Output directory from build_suitability_maps.py')
    ap.add_argument('ml_dir', help='Output directory from xgboost_suitability.py or ml_suitability.py')
    ap.add_argument('--outdir', required=True, help='Directory for blended outputs')
    ap.add_argument('--id-col', default='id')
    ap.add_argument('--x-col', default='lon')
    ap.add_argument('--y-col', default='lat')
    ap.add_argument('--ml-prob-power', type=float, default=0.5, help='Power applied to ML probability before gating the empirical map')
    ap.add_argument('--ml-floor', type=float, default=0.35, help='Minimum share of the empirical map retained even when ML probability is near zero')
    ap.add_argument('--mix-alpha', type=float, default=0.30, help='Linear blend weight reserved for the ML surface')
    ap.add_argument('--smooth-radius', type=int, default=1, help='NaN-aware box-filter radius in grid cells')
    ap.add_argument('--sharpen-amount', type=float, default=0.75, help='Unsharp-mask amount applied after smoothing to the gate blend')
    ap.add_argument('--preview-top-n', type=int, default=8)
    ap.add_argument('--preview-vmax', type=float, default=0.65)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    ml_dir = Path(args.ml_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_overall_path = base_dir / 'overall_suitability.csv'
    ml_overall_path = ml_dir / 'overall_suitability.csv'
    if not base_overall_path.exists():
        raise FileNotFoundError(f'Missing base overall_suitability.csv: {base_overall_path}')
    if not ml_overall_path.exists():
        raise FileNotFoundError(f'Missing ML overall_suitability.csv: {ml_overall_path}')

    base_overall = pd.read_csv(base_overall_path, low_memory=False)
    ml_overall = pd.read_csv(ml_overall_path, low_memory=False)
    ml_prefix = detect_column(ml_overall, ['overall_ml', 'overall_xgb'])

    blended_overall = blend_overall(
        base_df=base_overall,
        ml_df=ml_overall,
        id_col=args.id_col,
        x_col=args.x_col,
        y_col=args.y_col,
        ml_prefix=ml_prefix,
        floor=args.ml_floor,
        power=args.ml_prob_power,
        alpha=args.mix_alpha,
        smooth_radius=args.smooth_radius,
        sharpen_amount=args.sharpen_amount,
    )
    blended_overall.to_csv(outdir / 'overall_suitability.csv', index=False)

    preview_dir = outdir / 'previews'
    preview_point_map(blended_overall, 'blend_gate_adjusted', 'blended gate adjusted suitability', preview_dir / 'overall_blend_gate_adjusted.png', args.x_col, args.y_col, vmax=args.preview_vmax)
    preview_point_map(blended_overall, 'blend_gate_adjusted_sharpen', 'blended gate adjusted sharpened', preview_dir / 'overall_blend_gate_adjusted_sharpen.png', args.x_col, args.y_col, vmax=args.preview_vmax)
    preview_point_map(blended_overall, 'blend_mix_adjusted', 'blended mean adjusted suitability', preview_dir / 'overall_blend_mix_adjusted.png', args.x_col, args.y_col, vmax=args.preview_vmax)
    preview_point_map(blended_overall, 'blend_geo_adjusted', 'blended geometric adjusted suitability', preview_dir / 'overall_blend_geo_adjusted.png', args.x_col, args.y_col, vmax=args.preview_vmax)
    preview_point_map(blended_overall, 'blend_gate_adjusted_min', 'blended gate adjusted minimum overlap', preview_dir / 'overall_blend_gate_adjusted_min.png', args.x_col, args.y_col, vmax=args.preview_vmax)
    preview_point_map(blended_overall, 'blend_gate_adjusted_joint', 'blended gate adjusted joint suitability', preview_dir / 'overall_blend_gate_adjusted_joint.png', args.x_col, args.y_col, vmax=args.preview_vmax)

    base_species_dir = base_dir / 'by_species'
    ml_species_dir = ml_dir / 'by_species'
    out_species_dir = outdir / 'by_species'
    out_species_dir.mkdir(parents=True, exist_ok=True)
    species_rows = []
    if base_species_dir.exists() and ml_species_dir.exists():
        common = sorted({p.name for p in base_species_dir.glob('*.csv')} & {p.name for p in ml_species_dir.glob('*.csv')})
        for name in common:
            blended = blend_species(
                base_path=base_species_dir / name,
                ml_path=ml_species_dir / name,
                id_col=args.id_col,
                x_col=args.x_col,
                y_col=args.y_col,
                floor=args.ml_floor,
                power=args.ml_prob_power,
                alpha=args.mix_alpha,
                smooth_radius=args.smooth_radius,
                sharpen_amount=args.sharpen_amount,
            )
            if blended is None:
                continue
            blended.to_csv(out_species_dir / name, index=False)
            species_rows.append({
                'file': name,
                'grid_mean_blend_gate_adjusted': float(pd.to_numeric(blended['blend_gate_adjusted'], errors='coerce').mean()),
                'grid_mean_blend_mix_adjusted': float(pd.to_numeric(blended['blend_mix_adjusted'], errors='coerce').mean()),
                'grid_mean_blend_geo_adjusted': float(pd.to_numeric(blended['blend_geo_adjusted'], errors='coerce').mean()),
            })
        if species_rows:
            species_summary = pd.DataFrame(species_rows).sort_values(['grid_mean_blend_gate_adjusted', 'file'], ascending=[False, True]).reset_index(drop=True)
            species_summary.to_csv(outdir / 'species_blend_summary.csv', index=False)
            for row in species_summary.head(int(args.preview_top_n)).itertuples(index=False):
                dfp = pd.read_csv(out_species_dir / row.file, usecols=[args.x_col, args.y_col, 'blend_gate_adjusted'], low_memory=False)
                slug = safe_slug(Path(row.file).stem)
                preview_point_map(dfp, 'blend_gate_adjusted', f'{Path(row.file).stem} blended gate adjusted', preview_dir / 'by_species' / f'{slug}_blend_gate_adjusted.png', args.x_col, args.y_col, vmax=args.preview_vmax)

    manifest = {
        'base_dir': str(base_dir),
        'ml_dir': str(ml_dir),
        'outdir': str(outdir),
        'ml_probability_prefix': ml_prefix,
        'ml_prob_power': float(args.ml_prob_power),
        'ml_floor': float(args.ml_floor),
        'mix_alpha': float(args.mix_alpha),
        'smooth_radius': int(args.smooth_radius),
        'sharpen_amount': float(args.sharpen_amount),
        'recommended_primary_column': 'blend_gate_adjusted',
        'recommended_overlap_min_column': 'blend_gate_adjusted_min',
        'recommended_overlap_joint_column': 'blend_gate_adjusted_joint',
    }
    (outdir / 'manifest.json').write_text(pd.Series(manifest).to_json(indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
