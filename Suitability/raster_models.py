#!/usr/bin/env python3
# Example:
# py raster_models.py D:/Oregon_Suitability/oregon_grid_1000m_env/blended_suitability_maps4 --column blend_mix_adjusted --smooth-radius 1
# py raster_models.py D:/Oregon_Suitability/oregon_grid_1000m_env/blended_suitability_maps4 --columns blend_mix_adjusted,blend_gate_adjusted,blend_geo_adjusted --smooth-radius 1
# py raster_models.py D:/Suitability/project --columns blend_mix_adjusted,adjusted_score,core_score,likely_adjusted,ml_probability,reliability,reliability_factor --smooth-radius 1
# py raster_models.py D:/Suitability/project/previews_post --columns blend_mix_adjusted,blend_gate_adjusted,blend_geo_adjusted,blend_mix_adjusted --smooth-radius 1
# py raster_models.py D:/Suitability/project --csv community_model/community_by_species.csv --group-col species --columns adjusted_score,blend_gate_adjusted,blend_geo_adjusted,blend_mix_adjusted

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception as exc:
    raise RuntimeError("rasterio is required for GeoTIFF export") from exc


DEFAULT_COLUMN = "blend_mix_adjusted"
DEFAULT_CRS = "EPSG:4326"
DEFAULT_OUTPUT_SUBDIR = "qgis_rasters"
EXCLUDED_DIR_NAMES = {DEFAULT_OUTPUT_SUBDIR, "__pycache__"}
AUTO_GROUP_COLS = ("species", "genus", "family")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "value"


def normalize_input_root(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.resolve()

    input_path = input_path.resolve()
    direct_csv = input_path / "overall_suitability.csv"
    if direct_csv.exists():
        return input_path

    if input_path.name.lower().startswith("previews"):
        parent = input_path.parent
        if (parent / "overall_suitability.csv").exists():
            return parent

    return input_path


def resolve_csv_path(root: Path, csv_arg: Optional[str]) -> Path:
    if csv_arg:
        candidate = Path(csv_arg)
        if not candidate.is_absolute():
            candidate = (root / csv_arg).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    return (root / "overall_suitability.csv").resolve()


def should_skip_dir(path: Path) -> bool:
    names = {part.casefold() for part in path.parts}
    return any(name.casefold() in names for name in EXCLUDED_DIR_NAMES)


def csv_discovery_sort_key(root: Path, path: Path) -> Tuple[int, int, str]:
    rel = path.relative_to(root)
    name = path.name.casefold()
    if name == "overall_suitability.csv":
        priority = 0
    elif name == "community_overall.csv":
        priority = 1
    elif name == "community_by_species.csv":
        priority = 2
    elif name == "community_by_genus.csv":
        priority = 3
    elif name == "community_by_family.csv":
        priority = 4
    else:
        priority = 20
    return (priority, len(rel.parts), str(rel).casefold())


def discover_csv_paths(root: Path, recursive: bool) -> List[Path]:
    root = root.resolve()
    seen = set()
    out: List[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved).casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(resolved)

    top_csv = root / "overall_suitability.csv"
    if top_csv.exists():
        add(top_csv)

    if recursive:
        for path in root.rglob("*.csv"):
            if should_skip_dir(path.parent):
                continue
            add(path)
    else:
        for path in root.glob("*.csv"):
            add(path)

    out.sort(key=lambda path: csv_discovery_sort_key(root, path))
    return out


def estimate_regular_step(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    diffs = np.diff(np.sort(np.unique(values)))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def box_filter_nan(grid: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    radius = max(0, int(radius))
    arr = np.asarray(grid, dtype=np.float32)
    if radius <= 0:
        counts = np.isfinite(arr).astype(np.float32)
        return arr.copy(), counts

    valid = np.isfinite(arr)
    vals = np.where(valid, arr, 0.0).astype(np.float32, copy=False)
    cnts = valid.astype(np.float32, copy=False)

    def integral_image(a: np.ndarray) -> np.ndarray:
        return np.pad(a, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    iv = integral_image(vals)
    ic = integral_image(cnts)

    h, w = arr.shape
    y = np.arange(h)
    x = np.arange(w)
    y0 = np.maximum(0, y - radius)
    y1 = np.minimum(h, y + radius + 1)
    x0 = np.maximum(0, x - radius)
    x1 = np.minimum(w, x + radius + 1)

    sums = iv[y1[:, None], x1[None, :]] - iv[y0[:, None], x1[None, :]] - iv[y1[:, None], x0[None, :]] + iv[y0[:, None], x0[None, :]]
    counts = ic[y1[:, None], x1[None, :]] - ic[y0[:, None], x1[None, :]] - ic[y1[:, None], x0[None, :]] + ic[y0[:, None], x0[None, :]]

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    good = counts > 0
    out[good] = (sums[good] / counts[good]).astype(np.float32)
    return out, counts.astype(np.float32, copy=False)


def fill_small_holes(grid: np.ndarray, radius: int, min_neighbors: int = 3, passes: int = 1) -> np.ndarray:
    out = np.asarray(grid, dtype=np.float32).copy()
    radius = max(0, int(radius))
    passes = max(0, int(passes))
    if radius <= 0 or passes <= 0:
        return out

    for _ in range(passes):
        local_mean, local_count = box_filter_nan(out, radius)
        missing = ~np.isfinite(out)
        fill_mask = missing & np.isfinite(local_mean) & (local_count >= float(min_neighbors))
        if not np.any(fill_mask):
            break
        out[fill_mask] = local_mean[fill_mask]
    return out


def smooth_grid(grid: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(grid, dtype=np.float32).copy()
    smoothed, _ = box_filter_nan(grid, radius)
    return smoothed


def sharpen_grid(grid: np.ndarray, blur_radius: int, amount: float, clamp_min: Optional[float] = None, clamp_max: Optional[float] = None) -> np.ndarray:
    amt = float(max(0.0, amount))
    if amt <= 0:
        out = np.asarray(grid, dtype=np.float32).copy()
    else:
        base = np.asarray(grid, dtype=np.float32)
        blur = smooth_grid(base, max(1, int(blur_radius)))
        out = np.where(np.isfinite(base), base + amt * (base - blur), np.nan).astype(np.float32)

    if clamp_min is not None or clamp_max is not None:
        mn = -np.inf if clamp_min is None else float(clamp_min)
        mx = np.inf if clamp_max is None else float(clamp_max)
        valid = np.isfinite(out)
        out[valid] = np.clip(out[valid], mn, mx)
    return out


def infer_regular_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    round_decimals: Sequence[int] = (8, 7, 6, 5, 4, 3),
):
    x_vals = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    z_vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
    if not np.any(ok):
        raise RuntimeError(f"No valid numeric rows found for {value_col}")

    x_vals = x_vals[ok]
    y_vals = y_vals[ok]
    z_vals = z_vals[ok]

    n = x_vals.size
    if n < 4:
        raise RuntimeError(f"Not enough rows to infer a raster grid for {value_col}")

    max_cells = min(max(n * 4, 4096), 8_000_000)
    best_payload = None

    for decimals in round_decimals:
        xr = np.round(x_vals, decimals)
        yr = np.round(y_vals, decimals)

        xs = np.sort(np.unique(xr))
        ys = np.sort(np.unique(yr))
        if xs.size < 2 or ys.size < 2:
            continue

        cell_count = int(xs.size) * int(ys.size)
        if cell_count <= 0 or cell_count > max_cells:
            continue

        dx = estimate_regular_step(xs)
        dy = estimate_regular_step(ys)
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
        payload = {
            "decimals": int(decimals),
            "xs": xs,
            "ys": ys,
            "dx": float(dx),
            "dy": float(dy),
            "grid": grid,
            "filled_fraction": filled_fraction,
            "cell_count": cell_count,
        }

        if best_payload is None or payload["filled_fraction"] > best_payload["filled_fraction"]:
            best_payload = payload

        if filled_fraction >= 0.95:
            return payload

    if best_payload is None or best_payload["filled_fraction"] < 0.20:
        raise RuntimeError(f"Could not infer a regular grid for {value_col}")

    return best_payload


def write_geotiff(
    out_path: Path,
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    dx: float,
    dy: float,
    crs: str,
    nodata_value: float,
) -> Dict[str, object]:
    xmin = float(xs.min()) - float(dx) * 0.5
    ymax = float(ys.max()) + float(dy) * 0.5
    transform = from_origin(xmin, ymax, float(dx), float(dy))

    write_grid = np.flipud(np.asarray(grid, dtype=np.float32))
    write_grid = np.where(np.isfinite(write_grid), write_grid, np.float32(nodata_value)).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(write_grid.shape[0]),
        "width": int(write_grid.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float(nodata_value),
        "compress": "DEFLATE",
        "predictor": 2,
        "tiled": True,
        "blockxsize": min(256, int(write_grid.shape[1])),
        "blockysize": min(256, int(write_grid.shape[0])),
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(write_grid, 1)

    return {
        "width": int(write_grid.shape[1]),
        "height": int(write_grid.shape[0]),
        "xmin": float(xs.min()) - float(dx) * 0.5,
        "xmax": float(xs.max()) + float(dx) * 0.5,
        "ymin": float(ys.min()) - float(dy) * 0.5,
        "ymax": float(ys.max()) + float(dy) * 0.5,
        "dx": float(dx),
        "dy": float(dy),
        "crs": crs,
        "nodata": float(nodata_value),
    }


def choose_columns(header: Sequence[str], requested: Sequence[str], default_column: str) -> List[str]:
    cols = [str(c) for c in header]
    if requested:
        chosen = [c for c in requested if c in cols]
        if not chosen:
            raise KeyError(f"None of the requested columns were found: {requested}")
        return chosen

    if default_column in cols:
        return [default_column]

    fallbacks = [
        "blend_gate_adjusted",
        "blend_geo_adjusted",
        "overall_adjusted",
        "adjusted_score",
        "overall_ml",
        "overall_ml_suitability",
        "community_top_score",
    ]
    for col in fallbacks:
        if col in cols:
            return [col]

    raise KeyError(f"Could not find a default export column in CSV. Available columns: {cols[:40]}")


def normalize_group_series(series: pd.Series) -> pd.Series:
    out = series.where(series.notna(), "")
    return out.map(lambda value: str(value).strip())


def choose_group_values(group_text: pd.Series, requested: Sequence[str]) -> List[str]:
    values = []
    seen = set()

    for value in group_text.tolist():
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)

    values.sort(key=lambda value: value.casefold())

    if not requested:
        return values

    requested_map = {value.casefold(): value for value in requested}
    chosen = [value for value in values if value.casefold() in requested_map]
    if not chosen:
        raise KeyError(f"None of the requested group values were found in the CSV: {requested}")
    return chosen


def infer_group_col(csv_path: Path, header: Sequence[str], explicit_group_col: str) -> Optional[str]:
    header_set = {str(col) for col in header}
    explicit_group_col = str(explicit_group_col or "").strip()
    if explicit_group_col:
        if explicit_group_col in header_set:
            return explicit_group_col
        return None

    text = str(csv_path).replace("\\", "/").casefold()
    for group_col in AUTO_GROUP_COLS:
        if f"by_{group_col}" in text and group_col in header_set:
            return group_col

    return None


def build_out_path(outdir: Path, csv_path: Path, column: str, output_prefix: str = "", group_value: Optional[str] = None) -> Path:
    parts = []
    if str(output_prefix).strip():
        parts.append(safe_slug(output_prefix))
    if group_value is None:
        parts.append(safe_slug(csv_path.stem))
    else:
        parts.append(safe_slug(group_value))
    parts.append(safe_slug(column))
    return outdir / f"{'_'.join(parts)}.tif"


def export_surface(
    frame: pd.DataFrame,
    csv_path: Path,
    x_col: str,
    y_col: str,
    column: str,
    out_path: Path,
    crs: str,
    nodata: float,
    fill_radius: int,
    fill_min_neighbors: int,
    fill_passes: int,
    smooth_radius: int,
    sharpen_amount: float,
    clamp_0_1: bool,
) -> Dict[str, object]:
    payload = infer_regular_grid(frame[[x_col, y_col, column]].copy(), x_col, y_col, column)
    grid = payload["grid"]

    if fill_radius > 0:
        grid = fill_small_holes(
            grid,
            radius=fill_radius,
            min_neighbors=fill_min_neighbors,
            passes=fill_passes,
        )

    if smooth_radius > 0:
        grid = smooth_grid(grid, radius=smooth_radius)

    if sharpen_amount > 0:
        clamp_min = 0.0 if clamp_0_1 else None
        clamp_max = 1.0 if clamp_0_1 else None
        grid = sharpen_grid(
            grid,
            blur_radius=max(1, smooth_radius if smooth_radius > 0 else 1),
            amount=sharpen_amount,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

    if clamp_0_1:
        valid = np.isfinite(grid)
        grid[valid] = np.clip(grid[valid], 0.0, 1.0)

    meta = write_geotiff(
        out_path=out_path,
        grid=grid,
        xs=payload["xs"],
        ys=payload["ys"],
        dx=payload["dx"],
        dy=payload["dy"],
        crs=crs,
        nodata_value=nodata,
    )

    return {
        "csv": str(csv_path),
        "column": column,
        "out_tif": str(out_path),
        "grid_round_decimals": int(payload["decimals"]),
        "filled_fraction_before_filters": float(payload["filled_fraction"]),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "dx": float(meta["dx"]),
        "dy": float(meta["dy"]),
        "xmin": float(meta["xmin"]),
        "xmax": float(meta["xmax"]),
        "ymin": float(meta["ymin"]),
        "ymax": float(meta["ymax"]),
        "crs": str(meta["crs"]),
        "nodata": float(meta["nodata"]),
        "fill_radius": int(fill_radius),
        "smooth_radius": int(smooth_radius),
        "sharpen_amount": float(sharpen_amount),
        "clamp_0_1": bool(clamp_0_1),
    }


def prepare_csv_job(
    root: Path,
    csv_path: Path,
    x_col: str,
    y_col: str,
    requested_columns: Sequence[str],
    default_column: str,
    explicit_group_col: str,
) -> Optional[Dict[str, object]]:
    try:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as exc:
        log(f"[skip] {csv_path.relative_to(root)} :: could not read header :: {exc}")
        return None

    if x_col not in header or y_col not in header:
        log(f"[skip] {csv_path.relative_to(root)} :: missing coordinate columns {x_col},{y_col}")
        return None

    try:
        columns = choose_columns(header, requested_columns, default_column)
    except Exception as exc:
        log(f"[skip] {csv_path.relative_to(root)} :: {exc}")
        return None

    group_col = infer_group_col(csv_path, header, explicit_group_col)
    if str(explicit_group_col).strip() and not group_col:
        log(f"[skip] {csv_path.relative_to(root)} :: missing requested group column {explicit_group_col}")
        return None

    usecols = [x_col, y_col, *columns]
    if group_col:
        usecols.append(group_col)
    usecols = [col for col in usecols if col in header]

    return {
        "csv_path": csv_path,
        "header": header,
        "columns": columns,
        "group_col": group_col,
        "usecols": usecols,
    }


def grouped_source_counts(jobs: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for job in jobs:
        group_col = str(job.get("group_col") or "")
        if not group_col:
            continue
        counts[group_col] = counts.get(group_col, 0) + 1
    return counts


def choose_outdir_for_job(root_outdir: Path, job: Dict[str, object], group_counts: Dict[str, int]) -> Path:
    group_col = str(job.get("group_col") or "")
    if not group_col:
        return root_outdir

    grouped_outdir = root_outdir / f"by_{safe_slug(group_col)}"
    if group_counts.get(group_col, 0) <= 1:
        return grouped_outdir
    return grouped_outdir / safe_slug(Path(job["csv_path"]).stem)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export suitability CSV surfaces to GeoTIFF for QGIS.")
    ap.add_argument("input_path", help="Output directory containing model CSVs, a previews_post directory, or a CSV path")
    ap.add_argument("--csv", default=None, help="CSV path relative to input directory or absolute. When set, only this CSV is processed")
    ap.add_argument("--column", default=DEFAULT_COLUMN, help="Single default column to export when --columns is not set")
    ap.add_argument("--columns", default="", help="Comma-separated list of columns to export")
    ap.add_argument("--x-col", default="lon", help="Grid x column")
    ap.add_argument("--y-col", default="lat", help="Grid y column")
    ap.add_argument("--group-col", default="", help="Optional grouping column such as species, genus, or family")
    ap.add_argument("--group-values", default="", help="Optional comma-separated subset of group values to export")
    ap.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Recursively scan for matching CSVs under the input directory")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only scan the top level input directory")
    ap.add_argument("--crs", default=DEFAULT_CRS, help="Output CRS. Use EPSG:4326 for lon/lat grids")
    ap.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    ap.add_argument("--output-prefix", default="")
    ap.add_argument("--fill-radius", type=int, default=0, help="Fill tiny internal gaps before export")
    ap.add_argument("--fill-min-neighbors", type=int, default=3)
    ap.add_argument("--fill-passes", type=int, default=1)
    ap.add_argument("--smooth-radius", type=int, default=0, help="NaN-aware smoothing radius in grid cells")
    ap.add_argument("--sharpen-amount", type=float, default=0.0, help="Unsharp-mask amount after smoothing")
    ap.add_argument("--clamp-0-1", action="store_true", help="Clamp valid output values to [0, 1]")
    ap.add_argument("--nodata", type=float, default=-9999.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    input_path = normalize_input_root(Path(args.input_path))
    if input_path.is_file():
        root = input_path.parent
        csv_paths = [input_path]
    else:
        root = input_path
        if args.csv:
            csv_paths = [resolve_csv_path(root, args.csv)]
        else:
            csv_paths = discover_csv_paths(root, recursive=bool(args.recursive))

    csv_paths = [path for path in csv_paths if path.exists()]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {input_path}")

    requested_columns = parse_csv_list(args.columns)

    jobs: List[Dict[str, object]] = []
    for csv_path in csv_paths:
        job = prepare_csv_job(
            root=root,
            csv_path=csv_path,
            x_col=args.x_col,
            y_col=args.y_col,
            requested_columns=requested_columns,
            default_column=args.column,
            explicit_group_col=args.group_col,
        )
        if job is not None:
            jobs.append(job)

    if not jobs:
        raise RuntimeError("No matching CSVs were found with the requested coordinate and value columns")

    outdir = root / args.output_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    group_counts = grouped_source_counts(jobs)
    manifest_rows: List[Dict[str, object]] = []

    log(f"[scan] found {len(jobs)} matching CSV file(s)")

    for job in jobs:
        csv_path = Path(job["csv_path"])
        rel_csv = csv_path.relative_to(root)
        usecols = list(job["usecols"])
        columns = list(job["columns"])
        group_col = str(job.get("group_col") or "")

        log(f"[csv] {rel_csv}")
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        job_outdir = choose_outdir_for_job(outdir, job, group_counts)
        job_outdir.mkdir(parents=True, exist_ok=True)

        if not group_col:
            for col in columns:
                if col not in df.columns:
                    log(f"[skip] {rel_csv} :: missing column {col}")
                    continue

                try:
                    log(f"[grid] {rel_csv} :: {col}")
                    out_path = build_out_path(
                        outdir=job_outdir,
                        csv_path=csv_path,
                        column=col,
                        output_prefix=args.output_prefix,
                    )
                    row = export_surface(
                        frame=df,
                        csv_path=csv_path,
                        x_col=args.x_col,
                        y_col=args.y_col,
                        column=col,
                        out_path=out_path,
                        crs=args.crs,
                        nodata=float(args.nodata),
                        fill_radius=int(args.fill_radius),
                        fill_min_neighbors=int(args.fill_min_neighbors),
                        fill_passes=int(args.fill_passes),
                        smooth_radius=int(args.smooth_radius),
                        sharpen_amount=float(args.sharpen_amount),
                        clamp_0_1=bool(args.clamp_0_1),
                    )
                    row["relative_csv"] = str(rel_csv)
                    manifest_rows.append(row)
                    log(
                        f"[write] {rel_csv} :: {col} -> {out_path.relative_to(root)} "
                        f"size={row['width']}x{row['height']} "
                        f"cell={row['dx']:.10g}x{row['dy']:.10g} "
                        f"filled={row['filled_fraction_before_filters']:.3f}"
                    )
                except Exception as exc:
                    log(f"[skip] {rel_csv} :: {col} :: {exc}")
        else:
            group_text = normalize_group_series(df[group_col])
            group_keys = group_text.map(lambda value: value.casefold())
            try:
                group_values = choose_group_values(group_text, parse_csv_list(args.group_values))
            except Exception as exc:
                log(f"[skip] {rel_csv} :: {exc}")
                continue

            for group_value in group_values:
                group_key = group_value.casefold()
                subset = df.loc[group_keys == group_key].copy()
                if subset.empty:
                    continue

                for col in columns:
                    if col not in subset.columns:
                        log(f"[skip] {rel_csv} :: {group_value} :: missing column {col}")
                        continue

                    try:
                        log(f"[grid] {rel_csv} :: {group_value} :: {col}")
                        out_path = build_out_path(
                            outdir=job_outdir,
                            csv_path=csv_path,
                            column=col,
                            output_prefix=args.output_prefix,
                            group_value=group_value,
                        )
                        row = export_surface(
                            frame=subset,
                            csv_path=csv_path,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            column=col,
                            out_path=out_path,
                            crs=args.crs,
                            nodata=float(args.nodata),
                            fill_radius=int(args.fill_radius),
                            fill_min_neighbors=int(args.fill_min_neighbors),
                            fill_passes=int(args.fill_passes),
                            smooth_radius=int(args.smooth_radius),
                            sharpen_amount=float(args.sharpen_amount),
                            clamp_0_1=bool(args.clamp_0_1),
                        )
                        row["relative_csv"] = str(rel_csv)
                        row["group_col"] = group_col
                        row["group_value"] = group_value
                        manifest_rows.append(row)
                        log(
                            f"[write] {rel_csv} :: {group_value} :: {col} -> {out_path.relative_to(root)} "
                            f"size={row['width']}x{row['height']} "
                            f"cell={row['dx']:.10g}x{row['dy']:.10g} "
                            f"filled={row['filled_fraction_before_filters']:.3f}"
                        )
                    except Exception as exc:
                        log(f"[skip] {rel_csv} :: {group_value} :: {col} :: {exc}")

    if not manifest_rows:
        raise RuntimeError("No rasters were written")

    manifest_path = outdir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    log(f"[done] wrote {len(manifest_rows)} GeoTIFF(s) to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
