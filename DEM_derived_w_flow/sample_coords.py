#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform

# python sample_coords.py --coords coords.csv --index "D:\DEM_derived_w_flow\dem_flow_index.json" --layers all --out sampled.csv
# python sample_coords.py --coords coords.csv --index "D:\DEM_derived_w_flow\dem_flow_index.json" --layers all --out sampled.csv --fallback-radius-pixels 1

# expected format:
# id,lon,lat
# p0,-122.6765,45.5231
# p1,151.2093,-33.8688

CONT_CODES = ["af", "as", "au", "eu", "na", "sa"]
BIG_LAYERS = ["dem", "flowacc", "flowdir"]
DERIVED_LAYERS = ["slope_deg", "aspect_deg", "northness", "eastness"]
ALL_LAYERS = BIG_LAYERS + DERIVED_LAYERS


@dataclass(frozen=True)
class Point:
    pid: str
    lon: float
    lat: float


@dataclass(frozen=True)
class PlanItem:
    point_idx: int
    row: int
    col: int


@dataclass(frozen=True)
class PlanWindow:
    row_off: int
    col_off: int
    width: int
    height: int
    items: Tuple[PlanItem, ...]


@dataclass(frozen=True)
class SamplePlan:
    windows: Tuple[PlanWindow, ...]


class DatasetCache:
    def __init__(self):
        self._map: Dict[str, rasterio.DatasetReader] = {}

    def open(self, path: str):
        ds = self._map.get(path)
        if ds is None:
            ds = rasterio.open(path)
            self._map[path] = ds
        return ds

    def close_all(self):
        for ds in self._map.values():
            try:
                ds.close()
            except Exception:
                pass
        self._map.clear()


def norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def read_points_csv(path: str) -> List[Point]:
    f = os.sys.stdin if path == "-" else open(path, "r", newline="", encoding="utf-8")
    with f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("coords CSV must have header row like id,lon,lat")
        cols = {norm(c): c for c in reader.fieldnames}
        id_col = cols.get("id") or cols.get("pid") or cols.get("pointid") or cols.get("name")
        lon_col = cols.get("lon") or cols.get("longitude") or cols.get("x")
        lat_col = cols.get("lat") or cols.get("latitude") or cols.get("y")
        if not lon_col or not lat_col:
            raise SystemExit(f"Could not find lon/lat columns in headers: {reader.fieldnames}")
        pts: List[Point] = []
        for i, row in enumerate(reader):
            pid = str(row.get(id_col, i)).strip() if id_col else str(i)
            try:
                lon = float(row[lon_col])
                lat = float(row[lat_col])
            except Exception:
                continue
            if math.isfinite(lon) and math.isfinite(lat):
                pts.append(Point(pid=pid, lon=lon, lat=lat))
        if not pts:
            raise SystemExit("No valid coordinates parsed")
        return pts


def in_bounds(lon: float, lat: float, b: Sequence[float]) -> bool:
    return lon >= b[0] and lon <= b[2] and lat >= b[1] and lat <= b[3]


def find_continent(index: dict, lon: float, lat: float) -> Optional[str]:
    for cc in CONT_CODES:
        meta = index["continents"].get(cc, {}).get("big", {}).get("dem")
        if meta and in_bounds(lon, lat, meta["bounds_wgs84"]):
            return cc
    for cc in CONT_CODES:
        big = index["continents"].get(cc, {}).get("big", {})
        for layer in ("flowacc", "flowdir"):
            meta = big.get(layer)
            if meta and in_bounds(lon, lat, meta["bounds_wgs84"]):
                return cc
    return None


def tile_key(lon: float, lat: float, tile_deg: float, key_scale: int) -> str:
    x0 = math.floor(lon / tile_deg) * tile_deg
    y0 = math.floor(lat / tile_deg) * tile_deg
    kx = int(round(x0 * key_scale))
    ky = int(round(y0 * key_scale))
    return f"x{kx}_y{ky}"


def get_block_shape(ds: rasterio.DatasetReader) -> Tuple[int, int]:
    try:
        if ds.block_shapes and ds.block_shapes[0]:
            block_h, block_w = ds.block_shapes[0]
            if block_h > 0 and block_w > 0:
                return int(block_h), int(block_w)
    except Exception:
        pass
    return 512, 512


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        af = float(a)
        bf = float(b)
        if math.isnan(af) and math.isnan(bf):
            return True
        return af == bf
    except Exception:
        return a == b


def is_missing_value(value, nodata) -> bool:
    try:
        fv = float(value)
    except Exception:
        return False

    if math.isnan(fv):
        return True

    if nodata is None:
        return False

    return values_equal(fv, nodata)


def neighbor_offsets(radius: int) -> List[Tuple[int, int]]:
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            dist2 = dr * dr + dc * dc
            offsets.append((dist2, abs(dr) + abs(dc), dr, dc))
    offsets.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [(dr, dc) for _, _, dr, dc in offsets]


def sample_with_fallback(
    arr,
    r: int,
    c: int,
    row_off: int,
    col_off: int,
    ds_height: int,
    ds_width: int,
    nodata,
    offsets: Sequence[Tuple[int, int]],
) -> Optional[float]:
    local_r = r - row_off
    local_c = c - col_off
    value = arr[local_r, local_c]
    if not is_missing_value(value, nodata):
        return float(value)

    for dr, dc in offsets:
        rr = r + dr
        cc = c + dc
        if rr < 0 or cc < 0 or rr >= ds_height or cc >= ds_width:
            continue
        value = arr[rr - row_off, cc - col_off]
        if not is_missing_value(value, nodata):
            return float(value)

    return None


def dataset_layout_signature(
    ds: rasterio.DatasetReader,
    fallback_radius_pixels: int,
    superblock_size: int,
) -> Tuple[object, ...]:
    block_h, block_w = get_block_shape(ds)
    transform = tuple(float(x) for x in ds.transform)
    crs = "" if ds.crs is None else ds.crs.to_string()
    return (
        crs,
        ds.width,
        ds.height,
        transform,
        block_h,
        block_w,
        int(fallback_radius_pixels),
        int(superblock_size),
    )


def build_sample_plan(
    ds: rasterio.DatasetReader,
    points: Sequence[Tuple[float, float]],
    fallback_radius_pixels: int = 1,
    superblock_size: int = 2,
) -> SamplePlan:
    if not points:
        return SamplePlan(windows=tuple())

    if ds.crs is None or str(ds.crs).upper() == "EPSG:4326":
        coords = list(points)
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        tx, ty = rio_transform("EPSG:4326", ds.crs, xs, ys)
        coords = list(zip(tx, ty))

    block_h, block_w = get_block_shape(ds)
    sb = max(1, int(superblock_size))
    pending = []

    for i, (x, y) in enumerate(coords):
        try:
            r, c = ds.index(x, y)
        except Exception:
            continue
        if r < 0 or c < 0 or r >= ds.height or c >= ds.width:
            continue
        block_row = r // block_h
        block_col = c // block_w
        pending.append((block_row // sb, block_col // sb, r, c, i))

    if not pending:
        return SamplePlan(windows=tuple())

    pending.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    windows: List[PlanWindow] = []

    for (superblock_row, superblock_col), group_iter in groupby(pending, key=lambda x: (x[0], x[1])):
        group_items = list(group_iter)
        base_row_off = superblock_row * sb * block_h
        base_col_off = superblock_col * sb * block_w

        row_off = max(0, base_row_off - fallback_radius_pixels)
        col_off = max(0, base_col_off - fallback_radius_pixels)
        row_end = min(ds.height, base_row_off + sb * block_h + fallback_radius_pixels)
        col_end = min(ds.width, base_col_off + sb * block_w + fallback_radius_pixels)

        items = tuple(
            PlanItem(point_idx=i, row=r, col=c)
            for _, _, r, c, i in group_items
        )
        windows.append(
            PlanWindow(
                row_off=row_off,
                col_off=col_off,
                width=col_end - col_off,
                height=row_end - row_off,
                items=items,
            )
        )

    return SamplePlan(windows=tuple(windows))


def sample_many_with_plan(
    ds: rasterio.DatasetReader,
    plan: SamplePlan,
    n_points: int,
    fallback_radius_pixels: int = 1,
) -> List[Optional[float]]:
    results: List[Optional[float]] = [None] * n_points
    if not plan.windows:
        return results

    offsets = neighbor_offsets(fallback_radius_pixels) if fallback_radius_pixels > 0 else []
    nodata = ds.nodata

    for pw in plan.windows:
        window = Window(pw.col_off, pw.row_off, pw.width, pw.height)
        arr = ds.read(1, window=window, masked=False)
        for item in pw.items:
            results[item.point_idx] = sample_with_fallback(
                arr=arr,
                r=item.row,
                c=item.col,
                row_off=pw.row_off,
                col_off=pw.col_off,
                ds_height=ds.height,
                ds_width=ds.width,
                nodata=nodata,
                offsets=offsets,
            )

    return results


def sample_many(
    ds: rasterio.DatasetReader,
    points: Sequence[Tuple[float, float]],
    fallback_radius_pixels: int = 1,
    superblock_size: int = 2,
) -> List[Optional[float]]:
    plan = build_sample_plan(
        ds,
        points,
        fallback_radius_pixels=fallback_radius_pixels,
        superblock_size=superblock_size,
    )
    return sample_many_with_plan(
        ds,
        plan,
        n_points=len(points),
        fallback_radius_pixels=fallback_radius_pixels,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast batched coordinate sampler for DEM, flowacc, flowdir, slope, aspect, northness, eastness")
    ap.add_argument("--coords", required=True, help="CSV with id,lon,lat or lon,lat")
    ap.add_argument("--index", required=True, help="JSON built by build_dem_flow_index.py")
    ap.add_argument("--layers", default="all", help=f"Comma list or 'all'. Options: {','.join(ALL_LAYERS)}")
    ap.add_argument("--out", default="-", help="Output CSV path or '-' for stdout")
    ap.add_argument("--emit-tile-key", action="store_true")
    ap.add_argument("--emit-paths", action="store_true", help="Include resolved raster path columns for debugging")
    ap.add_argument(
        "--fallback-radius-pixels",
        type=int,
        default=3,
        help="If the exact pixel is nodata, search nearest valid neighbors within this pixel radius",
    )
    ap.add_argument(
        "--superblock-size",
        type=int,
        default=2,
        help="Read groups of nearby blocks together as larger windows; 1 disables block merging",
    )
    ap.add_argument(
        "--gdal-cache-mb",
        type=int,
        default=512,
        help="GDAL block cache size in MB for this process",
    )
    args = ap.parse_args()

    if args.fallback_radius_pixels < 0:
        raise SystemExit("--fallback-radius-pixels must be >= 0")
    if args.superblock_size < 1:
        raise SystemExit("--superblock-size must be >= 1")
    if args.gdal_cache_mb < 1:
        raise SystemExit("--gdal-cache-mb must be >= 1")

    pts = read_points_csv(args.coords)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))

    if args.layers.strip().lower() == "all":
        layers = ALL_LAYERS[:]
    else:
        layers = [s.strip().lower() for s in args.layers.split(",") if s.strip()]
        bad = [x for x in layers if x not in ALL_LAYERS]
        if bad:
            raise SystemExit(f"Unknown layers: {bad}; valid: {ALL_LAYERS}")

    tile_deg = float(index["tile_deg"])
    key_scale = int(index["key_scale"])
    need_tiles = any(layer in DERIVED_LAYERS for layer in layers)

    results: Dict[str, List[Optional[float]]] = {layer: [None] * len(pts) for layer in layers}
    continents: List[str] = [""] * len(pts)
    tile_keys: List[str] = [""] * len(pts)
    path_cols: Dict[str, List[str]] = {layer: [""] * len(pts) for layer in layers} if args.emit_paths else {}

    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, p in enumerate(pts):
        cc = find_continent(index, p.lon, p.lat)
        if not cc:
            continue
        continents[i] = cc
        key = tile_key(p.lon, p.lat, tile_deg, key_scale) if need_tiles else ""
        tile_keys[i] = key
        groups[(cc, key)].append(i)

    cache = DatasetCache()

    env_kwargs = {
        "GDAL_CACHEMAX": int(args.gdal_cache_mb),
    }

    with rasterio.Env(**env_kwargs):
        for cc in CONT_CODES:
            big = index["continents"].get(cc, {}).get("big", {})
            for layer in BIG_LAYERS:
                meta = big.get(layer)
                if meta:
                    cache.open(meta["path"])

        total_groups = len(groups)
        sorted_group_items = sorted(groups.items(), key=lambda kv: kv[0])

        for gnum, ((cc, key), idxs) in enumerate(sorted_group_items, 1):
            coords = [(pts[i].lon, pts[i].lat) for i in idxs]
            cont = index["continents"].get(cc, {})
            tile_entry = cont.get("tiles", {}).get(key, {})
            plan_cache: Dict[Tuple[object, ...], SamplePlan] = {}

            for layer in layers:
                if layer in BIG_LAYERS:
                    meta = cont.get("big", {}).get(layer)
                    if not meta:
                        continue
                    path = meta["path"]
                else:
                    path = tile_entry.get(layer)
                    if not path:
                        continue

                ds = cache.open(path)
                sig = dataset_layout_signature(
                    ds,
                    fallback_radius_pixels=args.fallback_radius_pixels,
                    superblock_size=args.superblock_size,
                )
                plan = plan_cache.get(sig)
                if plan is None:
                    plan = build_sample_plan(
                        ds,
                        coords,
                        fallback_radius_pixels=args.fallback_radius_pixels,
                        superblock_size=args.superblock_size,
                    )
                    plan_cache[sig] = plan

                vals = sample_many_with_plan(
                    ds,
                    plan,
                    n_points=len(coords),
                    fallback_radius_pixels=args.fallback_radius_pixels,
                )
                for j, v in zip(idxs, vals):
                    results[layer][j] = v
                    if args.emit_paths:
                        path_cols[layer][j] = path

            if gnum % 500 == 0 or gnum == total_groups:
                print(f"[{gnum}/{total_groups}] groups processed", file=os.sys.stderr)

    cache.close_all()

    out_f = os.sys.stdout if args.out == "-" else open(args.out, "w", newline="", encoding="utf-8")
    with out_f:
        header = ["id", "lon", "lat", "continent"]
        if args.emit_tile_key:
            header.append("tile_key")
        header.extend(layers)
        if args.emit_paths:
            header.extend([f"{layer}_path" for layer in layers])
        w = csv.writer(out_f)
        w.writerow(header)
        for i, p in enumerate(pts):
            row = [p.pid, f"{p.lon:.8f}", f"{p.lat:.8f}", continents[i]]
            if args.emit_tile_key:
                row.append(tile_keys[i])
            for layer in layers:
                v = results[layer][i]
                row.append("" if v is None else f"{v:.6g}")
            if args.emit_paths:
                for layer in layers:
                    row.append(path_cols[layer][i])
            w.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
