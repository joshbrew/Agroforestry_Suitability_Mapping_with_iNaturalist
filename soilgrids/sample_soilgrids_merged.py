#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from affine import Affine
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


_WORKER_DS_CACHE = {}
_WORKER_ENV = None
_WORKER_TRANSFORMER_CACHE = {}
_WORKER_ROWCOL_CACHE = {}
_NEIGHBOR_OFFSET_CACHE = {}
_ROWCOL_CACHE_MAX_ITEMS = 500000


def parse_csv_list(value):
    if not value:
        return []
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out



def infer_stack_datasets_from_file(tif_path):
    tif_path = Path(tif_path)
    datasets = {}
    stem = tif_path.stem.strip()
    stem_l = stem.lower()

    if not stem_l.endswith("_depthstack"):
        return datasets

    with rasterio.open(tif_path) as ds:
        parent_name = tif_path.parent.name.strip().lower()
        base_name = stem[:-len("_depthstack")]
        prop = base_name.split("_", 1)[0].lower() if "_" in base_name else parent_name
        stat = base_name.split("_", 1)[1] if "_" in base_name else ""
        inferred_depths = ["0-5cm", "5-15cm", "15-30cm"]

        for band_index in range(1, ds.count + 1):
            desc = ds.descriptions[band_index - 1] if ds.descriptions else None
            tags = ds.tags(band_index)
            logical_name = (desc or tags.get("logical_name") or "").strip()

            if not logical_name:
                depth = (tags.get("depth") or "").strip()
                if not depth and band_index <= len(inferred_depths):
                    depth = inferred_depths[band_index - 1]
                if depth:
                    logical_name = f"{prop}_{depth}_{stat}" if stat else f"{prop}_{depth}"
                else:
                    logical_name = f"{stem}_band{band_index}"

            ds_prop, ds_depth, ds_stat = split_dataset_name(logical_name)
            datasets[logical_name] = {
                "name": logical_name,
                "path": tif_path.resolve(),
                "prop": ds_prop or prop,
                "stem": logical_name,
                "band_index": int(band_index),
                "source_kind": "depthstack",
                "depth": ds_depth,
                "stat": ds_stat,
                "stack_name": stem,
            }

    return datasets


def discover_merged_tifs(root, prefer_depth_stacks=True):
    root = Path(root)
    single_depth = {}
    stack_datasets = {}

    for tif in sorted(root.rglob("*.tif")):
        if not tif.is_file():
            continue
        if tif.name.lower().endswith(".part"):
            continue

        stem = tif.stem.strip()
        stem_l = stem.lower()

        if stem_l.endswith("_depthstack"):
            stack_datasets.update(infer_stack_datasets_from_file(tif))
            continue

        dataset_name = stem
        parent_name = tif.parent.name.strip().lower()
        prop_name = "wrb" if dataset_name.lower().startswith("wrb_") else parent_name
        ds_prop, ds_depth, ds_stat = split_dataset_name(dataset_name)

        single_depth[dataset_name] = {
            "name": dataset_name,
            "path": tif.resolve(),
            "prop": ds_prop or prop_name,
            "stem": stem,
            "band_index": 1,
            "source_kind": "single",
            "depth": ds_depth,
            "stat": ds_stat,
        }

    if not prefer_depth_stacks:
        merged = dict(single_depth)
        merged.update({name: meta for name, meta in stack_datasets.items() if name not in merged})
        return merged

    merged = dict(single_depth)
    merged.update(stack_datasets)
    return merged


def split_dataset_name(name):
    parts = str(name).split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], "_".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return name, "", ""


def is_wrb_meta(meta):
    return str(meta.get("prop", "")).lower() == "wrb"


def wrb_dataset_names(all_datasets):
    names = [name for name, meta in all_datasets.items() if is_wrb_meta(meta)]
    names.sort()
    return names


def resolve_explicit_dataset_names(all_datasets, explicit_datasets):
    if not explicit_datasets:
        return []

    if len(explicit_datasets) == 1 and explicit_datasets[0].lower() == "all":
        return sorted(all_datasets.keys())

    lower_name_map = {name.lower(): name for name in all_datasets.keys()}

    requested = []
    missing = []

    for token in explicit_datasets:
        token = token.strip()
        token_l = token.lower()

        matches = []

        if token_l == "wrb":
            matches = wrb_dataset_names(all_datasets)
        elif token in all_datasets:
            matches = [token]
        elif token_l in lower_name_map:
            matches = [lower_name_map[token_l]]
        else:
            for name, meta in all_datasets.items():
                if meta["stem"].lower() == token_l:
                    matches.append(name)

        if not matches:
            missing.append(token)
            continue

        requested.extend(sorted(matches))

    if missing:
        raise KeyError(f"Requested dataset(s) not found: {', '.join(missing)}")

    return sorted(dict.fromkeys(requested))


def build_requested_dataset_names(all_datasets, props, depths, stat, explicit_datasets):
    if explicit_datasets:
        return resolve_explicit_dataset_names(all_datasets, explicit_datasets)

    if not props and not depths:
        return sorted(all_datasets.keys())

    available_by_prop = defaultdict(set)

    for name, meta in all_datasets.items():
        if is_wrb_meta(meta):
            available_by_prop["wrb"].add((meta["stem"], ""))
            continue

        prop, depth, ds_stat = split_dataset_name(name)
        available_by_prop[prop].add((depth, ds_stat))

    if not props:
        props = sorted(available_by_prop.keys())

    requested = []

    for prop in props:
        prop_l = prop.strip().lower()

        if prop_l not in available_by_prop:
            raise KeyError(f"Property not found: {prop}")

        if prop_l == "wrb":
            requested.extend(wrb_dataset_names(all_datasets))
            continue

        if depths:
            for depth in depths:
                candidate = f"{prop_l}_{depth}_{stat}" if stat else f"{prop_l}_{depth}"
                if candidate in all_datasets:
                    requested.append(candidate)
        else:
            for depth, ds_stat in sorted(available_by_prop[prop_l]):
                if stat and ds_stat != stat:
                    continue
                if depth:
                    requested.append(f"{prop_l}_{depth}_{ds_stat}" if ds_stat else f"{prop_l}_{depth}")
                else:
                    requested.append(prop_l)

    requested = sorted(dict.fromkeys(requested))
    missing = [name for name in requested if name not in all_datasets]
    if missing:
        raise KeyError(f"Requested dataset(s) not found: {', '.join(missing)}")
    return requested


def auto_pick_column(fieldnames, preferred_names):
    lookup = {str(name).lower(): name for name in fieldnames}
    for pref in preferred_names:
        got = lookup.get(str(pref).lower())
        if got is not None:
            return got
    return None


def resolve_columns(fieldnames, lon_col=None, lat_col=None, id_col=None):
    if lon_col is None:
        lon_col = auto_pick_column(fieldnames, ["lon", "lng", "longitude", "x", "decimalLongitude"])
    if lat_col is None:
        lat_col = auto_pick_column(fieldnames, ["lat", "latitude", "y", "decimalLatitude"])

    if lon_col is None or lat_col is None:
        raise ValueError(
            f"Could not infer lon/lat columns from header: {fieldnames}. "
            f"Pass --lon-col and --lat-col explicitly."
        )

    if id_col is None:
        id_col = auto_pick_column(fieldnames, ["id", "row_id", "occurrenceID", "gbifID"])

    if id_col is not None and id_col not in fieldnames:
        raise ValueError(f"--id-col '{id_col}' not found in CSV header")

    passthrough_cols = [c for c in fieldnames if c not in {lon_col, lat_col}]
    return lon_col, lat_col, id_col, passthrough_cols


def chunked_csv_reader(
    path,
    lon_col=None,
    lat_col=None,
    id_col=None,
    chunk_size=100000,
    start_row=0,
    max_rows=None,
):
    start_row = max(0, int(start_row or 0))
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows < 0:
            raise ValueError("max_rows must be >= 0")

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row")

        fields = list(reader.fieldnames)
        lon_col, lat_col, id_col, passthrough_cols = resolve_columns(
            fields,
            lon_col=lon_col,
            lat_col=lat_col,
            id_col=id_col,
        )

        rows = []
        seen = 0
        emitted = 0

        for rec in reader:
            if seen < start_row:
                seen += 1
                continue

            if max_rows is not None and emitted >= max_rows:
                break

            rows.append(rec)
            seen += 1
            emitted += 1

            if len(rows) >= chunk_size:
                yield rows, lon_col, lat_col, id_col, passthrough_cols
                rows = []

        if rows:
            yield rows, lon_col, lat_col, id_col, passthrough_cols


def canonical_crs_text(crs):
    if crs is None:
        return ""
    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs.to_wkt()


def get_transformer(src_crs, dst_crs):
    src = rasterio.crs.CRS.from_user_input(src_crs)
    dst = rasterio.crs.CRS.from_user_input(dst_crs)
    if src == dst:
        return None
    return Transformer.from_crs(src, dst, always_xy=True)


def _cache_lookup_or_compute(unique_vals, cache_dict, compute_fn, max_items=_ROWCOL_CACHE_MAX_ITEMS):
    mapped = np.empty(unique_vals.shape[0], dtype=np.int64)
    missing_idx = []
    missing_vals = []

    for i, val in enumerate(unique_vals):
        key = float(val)
        got = cache_dict.get(key)
        if got is None:
            missing_idx.append(i)
            missing_vals.append(key)
        else:
            mapped[i] = got

    if missing_vals:
        computed = compute_fn(np.asarray(missing_vals, dtype=np.float64))
        for idx, key, value in zip(missing_idx, missing_vals, computed):
            mapped[idx] = value
        if len(cache_dict) < max_items:
            room = max_items - len(cache_dict)
            for key, value in zip(missing_vals[:room], computed[:room]):
                cache_dict[key] = int(value)

    return mapped


def fast_rowcol(transform, xs, ys, rowcol_cache=None):
    rows = np.full(xs.shape[0], -1, dtype=np.int64)
    cols = np.full(xs.shape[0], -1, dtype=np.int64)

    coord_ok = np.isfinite(xs) & np.isfinite(ys)
    ok_idx = np.flatnonzero(coord_ok)
    if ok_idx.size == 0:
        return rows, cols

    xv = xs[ok_idx]
    yv = ys[ok_idx]

    a = transform.a
    b = transform.b
    c = transform.c
    d = transform.d
    e = transform.e
    f = transform.f

    if abs(b) < 1e-15 and abs(d) < 1e-15:
        use_memo = rowcol_cache is not None and xv.size >= 8192
        if use_memo:
            unique_x, inverse_x = np.unique(xv, return_inverse=True)
            unique_y, inverse_y = np.unique(yv, return_inverse=True)
            if unique_x.size <= xv.size * 0.85 or unique_y.size <= yv.size * 0.85:
                x_cache = rowcol_cache.setdefault("x_to_col", {})
                y_cache = rowcol_cache.setdefault("y_to_row", {})
                unique_cols = _cache_lookup_or_compute(
                    unique_x,
                    x_cache,
                    lambda vals: np.floor((vals - c) / a).astype(np.int64),
                )
                unique_rows = _cache_lookup_or_compute(
                    unique_y,
                    y_cache,
                    lambda vals: np.floor((vals - f) / e).astype(np.int64),
                )
                cols[ok_idx] = unique_cols[inverse_x]
                rows[ok_idx] = unique_rows[inverse_y]
                return rows, cols

        cols[ok_idx] = np.floor((xv - c) / a).astype(np.int64)
        rows[ok_idx] = np.floor((yv - f) / e).astype(np.int64)
        return rows, cols

    inv = ~transform
    cols_f, rows_f = inv * (xv, yv)
    cols[ok_idx] = np.floor(cols_f).astype(np.int64)
    rows[ok_idx] = np.floor(rows_f).astype(np.int64)
    return rows, cols


def get_block_shape(ds):
    if getattr(ds, "block_shapes", None):
        shape = ds.block_shapes[0]
        if shape and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return 256, 256


def compute_block_groups(rows, cols, width, height, block_h, block_w):
    valid = (
        np.isfinite(rows)
        & np.isfinite(cols)
        & (rows >= 0)
        & (cols >= 0)
        & (rows < height)
        & (cols < width)
    )
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return valid_idx, None, None, None, None, None

    vrows = rows[valid_idx]
    vcols = cols[valid_idx]
    brow = vrows // block_h
    bcol = vcols // block_w
    n_block_cols = int(math.ceil(width / block_w))
    codes = brow * n_block_cols + bcol

    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    split_at = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate(([0], split_at))
    ends = np.concatenate((split_at, [len(order)]))
    return valid_idx, vrows, vcols, brow, bcol, (order, starts, ends)


def affine_to_tuple(transform):
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def tuple_to_affine(values):
    return Affine(*values)


def make_grid_signature(crs_text, width, height, block_h, block_w, transform_values):
    return (
        str(crs_text),
        int(width),
        int(height),
        int(block_h),
        int(block_w),
        tuple(transform_values),
    )


def build_sampling_plan(transform, xs, ys, width, height, block_h, block_w, rowcol_cache=None):
    rows, cols = fast_rowcol(transform, xs, ys, rowcol_cache=rowcol_cache)
    rows = rows.astype(np.int64, copy=False)
    cols = cols.astype(np.int64, copy=False)

    coord_ok = np.isfinite(xs) & np.isfinite(ys)
    in_bounds_mask = (
        coord_ok
        & (rows >= 0)
        & (cols >= 0)
        & (rows < int(height))
        & (cols < int(width))
    )

    valid_idx, vrows, vcols, brow, bcol, block_groups = compute_block_groups(
        rows,
        cols,
        width,
        height,
        block_h,
        block_w,
    )

    if valid_idx.size == 0:
        order = np.empty(0, dtype=np.int64)
        starts = np.empty(0, dtype=np.int64)
        ends = np.empty(0, dtype=np.int64)
        vrows = np.empty(0, dtype=np.int64)
        vcols = np.empty(0, dtype=np.int64)
        brow = np.empty(0, dtype=np.int64)
        bcol = np.empty(0, dtype=np.int64)
    else:
        order, starts, ends = block_groups

    return {
        "rows": rows,
        "cols": cols,
        "valid_idx": valid_idx,
        "vrows": vrows,
        "vcols": vcols,
        "brow": brow,
        "bcol": bcol,
        "order": order,
        "starts": starts,
        "ends": ends,
        "block_h": int(block_h),
        "block_w": int(block_w),
        "width": int(width),
        "height": int(height),
        "size": int(xs.shape[0]),
        "in_bounds_mask": in_bounds_mask,
    }


def record_dataset_error(error_state, dataset_name, ds, exc):
    if error_state is None:
        return

    key = str(dataset_name or getattr(ds, "name", "unknown"))
    st = error_state.get(key)
    if st is None:
        st = {
            "dataset": key,
            "path": str(getattr(ds, "name", "")),
            "count": 0,
            "first_error": "",
        }
        error_state[key] = st

    st["count"] += 1
    if not st["first_error"]:
        st["first_error"] = str(exc)
        print(
            f"WARNING: dataset read failed, filling NaN for unreadable blocks: {key} :: {exc}",
            file=sys.stderr,
        )


def merge_error_states(dst, src):
    if not src:
        return
    for key, st in src.items():
        cur = dst.get(key)
        if cur is None:
            dst[key] = dict(st)
        else:
            cur["count"] += int(st.get("count", 0))
            if not cur.get("first_error"):
                cur["first_error"] = st.get("first_error", "")
            if not cur.get("path"):
                cur["path"] = st.get("path", "")


def build_neighbor_offsets(radius):
    radius = int(radius)
    cached = _NEIGHBOR_OFFSET_CACHE.get(radius)
    if cached is not None:
        return cached
    if radius <= 0:
        _NEIGHBOR_OFFSET_CACHE[radius] = []
        return []

    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            dist2 = dr * dr + dc * dc
            offsets.append((dist2, abs(dr) + abs(dc), dr, dc))

    offsets.sort()
    out = [(dr, dc) for _, _, dr, dc in offsets]
    _NEIGHBOR_OFFSET_CACHE[radius] = out
    return out



def read_block_array(
    ds,
    r0,
    c0,
    h,
    w,
    dataset_name=None,
    on_read_error="nan",
    error_state=None,
    block_cache=None,
    band_index=1,
):
    cache_key = (int(band_index), int(r0), int(c0), int(h), int(w))

    if block_cache is not None and cache_key in block_cache:
        return block_cache[cache_key]

    try:
        arr = ds.read(int(band_index), window=Window(c0, r0, w, h), masked=False)
    except Exception as exc:
        if on_read_error == "raise":
            raise

        record_dataset_error(error_state, dataset_name, ds, exc)
        if block_cache is not None:
            block_cache[cache_key] = None
        return None

    if block_cache is not None:
        block_cache[cache_key] = arr
    return arr



def sample_dataset_blocked(
    ds,
    sampling_plan,
    dataset_name=None,
    nodata_to_nan=True,
    on_read_error="nan",
    error_state=None,
    fallback_radius_pixels=1,
    progress_label=None,
    progress_every_seconds=10.0,
    band_index=1,
):
    out = np.full(sampling_plan["size"], np.nan, dtype=np.float64)

    valid_idx = sampling_plan["valid_idx"]
    if valid_idx.size == 0:
        return out

    vrows = sampling_plan["vrows"]
    vcols = sampling_plan["vcols"]
    brow = sampling_plan["brow"]
    bcol = sampling_plan["bcol"]
    order = sampling_plan["order"]
    starts = sampling_plan["starts"]
    ends = sampling_plan["ends"]
    block_h = sampling_plan["block_h"]
    block_w = sampling_plan["block_w"]
    rows = sampling_plan["rows"]
    cols = sampling_plan["cols"]

    block_cache = {}
    nodata = None
    if int(band_index) <= len(ds.nodatavals):
        nodata = ds.nodatavals[int(band_index) - 1]
    if nodata is None:
        nodata = ds.nodata
    nodata_f = float(nodata) if nodata is not None else None

    block_progress_state = None
    total_block_groups = int(len(starts))
    if progress_label:
        block_progress_state = make_progress_state(
            phase_label=f"{progress_label} subphase=read-blocks",
            total=total_block_groups,
            unit_name="block-groups",
            report_every_seconds=progress_every_seconds,
        )

    for block_idx, (start, end) in enumerate(zip(starts, ends), start=1):
        local = order[start:end]
        first = local[0]

        r0 = int(brow[first] * block_h)
        c0 = int(bcol[first] * block_w)
        h = min(block_h, ds.height - r0)
        w = min(block_w, ds.width - c0)

        arr = read_block_array(
            ds=ds,
            r0=r0,
            c0=c0,
            h=h,
            w=w,
            dataset_name=dataset_name,
            on_read_error=on_read_error,
            error_state=error_state,
            block_cache=block_cache,
            band_index=band_index,
        )
        if arr is not None:
            rr = vrows[local] - r0
            cc = vcols[local] - c0
            vals = arr[rr, cc].astype(np.float64, copy=False)
            out[valid_idx[local]] = vals

        maybe_emit_progress(
            block_progress_state,
            block_idx,
            extra=f"dataset={dataset_name} band={band_index}",
            force=(block_idx == total_block_groups),
        )

    if nodata_to_nan and nodata is not None:
        out[out == nodata_f] = np.nan

    fallback_radius_pixels = int(fallback_radius_pixels)
    if fallback_radius_pixels > 0:
        neighbor_offsets = build_neighbor_offsets(fallback_radius_pixels)
        fallback_candidates = np.flatnonzero(np.isnan(out) & sampling_plan["in_bounds_mask"])
        fallback_progress_state = None
        total_fallback = int(fallback_candidates.shape[0])
        if progress_label and total_fallback > 0:
            fallback_progress_state = make_progress_state(
                phase_label=f"{progress_label} subphase=fallback-search",
                total=total_fallback,
                unit_name="rows",
                report_every_seconds=progress_every_seconds,
            )

        for fallback_idx, i in enumerate(fallback_candidates, start=1):
            base_r = int(rows[i])
            base_c = int(cols[i])
            found = None

            for dr, dc in neighbor_offsets:
                rr = base_r + dr
                cc = base_c + dc

                if rr < 0 or cc < 0 or rr >= ds.height or cc >= ds.width:
                    continue

                r0 = int((rr // block_h) * block_h)
                c0 = int((cc // block_w) * block_w)
                h = min(block_h, ds.height - r0)
                w = min(block_w, ds.width - c0)

                arr = read_block_array(
                    ds=ds,
                    r0=r0,
                    c0=c0,
                    h=h,
                    w=w,
                    dataset_name=dataset_name,
                    on_read_error=on_read_error,
                    error_state=error_state,
                    block_cache=block_cache,
                    band_index=band_index,
                )
                if arr is None:
                    continue

                val = float(arr[rr - r0, cc - c0])

                if nodata is not None and val == nodata_f:
                    continue
                if not np.isfinite(val):
                    continue

                found = val
                break

            if found is not None:
                out[i] = found

            maybe_emit_progress(
                fallback_progress_state,
                fallback_idx,
                extra=f"dataset={dataset_name} band={band_index} radius={fallback_radius_pixels}",
                force=(fallback_idx == total_fallback),
            )

    return out

    vrows = sampling_plan["vrows"]
    vcols = sampling_plan["vcols"]
    brow = sampling_plan["brow"]
    bcol = sampling_plan["bcol"]
    order = sampling_plan["order"]
    starts = sampling_plan["starts"]
    ends = sampling_plan["ends"]
    block_h = sampling_plan["block_h"]
    block_w = sampling_plan["block_w"]
    rows = sampling_plan["rows"]
    cols = sampling_plan["cols"]

    block_cache = {}
    nodata = ds.nodata
    nodata_f = float(nodata) if nodata is not None else None

    block_progress_state = None
    total_block_groups = int(len(starts))
    if progress_label:
        block_progress_state = make_progress_state(
            phase_label=f"{progress_label} subphase=read-blocks",
            total=total_block_groups,
            unit_name="block-groups",
            report_every_seconds=progress_every_seconds,
        )

    for block_idx, (start, end) in enumerate(zip(starts, ends), start=1):
        local = order[start:end]
        first = local[0]

        r0 = int(brow[first] * block_h)
        c0 = int(bcol[first] * block_w)
        h = min(block_h, ds.height - r0)
        w = min(block_w, ds.width - c0)

        arr = read_block_array(
            ds=ds,
            r0=r0,
            c0=c0,
            h=h,
            w=w,
            dataset_name=dataset_name,
            on_read_error=on_read_error,
            error_state=error_state,
            block_cache=block_cache,
        )
        if arr is not None:
            rr = vrows[local] - r0
            cc = vcols[local] - c0
            vals = arr[rr, cc].astype(np.float64, copy=False)
            out[valid_idx[local]] = vals

        maybe_emit_progress(
            block_progress_state,
            block_idx,
            extra=f"dataset={dataset_name}",
            force=(block_idx == total_block_groups),
        )

    if nodata_to_nan and nodata is not None:
        out[out == nodata_f] = np.nan

    fallback_radius_pixels = int(fallback_radius_pixels)
    if fallback_radius_pixels > 0:
        neighbor_offsets = build_neighbor_offsets(fallback_radius_pixels)
        fallback_candidates = np.flatnonzero(np.isnan(out) & sampling_plan["in_bounds_mask"])
        fallback_progress_state = None
        total_fallback = int(fallback_candidates.shape[0])
        if progress_label and total_fallback > 0:
            fallback_progress_state = make_progress_state(
                phase_label=f"{progress_label} subphase=fallback-search",
                total=total_fallback,
                unit_name="rows",
                report_every_seconds=progress_every_seconds,
            )

        for fallback_idx, i in enumerate(fallback_candidates, start=1):
            base_r = int(rows[i])
            base_c = int(cols[i])
            found = None

            for dr, dc in neighbor_offsets:
                rr = base_r + dr
                cc = base_c + dc

                if rr < 0 or cc < 0 or rr >= ds.height or cc >= ds.width:
                    continue

                r0 = int((rr // block_h) * block_h)
                c0 = int((cc // block_w) * block_w)
                h = min(block_h, ds.height - r0)
                w = min(block_w, ds.width - c0)

                arr = read_block_array(
                    ds=ds,
                    r0=r0,
                    c0=c0,
                    h=h,
                    w=w,
                    dataset_name=dataset_name,
                    on_read_error=on_read_error,
                    error_state=error_state,
                    block_cache=block_cache,
                )
                if arr is None:
                    continue

                val = float(arr[rr - r0, cc - c0])

                if nodata is not None and val == nodata_f:
                    continue
                if not np.isfinite(val):
                    continue

                found = val
                break

            if found is not None:
                out[i] = found

            maybe_emit_progress(
                fallback_progress_state,
                fallback_idx,
                extra=f"dataset={dataset_name} radius={fallback_radius_pixels}",
                force=(fallback_idx == total_fallback),
            )

    return out


def parse_chunk_rows(rows, lon_col, lat_col):
    n = len(rows)
    lon = np.full(n, np.nan, dtype=np.float64)
    lat = np.full(n, np.nan, dtype=np.float64)
    valid = np.ones(n, dtype=bool)

    for i, rec in enumerate(rows):
        try:
            lon[i] = float(rec[lon_col])
            lat[i] = float(rec[lat_col])
        except Exception:
            valid[i] = False

    valid &= np.isfinite(lon) & np.isfinite(lat)
    valid &= (lon >= -180.0) & (lon <= 180.0) & (lat >= -90.0) & (lat <= 90.0)
    return lon, lat, valid


def format_cell(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    return str(value)


def split_into_batches(items, n_batches):
    if not items:
        return []
    n_batches = max(1, min(int(n_batches), len(items)))
    out = [[] for _ in range(n_batches)]
    for i, item in enumerate(items):
        out[i % n_batches].append(item)
    return [batch for batch in out if batch]


def ensure_worker_env(gdal_cache_mb):
    global _WORKER_ENV
    if _WORKER_ENV is None:
        _WORKER_ENV = rasterio.Env(GDAL_CACHEMAX=int(gdal_cache_mb))
        _WORKER_ENV.__enter__()


def get_cached_dataset(cache, dataset_path):
    key = str(Path(dataset_path).resolve()).lower()
    ds = cache.get(key)
    if ds is None:
        ds = rasterio.open(dataset_path)
        cache[key] = ds
    return ds


def close_cached_datasets(cache):
    for ds in cache.values():
        try:
            ds.close()
        except Exception:
            pass
    cache.clear()


def get_worker_dataset(dataset_path, gdal_cache_mb):
    ensure_worker_env(gdal_cache_mb)
    key = str(Path(dataset_path).resolve()).lower()
    ds = _WORKER_DS_CACHE.get(key)
    if ds is None:
        ds = rasterio.open(dataset_path)
        _WORKER_DS_CACHE[key] = ds
    return ds


def get_worker_transformer(src_crs_text, dst_crs_text):
    key = (str(src_crs_text), str(dst_crs_text))
    tr = _WORKER_TRANSFORMER_CACHE.get(key)
    if tr is None:
        tr = get_transformer(src_crs_text, dst_crs_text)
        _WORKER_TRANSFORMER_CACHE[key] = tr
    return tr


def get_worker_rowcol_cache(dataset_signature):
    cache = _WORKER_ROWCOL_CACHE.get(dataset_signature)
    if cache is None:
        cache = {}
        _WORKER_ROWCOL_CACHE[dataset_signature] = cache
    return cache


def transform_coords_array(lon, lat, valid_mask, transformer):
    xs = np.full(lon.shape[0], np.nan, dtype=np.float64)
    ys = np.full(lat.shape[0], np.nan, dtype=np.float64)

    ok_idx = np.flatnonzero(valid_mask)
    if ok_idx.size == 0:
        return xs, ys

    if transformer is None:
        xs[ok_idx] = lon[ok_idx]
        ys[ok_idx] = lat[ok_idx]
        return xs, ys

    tx, ty = transformer.transform(lon[ok_idx], lat[ok_idx])
    xs[ok_idx] = np.asarray(tx, dtype=np.float64)
    ys[ok_idx] = np.asarray(ty, dtype=np.float64)
    return xs, ys


def inspect_dataset_groups(dataset_specs):
    groups = defaultdict(list)
    for spec in dataset_specs:
        groups[spec["crs_text"]].append(spec)
    return groups


def inspect_sampling_groups(dataset_specs):
    groups = defaultdict(list)
    for spec in dataset_specs:
        groups[spec["grid_signature"]].append(spec)
    return groups


def sample_dataset_batch_worker(
    dataset_batch,
    input_crs_text,
    lon,
    lat,
    valid_coord,
    gdal_cache_mb,
    on_read_error,
    fallback_radius_pixels,
    progress_label=None,
    progress_every_seconds=10.0,
):
    results = {}
    error_state = {}
    worker_pid = os.getpid()

    crs_groups = inspect_dataset_groups(dataset_batch)
    sampling_groups = inspect_sampling_groups(dataset_batch)

    if progress_label:
        print(
            f"[progress] {progress_label} worker={worker_pid} batch-start datasets={len(dataset_batch):,} rows={lon.shape[0]:,}",
            file=sys.stderr,
            flush=True,
        )

    coords_by_crs = {}
    for crs_text in crs_groups.keys():
        transformer = get_worker_transformer(input_crs_text, crs_text)
        coords_by_crs[crs_text] = transform_coords_array(lon, lat, valid_coord, transformer)

    plans_by_grid = {}
    for grid_signature, specs in sampling_groups.items():
        spec0 = specs[0]
        xs, ys = coords_by_crs[spec0["crs_text"]]
        rowcol_cache = get_worker_rowcol_cache(grid_signature)
        plans_by_grid[grid_signature] = build_sampling_plan(
            transform=tuple_to_affine(spec0["transform_values"]),
            xs=xs,
            ys=ys,
            width=spec0["width"],
            height=spec0["height"],
            block_h=spec0["block_h"],
            block_w=spec0["block_w"],
            rowcol_cache=rowcol_cache,
        )

    for spec_idx, spec in enumerate(dataset_batch, start=1):
        plan = plans_by_grid[spec["grid_signature"]]
        dataset_progress_label = None
        dataset_started = time.time()
        if progress_label:
            dataset_progress_label = (
                f"{progress_label} worker={worker_pid} dataset={spec_idx}/{len(dataset_batch)} name={spec['name']}"
            )
            print(
                f"[progress] {dataset_progress_label} start rows={lon.shape[0]:,} "
                f"in_bounds={int(np.count_nonzero(plan['in_bounds_mask'])):,} "
                f"block_groups={len(plan['starts']):,}",
                file=sys.stderr,
                flush=True,
            )

        ds = get_worker_dataset(spec["path"], gdal_cache_mb)
        ds_crs_text = canonical_crs_text(ds.crs)
        if ds_crs_text != spec["crs_text"]:
            raise RuntimeError(
                f"Dataset CRS mismatch for {spec['name']}: expected {spec['crs_text']}, got {ds_crs_text}"
            )

        results[spec["name"]] = sample_dataset_blocked(
            ds=ds,
            sampling_plan=plan,
            dataset_name=spec["name"],
            on_read_error=on_read_error,
            error_state=error_state,
            fallback_radius_pixels=fallback_radius_pixels,
            progress_label=dataset_progress_label,
            progress_every_seconds=progress_every_seconds,
        )

        if progress_label:
            print(
                f"[progress] {dataset_progress_label} done elapsed={format_elapsed(time.time() - dataset_started)}",
                file=sys.stderr,
                flush=True,
            )

    if progress_label:
        print(
            f"[progress] {progress_label} worker={worker_pid} batch-done datasets={len(dataset_batch):,}",
            file=sys.stderr,
            flush=True,
        )

    return results, error_state


def format_result_column(values):
    return [format_cell(v) for v in values]


def build_output_fields(id_col, lon_col, lat_col, passthrough_cols, requested_names, add_diagnostics):
    output_fields = []
    if id_col:
        output_fields.append(id_col)
    output_fields.extend([lon_col, lat_col])
    output_fields.extend([c for c in passthrough_cols if c != id_col])
    if add_diagnostics:
        output_fields.extend([
            "soilgrids_input_ok",
            "soilgrids_any_hit",
            "soilgrids_crs_group_count",
        ])
    output_fields.extend(requested_names)
    return output_fields


def read_existing_header(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader, None)


def open_output_writer(out_path, output_fields, append=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_header = read_existing_header(out_path)
    should_write_header = True
    mode = "w"

    if append and existing_header is not None:
        if list(existing_header) != list(output_fields):
            raise ValueError(
                "Existing output header does not match this run. "
                f"Expected {output_fields} but found {existing_header}"
            )
        mode = "a"
        should_write_header = False
    elif append:
        mode = "a"
        should_write_header = existing_header is None

    out_f = open(out_path, mode, newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    if should_write_header:
        writer.writerow(output_fields)
    return out_f, writer, should_write_header


def resolve_row_window(args):
    start_row = max(0, int(args.start_row or 0))
    max_rows = None if args.max_rows in (None, 0) else int(args.max_rows)

    if args.job_size is not None:
        if int(args.job_size) <= 0:
            raise ValueError("--job-size must be > 0")
        if int(args.job_index) < 0:
            raise ValueError("--job-index must be >= 0")
        if max_rows is not None:
            raise ValueError("--max-rows cannot be combined with --job-size")

        start_row += int(args.job_index) * int(args.job_size)
        max_rows = int(args.job_size)

    return start_row, max_rows


def write_output_chunk(
    writer,
    rows,
    id_col,
    lon_col,
    lat_col,
    passthrough_cols,
    requested_names,
    formatted_dataset_cols,
    add_diagnostics,
    valid_coord,
    hit_mask,
    crs_group_count,
):
    out_rows = []
    passthrough_out = [c for c in passthrough_cols if c != id_col]

    for i, rec in enumerate(rows):
        out_row = []
        if id_col:
            out_row.append(rec.get(id_col, ""))
        out_row.append(rec.get(lon_col, ""))
        out_row.append(rec.get(lat_col, ""))
        for c in passthrough_out:
            out_row.append(rec.get(c, ""))
        if add_diagnostics:
            out_row.extend([
                "1" if bool(valid_coord[i]) else "0",
                "1" if bool(hit_mask[i]) else "0",
                str(crs_group_count),
            ])
        for name in requested_names:
            out_row.append(formatted_dataset_cols[name][i])
        out_rows.append(out_row)

    writer.writerows(out_rows)


def build_chunk_sampling_context(
    crs_groups,
    sampling_groups,
    input_crs_text,
    lon,
    lat,
    valid_coord,
    transformer_cache,
    rowcol_cache_store,
):
    coords_by_crs = {}
    for crs_text in crs_groups.keys():
        transformer = transformer_cache.get(crs_text)
        if transformer is None and crs_text not in transformer_cache:
            transformer = get_transformer(input_crs_text, crs_text)
            transformer_cache[crs_text] = transformer
        else:
            transformer = transformer_cache[crs_text]
        coords_by_crs[crs_text] = transform_coords_array(lon, lat, valid_coord, transformer)

    plans_by_grid = {}
    any_in_bounds = np.zeros(lon.shape[0], dtype=bool)
    for grid_signature, specs in sampling_groups.items():
        spec0 = specs[0]
        xs, ys = coords_by_crs[spec0["crs_text"]]
        rowcol_cache = rowcol_cache_store.setdefault(grid_signature, {})
        plan = build_sampling_plan(
            transform=tuple_to_affine(spec0["transform_values"]),
            xs=xs,
            ys=ys,
            width=spec0["width"],
            height=spec0["height"],
            block_h=spec0["block_h"],
            block_w=spec0["block_w"],
            rowcol_cache=rowcol_cache,
        )
        plans_by_grid[grid_signature] = plan
        any_in_bounds |= plan["in_bounds_mask"]

    return plans_by_grid, any_in_bounds


def sample_chunk_results(
    dataset_specs,
    crs_groups,
    sampling_groups,
    dataset_batches,
    input_crs_text,
    lon,
    lat,
    valid_coord,
    gdal_cache_mb,
    on_read_error,
    fallback_radius_pixels,
    executor,
    error_state,
    local_ds_cache,
    local_transformer_cache,
    local_rowcol_cache,
    progress_label=None,
    progress_every_datasets=8,
    progress_every_seconds=10.0,
):
    n = int(lon.shape[0])
    results = {
        spec["name"]: np.full(n, np.nan, dtype=np.float64)
        for spec in dataset_specs
    }

    if progress_label:
        print(
            f"[progress] {progress_label} planning-start rows={n:,} valid_coords={int(np.count_nonzero(valid_coord)):,} "
            f"crs_groups={len(crs_groups):,} sampling_groups={len(sampling_groups):,}",
            file=sys.stderr,
            flush=True,
        )

    plans_by_grid, any_in_bounds = build_chunk_sampling_context(
        crs_groups=crs_groups,
        sampling_groups=sampling_groups,
        input_crs_text=input_crs_text,
        lon=lon,
        lat=lat,
        valid_coord=valid_coord,
        transformer_cache=local_transformer_cache,
        rowcol_cache_store=local_rowcol_cache,
    )

    if progress_label:
        print(
            f"[progress] {progress_label} planning-done in_bounds={int(np.count_nonzero(any_in_bounds)):,}",
            file=sys.stderr,
            flush=True,
        )

    if executor is None:
        phase_started = time.time()
        last_report_time = phase_started
        last_report_completed = 0
        total_specs = len(dataset_specs)

        for spec_idx, spec in enumerate(dataset_specs, start=1):
            plan = plans_by_grid[spec["grid_signature"]]
            dataset_progress_label = None
            dataset_started = time.time()
            if progress_label:
                dataset_progress_label = f"{progress_label} dataset={spec_idx}/{total_specs} name={spec['name']}"
                print(
                    f"[progress] {dataset_progress_label} start rows={n:,} "
                    f"in_bounds={int(np.count_nonzero(plan['in_bounds_mask'])):,} "
                    f"block_groups={len(plan['starts']):,}",
                    file=sys.stderr,
                    flush=True,
                )

            ds = get_cached_dataset(local_ds_cache, spec["path"])
            ds_crs_text = canonical_crs_text(ds.crs)
            if ds_crs_text != spec["crs_text"]:
                raise RuntimeError(
                    f"Dataset CRS mismatch for {spec['name']}: expected {spec['crs_text']}, got {ds_crs_text}"
                )

            results[spec["name"]] = sample_dataset_blocked(
                ds=ds,
                sampling_plan=plan,
                dataset_name=spec["name"],
                on_read_error=on_read_error,
                error_state=error_state,
                fallback_radius_pixels=fallback_radius_pixels,
                progress_label=dataset_progress_label,
                progress_every_seconds=progress_every_seconds,
            )

            if progress_label:
                print(
                    f"[progress] {dataset_progress_label} done elapsed={format_elapsed(time.time() - dataset_started)}",
                    file=sys.stderr,
                    flush=True,
                )

            now = time.time()
            if progress_label and (
                spec_idx == total_specs
                or (spec_idx - last_report_completed) >= max(1, int(progress_every_datasets))
                or (now - last_report_time) >= max(1.0, float(progress_every_seconds))
            ):
                emit_progress_line(
                    phase_label=progress_label,
                    completed=spec_idx,
                    total=total_specs,
                    started_at=phase_started,
                    unit_name="datasets",
                    extra=f"current={spec['name']}",
                )
                last_report_time = now
                last_report_completed = spec_idx

        return results, any_in_bounds

    future_map = {
        executor.submit(
            sample_dataset_batch_worker,
            batch,
            input_crs_text,
            lon,
            lat,
            valid_coord,
            gdal_cache_mb,
            on_read_error,
            fallback_radius_pixels,
            progress_label,
            progress_every_seconds,
        ): batch
        for batch in dataset_batches
    }

    phase_started = time.time()
    last_report_time = phase_started
    last_report_completed = 0
    datasets_completed = 0
    total_specs = len(dataset_specs)
    total_batches = len(future_map)
    batches_completed = 0

    for fut in as_completed(future_map):
        batch = future_map[fut]
        batch_results, batch_errors = fut.result()
        merge_error_states(error_state, batch_errors)
        for name, arr in batch_results.items():
            results[name] = arr

        batches_completed += 1
        datasets_completed += len(batch)
        now = time.time()
        if progress_label and (
            batches_completed == total_batches
            or (datasets_completed - last_report_completed) >= max(1, int(progress_every_datasets))
            or (now - last_report_time) >= max(1.0, float(progress_every_seconds))
        ):
            batch_label = batch[0]["name"] if len(batch) == 1 else f"{batch[0]['name']}..{batch[-1]['name']}"
            emit_progress_line(
                phase_label=progress_label,
                completed=datasets_completed,
                total=total_specs,
                started_at=phase_started,
                unit_name="datasets",
                extra=f"batches={batches_completed}/{total_batches} last_batch={batch_label}",
            )
            last_report_time = now
            last_report_completed = datasets_completed

    return results, any_in_bounds


def format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600 + minutes * 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:04.1f}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def format_rate(count, seconds, unit_name):
    seconds = max(float(seconds), 1e-9)
    rate = float(count) / seconds
    if rate >= 10:
        return f"{rate:,.0f} {unit_name}/s"
    if rate >= 1:
        return f"{rate:,.1f} {unit_name}/s"
    return f"{rate:,.2f} {unit_name}/s"


def emit_progress_line(phase_label, completed, total, started_at, unit_name, extra=""):
    elapsed = time.time() - float(started_at)
    pct = 100.0 if total <= 0 else (100.0 * float(completed) / float(total))
    msg = (
        f"[progress] {phase_label} {completed:,}/{total:,} "
        f"({pct:.1f}%) elapsed={format_elapsed(elapsed)} "
        f"rate={format_rate(completed, elapsed, unit_name)}"
    )
    if extra:
        msg += f" {extra}"
    print(msg, file=sys.stderr, flush=True)


def make_progress_state(phase_label, total, unit_name, report_every_seconds=10.0, started_at=None):
    started_at = time.time() if started_at is None else float(started_at)
    return {
        "phase_label": str(phase_label),
        "total": int(total),
        "unit_name": str(unit_name),
        "started_at": started_at,
        "report_every_seconds": max(1.0, float(report_every_seconds)),
        "last_report_time": started_at,
        "last_report_completed": 0,
    }


def maybe_emit_progress(progress_state, completed, extra="", force=False):
    if not progress_state:
        return

    completed = int(completed)
    total = int(progress_state["total"])
    now = time.time()
    time_due = (now - float(progress_state["last_report_time"])) >= float(progress_state["report_every_seconds"])
    advanced = completed > int(progress_state["last_report_completed"])
    done = total > 0 and completed >= total

    if not (force or done or (time_due and advanced)):
        return

    emit_progress_line(
        phase_label=progress_state["phase_label"],
        completed=completed,
        total=total,
        started_at=progress_state["started_at"],
        unit_name=progress_state["unit_name"],
        extra=extra,
    )
    progress_state["last_report_time"] = now
    progress_state["last_report_completed"] = completed


def main():
    ap = argparse.ArgumentParser(
        description="Bulk coordinate lookup against merged SoilGrids GeoTIFFs"
    )
    ap.add_argument("--root", required=True, help="Path to merged SoilGrids GeoTIFF folder")
    ap.add_argument("--coords", required=True, help="CSV with lon/lat columns")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--input-crs", default="EPSG:4326", help="CRS of input coordinates")
    ap.add_argument("--lon-col", default=None, help="Longitude column name")
    ap.add_argument("--lat-col", default=None, help="Latitude column name")
    ap.add_argument("--id-col", default=None, help="Optional ID column to keep first in output")
    ap.add_argument(
        "--props",
        default="",
        help="Comma-separated properties, e.g. bdod,cec,clay,sand,silt,soc,phh2o,nitrogen,cfvo,wrb",
    )
    ap.add_argument(
        "--depths",
        default="",
        help="Comma-separated depths, e.g. 0-5cm,5-15cm,15-30cm",
    )
    ap.add_argument(
        "--datasets",
        default="",
        help="Explicit comma-separated dataset names, e.g. bdod_0-5cm_mean,clay_15-30cm_mean,wrb or all",
    )
    ap.add_argument(
        "--stat",
        default="mean",
        help="Suffix after depth when composing names from --props and --depths",
    )
    ap.add_argument("--chunk-size", type=int, default=100000, help="Rows per processing chunk")
    ap.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Skip this many input data rows before processing",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Process at most this many input data rows after --start-row; 0 means no limit",
    )
    ap.add_argument(
        "--job-size",
        type=int,
        default=None,
        help="Treat this run as a single fixed-size job window; commonly 100000",
    )
    ap.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Zero-based job number used with --job-size",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append rows to an existing output CSV instead of overwriting it; header must match if the file already exists",
    )
    ap.add_argument("--gdal-cache-mb", type=int, default=512, help="GDAL cache size in MB")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for parallel dataset sampling; use 1 for single-process mode",
    )
    ap.add_argument(
        "--on-read-error",
        choices=["nan", "raise"],
        default="nan",
        help="On raster read failure, either fill NaN and continue or raise immediately",
    )
    ap.add_argument(
        "--error-log",
        default="",
        help="Optional CSV path for dataset read errors; default is <out>.errors.csv when any occur",
    )
    ap.add_argument(
        "--fallback-radius-pixels",
        type=int,
        default=3,
        help="If the exact sampled pixel is nodata, run a second-pass search outward for the nearest valid neighbor within this pixel radius, only for rows still missing after the exact pass",
    )
    ap.add_argument(
        "--progress-every-seconds",
        type=float,
        default=10.0,
        help="Emit chunk and dataset progress at least this often while sampling",
    )
    ap.add_argument(
        "--progress-every-datasets",
        type=int,
        default=8,
        help="Emit dataset progress after this many datasets complete within a chunk phase",
    )
    ap.add_argument(
        "--add-diagnostics",
        action="store_true",
        help="Write diagnostic columns: soilgrids_input_ok, soilgrids_any_hit, soilgrids_crs_group_count",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List discovered merged TIFF datasets and exit",
    )
    ap.add_argument(
        "--no-prefer-depth-stacks",
        action="store_true",
        help="Ignore *_depthstack.tif files unless a single-depth TIFF is missing for that logical dataset",
    )

    args = ap.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.progress_every_seconds <= 0:
        raise ValueError("--progress-every-seconds must be > 0")
    if args.progress_every_datasets < 1:
        raise ValueError("--progress-every-datasets must be >= 1")

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    all_datasets = discover_merged_tifs(root, prefer_depth_stacks=(not args.no_prefer_depth_stacks))
    if not all_datasets:
        raise RuntimeError(f"No merged TIFFs found under: {root}")

    if args.list:
        for name in sorted(all_datasets.keys()):
            print(name)
        return

    props = parse_csv_list(args.props)
    depths = parse_csv_list(args.depths)
    explicit_datasets = parse_csv_list(args.datasets)

    requested_names = build_requested_dataset_names(
        all_datasets=all_datasets,
        props=props,
        depths=depths,
        stat=args.stat,
        explicit_datasets=explicit_datasets,
    )
    if not requested_names:
        raise RuntimeError("No datasets selected")

    print(f"Preparing {len(requested_names)} dataset(s)...", file=sys.stderr)

    dataset_specs = []
    with rasterio.Env(GDAL_CACHEMAX=args.gdal_cache_mb):
        for name in requested_names:
            dataset_path = all_datasets[name]["path"]
            with rasterio.open(dataset_path) as ds:
                if ds.crs is None:
                    raise RuntimeError(f"Dataset has no CRS: {dataset_path}")
                block_h, block_w = get_block_shape(ds)
                spec = {
                    "name": name,
                    "path": str(dataset_path),
                    "prop": all_datasets[name]["prop"],
                    "band_index": int(all_datasets[name].get("band_index", 1)),
                    "source_kind": all_datasets[name].get("source_kind", "single"),
                    "crs_text": canonical_crs_text(ds.crs),
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "block_h": int(block_h),
                    "block_w": int(block_w),
                    "transform_values": affine_to_tuple(ds.transform),
                }
                spec["grid_signature"] = make_grid_signature(
                    spec["crs_text"],
                    spec["width"],
                    spec["height"],
                    spec["block_h"],
                    spec["block_w"],
                    spec["transform_values"],
                )
                dataset_specs.append(spec)

    dataset_specs.sort(key=lambda d: d["name"])
    crs_groups = inspect_dataset_groups(dataset_specs)
    sampling_groups = inspect_sampling_groups(dataset_specs)
    print(f"Detected {len(crs_groups)} CRS group(s) across requested datasets", file=sys.stderr)
    for crs_text, specs in crs_groups.items():
        print(f"   {len(specs)} dataset(s) in {crs_text}", file=sys.stderr)
    print(f"Detected {len(sampling_groups)} sampling grid group(s)", file=sys.stderr)
    print(
        f"Fallback enabled: exact first pass, then retry only missing rows within radius {int(args.fallback_radius_pixels)}",
        file=sys.stderr,
    )

    dataset_batches = split_into_batches(dataset_specs, args.workers)
    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers)

    start_row, max_rows = resolve_row_window(args)
    print(
        f"Input row window: start_row={start_row:,}, end_row={'end' if max_rows is None else f'{start_row + max_rows:,}'}",
        file=sys.stderr,
    )

    local_ds_cache = {}
    local_transformer_cache = {}
    local_rowcol_cache = {}
    error_state = {}

    total_rows_written = 0
    chunk_counter = 0
    started_at = time.time()

    out_f = None
    try:
        for rows, lon_col, lat_col, id_col, passthrough_cols in chunked_csv_reader(
            args.coords,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
            id_col=args.id_col,
            chunk_size=args.chunk_size,
            start_row=start_row,
            max_rows=max_rows,
        ):
            chunk_counter += 1
            chunk_rows = len(rows)
            total_rows_written += chunk_rows
            chunk_label = f"chunk={chunk_counter} phase=sample rows={chunk_rows:,}"
            print(
                f"[progress] {chunk_label} total_rows={total_rows_written:,}",
                file=sys.stderr,
                flush=True,
            )

            lon, lat, valid_coord = parse_chunk_rows(rows, lon_col, lat_col)
            results, any_in_bounds = sample_chunk_results(
                dataset_specs=dataset_specs,
                crs_groups=crs_groups,
                sampling_groups=sampling_groups,
                dataset_batches=dataset_batches,
                input_crs_text=args.input_crs,
                lon=lon,
                lat=lat,
                valid_coord=valid_coord,
                gdal_cache_mb=args.gdal_cache_mb,
                on_read_error=args.on_read_error,
                fallback_radius_pixels=args.fallback_radius_pixels,
                executor=executor,
                error_state=error_state,
                local_ds_cache=local_ds_cache,
                local_transformer_cache=local_transformer_cache,
                local_rowcol_cache=local_rowcol_cache,
                progress_label=chunk_label,
                progress_every_datasets=args.progress_every_datasets,
                progress_every_seconds=args.progress_every_seconds,
            )

            hit_mask = np.zeros(chunk_rows, dtype=bool)
            for name in requested_names:
                hit_mask |= np.isfinite(results[name])

            formatted_dataset_cols = {
                name: format_result_column(results[name])
                for name in requested_names
            }

            output_fields = build_output_fields(
                id_col=id_col,
                lon_col=lon_col,
                lat_col=lat_col,
                passthrough_cols=passthrough_cols,
                requested_names=requested_names,
                add_diagnostics=args.add_diagnostics,
            )

            if out_f is None:
                out_f, writer, _ = open_output_writer(args.out, output_fields, append=args.append)
                print(
                    f"Opened output in {'append' if args.append else 'write'} mode: {args.out}",
                    file=sys.stderr,
                )
            else:
                writer = csv.writer(out_f)

            write_output_chunk(
                writer=writer,
                rows=rows,
                id_col=id_col,
                lon_col=lon_col,
                lat_col=lat_col,
                passthrough_cols=passthrough_cols,
                requested_names=requested_names,
                formatted_dataset_cols=formatted_dataset_cols,
                add_diagnostics=args.add_diagnostics,
                valid_coord=valid_coord,
                hit_mask=hit_mask,
                crs_group_count=len(crs_groups),
            )
            out_f.flush()

            emit_progress_line(
                phase_label="write-output",
                completed=chunk_counter,
                total=chunk_counter,
                started_at=started_at,
                unit_name="chunks",
                extra=(
                    f"rows_written={total_rows_written:,} "
                    f"valid_input={int(np.count_nonzero(valid_coord)):,} "
                    f"in_bounds={int(np.count_nonzero(any_in_bounds)):,} "
                    f"hits={int(np.count_nonzero(hit_mask)):,}"
                ),
            )

    finally:
        if out_f is not None:
            out_f.close()
        close_cached_datasets(local_ds_cache)
        if executor is not None:
            executor.shutdown(wait=True)

    if error_state:
        error_log_path = (
            Path(args.error_log)
            if args.error_log
            else Path(str(args.out) + ".errors.csv")
        )
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, "w", newline="", encoding="utf-8") as ef:
            w = csv.writer(ef)
            w.writerow(["dataset", "path", "count", "first_error"])
            for key in sorted(error_state.keys()):
                st = error_state[key]
                w.writerow([
                    st.get("dataset", ""),
                    st.get("path", ""),
                    int(st.get("count", 0)),
                    st.get("first_error", ""),
                ])
        print(
            f"Wrote dataset read error log: {error_log_path} ({len(error_state)} dataset(s) with read errors)",
            file=sys.stderr,
        )

    print(
        f"Done. chunks={chunk_counter:,} rows={total_rows_written:,} elapsed={format_elapsed(time.time() - started_at)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
