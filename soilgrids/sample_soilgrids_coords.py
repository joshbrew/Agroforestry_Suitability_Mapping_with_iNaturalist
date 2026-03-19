
#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from affine import Affine
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


BASE_URL_DEFAULT = "https://files.isric.org/soilgrids/latest/data"
TIFF_NAME_RE = re.compile(r"([A-Za-z0-9_.-]+\.tif)\b", re.IGNORECASE)

_WORKER_DS_CACHE = {}
_WORKER_ENV = None
_WORKER_TRANSFORMER_CACHE = {}
_WORKER_ROWCOL_CACHE = {}
_WORKER_TILE_INDEX_CACHE = {}
_LOCAL_TILE_INDEX_CACHE = {}
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


def discover_vrts(root):
    root = Path(root)
    datasets = {}

    for prop_dir in sorted(root.iterdir()):
        if not prop_dir.is_dir():
            continue

        prop_name = prop_dir.name.strip().lower()

        for vrt in sorted(prop_dir.glob("*.vrt")):
            stem = vrt.stem.strip()

            if prop_name == "wrb":
                dataset_name = f"wrb_{stem}"
            else:
                dataset_name = stem

            datasets[dataset_name] = {
                "path": vrt.resolve(),
                "prop": prop_name,
                "stem": stem,
            }

    return datasets


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


def extract_tif_names(text):
    if not text:
        return []
    out = []
    seen = set()
    for m in TIFF_NAME_RE.finditer(str(text)):
        name = m.group(1)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_url(base_url, rel_path):
    rel = str(rel_path).replace("\\", "/").lstrip("/")
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", rel)


def download_file(url, dest_path, retries=5, timeout=120, user_agent="Mozilla/5.0"):
    ensure_parent(dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            if dest_path.exists():
                dest_path.unlink()
            tmp_path.replace(dest_path)
            return
        except Exception as exc:
            last_err = exc
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"Failed to download {url} -> {dest_path}: {last_err}")


def _parse_vrt_rect(elem, tag_name):
    rect = None
    for child in elem:
        if str(child.tag).endswith(tag_name):
            rect = child
            break
    if rect is None:
        return None
    return {
        "xOff": int(float(rect.attrib.get("xOff", 0))),
        "yOff": int(float(rect.attrib.get("yOff", 0))),
        "xSize": int(float(rect.attrib.get("xSize", 0))),
        "ySize": int(float(rect.attrib.get("ySize", 0))),
    }


def parse_vrt_source_entries(vrt_path):
    vrt_path = Path(vrt_path)
    tree = ET.parse(vrt_path)
    root = tree.getroot()

    entries = []
    for elem in root.iter():
        tag = str(elem.tag)
        if not (tag.endswith("SimpleSource") or tag.endswith("ComplexSource")):
            continue

        src_file_elem = None
        for child in elem:
            if str(child.tag).endswith("SourceFilename"):
                src_file_elem = child
                break

        if src_file_elem is None:
            continue

        src_text = (src_file_elem.text or "").strip()
        if not src_text:
            continue

        rel_to_vrt = str(src_file_elem.attrib.get("relativeToVRT", "0")).strip() == "1"
        if rel_to_vrt:
            src_path = (vrt_path.parent / src_text).resolve()
        else:
            src_path = Path(src_text).resolve()

        entries.append({
            "path": src_path,
            "src_rect": _parse_vrt_rect(elem, "SrcRect"),
            "dst_rect": _parse_vrt_rect(elem, "DstRect"),
        })

    return entries


def parse_vrt_sources(vrt_path):
    return [entry["path"] for entry in parse_vrt_source_entries(vrt_path)]


def tile_index_path_for_dataset(index_dir, dataset_name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name).strip())
    return Path(index_dir) / f"{safe_name}.tile_index.npz"


def tile_index_is_fresh(index_path, vrt_path):
    index_path = Path(index_path)
    vrt_path = Path(vrt_path).resolve()
    if not index_path.exists():
        return False

    try:
        with np.load(index_path, allow_pickle=False) as data:
            stored_vrt_path = str(data["vrt_path"][0])
            stored_vrt_mtime_ns = int(data["vrt_mtime_ns"][0])
            stored_vrt_size = int(data["vrt_size"][0])
    except Exception:
        return False

    try:
        st = vrt_path.stat()
    except Exception:
        return False

    return (
        str(vrt_path) == stored_vrt_path
        and int(st.st_mtime_ns) == stored_vrt_mtime_ns
        and int(st.st_size) == stored_vrt_size
    )


def build_vrt_tile_index(vrt_path, index_path):
    vrt_path = Path(vrt_path).resolve()
    index_path = Path(index_path)
    ensure_parent(index_path)

    with rasterio.open(vrt_path) as vrt_ds:
        width = int(vrt_ds.width)
        height = int(vrt_ds.height)
        block_h, block_w = get_block_shape(vrt_ds)

    entries = parse_vrt_source_entries(vrt_path)
    if not entries:
        raise RuntimeError(f"No VRT sources found in {vrt_path}")

    source_meta_cache = {}
    paths = []
    row0 = []
    row1 = []
    col0 = []
    col1 = []
    src_xoff = []
    src_yoff = []
    src_xsize = []
    src_ysize = []
    dst_xoff = []
    dst_yoff = []
    dst_xsize = []
    dst_ysize = []

    for entry in entries:
        src_path = Path(entry["path"]).resolve()
        src_rect = entry.get("src_rect")
        dst_rect = entry.get("dst_rect")

        if dst_rect is None:
            continue

        dx = int(dst_rect["xOff"])
        dy = int(dst_rect["yOff"])
        dw = int(dst_rect["xSize"])
        dh = int(dst_rect["ySize"])
        if dw <= 0 or dh <= 0:
            continue

        src_info = source_meta_cache.get(str(src_path).lower())
        if src_info is None:
            with rasterio.open(src_path) as src_ds:
                src_info = {
                    "width": int(src_ds.width),
                    "height": int(src_ds.height),
                }
            source_meta_cache[str(src_path).lower()] = src_info

        if src_rect is None:
            sx = 0
            sy = 0
            sw = int(src_info["width"])
            sh = int(src_info["height"])
        else:
            sx = int(src_rect["xOff"])
            sy = int(src_rect["yOff"])
            sw = int(src_rect["xSize"])
            sh = int(src_rect["ySize"])

        if sw <= 0 or sh <= 0:
            continue

        paths.append(str(src_path))
        row0.append(dy)
        row1.append(dy + dh)
        col0.append(dx)
        col1.append(dx + dw)
        src_xoff.append(sx)
        src_yoff.append(sy)
        src_xsize.append(sw)
        src_ysize.append(sh)
        dst_xoff.append(dx)
        dst_yoff.append(dy)
        dst_xsize.append(dw)
        dst_ysize.append(dh)

    if not paths:
        raise RuntimeError(f"No usable VRT source rectangles found in {vrt_path}")

    n_block_cols = int(math.ceil(width / block_w))
    block_map = defaultdict(list)

    for src_id in range(len(paths)):
        br0 = max(0, int(row0[src_id]) // int(block_h))
        br1 = min((height - 1) // int(block_h), max(0, int(row1[src_id]) - 1) // int(block_h))
        bc0 = max(0, int(col0[src_id]) // int(block_w))
        bc1 = min((width - 1) // int(block_w), max(0, int(col1[src_id]) - 1) // int(block_w))
        for br in range(br0, br1 + 1):
            base = br * n_block_cols
            for bc in range(bc0, bc1 + 1):
                block_map[base + bc].append(src_id)

    block_codes = np.array(sorted(block_map.keys()), dtype=np.int64)
    block_starts = []
    block_ends = []
    block_src_ids_flat = []
    cursor = 0
    for code in block_codes:
        src_ids = sorted(set(int(v) for v in block_map[int(code)]))
        block_starts.append(cursor)
        block_src_ids_flat.extend(src_ids)
        cursor += len(src_ids)
        block_ends.append(cursor)

    st = vrt_path.stat()
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            vrt_path=np.array([str(vrt_path)]),
            vrt_mtime_ns=np.array([int(st.st_mtime_ns)], dtype=np.int64),
            vrt_size=np.array([int(st.st_size)], dtype=np.int64),
            width=np.array([width], dtype=np.int64),
            height=np.array([height], dtype=np.int64),
            block_h=np.array([int(block_h)], dtype=np.int64),
            block_w=np.array([int(block_w)], dtype=np.int64),
            n_block_cols=np.array([int(n_block_cols)], dtype=np.int64),
            paths=np.asarray(paths, dtype=np.str_),
            row0=np.asarray(row0, dtype=np.int64),
            row1=np.asarray(row1, dtype=np.int64),
            col0=np.asarray(col0, dtype=np.int64),
            col1=np.asarray(col1, dtype=np.int64),
            src_xoff=np.asarray(src_xoff, dtype=np.int64),
            src_yoff=np.asarray(src_yoff, dtype=np.int64),
            src_xsize=np.asarray(src_xsize, dtype=np.int64),
            src_ysize=np.asarray(src_ysize, dtype=np.int64),
            dst_xoff=np.asarray(dst_xoff, dtype=np.int64),
            dst_yoff=np.asarray(dst_yoff, dtype=np.int64),
            dst_xsize=np.asarray(dst_xsize, dtype=np.int64),
            dst_ysize=np.asarray(dst_ysize, dtype=np.int64),
            block_codes=block_codes,
            block_starts=np.asarray(block_starts, dtype=np.int64),
            block_ends=np.asarray(block_ends, dtype=np.int64),
            block_src_ids=np.asarray(block_src_ids_flat, dtype=np.int64),
        )
    tmp_path.replace(index_path)
    return index_path


def load_tile_index(index_path):
    index_path = Path(index_path).resolve()
    with np.load(index_path, allow_pickle=False) as data:
        tile_index = {
            "index_path": str(index_path),
            "vrt_path": str(data["vrt_path"][0]),
            "width": int(data["width"][0]),
            "height": int(data["height"][0]),
            "block_h": int(data["block_h"][0]),
            "block_w": int(data["block_w"][0]),
            "n_block_cols": int(data["n_block_cols"][0]),
            "paths": data["paths"],
            "row0": data["row0"],
            "row1": data["row1"],
            "col0": data["col0"],
            "col1": data["col1"],
            "src_xoff": data["src_xoff"],
            "src_yoff": data["src_yoff"],
            "src_xsize": data["src_xsize"],
            "src_ysize": data["src_ysize"],
            "dst_xoff": data["dst_xoff"],
            "dst_yoff": data["dst_yoff"],
            "dst_xsize": data["dst_xsize"],
            "dst_ysize": data["dst_ysize"],
            "block_codes": data["block_codes"],
            "block_starts": data["block_starts"],
            "block_ends": data["block_ends"],
            "block_src_ids": data["block_src_ids"],
        }

    tile_index["block_lookup"] = {
        int(code): (int(start), int(end))
        for code, start, end in zip(
            tile_index["block_codes"],
            tile_index["block_starts"],
            tile_index["block_ends"],
        )
    }
    tile_index["empty_src_ids"] = np.empty(0, dtype=np.int64)
    return tile_index


def get_local_tile_index(index_path):
    key = str(Path(index_path).resolve()).lower()
    cached = _LOCAL_TILE_INDEX_CACHE.get(key)
    if cached is None:
        cached = load_tile_index(index_path)
        _LOCAL_TILE_INDEX_CACHE[key] = cached
    return cached


def get_worker_tile_index(index_path):
    key = str(Path(index_path).resolve()).lower()
    cached = _WORKER_TILE_INDEX_CACHE.get(key)
    if cached is None:
        cached = load_tile_index(index_path)
        _WORKER_TILE_INDEX_CACHE[key] = cached
    return cached


def ensure_dataset_tile_index(vrt_path, dataset_name, index_dir, rebuild=False):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = tile_index_path_for_dataset(index_dir, dataset_name)
    if rebuild or not tile_index_is_fresh(index_path, vrt_path):
        build_vrt_tile_index(vrt_path=vrt_path, index_path=index_path)
        return index_path, True
    return index_path, False




def assign_block_points_to_sources_vectorized(tile_index, candidates, block_rows, block_cols):
    candidates = np.asarray(candidates, dtype=np.int64)
    n_points = int(block_rows.size)
    assigned = np.full(n_points, -1, dtype=np.int64)
    src_rows = np.full(n_points, -1, dtype=np.int64)
    src_cols = np.full(n_points, -1, dtype=np.int64)

    if n_points == 0 or candidates.size == 0:
        return assigned, src_rows, src_cols

    cand_row0 = tile_index["row0"][candidates]
    cand_row1 = tile_index["row1"][candidates]
    cand_col0 = tile_index["col0"][candidates]
    cand_col1 = tile_index["col1"][candidates]

    rows = np.asarray(block_rows, dtype=np.int64)[None, :]
    cols = np.asarray(block_cols, dtype=np.int64)[None, :]

    inside = (rows >= cand_row0[:, None])
    inside &= (rows < cand_row1[:, None])
    inside &= (cols >= cand_col0[:, None])
    inside &= (cols < cand_col1[:, None])

    has_match = np.any(inside, axis=0)
    if not np.any(has_match):
        return assigned, src_rows, src_cols

    first_match_idx = np.argmax(inside, axis=0)
    point_idx = np.flatnonzero(has_match)
    chosen_idx = first_match_idx[point_idx]
    chosen_src_ids = candidates[chosen_idx]

    assigned[point_idx] = chosen_src_ids

    chosen_dst_yoff = tile_index["dst_yoff"][chosen_src_ids]
    chosen_dst_xoff = tile_index["dst_xoff"][chosen_src_ids]
    chosen_dst_ysize = np.maximum(tile_index["dst_ysize"][chosen_src_ids], 1)
    chosen_dst_xsize = np.maximum(tile_index["dst_xsize"][chosen_src_ids], 1)
    chosen_src_yoff = tile_index["src_yoff"][chosen_src_ids]
    chosen_src_xoff = tile_index["src_xoff"][chosen_src_ids]
    chosen_src_ysize = tile_index["src_ysize"][chosen_src_ids]
    chosen_src_xsize = tile_index["src_xsize"][chosen_src_ids]

    matched_rows = np.asarray(block_rows, dtype=np.int64)[point_idx]
    matched_cols = np.asarray(block_cols, dtype=np.int64)[point_idx]
    dr = matched_rows - chosen_dst_yoff
    dc = matched_cols - chosen_dst_xoff

    src_rows[point_idx] = chosen_src_yoff + (dr * chosen_src_ysize) // chosen_dst_ysize
    src_cols[point_idx] = chosen_src_xoff + (dc * chosen_src_xsize) // chosen_dst_xsize

    return assigned, src_rows, src_cols
def tile_index_candidates(tile_index, brow, bcol):
    code = int(brow) * int(tile_index["n_block_cols"]) + int(bcol)
    span = tile_index["block_lookup"].get(code)
    if span is None:
        return tile_index["empty_src_ids"]
    start, end = span
    return tile_index["block_src_ids"][start:end]


def get_vrt_tile_map(vrt_path, vrt_cache):
    key = str(Path(vrt_path).resolve()).lower()
    cached = vrt_cache.get(key)
    if cached is not None:
        return cached

    tile_map = defaultdict(list)
    try:
        for src in parse_vrt_sources(vrt_path):
            tile_map[src.name.lower()].append(src)
    except Exception as exc:
        tile_map = None
        print(f"WARNING: could not parse VRT for repair {vrt_path}: {exc}", file=sys.stderr)

    vrt_cache[key] = tile_map
    return tile_map


def resolve_repair_targets_from_exception(ds, exc, repair_root, vrt_cache):
    ds_path = Path(str(getattr(ds, "name", ""))).resolve()
    tif_names = extract_tif_names(str(exc))
    if not tif_names:
        return []

    targets = []

    if ds_path.suffix.lower() == ".vrt":
        tile_map = get_vrt_tile_map(ds_path, vrt_cache)
        if tile_map:
            for tif_name in tif_names:
                matches = tile_map.get(tif_name.lower(), [])
                for local_path in matches:
                    try:
                        rel_path = local_path.resolve().relative_to(repair_root)
                    except Exception:
                        continue
                    targets.append((local_path.resolve(), rel_path))
    elif ds_path.suffix.lower() == ".tif":
        try:
            rel_path = ds_path.resolve().relative_to(repair_root)
            targets.append((ds_path.resolve(), rel_path))
        except Exception:
            pass

    deduped = []
    seen = set()
    for local_path, rel_path in targets:
        key = str(local_path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((local_path, rel_path))

    return deduped


def attempt_repair_from_exception(
    ds,
    dataset_name,
    exc,
    repair_root,
    repair_base_url,
    vrt_cache,
    repair_state,
    backup_bad,
    retries,
    timeout,
):
    if repair_root is None:
        return False

    targets = resolve_repair_targets_from_exception(ds, exc, repair_root, vrt_cache)
    if not targets:
        return False

    repaired_any = False
    downloaded = repair_state.setdefault("downloaded", set())
    failed = repair_state.setdefault("failed", set())

    for local_path, rel_path in targets:
        local_key = str(local_path).lower()
        if local_key in downloaded:
            repaired_any = True
            continue
        if local_key in failed:
            continue

        url = normalize_url(repair_base_url, rel_path)
        try:
            print(f"Repairing unreadable tile for {dataset_name}: {local_path.name}", file=sys.stderr)
            print(f"  {url}", file=sys.stderr)

            if backup_bad and local_path.exists():
                backup_path = local_path.with_suffix(local_path.suffix + ".bad")
                if backup_path.exists():
                    backup_path.unlink()
                local_path.replace(backup_path)

            download_file(
                url=url,
                dest_path=local_path,
                retries=retries,
                timeout=timeout,
            )
            downloaded.add(local_key)
            repaired_any = True
        except Exception as repair_exc:
            failed.add(local_key)
            print(
                f"WARNING: failed to repair tile for {dataset_name}: {local_path} :: {repair_exc}",
                file=sys.stderr,
            )

    return repaired_any


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
    repair_read_errors=False,
    repair_root=None,
    repair_base_url=BASE_URL_DEFAULT,
    vrt_cache=None,
    repair_state=None,
    repair_backup_bad=False,
    repair_retries=5,
    repair_timeout=120,
    block_cache=None,
):
    cache_key = (int(r0), int(c0), int(h), int(w))

    if block_cache is not None and cache_key in block_cache:
        return block_cache[cache_key]

    try:
        arr = ds.read(1, window=Window(c0, r0, w, h), masked=False)
    except Exception as exc:
        repaired = False

        if repair_read_errors:
            repaired = attempt_repair_from_exception(
                ds=ds,
                dataset_name=dataset_name,
                exc=exc,
                repair_root=repair_root,
                repair_base_url=repair_base_url,
                vrt_cache=vrt_cache if vrt_cache is not None else {},
                repair_state=repair_state if repair_state is not None else {},
                backup_bad=repair_backup_bad,
                retries=repair_retries,
                timeout=repair_timeout,
            )

        if repaired:
            try:
                arr = ds.read(1, window=Window(c0, r0, w, h), masked=False)
            except Exception as exc2:
                exc = exc2
            else:
                if block_cache is not None:
                    block_cache[cache_key] = arr
                return arr

        if on_read_error == "raise":
            raise

        record_dataset_error(error_state, dataset_name, ds, exc)
        if block_cache is not None:
            block_cache[cache_key] = None
        return None

    if block_cache is not None:
        block_cache[cache_key] = arr
    return arr


def sample_pixel_coords_from_dataset(
    ds,
    pixel_rows,
    pixel_cols,
    dataset_name=None,
    nodata_to_nan=True,
    on_read_error="nan",
    error_state=None,
    repair_read_errors=False,
    repair_root=None,
    repair_base_url=BASE_URL_DEFAULT,
    vrt_cache=None,
    repair_state=None,
    repair_backup_bad=False,
    repair_retries=5,
    repair_timeout=120,
    block_cache=None,
    progress_label=None,
    progress_every_seconds=10.0,
):
    pixel_rows = np.asarray(pixel_rows, dtype=np.int64)
    pixel_cols = np.asarray(pixel_cols, dtype=np.int64)
    out = np.full(pixel_rows.shape[0], np.nan, dtype=np.float64)
    if pixel_rows.size == 0:
        return out

    block_h, block_w = get_block_shape(ds)
    valid_idx, vrows, vcols, brow, bcol, block_groups = compute_block_groups(
        pixel_rows,
        pixel_cols,
        ds.width,
        ds.height,
        block_h,
        block_w,
    )
    if valid_idx.size == 0:
        return out

    order, starts, ends = block_groups
    nodata = ds.nodata
    nodata_f = float(nodata) if nodata is not None else None
    if block_cache is None:
        block_cache = {}

    progress_state = None
    total_block_groups = int(len(starts))
    if progress_label:
        progress_state = make_progress_state(
            phase_label=progress_label,
            total=total_block_groups,
            unit_name="blocks",
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
            repair_read_errors=repair_read_errors,
            repair_root=repair_root,
            repair_base_url=repair_base_url,
            vrt_cache=vrt_cache,
            repair_state=repair_state,
            repair_backup_bad=repair_backup_bad,
            repair_retries=repair_retries,
            repair_timeout=repair_timeout,
            block_cache=block_cache,
        )
        if arr is not None:
            rr = vrows[local] - r0
            cc = vcols[local] - c0
            vals = arr[rr, cc].astype(np.float64, copy=False)
            out[valid_idx[local]] = vals

        maybe_emit_progress(
            progress_state,
            block_idx,
            extra=f"dataset={dataset_name} points={pixel_rows.size:,}",
            force=(block_idx == total_block_groups),
        )

    if nodata_to_nan and nodata is not None:
        out[out == nodata_f] = np.nan

    return out

def open_source_dataset(dataset_path, open_mode, gdal_cache_mb=None, local_ds_cache=None):
    if open_mode == "worker":
        return get_worker_dataset(dataset_path, gdal_cache_mb)
    return get_cached_dataset(local_ds_cache, dataset_path)


def sample_dataset_blocked_tile_index(
    tile_index,
    sampling_plan,
    dataset_name=None,
    nodata_to_nan=True,
    on_read_error="nan",
    error_state=None,
    repair_read_errors=False,
    repair_root=None,
    repair_base_url=BASE_URL_DEFAULT,
    vrt_cache=None,
    repair_state=None,
    repair_backup_bad=False,
    repair_retries=5,
    repair_timeout=120,
    fallback_radius_pixels=1,
    vrt_ds=None,
    open_mode="local",
    gdal_cache_mb=None,
    local_ds_cache=None,
    progress_label=None,
    progress_every_seconds=10.0,
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

    source_block_caches = {}
    block_progress_state = None
    total_block_groups = int(len(starts))
    if progress_label:
        block_progress_state = make_progress_state(
            phase_label=f"{progress_label} subphase=assign-source-blocks",
            total=total_block_groups,
            unit_name="block-groups",
            report_every_seconds=progress_every_seconds,
        )

    for block_idx, (start, end) in enumerate(zip(starts, ends), start=1):
        local = order[start:end]
        first = local[0]
        candidates = tile_index_candidates(tile_index, int(brow[first]), int(bcol[first]))
        assigned_ids = np.empty(0, dtype=np.int64)
        matched_points = 0

        if candidates.size > 0:
            block_rows = vrows[local]
            block_cols = vcols[local]
            assigned, src_rows, src_cols = assign_block_points_to_sources_vectorized(
                tile_index=tile_index,
                candidates=candidates,
                block_rows=block_rows,
                block_cols=block_cols,
            )

            assigned_ids = np.unique(assigned[assigned >= 0])
            matched_points = int(np.count_nonzero(assigned >= 0))
            for src_id in assigned_ids:
                mask = assigned == int(src_id)
                source_path = str(tile_index["paths"][int(src_id)])
                source_ds = open_source_dataset(
                    dataset_path=source_path,
                    open_mode=open_mode,
                    gdal_cache_mb=gdal_cache_mb,
                    local_ds_cache=local_ds_cache,
                )
                source_cache = source_block_caches.setdefault(str(Path(source_path).resolve()).lower(), {})
                source_progress_label = None
                if progress_label and int(np.count_nonzero(mask)) >= 10000:
                    source_progress_label = (
                        f"{progress_label} subphase=source-read source={Path(source_path).name}"
                    )
                vals = sample_pixel_coords_from_dataset(
                    ds=source_ds,
                    pixel_rows=src_rows[mask],
                    pixel_cols=src_cols[mask],
                    dataset_name=dataset_name,
                    nodata_to_nan=nodata_to_nan,
                    on_read_error=on_read_error,
                    error_state=error_state,
                    repair_read_errors=repair_read_errors,
                    repair_root=repair_root,
                    repair_base_url=repair_base_url,
                    vrt_cache=vrt_cache,
                    repair_state=repair_state,
                    repair_backup_bad=repair_backup_bad,
                    repair_retries=repair_retries,
                    repair_timeout=repair_timeout,
                    block_cache=source_cache,
                    progress_label=source_progress_label,
                    progress_every_seconds=progress_every_seconds,
                )
                out[valid_idx[local[mask]]] = vals

        maybe_emit_progress(
            block_progress_state,
            block_idx,
            extra=(
                f"dataset={dataset_name} candidates={candidates.size:,} "
                f"matched_points={matched_points:,} sources={assigned_ids.size:,}"
            ),
            force=(block_idx == total_block_groups),
        )

    fallback_radius_pixels = int(fallback_radius_pixels)
    if fallback_radius_pixels > 0 and vrt_ds is not None:
        neighbor_offsets = build_neighbor_offsets(fallback_radius_pixels)
        fallback_candidates = np.flatnonzero(np.isnan(out) & sampling_plan["in_bounds_mask"])
        block_h, block_w = get_block_shape(vrt_ds)
        vrt_block_cache = {}
        nodata = vrt_ds.nodata
        nodata_f = float(nodata) if nodata is not None else None

        rows = sampling_plan["rows"]
        cols = sampling_plan["cols"]
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
                if rr < 0 or cc < 0 or rr >= vrt_ds.height or cc >= vrt_ds.width:
                    continue

                r0 = int((rr // block_h) * block_h)
                c0 = int((cc // block_w) * block_w)
                h = min(block_h, vrt_ds.height - r0)
                w = min(block_w, vrt_ds.width - c0)

                arr = read_block_array(
                    ds=vrt_ds,
                    r0=r0,
                    c0=c0,
                    h=h,
                    w=w,
                    dataset_name=dataset_name,
                    on_read_error=on_read_error,
                    error_state=error_state,
                    repair_read_errors=repair_read_errors,
                    repair_root=repair_root,
                    repair_base_url=repair_base_url,
                    vrt_cache=vrt_cache,
                    repair_state=repair_state,
                    repair_backup_bad=repair_backup_bad,
                    repair_retries=repair_retries,
                    repair_timeout=repair_timeout,
                    block_cache=vrt_block_cache,
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

def sample_dataset_blocked(
    ds,
    sampling_plan,
    dataset_name=None,
    nodata_to_nan=True,
    on_read_error="nan",
    error_state=None,
    repair_read_errors=False,
    repair_root=None,
    repair_base_url=BASE_URL_DEFAULT,
    vrt_cache=None,
    repair_state=None,
    repair_backup_bad=False,
    repair_retries=5,
    repair_timeout=120,
    fallback_radius_pixels=1,
    progress_label=None,
    progress_every_seconds=10.0,
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
            repair_read_errors=repair_read_errors,
            repair_root=repair_root,
            repair_base_url=repair_base_url,
            vrt_cache=vrt_cache,
            repair_state=repair_state,
            repair_backup_bad=repair_backup_bad,
            repair_retries=repair_retries,
            repair_timeout=repair_timeout,
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
                    repair_read_errors=repair_read_errors,
                    repair_root=repair_root,
                    repair_base_url=repair_base_url,
                    vrt_cache=vrt_cache,
                    repair_state=repair_state,
                    repair_backup_bad=repair_backup_bad,
                    repair_retries=repair_retries,
                    repair_timeout=repair_timeout,
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
        sampler_mode = str(spec.get("sampler_mode", "vrt"))
        plan = plans_by_grid[spec["grid_signature"]]
        dataset_progress_label = None
        dataset_started = time.time()
        if progress_label:
            dataset_progress_label = (
                f"{progress_label} worker={worker_pid} dataset={spec_idx}/{len(dataset_batch)} name={spec['name']}"
            )
            print(
                f"[progress] {dataset_progress_label} start mode={sampler_mode} "
                f"rows={lon.shape[0]:,} in_bounds={int(np.count_nonzero(plan['in_bounds_mask'])):,} "
                f"block_groups={len(plan['starts']):,}",
                file=sys.stderr,
                flush=True,
            )

        if sampler_mode == "tile-index":
            tile_index = get_worker_tile_index(spec["tile_index_path"])
            vrt_ds = None
            if int(fallback_radius_pixels) > 0:
                vrt_ds = get_worker_dataset(spec["path"], gdal_cache_mb)
            results[spec["name"]] = sample_dataset_blocked_tile_index(
                tile_index=tile_index,
                sampling_plan=plan,
                dataset_name=spec["name"],
                on_read_error=on_read_error,
                error_state=error_state,
                repair_read_errors=False,
                fallback_radius_pixels=fallback_radius_pixels,
                vrt_ds=vrt_ds,
                open_mode="worker",
                gdal_cache_mb=gdal_cache_mb,
                progress_label=dataset_progress_label,
                progress_every_seconds=progress_every_seconds,
            )
        else:
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
                repair_read_errors=False,
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
    root,
    vrt_cache,
    repair_state,
    repair_backup_bad,
    repair_retries,
    repair_timeout,
    local_ds_cache,
    local_transformer_cache,
    local_rowcol_cache,
    repair_read_errors,
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
            sampler_mode = str(spec.get("sampler_mode", "vrt"))
            plan = plans_by_grid[spec["grid_signature"]]
            dataset_progress_label = None
            dataset_started = time.time()
            if progress_label:
                dataset_progress_label = f"{progress_label} dataset={spec_idx}/{total_specs} name={spec['name']}"
                print(
                    f"[progress] {dataset_progress_label} start mode={sampler_mode} "
                    f"rows={n:,} in_bounds={int(np.count_nonzero(plan['in_bounds_mask'])):,} "
                    f"block_groups={len(plan['starts']):,}",
                    file=sys.stderr,
                    flush=True,
                )

            if sampler_mode == "tile-index":
                tile_index = get_local_tile_index(spec["tile_index_path"])
                vrt_ds = None
                if int(fallback_radius_pixels) > 0:
                    vrt_ds = get_cached_dataset(local_ds_cache, spec["path"])
                results[spec["name"]] = sample_dataset_blocked_tile_index(
                    tile_index=tile_index,
                    sampling_plan=plan,
                    dataset_name=spec["name"],
                    on_read_error=on_read_error,
                    error_state=error_state,
                    repair_read_errors=repair_read_errors,
                    repair_root=root,
                    repair_base_url=BASE_URL_DEFAULT,
                    vrt_cache=vrt_cache,
                    repair_state=repair_state,
                    repair_backup_bad=repair_backup_bad,
                    repair_retries=repair_retries,
                    repair_timeout=repair_timeout,
                    fallback_radius_pixels=fallback_radius_pixels,
                    vrt_ds=vrt_ds,
                    open_mode="local",
                    gdal_cache_mb=gdal_cache_mb,
                    local_ds_cache=local_ds_cache,
                    progress_label=dataset_progress_label,
                    progress_every_seconds=progress_every_seconds,
                )
            else:
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
                    repair_read_errors=repair_read_errors,
                    repair_root=root,
                    repair_base_url=BASE_URL_DEFAULT,
                    vrt_cache=vrt_cache,
                    repair_state=repair_state,
                    repair_backup_bad=repair_backup_bad,
                    repair_retries=repair_retries,
                    repair_timeout=repair_timeout,
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

def compute_retry_indices(results, requested_names, valid_coord, retryable_mask):
    missing_mask = np.zeros(valid_coord.shape[0], dtype=bool)
    for name in requested_names:
        missing_mask |= ~np.isfinite(results[name])
    missing_mask &= valid_coord
    if retryable_mask is not None:
        missing_mask &= retryable_mask
    return np.flatnonzero(missing_mask)


def merge_retry_results(base_results, retry_results, retry_idx, requested_names):
    if retry_idx.size == 0:
        return 0

    filled = 0
    for name in requested_names:
        base_arr = base_results[name]
        retry_arr = retry_results[name]
        if retry_arr.shape[0] != retry_idx.shape[0]:
            raise RuntimeError(
                f"Retry result length mismatch for {name}: expected {retry_idx.shape[0]}, got {retry_arr.shape[0]}"
            )

        base_missing = ~np.isfinite(base_arr[retry_idx])
        if not np.any(base_missing):
            continue

        retry_good = np.isfinite(retry_arr)
        fill_mask = base_missing & retry_good
        if not np.any(fill_mask):
            continue

        fill_idx = retry_idx[fill_mask]
        base_arr[fill_idx] = retry_arr[fill_mask]
        filled += int(np.count_nonzero(fill_mask))

    return filled


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



def format_eta(remaining_count, rate_per_second):
    if remaining_count is None or rate_per_second is None or rate_per_second <= 0:
        return "n/a"
    return format_elapsed(float(remaining_count) / float(rate_per_second))



def count_missing_results(results, requested_names, valid_coord=None, retryable_mask=None):
    if not requested_names:
        return 0

    first = results[requested_names[0]]
    missing_mask = np.zeros(first.shape[0], dtype=bool)
    for name in requested_names:
        missing_mask |= ~np.isfinite(results[name])

    if valid_coord is not None:
        missing_mask &= valid_coord
    if retryable_mask is not None:
        missing_mask &= retryable_mask
    return int(np.count_nonzero(missing_mask))



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
        description="Bulk coordinate lookup against SoilGrids VRT mosaics with explicit CRS transforms"
    )
    ap.add_argument("--root", required=True, help="Path to SoilGrids data folder")
    ap.add_argument("--coords", default=None, help="CSV with lon/lat columns; optional with --index-only")
    ap.add_argument("--out", default=None, help="Output CSV path; optional with --index-only")
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
    ap.add_argument(
        "--sampler",
        choices=["auto", "vrt", "tile-index"],
        default="auto",
        help="Sampling backend. auto builds or reuses a sidecar tile index and uses it when available",
    )
    ap.add_argument(
        "--index-dir",
        default="",
        help="Directory for persistent tile indexes; defaults to <root>/.soilgrids_tile_index",
    )
    ap.add_argument(
        "--build-index",
        action="store_true",
        help="Build missing tile indexes before sampling",
    )
    ap.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding tile indexes even if they already exist and look fresh",
    )
    ap.add_argument(
        "--index-only",
        action="store_true",
        help="Build or refresh tile indexes for the selected datasets and exit without sampling",
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
        "--repair-read-errors",
        action="store_true",
        help="If a VRT-backed TIFF block is unreadable, try re-downloading the specific bad local TIFF and retry once",
    )
    ap.add_argument(
        "--repair-base-url",
        default=BASE_URL_DEFAULT,
        help="Remote SoilGrids base URL used for targeted tile repair",
    )
    ap.add_argument(
        "--repair-backup-bad",
        action="store_true",
        help="Rename unreadable local TIFFs to .bad before downloading replacements",
    )
    ap.add_argument(
        "--repair-retries",
        type=int,
        default=5,
        help="Download retry count for targeted tile repair",
    )
    ap.add_argument(
        "--repair-timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds for targeted tile repair",
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
        help="List discovered VRT datasets and exit",
    )

    args = ap.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > 1 and args.repair_read_errors:
        raise ValueError("--workers > 1 is not compatible with --repair-read-errors in this version")

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    all_datasets = discover_vrts(root)
    if not all_datasets:
        raise RuntimeError(f"No VRTs found under: {root}")

    if args.list:
        for name in sorted(all_datasets.keys()):
            print(name)
        return

    index_dir = Path(args.index_dir).expanduser().resolve() if args.index_dir else (root / ".soilgrids_tile_index")

    if not args.index_only:
        if not args.coords:
            raise ValueError("--coords is required unless --index-only is used")
        if not args.out:
            raise ValueError("--out is required unless --index-only is used")

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

    if args.progress_every_seconds <= 0:
        raise ValueError("--progress-every-seconds must be > 0")
    if args.progress_every_datasets < 1:
        raise ValueError("--progress-every-datasets must be >= 1")

    tile_index_paths = {}
    built_index_count = 0
    reused_index_count = 0
    sampler_request = str(args.sampler).lower()

    if sampler_request in {"auto", "tile-index"} or args.build_index or args.index_only:
        print(f"Ensuring tile indexes under {index_dir}", file=sys.stderr)
        index_started = time.time()
        last_index_report_time = index_started
        last_index_report_completed = 0
        for idx, name in enumerate(requested_names, start=1):
            index_path, built_now = ensure_dataset_tile_index(
                vrt_path=all_datasets[name]["path"],
                dataset_name=name,
                index_dir=index_dir,
                rebuild=args.rebuild_index,
            )
            tile_index_paths[name] = str(index_path)
            if built_now:
                built_index_count += 1
            else:
                reused_index_count += 1

            now = time.time()
            if (
                idx == len(requested_names)
                or (idx - last_index_report_completed) >= max(1, int(args.progress_every_datasets))
                or (now - last_index_report_time) >= max(1.0, float(args.progress_every_seconds))
            ):
                emit_progress_line(
                    phase_label="index-build",
                    completed=idx,
                    total=len(requested_names),
                    started_at=index_started,
                    unit_name="datasets",
                    extra=f"built={built_index_count:,} reused={reused_index_count:,} current={name}",
                )
                last_index_report_time = now
                last_index_report_completed = idx
        print(
            f"Tile index status: built={built_index_count}, reused={reused_index_count}",
            file=sys.stderr,
        )

    if args.index_only:
        return

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
                    "crs_text": canonical_crs_text(ds.crs),
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "block_h": int(block_h),
                    "block_w": int(block_w),
                    "transform_values": affine_to_tuple(ds.transform),
                    "sampler_mode": "vrt",
                }
                tile_index_path = tile_index_paths.get(name)
                if sampler_request == "tile-index":
                    if not tile_index_path:
                        raise RuntimeError(f"Tile index not available for {name}")
                    spec["sampler_mode"] = "tile-index"
                    spec["tile_index_path"] = str(tile_index_path)
                elif sampler_request == "auto" and tile_index_path:
                    spec["sampler_mode"] = "tile-index"
                    spec["tile_index_path"] = str(tile_index_path)

                spec["grid_signature"] = make_grid_signature(
                    spec["crs_text"],
                    spec["width"],
                    spec["height"],
                    spec["block_h"],
                    spec["block_w"],
                    spec["transform_values"],
                )
                dataset_specs.append(spec)

    crs_groups = inspect_dataset_groups(dataset_specs)
    sampling_groups = inspect_sampling_groups(dataset_specs)

    print(f"Detected {len(crs_groups)} CRS group(s) across requested datasets", file=sys.stderr)
    for crs_text, specs in crs_groups.items():
        print(f"  {len(specs):3d} dataset(s) in {crs_text}", file=sys.stderr)
    print(f"Detected {len(sampling_groups)} sampling grid group(s)", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tile_index_spec_count = sum(1 for spec in dataset_specs if spec.get("sampler_mode") == "tile-index")
    if tile_index_spec_count:
        print(
            f"Using tile-index sampling for {tile_index_spec_count:,} of {len(dataset_specs):,} dataset(s)",
            file=sys.stderr,
        )

    start_row, max_rows = resolve_row_window(args)
    if max_rows == 0:
        return

    input_crs_obj = rasterio.crs.CRS.from_user_input(args.input_crs)
    input_crs_text = canonical_crs_text(input_crs_obj)

    error_state = {}
    vrt_cache = {}
    repair_state = {"downloaded": set(), "failed": set()}
    local_ds_cache = {}
    local_transformer_cache = {}
    local_rowcol_cache = {}
    executor = None
    out_f = None
    writer = None
    total_rows = 0
    chunk_idx = 0
    run_started = time.time()

    try:
        if args.workers > 1:
            max_workers = min(args.workers, len(dataset_specs))
            executor = ProcessPoolExecutor(max_workers=max_workers)
            dataset_batches = split_into_batches(dataset_specs, max_workers)
            print(
                f"Using {max_workers} worker processes; GDAL cache is per worker, "
                f"so total cache can reach about {max_workers * args.gdal_cache_mb:,} MB",
                file=sys.stderr,
            )
        else:
            dataset_batches = [list(dataset_specs)]

        if max(0, args.fallback_radius_pixels) > 0:
            print(
                f"Fallback enabled: exact first pass, then retry only missing rows within radius {max(0, args.fallback_radius_pixels)}",
                file=sys.stderr,
            )

        row_window_end = None if max_rows is None else start_row + max_rows
        print(
            f"Input row window: start_row={start_row:,}, end_row={'EOF' if row_window_end is None else format(row_window_end, ',')}",
            file=sys.stderr,
        )
        if max_rows is not None:
            est_chunks = max(1, int(math.ceil(max_rows / max(1, args.chunk_size))))
            print(
                f"Planned chunk count: about {est_chunks:,} chunk(s) at chunk_size={max(1, args.chunk_size):,}",
                file=sys.stderr,
            )

        for rows, lon_col, lat_col, id_col, passthrough_cols in chunked_csv_reader(
            args.coords,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
            id_col=args.id_col,
            chunk_size=args.chunk_size,
            start_row=start_row,
            max_rows=max_rows,
        ):
            chunk_idx += 1
            chunk_started = time.time()
            n = len(rows)
            total_rows += n

            lon, lat, valid_coord = parse_chunk_rows(rows, lon_col, lat_col)
            valid_count = int(np.count_nonzero(valid_coord))
            print(
                f"[chunk {chunk_idx}] start rows={n:,} valid_coords={valid_count:,} processed_rows={total_rows:,}",
                file=sys.stderr,
            )

            retry_rows = 0
            retry_filled = 0

            exact_started = time.time()
            results, retryable_mask = sample_chunk_results(
                dataset_specs=dataset_specs,
                crs_groups=crs_groups,
                sampling_groups=sampling_groups,
                dataset_batches=dataset_batches,
                input_crs_text=input_crs_text,
                lon=lon,
                lat=lat,
                valid_coord=valid_coord,
                gdal_cache_mb=args.gdal_cache_mb,
                on_read_error=args.on_read_error,
                fallback_radius_pixels=0,
                executor=executor,
                error_state=error_state,
                root=root,
                vrt_cache=vrt_cache,
                repair_state=repair_state,
                repair_backup_bad=args.repair_backup_bad,
                repair_retries=max(1, args.repair_retries),
                repair_timeout=max(1, args.repair_timeout),
                local_ds_cache=local_ds_cache,
                local_transformer_cache=local_transformer_cache,
                local_rowcol_cache=local_rowcol_cache,
                repair_read_errors=args.repair_read_errors,
                progress_label=f"chunk={chunk_idx} phase=exact",
                progress_every_datasets=args.progress_every_datasets,
                progress_every_seconds=args.progress_every_seconds,
            )
            exact_elapsed = time.time() - exact_started
            missing_after_exact = count_missing_results(
                results=results,
                requested_names=requested_names,
                valid_coord=valid_coord,
                retryable_mask=retryable_mask,
            )
            print(
                f"[chunk {chunk_idx}] exact done elapsed={format_elapsed(exact_elapsed)} "
                f"rate={format_rate(n, exact_elapsed, 'rows')} missing_after_exact={missing_after_exact:,}",
                file=sys.stderr,
            )

            fallback_radius_pixels = max(0, args.fallback_radius_pixels)
            if fallback_radius_pixels > 0:
                retry_idx = compute_retry_indices(
                    results=results,
                    requested_names=requested_names,
                    valid_coord=valid_coord,
                    retryable_mask=retryable_mask,
                )
                retry_rows = int(retry_idx.shape[0])

                if retry_rows > 0:
                    print(
                        f"[chunk {chunk_idx}] retry start rows={retry_rows:,} radius={fallback_radius_pixels}",
                        file=sys.stderr,
                    )
                    retry_started = time.time()
                    retry_results, _ = sample_chunk_results(
                        dataset_specs=dataset_specs,
                        crs_groups=crs_groups,
                        sampling_groups=sampling_groups,
                        dataset_batches=dataset_batches,
                        input_crs_text=input_crs_text,
                        lon=lon[retry_idx],
                        lat=lat[retry_idx],
                        valid_coord=valid_coord[retry_idx],
                        gdal_cache_mb=args.gdal_cache_mb,
                        on_read_error=args.on_read_error,
                        fallback_radius_pixels=fallback_radius_pixels,
                        executor=executor,
                        error_state=error_state,
                        root=root,
                        vrt_cache=vrt_cache,
                        repair_state=repair_state,
                        repair_backup_bad=args.repair_backup_bad,
                        repair_retries=max(1, args.repair_retries),
                        repair_timeout=max(1, args.repair_timeout),
                        local_ds_cache=local_ds_cache,
                        local_transformer_cache=local_transformer_cache,
                        local_rowcol_cache=local_rowcol_cache,
                        repair_read_errors=args.repair_read_errors,
                        progress_label=f"chunk={chunk_idx} phase=retry",
                        progress_every_datasets=args.progress_every_datasets,
                        progress_every_seconds=args.progress_every_seconds,
                    )
                    retry_elapsed = time.time() - retry_started
                    retry_filled = merge_retry_results(
                        base_results=results,
                        retry_results=retry_results,
                        retry_idx=retry_idx,
                        requested_names=requested_names,
                    )
                    print(
                        f"[chunk {chunk_idx}] retry done elapsed={format_elapsed(retry_elapsed)} "
                        f"rate={format_rate(retry_rows, retry_elapsed, 'rows')} retry_filled={retry_filled:,}",
                        file=sys.stderr,
                    )

            if writer is None:
                output_fields = build_output_fields(
                    id_col=id_col,
                    lon_col=lon_col,
                    lat_col=lat_col,
                    passthrough_cols=passthrough_cols,
                    requested_names=requested_names,
                    add_diagnostics=args.add_diagnostics,
                )
                out_f, writer, wrote_header = open_output_writer(
                    out_path=out_path,
                    output_fields=output_fields,
                    append=args.append,
                )
                mode_name = "append" if args.append else "write"
                header_name = "yes" if wrote_header else "no"
                print(
                    f"Opened output in {mode_name} mode: {out_path} (wrote_header={header_name})",
                    file=sys.stderr,
                )

            formatted_dataset_cols = {
                name: format_result_column(results[name])
                for name in requested_names
            }

            hit_mask = None
            if args.add_diagnostics:
                hit_mask = np.zeros(n, dtype=bool)
                for name in requested_names:
                    hit_mask |= np.isfinite(results[name])

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

            if out_f is not None:
                out_f.flush()

            overall_elapsed = time.time() - run_started
            overall_rate = None if overall_elapsed <= 0 else (float(total_rows) / float(overall_elapsed))
            remaining_rows = None if max_rows is None else max(0, int(max_rows) - int(total_rows))
            remaining_missing = count_missing_results(
                results=results,
                requested_names=requested_names,
                valid_coord=valid_coord,
                retryable_mask=retryable_mask,
            )
            print(
                f"[chunk {chunk_idx}] complete rows={n:,} total_rows={total_rows:,} "
                f"retry_rows={retry_rows:,} retry_filled={retry_filled:,} remaining_missing={remaining_missing:,} "
                f"chunk_elapsed={format_elapsed(time.time() - chunk_started)} "
                f"overall_elapsed={format_elapsed(overall_elapsed)} "
                f"overall_rate={format_rate(total_rows, overall_elapsed, 'rows')} "
                f"eta={format_eta(remaining_rows, overall_rate)}",
                file=sys.stderr,
            )

        if error_state:
            error_log_path = (
                Path(args.error_log)
                if args.error_log
                else out_path.with_suffix(out_path.suffix + ".errors.csv")
            )

            with open(error_log_path, "w", newline="", encoding="utf-8") as ef:
                err_writer = csv.DictWriter(
                    ef,
                    fieldnames=["dataset", "path", "count", "first_error"],
                )
                err_writer.writeheader()
                for key in sorted(error_state.keys()):
                    err_writer.writerow(error_state[key])

            total_error_hits = sum(v["count"] for v in error_state.values())
            print(
                f"Wrote dataset read error log: {error_log_path} "
                f"(datasets={len(error_state)}, block_failures={total_error_hits})",
                file=sys.stderr,
            )

        if args.repair_read_errors:
            repaired_count = len(repair_state.get("downloaded", set()))
            failed_count = len(repair_state.get("failed", set()))
            print(
                f"Repair summary: downloaded={repaired_count}, failed={failed_count}",
                file=sys.stderr,
            )

        total_elapsed = time.time() - run_started
        print(
            f"Completed sampling rows={total_rows:,} elapsed={format_elapsed(total_elapsed)} "
            f"avg_rate={format_rate(total_rows, total_elapsed, 'rows')}",
            file=sys.stderr,
        )

    finally:
        if out_f is not None:
            out_f.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        close_cached_datasets(local_ds_cache)


if __name__ == "__main__":
    main()
