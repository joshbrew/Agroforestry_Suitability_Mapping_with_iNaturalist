#!/usr/bin/env python3
import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform

"""
python sample_twi_coords_rewritten.py ^
  --tif twi_edtm_120m.tif ^
  --input coords.csv ^
  --output coords_with_twi.csv ^
  --mode auto ^
  --chunk-size 250000 ^
  --superblock-size 4 ^
  --fallback-radius-pixels 3 ^
  --gdal-cache-mb 2048
"""

DEFAULT_LON_NAMES = (
    "lon",
    "longitude",
    "x",
    "decimalLongitude",
    "LONGITUDE",
    "LON",
    "X",
)

DEFAULT_LAT_NAMES = (
    "lat",
    "latitude",
    "y",
    "decimalLatitude",
    "LATITUDE",
    "LAT",
    "Y",
)

OUT_EXTRA_FIELDS = [
    "twi_value",
    "twi_row",
    "twi_col",
    "twi_in_bounds",
    "twi_is_nodata",
    "twi_ok",
    "twi_error",
]


class BucketWriterCache:
    def __init__(self, root: str, fieldnames: Sequence[str]):
        self.root = root
        self.fieldnames = list(fieldnames)
        self._writers: Dict[int, Tuple[object, csv.DictWriter]] = {}
        self._paths: Dict[int, str] = {}
        os.makedirs(self.root, exist_ok=True)

    def _path_for_bucket(self, bucket_id: int) -> str:
        return os.path.join(self.root, f"bucket_{bucket_id:06d}.csv")

    def write(self, bucket_id: int, row: Dict[str, str]) -> None:
        entry = self._writers.get(bucket_id)
        if entry is None:
            path = self._path_for_bucket(bucket_id)
            is_new = not os.path.exists(path)
            f = open(path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if is_new:
                writer.writeheader()
            self._writers[bucket_id] = (f, writer)
            self._paths[bucket_id] = path
            entry = (f, writer)
        _, writer = entry
        writer.writerow(row)

    def close(self) -> None:
        for f, _ in self._writers.values():
            try:
                f.close()
            except Exception:
                pass
        self._writers.clear()

    def bucket_paths(self) -> List[str]:
        return [self._paths[k] for k in sorted(self._paths)]


class ShardWriterCache:
    def __init__(self, root: str, fieldnames: Sequence[str]):
        self.root = root
        self.fieldnames = list(fieldnames)
        self._writers: Dict[str, Tuple[object, csv.DictWriter]] = {}
        self._paths: Dict[str, str] = {}
        os.makedirs(self.root, exist_ok=True)

    def _path_for_key(self, shard_key: str) -> str:
        return os.path.join(self.root, f"{shard_key}.csv")

    def write_many(self, shard_key: str, rows: Sequence[Dict[str, str]]) -> None:
        if not rows:
            return
        entry = self._writers.get(shard_key)
        if entry is None:
            path = self._path_for_key(shard_key)
            is_new = not os.path.exists(path)
            f = open(path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if is_new:
                writer.writeheader()
            self._writers[shard_key] = (f, writer)
            self._paths[shard_key] = path
            entry = (f, writer)
        _, writer = entry
        for row in rows:
            writer.writerow(row)

    def close(self) -> None:
        for f, _ in self._writers.values():
            try:
                f.close()
            except Exception:
                pass
        self._writers.clear()

    def shard_paths(self) -> List[str]:
        return [self._paths[k] for k in sorted(self._paths)]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def detect_column(fieldnames: Sequence[str], explicit_name: str, fallbacks: Sequence[str]) -> str:
    if explicit_name:
        if explicit_name not in fieldnames:
            raise ValueError(
                f"Column '{explicit_name}' not found. Available columns: {', '.join(fieldnames)}"
            )
        return explicit_name

    lowered = {name.lower(): name for name in fieldnames}
    for candidate in fallbacks:
        hit = lowered.get(candidate.lower())
        if hit is not None:
            return hit

    raise ValueError(
        f"Could not auto-detect column. Available columns: {', '.join(fieldnames)}"
    )


def parse_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return float(s)


def format_value(value):
    try:
        fv = float(value)
    except Exception:
        return str(value)

    if math.isnan(fv):
        return ""

    if fv.is_integer():
        return str(int(fv))

    return format(fv, ".10g")


def get_block_shape(ds):
    try:
        if ds.block_shapes and ds.block_shapes[0]:
            block_h, block_w = ds.block_shapes[0]
            if block_h > 0 and block_w > 0:
                return int(block_h), int(block_w)
    except Exception:
        pass

    return 512, 512


def unique_fieldnames(fieldnames: Sequence[str], extra_fields: Sequence[str]) -> List[str]:
    out = list(fieldnames)
    for name in extra_fields:
        if name not in out:
            out.append(name)
    return out


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


def is_missing_value_scalar(value, nodata) -> bool:
    try:
        fv = float(value)
    except Exception:
        return False

    if math.isnan(fv):
        return True

    if nodata is None:
        return False

    try:
        return fv == float(nodata)
    except Exception:
        return False


def is_missing_array(values: np.ndarray, nodata) -> np.ndarray:
    missing = np.isnan(values)
    if nodata is None:
        return missing
    try:
        return missing | (values == float(nodata))
    except Exception:
        return missing


def iter_csv_chunks(reader: csv.DictReader, chunk_size: int):
    chunk = []
    for row in reader:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def count_input_rows(path: str) -> int:
    total = 0
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        for _ in reader:
            total += 1
    return total


def make_output_row(row: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)
    out["twi_value"] = ""
    out["twi_row"] = ""
    out["twi_col"] = ""
    out["twi_in_bounds"] = "0"
    out["twi_is_nodata"] = ""
    out["twi_ok"] = "0"
    out["twi_error"] = ""
    return out


def get_inv_transform(ds) -> Affine:
    return ~ds.transform


def maybe_transform_coords(ds, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ds.crs is None:
        return xs, ys
    crs_str = str(ds.crs).upper()
    if crs_str in ("EPSG:4326", "OGC:CRS84"):
        return xs, ys
    tx, ty = rio_transform("EPSG:4326", ds.crs, xs.tolist(), ys.tolist())
    return np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)


def vectorized_rowcol(inv: Affine, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cols_f = inv.a * xs + inv.b * ys + inv.c
    rows_f = inv.d * xs + inv.e * ys + inv.f
    cols = np.floor(cols_f).astype(np.int64)
    rows = np.floor(rows_f).astype(np.int64)
    return rows, cols


def sample_with_fallback_scalar(arr, r, c, row_off, col_off, ds_height, ds_width, nodata, offsets):
    local_r = r - row_off
    local_c = c - col_off
    value = arr[local_r, local_c]
    if not is_missing_value_scalar(value, nodata):
        return r, c, value

    for dr, dc in offsets:
        rr = r + dr
        cc = c + dc
        if rr < 0 or cc < 0 or rr >= ds_height or cc >= ds_width:
            continue
        local_rr = rr - row_off
        local_cc = cc - col_off
        if local_rr < 0 or local_cc < 0 or local_rr >= arr.shape[0] or local_cc >= arr.shape[1]:
            continue
        value = arr[local_rr, local_cc]
        if not is_missing_value_scalar(value, nodata):
            return rr, cc, value

    return None, None, None


def apply_samples_to_outputs(
    ds,
    outputs: List[Dict[str, str]],
    pending_indices: np.ndarray,
    rows_idx: np.ndarray,
    cols_idx: np.ndarray,
    band: int,
    block_h: int,
    block_w: int,
    nodata,
    fallback_radius_pixels: int,
    offsets: Sequence[Tuple[int, int]],
    superblock_size: int,
) -> None:
    if pending_indices.size == 0:
        return

    sbh = max(1, superblock_size) * block_h
    sbw = max(1, superblock_size) * block_w

    sb_rows = rows_idx // sbh
    sb_cols = cols_idx // sbw
    order = np.lexsort((sb_cols, sb_rows))

    sorted_pending = pending_indices[order]
    sorted_rows = rows_idx[order]
    sorted_cols = cols_idx[order]
    sorted_sb_rows = sb_rows[order]
    sorted_sb_cols = sb_cols[order]

    group_start = 0
    n = sorted_pending.size
    while group_start < n:
        sb_row = int(sorted_sb_rows[group_start])
        sb_col = int(sorted_sb_cols[group_start])
        group_end = group_start + 1
        while (
            group_end < n
            and int(sorted_sb_rows[group_end]) == sb_row
            and int(sorted_sb_cols[group_end]) == sb_col
        ):
            group_end += 1

        grp_pending = sorted_pending[group_start:group_end]
        grp_rows = sorted_rows[group_start:group_end]
        grp_cols = sorted_cols[group_start:group_end]

        base_row_off = sb_row * sbh
        base_col_off = sb_col * sbw

        row_off = max(0, base_row_off - fallback_radius_pixels)
        col_off = max(0, base_col_off - fallback_radius_pixels)
        row_end = min(ds.height, base_row_off + sbh + fallback_radius_pixels)
        col_end = min(ds.width, base_col_off + sbw + fallback_radius_pixels)

        window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
        arr = ds.read(band, window=window, masked=False)

        local_rows = grp_rows - row_off
        local_cols = grp_cols - col_off
        values = arr[local_rows, local_cols]
        missing = is_missing_array(values, nodata)

        for rel_i, out_idx in enumerate(grp_pending.tolist()):
            out = outputs[out_idx]
            out["twi_ok"] = "1"
            out["twi_row"] = str(int(grp_rows[rel_i]))
            out["twi_col"] = str(int(grp_cols[rel_i]))

            if not missing[rel_i]:
                out["twi_is_nodata"] = "0"
                out["twi_value"] = format_value(values[rel_i])
                continue

            if fallback_radius_pixels > 0:
                rr, cc, value = sample_with_fallback_scalar(
                    arr=arr,
                    r=int(grp_rows[rel_i]),
                    c=int(grp_cols[rel_i]),
                    row_off=row_off,
                    col_off=col_off,
                    ds_height=ds.height,
                    ds_width=ds.width,
                    nodata=nodata,
                    offsets=offsets,
                )
                if value is not None:
                    out["twi_is_nodata"] = "0"
                    out["twi_row"] = str(int(rr))
                    out["twi_col"] = str(int(cc))
                    out["twi_value"] = format_value(value)
                    continue

            out["twi_is_nodata"] = "1"
            out["twi_value"] = ""

        group_start = group_end


def prepare_chunk_rows(ds, chunk_rows, lon_col, lat_col):
    outputs = [make_output_row(row) for row in chunk_rows]
    xs = np.full(len(chunk_rows), np.nan, dtype=np.float64)
    ys = np.full(len(chunk_rows), np.nan, dtype=np.float64)
    valid = np.zeros(len(chunk_rows), dtype=bool)

    for i, row in enumerate(chunk_rows):
        try:
            lon = parse_float(row.get(lon_col))
            lat = parse_float(row.get(lat_col))
        except Exception:
            lon = None
            lat = None

        if lon is None or lat is None:
            outputs[i]["twi_error"] = "bad_coordinate"
            continue

        xs[i] = lon
        ys[i] = lat
        valid[i] = True

    if not valid.any():
        return outputs, np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    valid_idx = np.flatnonzero(valid)
    tx, ty = maybe_transform_coords(ds, xs[valid], ys[valid])
    inv = get_inv_transform(ds)
    rows_idx, cols_idx = vectorized_rowcol(inv, tx, ty)
    in_bounds = (
        (rows_idx >= 0)
        & (cols_idx >= 0)
        & (rows_idx < ds.height)
        & (cols_idx < ds.width)
    )

    pending_indices = []
    pending_rows = []
    pending_cols = []

    for rel_i, row_i in enumerate(valid_idx.tolist()):
        out = outputs[row_i]
        out["twi_row"] = str(int(rows_idx[rel_i]))
        out["twi_col"] = str(int(cols_idx[rel_i]))
        if not in_bounds[rel_i]:
            out["twi_error"] = "out_of_bounds"
            continue
        out["twi_in_bounds"] = "1"
        pending_indices.append(row_i)
        pending_rows.append(int(rows_idx[rel_i]))
        pending_cols.append(int(cols_idx[rel_i]))

    return (
        outputs,
        np.asarray(pending_indices, dtype=np.int64),
        np.asarray(pending_rows, dtype=np.int64),
        np.asarray(pending_cols, dtype=np.int64),
    )


def process_chunk_memory(
    ds,
    writer,
    chunk_rows,
    lon_col,
    lat_col,
    band,
    block_h,
    block_w,
    nodata,
    fallback_radius_pixels,
    offsets,
    superblock_size,
):
    outputs, pending_indices, pending_rows, pending_cols = prepare_chunk_rows(ds, chunk_rows, lon_col, lat_col)
    apply_samples_to_outputs(
        ds=ds,
        outputs=outputs,
        pending_indices=pending_indices,
        rows_idx=pending_rows,
        cols_idx=pending_cols,
        band=band,
        block_h=block_h,
        block_w=block_w,
        nodata=nodata,
        fallback_radius_pixels=fallback_radius_pixels,
        offsets=offsets,
        superblock_size=superblock_size,
    )
    for out in outputs:
        writer.writerow(out)


def stream_memory_mode(
    ds,
    args,
    lon_col,
    lat_col,
    out_fields,
    block_h,
    block_w,
    nodata,
    offsets,
):
    with open(args.input, "r", newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        with open(args.output, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()

            total = 0
            for chunk in iter_csv_chunks(reader, args.chunk_size):
                process_chunk_memory(
                    ds=ds,
                    writer=writer,
                    chunk_rows=chunk,
                    lon_col=lon_col,
                    lat_col=lat_col,
                    band=args.band,
                    block_h=block_h,
                    block_w=block_w,
                    nodata=nodata,
                    fallback_radius_pixels=args.fallback_radius_pixels,
                    offsets=offsets,
                    superblock_size=args.superblock_size,
                )
                total += len(chunk)
                log(f"processed {total} rows")


def spatial_shard_key(r: int, c: int, block_h: int, block_w: int, shard_superblock_size: int) -> str:
    sbh = max(1, shard_superblock_size) * block_h
    sbw = max(1, shard_superblock_size) * block_w
    sr = r // sbh
    sc = c // sbw
    return f"sr{int(sr):08d}_sc{int(sc):08d}"


def build_spatial_shards(
    ds,
    args,
    lon_col,
    lat_col,
    out_fields,
    block_h,
    block_w,
    row_bucket_writers: BucketWriterCache,
    shard_writers: ShardWriterCache,
) -> int:
    shard_fieldnames = ["__rowid__", "__r__", "__c__"] + list(out_fields[:-len(OUT_EXTRA_FIELDS)])
    if shard_writers.fieldnames != shard_fieldnames:
        raise RuntimeError("Shard writer fieldnames mismatch")

    total_rows = 0
    valid_for_shards = 0

    with open(args.input, "r", newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        rowid_base = 0

        for chunk in iter_csv_chunks(reader, args.chunk_size):
            outputs, pending_indices, pending_rows, pending_cols = prepare_chunk_rows(ds, chunk, lon_col, lat_col)

            spatial_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)

            pending_lookup = {int(idx): (int(r), int(c)) for idx, r, c in zip(pending_indices.tolist(), pending_rows.tolist(), pending_cols.tolist())}

            for local_i, out in enumerate(outputs):
                rowid = rowid_base + local_i
                bucket_id = rowid // args.merge_bucket_rows
                if local_i not in pending_lookup:
                    final_row = {"__rowid__": str(rowid)}
                    final_row.update(out)
                    row_bucket_writers.write(bucket_id, final_row)
                    continue

                r, c = pending_lookup[local_i]
                shard_key = spatial_shard_key(r, c, block_h, block_w, args.shard_superblock_size)
                shard_row = {
                    "__rowid__": str(rowid),
                    "__r__": str(r),
                    "__c__": str(c),
                }
                for name in out_fields[:-len(OUT_EXTRA_FIELDS)]:
                    shard_row[name] = out.get(name, "")
                spatial_groups[shard_key].append(shard_row)
                valid_for_shards += 1

            for shard_key, rows in spatial_groups.items():
                shard_writers.write_many(shard_key, rows)

            rowid_base += len(chunk)
            total_rows += len(chunk)
            log(f"shard pass assigned {total_rows} rows")

    return valid_for_shards


def process_one_shard(
    ds,
    shard_path: str,
    out_fields,
    band,
    block_h,
    block_w,
    nodata,
    fallback_radius_pixels,
    offsets,
    superblock_size,
    row_bucket_writers: BucketWriterCache,
    merge_bucket_rows: int,
):
    with open(shard_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if not records:
        return 0

    outputs = [make_output_row({name: rec.get(name, "") for name in out_fields[:-len(OUT_EXTRA_FIELDS)]}) for rec in records]
    pending_indices = np.arange(len(records), dtype=np.int64)
    pending_rows = np.asarray([int(rec["__r__"]) for rec in records], dtype=np.int64)
    pending_cols = np.asarray([int(rec["__c__"]) for rec in records], dtype=np.int64)

    for i, out in enumerate(outputs):
        out["twi_in_bounds"] = "1"
        out["twi_row"] = records[i]["__r__"]
        out["twi_col"] = records[i]["__c__"]

    apply_samples_to_outputs(
        ds=ds,
        outputs=outputs,
        pending_indices=pending_indices,
        rows_idx=pending_rows,
        cols_idx=pending_cols,
        band=band,
        block_h=block_h,
        block_w=block_w,
        nodata=nodata,
        fallback_radius_pixels=fallback_radius_pixels,
        offsets=offsets,
        superblock_size=superblock_size,
    )

    for rec, out in zip(records, outputs):
        rowid = int(rec["__rowid__"])
        bucket_id = rowid // merge_bucket_rows
        final_row = {"__rowid__": str(rowid)}
        final_row.update(out)
        row_bucket_writers.write(bucket_id, final_row)

    return len(records)


def merge_row_buckets(bucket_paths: Sequence[str], output_path: str, out_fields: Sequence[str]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(out_fields))
        writer.writeheader()

        for bucket_path in sorted(bucket_paths):
            with open(bucket_path, "r", newline="", encoding="utf-8-sig") as f_in:
                reader = csv.DictReader(f_in)
                rows = list(reader)
            rows.sort(key=lambda row: int(row["__rowid__"]))
            for row in rows:
                out = {name: row.get(name, "") for name in out_fields}
                writer.writerow(out)


def run_shard_mode(
    ds,
    args,
    lon_col,
    lat_col,
    out_fields,
    block_h,
    block_w,
    nodata,
    offsets,
):
    temp_root = tempfile.mkdtemp(prefix="twi_sample_")
    shard_dir = os.path.join(temp_root, "spatial_shards")
    bucket_dir = os.path.join(temp_root, "row_buckets")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(bucket_dir, exist_ok=True)

    shard_fieldnames = ["__rowid__", "__r__", "__c__"] + list(out_fields[:-len(OUT_EXTRA_FIELDS)])
    bucket_fieldnames = ["__rowid__"] + list(out_fields)

    shard_writers = ShardWriterCache(shard_dir, shard_fieldnames)
    row_bucket_writers = BucketWriterCache(bucket_dir, bucket_fieldnames)

    try:
        valid_for_shards = build_spatial_shards(
            ds=ds,
            args=args,
            lon_col=lon_col,
            lat_col=lat_col,
            out_fields=out_fields,
            block_h=block_h,
            block_w=block_w,
            row_bucket_writers=row_bucket_writers,
            shard_writers=shard_writers,
        )
        shard_writers.close()

        shard_paths = shard_writers.shard_paths()
        log(f"built {len(shard_paths)} spatial shard files for {valid_for_shards} in-bounds rows")

        processed = 0
        for shard_path in shard_paths:
            processed += process_one_shard(
                ds=ds,
                shard_path=shard_path,
                out_fields=out_fields,
                band=args.band,
                block_h=block_h,
                block_w=block_w,
                nodata=nodata,
                fallback_radius_pixels=args.fallback_radius_pixels,
                offsets=offsets,
                superblock_size=args.superblock_size,
                row_bucket_writers=row_bucket_writers,
                merge_bucket_rows=args.merge_bucket_rows,
            )
            log(f"sampled {processed} shard rows")

        row_bucket_writers.close()
        merge_row_buckets(row_bucket_writers.bucket_paths(), args.output, out_fields)
    finally:
        try:
            shard_writers.close()
        except Exception:
            pass
        try:
            row_bucket_writers.close()
        except Exception:
            pass
        if args.keep_temp:
            log(f"kept temp dir: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True, help="Path to TWI GeoTIFF/COG")
    ap.add_argument("--input", required=True, help="Input CSV with lon/lat columns")
    ap.add_argument("--output", required=True, help="Output CSV with sampled values")
    ap.add_argument("--lon-col", default="", help="Longitude column name")
    ap.add_argument("--lat-col", default="", help="Latitude column name")
    ap.add_argument("--band", type=int, default=1, help="Raster band to sample")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=250000,
        help="Rows per chunk to keep memory bounded",
    )
    ap.add_argument(
        "--fallback-radius-pixels",
        type=int,
        default=3,
        help="If the exact pixel is nodata, search nearest valid neighbors within this pixel radius",
    )
    ap.add_argument(
        "--superblock-size",
        type=int,
        default=4,
        help="Read groups of this many raster blocks at once",
    )
    ap.add_argument(
        "--mode",
        choices=["auto", "memory", "shard"],
        default="auto",
        help="Use in-memory chunking for smaller jobs and coarse spatial sharding for larger jobs",
    )
    ap.add_argument(
        "--shard-threshold-points",
        type=int,
        default=400000,
        help="Auto mode switches to shard mode at or above this many input rows",
    )
    ap.add_argument(
        "--shard-superblock-size",
        type=int,
        default=16,
        help="Coarse spatial shard size in raster blocks",
    )
    ap.add_argument(
        "--merge-bucket-rows",
        type=int,
        default=200000,
        help="Number of original input rows per final merge bucket in shard mode",
    )
    ap.add_argument(
        "--gdal-cache-mb",
        type=int,
        default=1024,
        help="GDAL block cache size in MB",
    )
    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep shard temp files for debugging",
    )
    args = ap.parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.fallback_radius_pixels < 0:
        raise ValueError("--fallback-radius-pixels must be >= 0")
    if args.superblock_size < 1:
        raise ValueError("--superblock-size must be >= 1")
    if args.shard_superblock_size < 1:
        raise ValueError("--shard-superblock-size must be >= 1")
    if args.merge_bucket_rows < 1:
        raise ValueError("--merge-bucket-rows must be >= 1")
    if args.shard_threshold_points < 1:
        raise ValueError("--shard-threshold-points must be >= 1")

    with open(args.input, "r", newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row")
        input_fieldnames = list(reader.fieldnames)

    lon_col = detect_column(input_fieldnames, args.lon_col, DEFAULT_LON_NAMES)
    lat_col = detect_column(input_fieldnames, args.lat_col, DEFAULT_LAT_NAMES)
    out_fields = unique_fieldnames(input_fieldnames, OUT_EXTRA_FIELDS)

    total_rows = None
    mode = args.mode
    if mode == "auto":
        total_rows = count_input_rows(args.input)
        mode = "shard" if total_rows >= args.shard_threshold_points else "memory"
        log(f"auto mode selected {mode} for {total_rows} rows")

    offsets = neighbor_offsets(args.fallback_radius_pixels) if args.fallback_radius_pixels > 0 else []

    env_kwargs = {
        "GDAL_CACHEMAX": int(args.gdal_cache_mb),
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
    }

    with rasterio.Env(**env_kwargs):
        with rasterio.open(args.tif) as ds:
            if args.band < 1 or args.band > ds.count:
                raise ValueError(f"--band must be between 1 and {ds.count}")

            block_h, block_w = get_block_shape(ds)
            nodata = ds.nodata

            if mode == "memory":
                stream_memory_mode(
                    ds=ds,
                    args=args,
                    lon_col=lon_col,
                    lat_col=lat_col,
                    out_fields=out_fields,
                    block_h=block_h,
                    block_w=block_w,
                    nodata=nodata,
                    offsets=offsets,
                )
            else:
                run_shard_mode(
                    ds=ds,
                    args=args,
                    lon_col=lon_col,
                    lat_col=lat_col,
                    out_fields=out_fields,
                    block_h=block_h,
                    block_w=block_w,
                    nodata=nodata,
                    offsets=offsets,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
