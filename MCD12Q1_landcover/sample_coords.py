#!/usr/bin/env python3
import argparse
import csv
import math
from itertools import groupby
from pathlib import Path

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform

# Example call with your VRT:
# py sample_coords_neighbor.py mcd12q1_lc_type1.vrt coords.csv coords_sampled.csv --x-col lon --y-col lat --value-col mcd12q1

# If your raster uses 0 as no-data:
# py sample_coords_neighbor.py mcd12q1_lc_type1.vrt coords.csv coords_sampled.csv --x-col lon --y-col lat --value-col mcd12q1 --nodata-id 0

# With nearest-neighbor fallback around nodata cells:
# py sample_coords_neighbor.py mcd12q1_lc_type1.vrt coords.csv coords_sampled.csv --x-col lon --y-col lat --value-col mcd12q1 --fallback-radius-pixels 1


def load_lookup(path, id_field):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lookup CSV not found: {path}")

    lookup = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(id_field)
            if raw is None or raw == "":
                continue
            try:
                key = int(float(raw))
            except ValueError:
                continue
            lookup[key] = row
    return lookup


def batched_rows(reader, batch_size):
    batch = []
    for row in reader:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def values_equal(a, b):
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


def is_missing_value(value, src_nodata, user_nodata):
    if user_nodata is not None and values_equal(value, user_nodata):
        return True
    if src_nodata is not None and values_equal(value, src_nodata):
        return True
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def format_cell(cell):
    try:
        fcell = float(cell)
        if math.isnan(fcell):
            return ""
        if fcell.is_integer():
            return int(fcell)
        return f"{fcell:.10g}"
    except Exception:
        return cell


def get_block_shape(src):
    try:
        if src.block_shapes and src.block_shapes[0]:
            block_h, block_w = src.block_shapes[0]
            if block_h > 0 and block_w > 0:
                return int(block_h), int(block_w)
    except Exception:
        pass
    return 512, 512


def neighbor_offsets(radius):
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            dist2 = dr * dr + dc * dc
            offsets.append((dist2, abs(dr) + abs(dc), dr, dc))
    offsets.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [(dr, dc) for _, _, dr, dc in offsets]


def sample_with_fallback(arr, r, c, row_off, col_off, src_height, src_width, src_nodata, user_nodata, offsets):
    local_r = r - row_off
    local_c = c - col_off
    cell = arr[local_r, local_c]
    if not is_missing_value(cell, src_nodata, user_nodata):
        return cell

    for dr, dc in offsets:
        rr = r + dr
        cc = c + dc
        if rr < 0 or cc < 0 or rr >= src_height or cc >= src_width:
            continue
        cell = arr[rr - row_off, cc - col_off]
        if not is_missing_value(cell, src_nodata, user_nodata):
            return cell

    return None


def main():
    ap = argparse.ArgumentParser(
        description="Sample a raster or VRT at point coordinates in a CSV and append the sampled value to a new CSV."
    )
    ap.add_argument("raster", help="Input raster or VRT path")
    ap.add_argument("coords_csv", help="Input CSV containing coordinates")
    ap.add_argument("out_csv", help="Output CSV path")
    ap.add_argument("--lookup-csv", help="Optional lookup CSV keyed by sampled raster value")
    ap.add_argument("--x-col", default="lon", help="X/longitude column name")
    ap.add_argument("--y-col", default="lat", help="Y/latitude column name")
    ap.add_argument(
        "--input-crs",
        default="EPSG:4326",
        help="CRS of input coordinates, for example EPSG:4326",
    )
    ap.add_argument(
        "--lookup-id-col",
        default="id",
        help="ID column in the lookup CSV corresponding to the sampled raster value",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Rows to process per batch",
    )
    ap.add_argument(
        "--nodata-id",
        type=float,
        default=None,
        help="Treat this raster value as empty/no hit",
    )
    ap.add_argument(
        "--band",
        type=int,
        default=1,
        help="Raster band to sample",
    )
    ap.add_argument(
        "--value-col",
        default="sample_value",
        help="Name of the appended sampled value column",
    )
    ap.add_argument(
        "--hit-col",
        default="sample_found",
        help="Name of the appended hit flag column",
    )
    ap.add_argument(
        "--fallback-radius-pixels",
        type=int,
        default=3,
        help="If the exact pixel is nodata, search nearest valid neighbors within this pixel radius",
    )
    args = ap.parse_args()

    raster_path = Path(args.raster)
    coords_path = Path(args.coords_csv)
    out_path = Path(args.out_csv)

    if not raster_path.exists():
        raise SystemExit(f"Raster not found: {raster_path}")
    if not coords_path.exists():
        raise SystemExit(f"Coordinate CSV not found: {coords_path}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.fallback_radius_pixels < 0:
        raise SystemExit("--fallback-radius-pixels must be >= 0")

    lookup = load_lookup(args.lookup_csv, args.lookup_id_col) if args.lookup_csv else None
    offsets = neighbor_offsets(args.fallback_radius_pixels) if args.fallback_radius_pixels > 0 else []

    with rasterio.open(raster_path) as src, coords_path.open("r", newline="", encoding="utf-8-sig") as fin:
        if src.crs is None:
            raise SystemExit(f"Raster/VRT has no CRS: {raster_path}")
        if args.band < 1 or args.band > src.count:
            raise SystemExit(f"Requested band {args.band} is out of range. Raster has {src.count} band(s).")

        block_h, block_w = get_block_shape(src)

        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header row")
        if args.x_col not in reader.fieldnames:
            raise SystemExit(f"Missing x column: {args.x_col}")
        if args.y_col not in reader.fieldnames:
            raise SystemExit(f"Missing y column: {args.y_col}")

        extra_cols = [args.value_col, args.hit_col]
        lookup_fields = []
        if lookup:
            sample_lookup = next(iter(lookup.values())) if lookup else None
            if sample_lookup:
                lookup_fields = [
                    f"lookup_{k}" for k in sample_lookup.keys() if k != args.lookup_id_col
                ]
                extra_cols.extend(lookup_fields)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=list(reader.fieldnames) + extra_cols)
            writer.writeheader()

            total = 0
            hit_count = 0

            for batch in batched_rows(reader, args.batch_size):
                sampled_values = [None] * len(batch)
                tx = []
                ty = []
                valid_positions = []

                for i, row in enumerate(batch):
                    x = parse_float(row.get(args.x_col))
                    y = parse_float(row.get(args.y_col))
                    if x is None or y is None:
                        continue
                    tx.append(x)
                    ty.append(y)
                    valid_positions.append(i)

                if tx:
                    proj_x, proj_y = transform(args.input_crs, src.crs, tx, ty)

                    pending = []
                    for pos, x, y in zip(valid_positions, proj_x, proj_y):
                        try:
                            r, c = src.index(x, y)
                        except Exception:
                            continue
                        if r < 0 or c < 0 or r >= src.height or c >= src.width:
                            continue
                        pending.append((r // block_h, c // block_w, r, c, pos))

                    pending.sort(key=lambda x: (x[0], x[1]))

                    for (block_row, block_col), group_iter in groupby(pending, key=lambda x: (x[0], x[1])):
                        group_items = list(group_iter)
                        base_row_off = block_row * block_h
                        base_col_off = block_col * block_w

                        row_off = max(0, base_row_off - args.fallback_radius_pixels)
                        col_off = max(0, base_col_off - args.fallback_radius_pixels)
                        row_end = min(src.height, base_row_off + block_h + args.fallback_radius_pixels)
                        col_end = min(src.width, base_col_off + block_w + args.fallback_radius_pixels)

                        window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
                        arr = src.read(args.band, window=window, masked=False)

                        for _, _, r, c, pos in group_items:
                            cell = sample_with_fallback(
                                arr=arr,
                                r=r,
                                c=c,
                                row_off=row_off,
                                col_off=col_off,
                                src_height=src.height,
                                src_width=src.width,
                                src_nodata=src.nodata,
                                user_nodata=args.nodata_id,
                                offsets=offsets,
                            )
                            sampled_values[pos] = cell

                for i, row in enumerate(batch):
                    cell = sampled_values[i]

                    if cell is None:
                        row[args.value_col] = ""
                        row[args.hit_col] = 0
                    else:
                        row[args.value_col] = format_cell(cell)
                        row[args.hit_col] = 1

                    if lookup_fields:
                        for out_name in lookup_fields:
                            row[out_name] = ""

                        if cell is not None:
                            try:
                                key = int(round(float(cell)))
                            except Exception:
                                key = None

                            if key is not None and key in lookup:
                                raw = lookup[key]
                                for k, v in raw.items():
                                    if k == args.lookup_id_col:
                                        continue
                                    row[f"lookup_{k}"] = v

                    writer.writerow(row)
                    total += 1
                    if cell is not None:
                        hit_count += 1

                print(f"processed={total:,} hits={hit_count:,}")

    print(f"\nWrote: {out_path}")
    print(f"Rows processed: {total:,}")
    print(f"Rows with hit: {hit_count:,}")


if __name__ == "__main__":
    main()
