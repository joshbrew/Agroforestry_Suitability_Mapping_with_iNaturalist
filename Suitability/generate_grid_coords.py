#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


"""
python generate_grid_coords.py --geojson D:\Oregon_Suitability\oregon.geojson --out D:\Oregon_Suitability\oregon_grid_250m.csv --resolution-m 250 --include-projected-cols --include-grid-index
"""

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate a regular point grid inside a GeoJSON boundary at a specified meter resolution."
    )
    ap.add_argument("--geojson", required=True, help="Input GeoJSON boundary file")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument(
        "--resolution-m",
        type=float,
        required=True,
        help="Grid spacing in meters",
    )
    ap.add_argument(
        "--work-crs",
        default="EPSG:5070",
        help="Projected CRS used to build the meter grid (default: EPSG:5070)",
    )
    ap.add_argument(
        "--target-crs",
        default="EPSG:4326",
        help="Output coordinate CRS for lon/lat columns (default: EPSG:4326)",
    )
    ap.add_argument(
        "--input-crs",
        default=None,
        help="Override input CRS if the GeoJSON has no CRS metadata",
    )
    ap.add_argument(
        "--id-prefix",
        default="p",
        help="Prefix for output point ids (default: p)",
    )
    ap.add_argument(
        "--include-projected-cols",
        action="store_true",
        help="Also write x_m,y_m columns in the working projected CRS",
    )
    ap.add_argument(
        "--include-grid-index",
        action="store_true",
        help="Also write ix,iy grid index columns",
    )
    return ap.parse_args()


def frange_aligned(min_v: float, max_v: float, step: float):
    start = math.floor(min_v / step) * step + (step * 0.5)
    vals = []
    v = start
    eps = step * 1e-9
    while v <= max_v + eps:
        vals.append(v)
        v += step
    return np.asarray(vals, dtype=np.float64)


def main():
    args = parse_args()

    if args.resolution_m <= 0:
        raise SystemExit("--resolution-m must be > 0")

    in_path = Path(args.geojson)
    out_path = Path(args.out)

    gdf = gpd.read_file(in_path)
    if gdf.empty:
        raise SystemExit(f"No features found in {in_path}")

    if gdf.crs is None:
        if not args.input_crs:
            raise SystemExit(
                "Input GeoJSON has no CRS. Pass --input-crs, or ensure the file has CRS metadata."
            )
        gdf = gdf.set_crs(args.input_crs)

    work_crs = CRS.from_user_input(args.work_crs)
    target_crs = CRS.from_user_input(args.target_crs)

    gdf = gdf.to_crs(work_crs)

    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    if geom.is_empty:
        raise SystemExit("Boundary geometry is empty after loading/reprojection")

    prepared = prep(geom)
    minx, miny, maxx, maxy = geom.bounds

    xs = frange_aligned(minx, maxx, args.resolution_m)
    ys = frange_aligned(miny, maxy, args.resolution_m)

    accepted_x = []
    accepted_y = []
    accepted_ix = []
    accepted_iy = []

    total_candidates = 0
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            total_candidates += 1
            pt = Point(float(x), float(y))
            if prepared.intersects(pt):
                accepted_x.append(float(x))
                accepted_y.append(float(y))
                accepted_ix.append(ix)
                accepted_iy.append(iy)

    if not accepted_x:
        raise SystemExit("No grid points fell inside the boundary. Try a smaller resolution.")

    transformer = Transformer.from_crs(work_crs, target_crs, always_xy=True)
    out_lon, out_lat = transformer.transform(accepted_x, accepted_y)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "lon", "lat"]
    if args.include_protected_cols if False else False:
        pass
    if args.include_projected_cols:
        fieldnames.extend(["x_m", "y_m"])
    if args.include_grid_index:
        fieldnames.extend(["ix", "iy"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i in range(len(accepted_x)):
            row = {
                "id": f"{args.id_prefix}{i}",
                "lon": f"{float(out_lon[i]):.8f}",
                "lat": f"{float(out_lat[i]):.8f}",
            }
            if args.include_projected_cols:
                row["x_m"] = f"{accepted_x[i]:.3f}"
                row["y_m"] = f"{accepted_y[i]:.3f}"
            if args.include_grid_index:
                row["ix"] = int(accepted_ix[i])
                row["iy"] = int(accepted_iy[i])
            w.writerow(row)

    print(
        f"Wrote {len(accepted_x):,} points to {out_path} "
        f"from {total_candidates:,} candidates at {args.resolution_m:g} m spacing"
    )


if __name__ == "__main__":
    main()