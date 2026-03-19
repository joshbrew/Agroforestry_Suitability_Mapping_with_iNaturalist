#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

"""
python sample_cogs_from_coords.py ^
  --cog-root D:/terraclimate/terraclimate_cogs_global ^
  --coords coords.csv ^
  --vars ppt,tmax,tmin ^
  --year 2018,2019,2020 ^
  --out sampled.csv

python sample_cogs_from_coords.py ^
  --cog-root D:/terraclimate/terraclimate_cogs_global ^
  --coords coords.csv ^
  --vars ppt ^
  --year 2015-2020 ^
  --aggregate mean ^
  --out sampled.csv

"""

import numpy as np

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.transform import rowcol
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import Window
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install rasterio numpy\n\n"
        f"Original error: {e}"
    )

KNOWN_VARS = [
    "aet", "def", "pet", "ppt", "q", "soil", "srad", "swe",
    "tmax", "tmin", "vap", "ws", "vpd", "pdsi",
]

RX_YEAR = re.compile(
    r"^(?P<var>[a-z0-9]+)_(?P<year>\d{4})_global_12band\.cog\.tif$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Point:
    pid: str
    lon: float
    lat: float


def _norm_header(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _is_nan_scalar(x) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def read_points_csv(path: str) -> List[Point]:
    f = sys.stdin if path == "-" else open(path, "r", newline="", encoding="utf-8")
    with f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise SystemExit("Input has no header row. Provide a CSV with headers like: id,lon,lat")

        fields = [_norm_header(x) for x in reader.fieldnames]
        idx = {fields[i]: reader.fieldnames[i] for i in range(len(fields))}

        id_col = None
        for k in ("id", "pid", "pointid", "name"):
            if k in idx:
                id_col = idx[k]
                break

        lon_col = None
        for k in ("lon", "long", "longitude", "x"):
            if k in idx:
                lon_col = idx[k]
                break

        lat_col = None
        for k in ("lat", "latitude", "y"):
            if k in idx:
                lat_col = idx[k]
                break

        if lon_col is None or lat_col is None:
            raise SystemExit(
                f"Could not find lon/lat columns in headers: {reader.fieldnames}\n"
                "Need columns like lon,lat (or longitude,latitude)."
            )

        pts: List[Point] = []
        for i, row in enumerate(reader):
            pid = str(row[id_col]).strip() if id_col and row.get(id_col) not in (None, "") else str(i)
            try:
                lon = float(row[lon_col])
                lat = float(row[lat_col])
            except Exception:
                continue
            if not (math.isfinite(lon) and math.isfinite(lat)):
                continue
            pts.append(Point(pid=pid, lon=lon, lat=lat))

        if not pts:
            raise SystemExit("No valid points parsed from input.")
        return pts


def parse_year_spec(raw: str) -> List[Optional[int]]:
    text = str(raw or "").strip().lower()
    if not text:
        return [None]

    out: List[Optional[int]] = []
    seen = set()

    def add_year(value: Optional[int]) -> None:
        key = "latest" if value is None else int(value)
        if key in seen:
            return
        seen.add(key)
        out.append(value)

    for part in text.split(","):
        token = part.strip().lower()
        if not token:
            continue

        if token == "latest":
            add_year(None)
            continue

        m = re.match(r"^(\d{4})\s*-\s*(\d{4})$", token)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            step = 1 if b >= a else -1
            for y in range(a, b + step, step):
                add_year(y)
            continue

        try:
            add_year(int(token))
        except Exception:
            raise ValueError(
                f"Bad --year token '{token}'. Use a year, comma list, range like 2018-2020, or latest."
            )

    if not out:
        return [None]
    return out


def find_latest_year_for_var(cog_root: Path, var: str) -> Optional[int]:
    vdir = cog_root / var
    if not vdir.exists():
        return None

    best: Optional[int] = None
    for p in vdir.glob(f"{var}_????_global_12band.cog.tif"):
        m = RX_YEAR.match(p.name)
        if not m:
            continue
        y = int(m.group("year"))
        if best is None or y > best:
            best = y
    return best


def resolve_cog_path(cog_root: Path, var: str, year: Optional[int]) -> Tuple[Path, int]:
    if year is None:
        y = find_latest_year_for_var(cog_root, var)
        if y is None:
            raise FileNotFoundError(f"No COGs found for var '{var}' under {cog_root / var}")
        year = y

    p = cog_root / var / f"{var}_{year:04d}_global_12band.cog.tif"
    if not p.exists():
        raise FileNotFoundError(f"Missing COG: {p}")
    return p, year


def _window_value_with_fallback(ds, x: float, y: float, fallback_radius_pixels: int) -> np.ndarray:
    try:
        row, col = rowcol(ds.transform, x, y)
    except Exception:
        return np.full(12, np.nan, dtype=np.float64)

    if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
        return np.full(12, np.nan, dtype=np.float64)

    radius = max(0, int(fallback_radius_pixels))
    r0 = max(0, row - radius)
    c0 = max(0, col - radius)
    r1 = min(ds.height - 1, row + radius)
    c1 = min(ds.width - 1, col + radius)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    arr = ds.read(indexes=list(range(1, 13)), window=Window(c0, r0, w, h), masked=True)
    if arr.ndim != 3 or arr.shape[0] < 12:
        return np.full(12, np.nan, dtype=np.float64)
    arr = arr[:12]

    local_r = row - r0
    local_c = col - c0

    center = arr[:, local_r, local_c]
    if not np.ma.getmaskarray(center).all():
        return np.ma.asarray(center, dtype=np.float64).filled(np.nan)

    if radius == 0:
        return np.full(12, np.nan, dtype=np.float64)

    candidates: List[Tuple[int, int, int]] = []
    for rr in range(h):
        for cc in range(w):
            dr = rr - local_r
            dc = cc - local_c
            if dr == 0 and dc == 0:
                continue
            d2 = dr * dr + dc * dc
            if d2 > radius * radius:
                continue
            candidates.append((d2, abs(dr) + abs(dc), rr * w + cc))

    candidates.sort()

    for _, _, flat_idx in candidates:
        rr = flat_idx // w
        cc = flat_idx % w
        pix = arr[:, rr, cc]
        if not np.ma.getmaskarray(pix).all():
            return np.ma.asarray(pix, dtype=np.float64).filled(np.nan)

    return np.full(12, np.nan, dtype=np.float64)


def sample_var_points(
    cog_path: Path,
    year: int,
    points: Sequence[Point],
    in_crs: str,
    aggregate: str,
    fallback_radius_pixels: int,
) -> Tuple[int, np.ndarray]:
    with rasterio.Env():
        try:
            with rasterio.open(cog_path) as ds:
                xs = np.array([p.lon for p in points], dtype=np.float64)
                ys = np.array([p.lat for p in points], dtype=np.float64)

                ds_crs = ds.crs
                if ds_crs is None:
                    raise RuntimeError(f"Dataset has no CRS: {cog_path}")

                if str(ds_crs).lower() != str(in_crs).lower():
                    tx, ty = rio_transform(in_crs, ds_crs, xs.tolist(), ys.tolist())
                    xs2 = np.asarray(tx, dtype=np.float64)
                    ys2 = np.asarray(ty, dtype=np.float64)
                else:
                    xs2 = xs
                    ys2 = ys

                samp = np.empty((len(points), 12), dtype=np.float64)
                for i in range(len(points)):
                    samp[i, :] = _window_value_with_fallback(
                        ds,
                        float(xs2[i]),
                        float(ys2[i]),
                        fallback_radius_pixels,
                    )
        except RasterioIOError as e:
            raise RuntimeError(f"Failed opening {cog_path}: {e}")

    agg = aggregate.lower().strip()
    if agg == "none":
        return year, samp
    if agg == "mean":
        with np.errstate(invalid="ignore"):
            return year, np.nanmean(samp, axis=1)
    if agg == "sum":
        out = np.nansum(samp, axis=1)
        all_nan = np.isnan(samp).all(axis=1)
        out = out.astype(np.float64, copy=False)
        out[all_nan] = np.nan
        return year, out
    if agg == "min":
        with np.errstate(invalid="ignore"):
            return year, np.nanmin(samp, axis=1)
    if agg == "max":
        with np.errstate(invalid="ignore"):
            return year, np.nanmax(samp, axis=1)
    raise ValueError(f"Unknown aggregate: {aggregate}")


def write_output_csv(
    out_path: str,
    points: Sequence[Point],
    var_results: Dict[str, List[Tuple[int, np.ndarray]]],
    aggregate: str,
) -> None:
    f = sys.stdout if out_path == "-" else open(out_path, "w", newline="", encoding="utf-8")
    with f:
        writer = csv.writer(f)
        agg = aggregate.lower().strip()
        multi_year = any(len(entries) > 1 for entries in var_results.values())

        header = ["id", "lon", "lat"]
        for var, entries in var_results.items():
            if multi_year:
                for year, _vals in entries:
                    if agg == "none":
                        header.extend([f"{var}_{year}_m{m:02d}" for m in range(1, 13)])
                    else:
                        header.append(f"{var}_{year}_{agg}")
            else:
                year, _vals = entries[0]
                header.append(f"{var}_year")
                if agg == "none":
                    header.extend([f"{var}_m{m:02d}" for m in range(1, 13)])
                else:
                    header.append(f"{var}_{agg}")
        writer.writerow(header)

        for i, p in enumerate(points):
            row = [p.pid, f"{p.lon:.8f}", f"{p.lat:.8f}"]
            for var, entries in var_results.items():
                if multi_year:
                    for _year, vals in entries:
                        if agg == "none":
                            v = vals[i]
                            row.extend(["" if _is_nan_scalar(x) else f"{x:.6g}" for x in v.tolist()])
                        else:
                            x = float(vals[i])
                            row.append("" if math.isnan(x) else f"{x:.6g}")
                else:
                    year, vals = entries[0]
                    row.append(str(year))
                    if agg == "none":
                        v = vals[i]
                        row.extend(["" if _is_nan_scalar(x) else f"{x:.6g}" for x in v.tolist()])
                    else:
                        x = float(vals[i])
                        row.append("" if math.isnan(x) else f"{x:.6g}")
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample TerraClimate yearly 12-band COGs at nearest pixel for points.")
    ap.add_argument("--cog-root", required=True, help="Root folder containing <var>/<var>_<year>_global_12band.cog.tif")
    ap.add_argument("--coords", required=True, help="CSV path (or '-' for stdin) with lon/lat columns, optional id.")
    ap.add_argument("--vars", default="all", help="Comma list (e.g. ppt,tmax) or 'all' (default).")
    ap.add_argument("--year", default="latest", help="Year, comma list, range like 2018-2020, or 'latest'.")
    ap.add_argument("--in-crs", default="EPSG:4326", help="Input coordinate CRS (default EPSG:4326 lon/lat).")
    ap.add_argument("--aggregate", default="none", choices=["none", "mean", "sum", "min", "max"], help="If set, reduce 12 months to a single value per var.")
    ap.add_argument("--out", default="-", help="Output CSV path (or '-' for stdout).")
    ap.add_argument("--gdal-cache-mb", type=int, default=512, help="GDAL cache size in MB (default 512).")
    ap.add_argument("--fallback-radius-pixels", type=int, default=3, help="Fallback search radius in pixels when the exact cell is nodata.")
    args = ap.parse_args()

    os.environ["GDAL_CACHEMAX"] = str(max(16, int(args.gdal_cache_mb)))

    cog_root = Path(args.cog_root).resolve()
    if not cog_root.exists():
        print(f"Missing --cog-root folder: {cog_root}", file=sys.stderr)
        return 2

    points = read_points_csv(args.coords)

    if args.vars.strip().lower() == "all":
        vars_sel = KNOWN_VARS
    else:
        vars_sel = [v.strip().lower() for v in args.vars.split(",") if v.strip()]
        bad = [v for v in vars_sel if v not in KNOWN_VARS]
        if bad:
            print(f"Unknown vars: {bad}. Known: {KNOWN_VARS}", file=sys.stderr)
            return 2

    try:
        year_specs = parse_year_spec(args.year)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    var_results: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for var in vars_sel:
        entries: List[Tuple[int, np.ndarray]] = []
        for year_spec in year_specs:
            try:
                cog_path, resolved_year = resolve_cog_path(cog_root, var, year_spec)
                _resolved_year, vals = sample_var_points(
                    cog_path=cog_path,
                    year=resolved_year,
                    points=points,
                    in_crs=args.in_crs,
                    aggregate=args.aggregate,
                    fallback_radius_pixels=args.fallback_radius_pixels,
                )
                entries.append((resolved_year, vals))
                print(f"[ok] {var} {resolved_year}: {cog_path.name}", file=sys.stderr)
            except Exception as e:
                label = "latest" if year_spec is None else str(year_spec)
                print(f"[err] {var} {label}: {e}", file=sys.stderr)
                return 3
        var_results[var] = entries

    write_output_csv(args.out, points, var_results, args.aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())