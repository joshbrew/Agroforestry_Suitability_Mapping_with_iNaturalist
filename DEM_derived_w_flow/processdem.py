#!/usr/bin/env python3
import os
import math
import argparse
import tempfile
import subprocess
from shutil import which
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from rasterio.windows import Window


def abspath(p: str) -> str:
  return os.path.abspath(os.path.expanduser(p))


def run(cmd: List[str]) -> str:
  p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  if p.returncode != 0:
    raise RuntimeError(
      "Command failed:\n"
      + " ".join(cmd)
      + "\n\nSTDOUT:\n"
      + p.stdout
      + "\n\nSTDERR:\n"
      + p.stderr
    )
  return p.stdout


def have(name: str) -> bool:
  return which(name) is not None


def require_gdal():
  missing = []
  for exe in ["gdalwarp", "gdaldem", "gdalinfo"]:
    if not have(exe):
      missing.append(exe)
  if missing:
    raise RuntimeError("Missing GDAL CLI on PATH: " + ", ".join(missing))


def list_dem_subfolders(dem_root: str, only: Optional[str]) -> List[str]:
  dem_root = abspath(dem_root)
  names = []
  for n in sorted(os.listdir(dem_root)):
    p = os.path.join(dem_root, n)
    if os.path.isdir(p) and n.lower().endswith("_dem_3s"):
      names.append(n)
  if only:
    wanted = set([x.strip() for x in only.split(",") if x.strip()])
    names = [n for n in names if n in wanted]
  if not names:
    raise RuntimeError(f"No *_dem_3s subfolders found under {dem_root}")
  return names


def pick_tif(folder_path: str, folder_name: str) -> str:
  folder_path = abspath(folder_path)
  exact = os.path.join(folder_path, f"{folder_name}.tif")
  if os.path.exists(exact):
    return exact
  exact2 = os.path.join(folder_path, f"{folder_name}.tiff")
  if os.path.exists(exact2):
    return exact2
  cands = []
  for f in os.listdir(folder_path):
    fl = f.lower()
    if fl.endswith(".tif") or fl.endswith(".tiff"):
      cands.append(os.path.join(folder_path, f))
  if not cands:
    raise RuntimeError(f"No .tif found in {folder_path}")
  cands.sort(key=lambda p: os.path.getsize(p), reverse=True)
  return cands[0]


def raster_bounds_wgs84(src_path: str) -> Tuple[float, float, float, float]:
  with rasterio.open(src_path) as ds:
    b = ds.bounds
    return (float(b.left), float(b.bottom), float(b.right), float(b.top))


def align_down(x: float, step: float) -> float:
  return math.floor(x / step) * step


def align_up(x: float, step: float) -> float:
  return math.ceil(x / step) * step


def gdal_extract_tile(src: str, dst: str, te: Tuple[float, float, float, float]) -> None:
  minx, miny, maxx, maxy = te
  cmd = [
    "gdalwarp",
    "-overwrite",
    "-multi",
    "-wo", "NUM_THREADS=ALL_CPUS",
    "-te", str(minx), str(miny), str(maxx), str(maxy),
    "-r", "bilinear",
    "-co", "TILED=YES",
    "-co", "COMPRESS=ZSTD",
    "-co", "PREDICTOR=2",
    "-co", "BIGTIFF=YES",
    src, dst
  ]
  run(cmd)


def gdaldem_slope(src: str, dst: str) -> None:
  cmd = [
    "gdaldem", "slope",
    src, dst,
    "-s", "111120",     # approx meters per degree (lat/lon). good enough for 3s; Oregon pass later can be projected
    "-compute_edges",
    "-co", "TILED=YES",
    "-co", "COMPRESS=ZSTD",
    "-co", "PREDICTOR=2",
    "-co", "BIGTIFF=YES",
  ]
  run(cmd)


def gdaldem_aspect(src: str, dst: str) -> None:
  cmd = [
    "gdaldem", "aspect",
    src, dst,
    "-compute_edges",
    "-co", "TILED=YES",
    "-co", "COMPRESS=ZSTD",
    "-co", "PREDICTOR=2",
    "-co", "BIGTIFF=YES",
  ]
  run(cmd)


def window_iter(width: int, height: int, chunk: int):
  for row0 in range(0, height, chunk):
    h = min(chunk, height - row0)
    for col0 in range(0, width, chunk):
      w = min(chunk, width - col0)
      yield Window(col0, row0, w, h)


def make_like(src_ds, dtype, nodata):
  profile = src_ds.profile.copy()
  profile.update(
    dtype=dtype,
    nodata=nodata,
    count=1,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    bigtiff="YES",
    compress="ZSTD",
    predictor=2,
  )
  return profile


def compute_north_east_from_aspect(aspect_path: str, north_out: str, east_out: str, chunk: int):
  aspect_path = abspath(aspect_path)
  north_out = abspath(north_out)
  east_out = abspath(east_out)

  with rasterio.open(aspect_path) as src:
    profile_n = make_like(src, dtype="float32", nodata=np.nan)
    profile_e = make_like(src, dtype="float32", nodata=np.nan)

    with rasterio.open(north_out, "w", **profile_n) as dst_n, rasterio.open(east_out, "w", **profile_e) as dst_e:
      src_nodata = src.nodata
      for win in window_iter(src.width, src.height, chunk):
        a = src.read(1, window=win).astype(np.float32)
        mask = np.zeros_like(a, dtype=bool)
        if src_nodata is not None:
          mask |= (a == src_nodata)

        a_rad = np.deg2rad(a)
        north = np.cos(a_rad).astype(np.float32)
        east = np.sin(a_rad).astype(np.float32)

        if mask.any():
          north[mask] = np.nan
          east[mask] = np.nan

        dst_n.write(north, 1, window=win)
        dst_e.write(east, 1, window=win)


def compute_twi_from_slope_and_aca(
  slope_deg_path: str,
  upstream_area_path: str,
  out_path: str,
  chunk: int,
):
  slope_deg_path = abspath(slope_deg_path)
  upstream_area_path = abspath(upstream_area_path)
  out_path = abspath(out_path)

  with rasterio.open(slope_deg_path) as sds, rasterio.open(upstream_area_path) as ads:
    if sds.width != ads.width or sds.height != ads.height or sds.transform != ads.transform:
      raise RuntimeError("slope and upstream area rasters must match shape/transform")

    prof = make_like(sds, dtype="float32", nodata=np.nan)
    s_nodata = sds.nodata
    a_nodata = ads.nodata

    with rasterio.open(out_path, "w", **prof) as out_ds:
      for win in window_iter(sds.width, sds.height, chunk):
        slope_deg = sds.read(1, window=win).astype(np.float32)
        area = ads.read(1, window=win).astype(np.float32)

        mask = np.zeros_like(slope_deg, dtype=bool)
        if s_nodata is not None:
          mask |= (slope_deg == s_nodata)
        if a_nodata is not None:
          mask |= (area == a_nodata)

        slope_rad = np.deg2rad(slope_deg)
        tan_s = np.maximum(np.tan(slope_rad), 1.0e-6)

        # upstream area should be in area units already (ACA preferred)
        a = np.maximum(area, 0.0) + 1.0
        twi = np.log(a / tan_s).astype(np.float32)

        if mask.any():
          twi[mask] = np.nan

        out_ds.write(twi, 1, window=win)


def process_continent(
  dem_path: str,
  out_dir: str,
  tile_deg: float,
  chunk: int,
  do_twi: bool,
  aca_path: Optional[str],
  skip_existing: bool,
):
  dem_path = abspath(dem_path)
  out_dir = abspath(out_dir)
  os.makedirs(out_dir, exist_ok=True)

  left, bottom, right, top = raster_bounds_wgs84(dem_path)
  x0 = align_down(left, tile_deg)
  y0 = align_down(bottom, tile_deg)
  x1 = align_up(right, tile_deg)
  y1 = align_up(top, tile_deg)

  xs = np.arange(x0, x1, tile_deg, dtype=np.float64)
  ys = np.arange(y0, y1, tile_deg, dtype=np.float64)

  for j, y in enumerate(ys):
    for i, x in enumerate(xs):
      te = (float(x), float(y), float(x + tile_deg), float(y + tile_deg))
      tag = f"x{int(round(x*100)):d}_y{int(round(y*100)):d}"
      tile_base = os.path.join(out_dir, tag)

      tile_dem = tile_base + "_dem.tif"
      tile_slope = tile_base + "_slope_deg.tif"
      tile_aspect = tile_base + "_aspect_deg.tif"
      tile_north = tile_base + "_northness.tif"
      tile_east = tile_base + "_eastness.tif"
      tile_twi = tile_base + "_twi.tif"

      if skip_existing and os.path.exists(tile_east) and (not do_twi or os.path.exists(tile_twi)):
        continue

      with tempfile.TemporaryDirectory() as tmp:
        tmp_dem = os.path.join(tmp, "dem.tif")
        gdal_extract_tile(dem_path, tmp_dem, te)

        if not os.path.exists(tmp_dem) or os.path.getsize(tmp_dem) < 1024:
          continue

        gdaldem_slope(tmp_dem, tile_slope)
        gdaldem_aspect(tmp_dem, tile_aspect)
        compute_north_east_from_aspect(tile_aspect, tile_north, tile_east, chunk=chunk)

        if do_twi:
          if not aca_path:
            raise RuntimeError("--twi requires --aca path (upstream area raster)")
          # Extract matching upstream-area tile to temp and compute TWI from it
          tmp_aca = os.path.join(tmp, "aca.tif")
          gdal_extract_tile(abspath(aca_path), tmp_aca, te)
          compute_twi_from_slope_and_aca(tile_slope, tmp_aca, tile_twi, chunk=chunk)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dem-root", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--subfolders", default=None)
  ap.add_argument("--tile-deg", type=float, default=5.0, help="Tile size in degrees (e.g. 5 or 10).")
  ap.add_argument("--chunk", type=int, default=1024)
  ap.add_argument("--skip-existing", action="store_true")
  ap.add_argument("--twi", action="store_true", help="Compute TWI (requires --aca)")
  ap.add_argument("--aca", default=None, help="Upstream area raster (ACA) aligned to DEM (HydroSHEDS preferred).")
  args = ap.parse_args()

  require_gdal()

  dem_root = abspath(args.dem_root)
  out_root = abspath(args.out)
  os.makedirs(out_root, exist_ok=True)

  subs = list_dem_subfolders(dem_root, only=args.subfolders)
  print("Will process subfolders:")
  for n in subs:
    print("  -", n)

  for n in subs:
    folder_path = os.path.join(dem_root, n)
    dem_path = pick_tif(folder_path, n)
    out_dir = os.path.join(out_root, n)

    print("\n==============================")
    print("Processing:", n)
    print("DEM:", dem_path)
    print("Output:", out_dir)

    process_continent(
      dem_path=dem_path,
      out_dir=out_dir,
      tile_deg=float(args.tile_deg),
      chunk=int(args.chunk),
      do_twi=bool(args.twi),
      aca_path=args.aca,
      skip_existing=bool(args.skip_existing),
    )

  print("\nAll done.")


if __name__ == "__main__":
  main()