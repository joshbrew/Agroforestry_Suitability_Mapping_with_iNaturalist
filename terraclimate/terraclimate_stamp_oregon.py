#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


RX_COG = re.compile(r"^(?P<var>[a-z0-9]+)_(?P<year>\d{4})_global_12band\.cog\.tif$", re.IGNORECASE)
RX_NC = re.compile(r"^TerraClimate_(?P<var>[A-Za-z0-9]+)_(?P<year>\d{4})\.nc$", re.IGNORECASE)


@dataclass(frozen=True)
class Inp:
    path: Path
    var: str
    year: int


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: List[str]) -> str:
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


def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    lonmin, lonmax, latmin, latmax = [float(x.strip()) for x in s.split(",")]
    return lonmin, lonmax, latmin, latmax


def list_inputs(mode: str, in_dir: Path, only_vars: Optional[set]) -> List[Inp]:
    out: List[Inp] = []
    if mode == "cogs":
        for p in sorted(in_dir.rglob("*.tif")):
            m = RX_COG.match(p.name)
            if not m:
                continue
            var = m.group("var").lower()
            year = int(m.group("year"))
            if only_vars and var not in only_vars:
                continue
            out.append(Inp(p, var, year))
    else:
        for p in sorted(in_dir.glob("*.nc")):
            m = RX_NC.match(p.name)
            if not m:
                continue
            var = m.group("var").lower()
            year = int(m.group("year"))
            if only_vars and var not in only_vars:
                continue
            out.append(Inp(p, var, year))
    return out


def dataset_for_var(nc_path: Path, var: str, driver_path: str) -> str:
    cfg: List[str] = []
    if driver_path:
        cfg = ["--config", "GDAL_DRIVER_PATH", driver_path]

    txt = run_cmd(["gdalinfo", *cfg, str(nc_path)])

    sub_names: List[str] = []
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("SUBDATASET_") and "_NAME=" in line:
            _, rhs = line.split("=", 1)
            sub_names.append(rhs.strip())

    if not sub_names:
        return str(nc_path)

    want = f":{var}".lower()
    for name in sub_names:
        if name.lower().endswith(want):
            return name

    return sub_names[0]


def stamp_one(
    inp: Inp,
    out_root: Path,
    mode: str,
    bbox: Tuple[float, float, float, float],
    driver_path: str,
    zstd_level: int,
    monthly: bool,
    force: bool,
) -> Tuple[Path, str]:
    lonmin, lonmax, latmin, latmax = bbox

    out_dir = out_root / inp.var
    out_dir.mkdir(parents=True, exist_ok=True)

    year_out = out_dir / f"{inp.var}_{inp.year:04d}_oregon_12band.cog.tif"
    if year_out.exists() and year_out.stat().st_size > 0 and not force and not monthly:
        return year_out, "skip"

    cfg: List[str] = []
    if driver_path:
        cfg = ["--config", "GDAL_DRIVER_PATH", driver_path]

    src = str(inp.path) if mode == "cogs" else dataset_for_var(inp.path, inp.var, driver_path)

    if (not year_out.exists()) or force:
        run_cmd(
            [
                "gdalwarp",
                *cfg,
                "-te",
                str(lonmin),
                str(latmin),
                str(lonmax),
                str(latmax),
                "-t_srs",
                "EPSG:4326",
                "-r",
                "bilinear",
                "-of",
                "COG",
                "-co",
                "COMPRESS=ZSTD",
                "-co",
                f"ZSTD_LEVEL={zstd_level}",
                "-co",
                "RESAMPLING=BILINEAR",
                "-co",
                "BIGTIFF=IF_SAFER",
                "--config",
                "GDAL_NUM_THREADS",
                "ALL_CPUS",
                src,
                str(year_out),
            ]
        )

    if monthly:
        for m in range(1, 13):
            mo_out = out_dir / f"{inp.var}_{inp.year:04d}_{m:02d}_oregon.cog.tif"
            if mo_out.exists() and mo_out.stat().st_size > 0 and not force:
                continue
            run_cmd(
                [
                    "gdal_translate",
                    *cfg,
                    "-b",
                    str(m),
                    str(year_out),
                    str(mo_out),
                    "-of",
                    "COG",
                    "-co",
                    "COMPRESS=ZSTD",
                    "-co",
                    f"ZSTD_LEVEL={zstd_level}",
                    "-co",
                    "RESAMPLING=BILINEAR",
                    "-co",
                    "BIGTIFF=IF_SAFER",
                    "--config",
                    "GDAL_NUM_THREADS",
                    "ALL_CPUS",
                ]
            )

    return year_out, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="Stamp Oregon from TerraClimate (from global COGs or directly from NetCDF).")
    ap.add_argument("--from", dest="mode", choices=["cogs", "nc"], required=True)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--bbox", default="-125,-116,42,46.5", help="lonmin,lonmax,latmin,latmax")
    ap.add_argument("--vars", default="", help="Comma list (default: all)")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gdal-driver-path", default=os.environ.get("GDAL_DRIVER_PATH", ""))
    args = ap.parse_args()

    for t in ("gdalinfo", "gdalwarp", "gdal_translate"):
        if not have_tool(t):
            print(f"Missing GDAL tool on PATH: {t}", file=sys.stderr)
            return 2

    only_vars = None
    if args.vars.strip():
        only_vars = {v.strip().lower() for v in args.vars.split(",") if v.strip()}

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = parse_bbox(args.bbox)

    inputs = list_inputs(args.mode, in_dir, only_vars)
    if not inputs:
        print("No inputs found for that mode/path.", file=sys.stderr)
        return 1

    total = len(inputs)
    print(f"inputs: {total}, workers: {args.workers}, monthly={args.monthly}")
    print(f"bbox: {bbox}")
    print(f"GDAL_DRIVER_PATH: {args.gdal_driver_path or '(empty)'}")
    print()

    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                stamp_one,
                inp,
                out_dir,
                args.mode,
                bbox,
                args.gdal_driver_path,
                args.zstd_level,
                args.monthly,
                args.force,
            )
            for inp in inputs
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out_path, st = fut.result()
                if st == "ok":
                    ok += 1
                else:
                    skip += 1
                print(f"[{i}/{total}] {st}: {out_path}")
            except Exception as e:
                err += 1
                print(f"[{i}/{total}] ERROR: {e}", file=sys.stderr)

    print(f"\nok={ok} skip={skip} err={err}")
    return 0 if err == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())