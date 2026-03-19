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


RX_FILE = re.compile(r"^TerraClimate_(?P<var>[A-Za-z0-9]+)_(?P<year>\d{4})\.nc$", re.IGNORECASE)


@dataclass(frozen=True)
class Job:
    nc_path: Path
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


def list_jobs(in_dir: Path, only_vars: Optional[set]) -> List[Job]:
    jobs: List[Job] = []
    for p in sorted(in_dir.glob("*.nc")):
        m = RX_FILE.match(p.name)
        if not m:
            continue
        var = m.group("var").lower()
        year = int(m.group("year"))
        if only_vars and var not in only_vars:
            continue
        jobs.append(Job(nc_path=p, var=var, year=year))
    return jobs


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
        # Common TerraClimate case: GDAL opens the single raster variable directly.
        return str(nc_path)

    want = f":{var}".lower()
    for name in sub_names:
        if name.lower().endswith(want):
            return name

    return sub_names[0]


def convert_one(job: Job, out_root: Path, driver_path: str, zstd_level: int, force: bool) -> Tuple[Path, str]:
    out_dir = out_root / job.var
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job.var}_{job.year:04d}_global_12band.cog.tif"

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return out_path, "skip"

    cfg: List[str] = []
    if driver_path:
        cfg = ["--config", "GDAL_DRIVER_PATH", driver_path]

    src = dataset_for_var(job.nc_path, job.var, driver_path)

    run_cmd(
        [
            "gdal_translate",
            *cfg,
            src,
            str(out_path),
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

    return out_path, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk convert TerraClimate yearly NetCDFs to global 12-band COGs.")
    ap.add_argument("in_dir", help="Folder containing TerraClimate_*.nc")
    ap.add_argument("out_dir", help="Output folder for COGs")
    ap.add_argument("--vars", default="", help="Comma list like aet,ppt,tmax (default: all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gdal-driver-path", default=os.environ.get("GDAL_DRIVER_PATH", ""))
    args = ap.parse_args()

    for t in ("gdalinfo", "gdal_translate"):
        if not have_tool(t):
            print(f"Missing GDAL tool on PATH: {t}", file=sys.stderr)
            return 2

    only_vars = None
    if args.vars.strip():
        only_vars = {v.strip().lower() for v in args.vars.split(",") if v.strip()}

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = list_jobs(in_dir, only_vars)
    if not jobs:
        print("No matching TerraClimate_*.nc files found.", file=sys.stderr)
        return 1

    total = len(jobs)
    print(f"jobs: {total}, workers: {args.workers}")
    print(f"GDAL_DRIVER_PATH: {args.gdal_driver_path or '(empty)'}")
    print()

    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(convert_one, j, out_dir, args.gdal_driver_path, args.zstd_level, args.force) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out_path, st = fut.result()
                if st == "ok":
                    ok += 1
                else:
                    skip += 1
                print(f"[{i}/{total}] {st}: {out_path.name}")
            except Exception as e:
                err += 1
                print(f"[{i}/{total}] ERROR: {e}", file=sys.stderr)

    print(f"\nok={ok} skip={skip} err={err}")
    return 0 if err == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())