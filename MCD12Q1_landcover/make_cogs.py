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
from typing import Dict, List, Optional, Tuple

# USAGE, e.g.
# python make_cogs.py MCD12Q1_2024 MCD12Q1_2024_cogs --workers 6 --vrt

@dataclass(frozen=True)
class Job:
    hdf_path: Path
    out_path: Path
    tmp_path: Path


_RX_SUB_NAME = re.compile(r"^SUBDATASET_(\d+)_NAME=(.+)$", re.IGNORECASE)
_RX_SUB_DESC = re.compile(r"^SUBDATASET_(\d+)_DESC=(.+)$", re.IGNORECASE)


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: List[str]) -> None:
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


def file_nonempty(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except FileNotFoundError:
        return False


def gdalinfo_text(hdf_path: Path, driver_path: str) -> str:
    cmd: List[str] = ["gdalinfo"]
    if driver_path:
        cmd += ["--config", "GDAL_DRIVER_PATH", driver_path]
    cmd += [str(hdf_path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"gdalinfo failed for {hdf_path}\n\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}"
        )
    return p.stdout


def parse_subdatasets(txt: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    names: Dict[str, str] = {}
    descs: Dict[str, str] = {}
    for line in txt.splitlines():
        line = line.strip()
        m = _RX_SUB_NAME.match(line)
        if m:
            names[m.group(1)] = m.group(2).strip()
            continue
        m = _RX_SUB_DESC.match(line)
        if m:
            descs[m.group(1)] = m.group(2).strip()
            continue
    return names, descs


def pick_lc_type1_subdataset(names: Dict[str, str], descs: Dict[str, str]) -> Optional[str]:
    if not names:
        return None

    want = ("LC_Type1", "Land_Cover_Type_1", "IGBP")

    for _, v in names.items():
        vv = v.lower()
        if any(t.lower() in vv for t in want):
            return v

    for k, d in descs.items():
        dd = d.lower()
        if any(t.lower() in dd for t in want):
            return names.get(k)

    return None


def gdal_subdataset_lc_type1(hdf_path: Path, driver_path: str) -> Optional[str]:
    txt = gdalinfo_text(hdf_path, driver_path)
    names, descs = parse_subdatasets(txt)
    return pick_lc_type1_subdataset(names, descs)


def make_one_cog(
    job: Job,
    driver_path: str,
    zstd_level: int,
    ov_levels: List[int],
    force: bool,
) -> Tuple[Path, str]:
    if file_nonempty(job.out_path) and not force:
        return (job.out_path, "skip")

    sd = gdal_subdataset_lc_type1(job.hdf_path, driver_path)
    if not sd:
        return (job.out_path, f"warn(no LC_Type1): {job.hdf_path.name}")

    job.out_path.parent.mkdir(parents=True, exist_ok=True)

    if job.tmp_path.exists():
        try:
            job.tmp_path.unlink()
        except OSError:
            pass

    cfg: List[str] = []
    if driver_path:
        cfg = ["--config", "GDAL_DRIVER_PATH", driver_path]

    # 1) Extract LC_Type1 to tiled GeoTIFF
    run_cmd(
        [
            "gdal_translate",
            *cfg,
            sd,
            str(job.tmp_path),
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=ZSTD",
            "-co",
            f"ZSTD_LEVEL={zstd_level}",
            "-co",
            "BIGTIFF=IF_SAFER",
            "--config",
            "GDAL_NUM_THREADS",
            "ALL_CPUS",
        ]
    )

    # 2) Overviews (categorical -> nearest)
    run_cmd(
        [
            "gdaladdo",
            *cfg,
            "-r",
            "nearest",
            str(job.tmp_path),
            *[str(x) for x in ov_levels],
            "--config",
            "GDAL_NUM_THREADS",
            "ALL_CPUS",
            "--config",
            "COMPRESS_OVERVIEW",
            "ZSTD",
            "--config",
            "ZSTD_LEVEL_OVERVIEW",
            str(zstd_level),
            "--config",
            "BIGTIFF_OVERVIEW",
            "IF_SAFER",
        ]
    )

    # 3) Convert to COG, copy overviews
    run_cmd(
        [
            "gdal_translate",
            *cfg,
            str(job.tmp_path),
            str(job.out_path),
            "-of",
            "COG",
            "-co",
            "COMPRESS=ZSTD",
            "-co",
            f"ZSTD_LEVEL={zstd_level}",
            "-co",
            "RESAMPLING=NEAREST",
            "-co",
            "COPY_SRC_OVERVIEWS=YES",
            "-co",
            "BIGTIFF=IF_SAFER",
            "--config",
            "GDAL_NUM_THREADS",
            "ALL_CPUS",
        ]
    )

    try:
        job.tmp_path.unlink()
    except OSError:
        pass

    return (job.out_path, "ok")


def iter_hdf_files(root: Path) -> List[Path]:
    exts = {".hdf", ".hdf4"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def build_jobs(in_dir: Path, out_dir: Path) -> List[Job]:
    jobs: List[Job] = []
    for hdf in iter_hdf_files(in_dir):
        stem = hdf.stem
        out = out_dir / f"{stem}.LC_Type1.cog.tif"
        tmp = out_dir / f"{stem}.tmp.tif"
        jobs.append(Job(hdf_path=hdf, out_path=out, tmp_path=tmp))
    return jobs


def check_gdal_support(driver_path: str) -> None:
    cfg: List[str] = []
    if driver_path:
        cfg = ["--config", "GDAL_DRIVER_PATH", driver_path]
    p = subprocess.run(
        ["gdalinfo", *cfg, "--formats"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError("gdalinfo --formats failed:\n" + p.stderr)

    txt = (p.stdout + "\n" + p.stderr).lower()
    if "hdf4" not in txt:
        raise RuntimeError(
            "GDAL does not appear to have HDF4 support enabled (HDF4 not listed in gdalinfo --formats)."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bulk convert MODIS MCD12Q1 HDF tiles to fast COGs (LC_Type1) with nearest-neighbor overviews."
    )
    ap.add_argument("in_dir", nargs="?", default="MCD12Q1_2024", help="Folder containing .hdf tiles")
    ap.add_argument("out_dir", nargs="?", default=None, help="Output folder for .cog.tif files")
    ap.add_argument(
        "--gdal-driver-path",
        default=os.environ.get("GDAL_DRIVER_PATH", ""),
        help="Folder containing GDAL plugin drivers (e.g. gdal_HDF4.dll).",
    )
    ap.add_argument("--zstd-level", type=int, default=9, help="ZSTD compression level")
    ap.add_argument("--ov-levels", default="2,4,8,16,32", help="Overview levels, comma-separated")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel workers",
    )
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    ap.add_argument("--vrt", action="store_true", help="Build a VRT mosaic over outputs at the end")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path(f"{args.in_dir}_cogs").resolve()
    )
    driver_path = (args.gdal_driver_path or "").strip()

    if not in_dir.is_dir():
        print(f"Input dir not found: {in_dir}", file=sys.stderr)
        return 2

    for tool in ("gdalinfo", "gdal_translate", "gdaladdo"):
        if not have_tool(tool):
            print(f"Missing GDAL CLI tool on PATH: {tool}", file=sys.stderr)
            return 2

    try:
        ov_levels = [int(x) for x in args.ov_levels.split(",") if x.strip()]
    except ValueError:
        print("Bad --ov-levels, expected comma-separated ints like 2,4,8,16,32", file=sys.stderr)
        return 2
    if not ov_levels:
        print("No overview levels provided", file=sys.stderr)
        return 2

    try:
        check_gdal_support(driver_path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    jobs = build_jobs(in_dir, out_dir)
    if not jobs:
        print("No .hdf files found.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    print(f"IN : {in_dir}")
    print(f"OUT: {out_dir}")
    print(f"GDAL_DRIVER_PATH: {driver_path or '(empty)'}")
    print(f"tiles: {total}")
    print(f"workers: {args.workers}")
    print(f"zstd_level: {args.zstd_level}")
    print(f"ov_levels: {ov_levels}")
    print()

    done_ok = 0
    done_skip = 0
    done_warn = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                make_one_cog,
                job,
                driver_path,
                args.zstd_level,
                ov_levels,
                args.force,
            )
            for job in jobs
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out_path, status = fut.result()
            except Exception as e:
                print(f"[{i}/{total}] ERROR: {e}", file=sys.stderr)
                continue

            if status == "ok":
                done_ok += 1
            elif status == "skip":
                done_skip += 1
            else:
                done_warn += 1

            print(f"[{i}/{total}] {status}: {out_path.name}")

    print()
    print(f"ok: {done_ok}, skip: {done_skip}, warn: {done_warn}")

    if args.vrt:
        vrt_path = out_dir / "mcd12q1_lc_type1.vrt"
        cog_list = sorted(str(p) for p in out_dir.glob("*.cog.tif"))
        if cog_list:
            run_cmd(["gdalbuildvrt", str(vrt_path), *cog_list])
            print(f"VRT: {vrt_path}")
        else:
            print("No COGs to build VRT from.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

