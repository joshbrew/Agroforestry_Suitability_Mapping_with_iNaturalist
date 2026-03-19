#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ex:
# python cogify_geotiffs.py --root "D:\DEM_derived_w_flow" --workers 6
# python cogify_geotiffs.py --root "D:\DEM_derived_w_flow" --out-root "D:\DEM_derived_w_flow_COG" --workers 6


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None

def run_cmd(cmd):
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

def is_probably_cog(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".cog.tif") or n.endswith(".cog.tiff")

def default_resampling_for_name(name: str) -> str:
    nl = name.lower()
    # Flow direction and other categorical rasters should use NEAREST overviews
    if "flowdir" in nl or "dir_3s" in nl or "_dir_" in nl or "fdr" in nl:
        return "NEAREST"
    return "AVERAGE"

def cogify_one(src: Path, dst: Path, compress: str, zstd_level: int, resampling: str, force: bool) -> str:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Prefer GDAL's COG driver
    cmd = [
        "gdal_translate",
        str(src),
        str(dst),
        "-of", "COG",
        "-co", f"COMPRESS={compress}",
        "-co", "BIGTIFF=IF_SAFER",
        "-co", f"RESAMPLING={resampling}",
        "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
    ]
    if compress.upper() == "ZSTD":
        cmd += ["-co", f"ZSTD_LEVEL={int(zstd_level)}"]

    try:
        run_cmd(cmd)
        return "ok"
    except Exception as e:
        # Fallback: GTiff + external overviews, then copy as “COG-ish”
        # This keeps things usable even if COG driver is missing.
        tmp = dst.with_suffix(dst.suffix + ".tmp.tif")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

        predictor = "2"
        # Predictor 3 is for floating point (if you want to get fancy you can detect dtype via gdalinfo -json)
        # Keeping predictor=2 works fine broadly.
        cmd2 = [
            "gdal_translate",
            str(src),
            str(tmp),
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", f"COMPRESS={compress}",
            "-co", f"PREDICTOR={predictor}",
            "-co", "BIGTIFF=IF_SAFER",
        ]
        if compress.upper() == "ZSTD":
            cmd2 += ["-co", f"ZSTD_LEVEL={int(zstd_level)}"]
        run_cmd(cmd2)

        # build overviews
        run_cmd([
            "gdaladdo",
            "-r", resampling.lower(),
            "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
            str(tmp),
            "2", "4", "8", "16", "32"
        ])

        # move into place
        if dst.exists():
            dst.unlink()
        tmp.replace(dst)
        return f"ok_fallback({type(e).__name__})"

def iter_tifs(root: Path):
    for p in root.rglob("*.tif"):
        if p.name.lower().endswith(".tif.aux.xml"):
            continue
        yield p
    for p in root.rglob("*.tiff"):
        if p.name.lower().endswith(".tiff.aux.xml"):
            continue
        yield p

def main():
    ap = argparse.ArgumentParser(description="Convert GeoTIFFs (big + tiles) to Cloud Optimized GeoTIFFs (*.cog.tif).")
    ap.add_argument("--root", required=True, help="Root folder, e.g. D:\\DEM_derived_w_flow")
    ap.add_argument("--out-root", default="", help="If set, mirror output tree under this folder. Otherwise write next to inputs.")
    ap.add_argument("--compress", default="ZSTD", choices=["ZSTD", "DEFLATE", "LZW"])
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-under", default="DEM,FlowAccumulation,FlowDir,Derived_Aspect_Slope_Eastness_Northness",
                    help="Comma list of subfolders under root to scan.")
    ap.add_argument("--resampling", default="auto", choices=["auto", "AVERAGE", "NEAREST", "BILINEAR"])
    args = ap.parse_args()

    for t in ("gdal_translate", "gdaladdo"):
        if not have_tool(t):
            print(f"Missing GDAL tool on PATH: {t}", file=sys.stderr)
            return 2

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 2

    scan_subs = [s.strip() for s in args.only_under.split(",") if s.strip()]
    scan_roots = [root / s for s in scan_subs if (root / s).exists()]
    if not scan_roots:
        print("No scan roots found under --root. Check --only-under.", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve() if args.out_root.strip() else None

    jobs = []
    for sr in scan_roots:
        for src in iter_tifs(sr):
            if is_probably_cog(src):
                continue
            if src.name.lower().endswith(".aux.xml"):
                continue
            if out_root is None:
                dst = src.with_name(src.stem + ".cog" + src.suffix)
            else:
                rel = src.relative_to(root)
                dst = out_root / rel.parent / (src.stem + ".cog" + src.suffix)
            jobs.append((src, dst))

    if not jobs:
        print("No GeoTIFFs found to convert.")
        return 0

    print(f"jobs: {len(jobs)}, workers: {args.workers}")

    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = []
        for src, dst in jobs:
            resamp = default_resampling_for_name(src.name) if args.resampling == "auto" else args.resampling
            futs.append(ex.submit(cogify_one, src, dst, args.compress, args.zstd_level, resamp, bool(args.force)))

        for i, fut in enumerate(as_completed(futs), 1):
            try:
                st = fut.result()
                if st.startswith("ok"):
                    ok += 1
                elif st == "skip":
                    skip += 1
                else:
                    ok += 1
                if i % 50 == 0 or i == len(futs):
                    print(f"[{i}/{len(futs)}] ok={ok} skip={skip} err={err}")
            except Exception as e:
                err += 1
                print(f"[{i}/{len(futs)}] ERROR: {e}", file=sys.stderr)

    print(f"done. ok={ok} skip={skip} err={err}")
    return 0 if err == 0 else 3

if __name__ == "__main__":
    raise SystemExit(main())