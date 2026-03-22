#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# sample commands:
# python cogify_geotiffs.py --root "D:\\DEM_derived_w_flow" --workers 6
# python cogify_geotiffs.py --root "D:\\DEM_derived_w_flow" --out-root "D:\\DEM_derived_w_flow_COG" --workers 6
# python cogify_geotiffs.py --root "D:\\DEM_derived_w_flow" --merge-derived --workers 2
# python cogify_geotiffs.py --root "D:\\DEM_derived_w_flow" --merge-derived --subfolders af_dem_3s,as_dem_3s --workers 2
# python cogify_geotiffs.py --root "D:\\DEM_derived_w_flow_COG" --merge-derived --merge-source cog --workers 2
# python cogify_geotiffs.py --root "D:\\DEM_derived_w_flow" --merge-derived --merge-source raw --out-root "D:\\DEM_derived_w_flow_MERGED" --workers 2

DERIVED_TILE_PREFIX_RE = re.compile(r"^x-?\d+_y-?\d+_")
DEFAULT_DERIVED_VARIABLES = ("slope_deg", "aspect_deg", "northness", "eastness")


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: Sequence[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(str(x) for x in cmd)
            + "\n\nSTDOUT:\n"
            + p.stdout
            + "\n\nSTDERR:\n"
            + p.stderr
        )
    return p.stdout


def is_probably_cog(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".cog.tif") or n.endswith(".cog.tiff")


def strip_tif_and_cog_suffix(name: str) -> str:
    nl = name.lower()
    if nl.endswith(".tiff"):
        nl = nl[:-5]
    elif nl.endswith(".tif"):
        nl = nl[:-4]
    if nl.endswith(".cog"):
        nl = nl[:-4]
    return nl


def default_resampling_for_name(name: str) -> str:
    nl = name.lower()
    if "flowdir" in nl or "dir_3s" in nl or "_dir_" in nl or "fdr" in nl:
        return "NEAREST"
    if "aspect" in nl:
        return "NEAREST"
    return "AVERAGE"


def cogify_one(src: Path, dst: Path, compress: str, zstd_level: int, resampling: str, force: bool) -> str:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdal_translate",
        str(src),
        str(dst),
        "-of",
        "COG",
        "-co",
        f"COMPRESS={compress}",
        "-co",
        "BIGTIFF=IF_SAFER",
        "-co",
        f"RESAMPLING={resampling}",
        "--config",
        "GDAL_NUM_THREADS",
        "ALL_CPUS",
    ]
    if compress.upper() == "ZSTD":
        cmd += ["-co", f"ZSTD_LEVEL={int(zstd_level)}"]

    try:
        run_cmd(cmd)
        return "ok"
    except Exception as e:
        tmp = dst.with_suffix(dst.suffix + ".tmp.tif")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

        predictor = "2"
        cmd2 = [
            "gdal_translate",
            str(src),
            str(tmp),
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            f"COMPRESS={compress}",
            "-co",
            f"PREDICTOR={predictor}",
            "-co",
            "BIGTIFF=IF_SAFER",
        ]
        if compress.upper() == "ZSTD":
            cmd2 += ["-co", f"ZSTD_LEVEL={int(zstd_level)}"]
        run_cmd(cmd2)

        run_cmd([
            "gdaladdo",
            "-r",
            resampling.lower(),
            "--config",
            "GDAL_NUM_THREADS",
            "ALL_CPUS",
            str(tmp),
            "2",
            "4",
            "8",
            "16",
            "32",
        ])

        if dst.exists():
            dst.unlink()
        tmp.replace(dst)
        return f"ok_fallback({type(e).__name__})"


def iter_tifs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.tif"):
        if p.name.lower().endswith(".tif.aux.xml"):
            continue
        yield p
    for p in root.rglob("*.tiff"):
        if p.name.lower().endswith(".tiff.aux.xml"):
            continue
        yield p


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def require_tools(names: Sequence[str]) -> None:
    missing = [name for name in names if not have_tool(name)]
    if missing:
        raise RuntimeError("Missing GDAL tool on PATH: " + ", ".join(missing))


def resolve_merge_parent(root: Path, merge_under: str) -> Path:
    merge_under = merge_under.strip()
    candidate = root / merge_under if merge_under else root
    if candidate.exists() and candidate.is_dir():
        return candidate

    subdirs = [p for p in root.iterdir() if p.is_dir() and p.name.lower().endswith("_dem_3s")]
    if subdirs:
        return root

    raise RuntimeError(f"Could not find merge folder under {root}: {merge_under}")


def list_merge_subfolders(merge_parent: Path, only: Optional[str]) -> List[str]:
    names = [p.name for p in sorted(merge_parent.iterdir()) if p.is_dir() and p.name.lower().endswith("_dem_3s")]
    if only:
        wanted = set(parse_csv_list(only))
        names = [n for n in names if n in wanted]
    if not names:
        raise RuntimeError(f"No *_dem_3s subfolders found under {merge_parent}")
    return names


def detect_derived_variable(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    if path.name.lower().endswith(".aux.xml"):
        return None
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    stem = strip_tif_and_cog_suffix(path.name)
    if not DERIVED_TILE_PREFIX_RE.match(stem):
        return None

    for var in DEFAULT_DERIVED_VARIABLES + ("twi",):
        if stem.endswith("_" + var):
            return var
    return None


def choose_source_set(cog_paths: List[Path], raw_paths: List[Path], source_mode: str) -> List[Path]:
    cog_sorted = sorted(cog_paths)
    raw_sorted = sorted(raw_paths)

    if source_mode == "cog":
        return cog_sorted
    if source_mode == "raw":
        return raw_sorted

    if len(cog_sorted) >= len(raw_sorted) and cog_sorted:
        return cog_sorted
    if raw_sorted:
        return raw_sorted
    return []


def make_merge_output_path(
    root: Path,
    out_root: Optional[Path],
    merge_parent: Path,
    subfolder: str,
    variable: str,
) -> Path:
    name = f"{subfolder}_{variable}.merged.cog.tif"
    if out_root is None:
        return merge_parent / subfolder / name
    rel_parent = merge_parent.relative_to(root)
    return out_root / rel_parent / subfolder / name


def build_vrt(vrt_path: Path, sources: Sequence[Path]) -> None:
    if not sources:
        raise RuntimeError(f"No sources supplied for VRT: {vrt_path}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        list_path = Path(tf.name)
        for src in sources:
            tf.write(str(src))
            tf.write("\n")

    try:
        run_cmd([
            "gdalbuildvrt",
            "-overwrite",
            "-input_file_list",
            str(list_path),
            str(vrt_path),
        ])
    finally:
        try:
            list_path.unlink()
        except Exception:
            pass


def merge_tiles_to_cog(
    sources: Sequence[Path],
    dst: Path,
    compress: str,
    zstd_level: int,
    resampling: str,
    force: bool,
) -> str:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return "skip"

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="merge_derived_") as td:
        vrt_path = Path(td) / "merge.vrt"
        build_vrt(vrt_path, sources)
        return cogify_one(vrt_path, dst, compress, zstd_level, resampling, force)


def collect_merge_jobs(
    root: Path,
    merge_parent: Path,
    out_root: Optional[Path],
    subfolders: Sequence[str],
    variables: Sequence[str],
    source_mode: str,
) -> List[Tuple[str, str, List[Path], Path]]:
    jobs: List[Tuple[str, str, List[Path], Path]] = []
    allowed = set(variables)

    for subfolder in subfolders:
        subdir = merge_parent / subfolder
        grouped: Dict[str, Dict[str, List[Path]]] = {
            var: {"cog": [], "raw": []}
            for var in variables
        }

        for path in iter_tifs(subdir):
            var = detect_derived_variable(path)
            if var is None or var not in allowed:
                continue
            if is_probably_cog(path):
                grouped[var]["cog"].append(path)
            else:
                grouped[var]["raw"].append(path)

        for var in variables:
            selected = choose_source_set(grouped[var]["cog"], grouped[var]["raw"], source_mode)
            if not selected:
                continue
            dst = make_merge_output_path(root, out_root, merge_parent, subfolder, var)
            jobs.append((subfolder, var, selected, dst))

    return jobs


def run_cogify_mode(args: argparse.Namespace) -> int:
    require_tools(("gdal_translate", "gdaladdo"))

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

    jobs: List[Tuple[Path, Path]] = []
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


def run_merge_mode(args: argparse.Namespace) -> int:
    require_tools(("gdal_translate", "gdaladdo", "gdalbuildvrt"))

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve() if args.out_root.strip() else None
    variables = parse_csv_list(args.merge_variables) or list(DEFAULT_DERIVED_VARIABLES)
    merge_parent = resolve_merge_parent(root, args.merge_under)
    subfolders = list_merge_subfolders(merge_parent, args.subfolders)
    jobs = collect_merge_jobs(root, merge_parent, out_root, subfolders, variables, args.merge_source)

    if not jobs:
        print("No derived tile groups found to merge.")
        return 0

    print(f"merge parent: {merge_parent}")
    print(f"subfolders: {', '.join(subfolders)}")
    print(f"variables: {', '.join(variables)}")
    print(f"merge jobs: {len(jobs)}, workers: {args.workers}, source={args.merge_source}")

    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut_to_meta = {}
        for subfolder, variable, sources, dst in jobs:
            src_name_for_resamp = f"{subfolder}_{variable}"
            resamp = default_resampling_for_name(src_name_for_resamp) if args.resampling == "auto" else args.resampling
            fut = ex.submit(
                merge_tiles_to_cog,
                sources,
                dst,
                args.compress,
                args.zstd_level,
                resamp,
                bool(args.force),
            )
            fut_to_meta[fut] = (subfolder, variable, len(sources), dst)

        for i, fut in enumerate(as_completed(fut_to_meta), 1):
            subfolder, variable, count, dst = fut_to_meta[fut]
            try:
                st = fut.result()
                if st.startswith("ok"):
                    ok += 1
                elif st == "skip":
                    skip += 1
                else:
                    ok += 1
                print(f"[{i}/{len(fut_to_meta)}] {subfolder} {variable} tiles={count} status={st} out={dst}")
            except Exception as e:
                err += 1
                print(f"[{i}/{len(fut_to_meta)}] ERROR {subfolder} {variable}: {e}", file=sys.stderr)

    print(f"done. ok={ok} skip={skip} err={err}")
    return 0 if err == 0 else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert GeoTIFFs to COGs and optionally merge derived DEM tiles back into per-variable COG mosaics.")
    ap.add_argument("--root", required=True, help="Root folder, e.g. D:\\DEM_derived_w_flow")
    ap.add_argument("--out-root", default="", help="If set, mirror output tree under this folder. Otherwise write next to inputs.")
    ap.add_argument("--compress", default="ZSTD", choices=["ZSTD", "DEFLATE", "LZW"])
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--only-under",
        default="DEM,FlowAccumulation,FlowDir,Derived_Aspect_Slope_Eastness_Northness",
        help="Comma list of subfolders under root to scan in normal cogify mode.",
    )
    ap.add_argument("--resampling", default="auto", choices=["auto", "AVERAGE", "NEAREST", "BILINEAR"])
    ap.add_argument("--merge-derived", action="store_true", help="Merge derived slope/aspect/northness/eastness tile sets back into one COG per variable, per *_dem_3s subfolder.")
    ap.add_argument(
        "--merge-under",
        default="Derived_Aspect_Slope_Eastness_Northness",
        help="Folder under --root containing *_dem_3s derived tile subfolders. If --root already points there, that is used directly.",
    )
    ap.add_argument(
        "--merge-variables",
        default=",".join(DEFAULT_DERIVED_VARIABLES),
        help="Comma list of derived variables to merge, e.g. slope_deg,aspect_deg,northness,eastness",
    )
    ap.add_argument(
        "--merge-source",
        default="auto",
        choices=["auto", "cog", "raw"],
        help="Choose merge inputs from .cog.tif tiles, raw .tif tiles, or auto. Auto picks the larger available set and prefers COG on ties.",
    )
    ap.add_argument(
        "--subfolders",
        default=None,
        help="Comma list of *_dem_3s subfolders to merge, e.g. af_dem_3s,as_dem_3s",
    )
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    try:
        if args.merge_derived:
            return run_merge_mode(args)
        return run_cogify_mode(args)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
