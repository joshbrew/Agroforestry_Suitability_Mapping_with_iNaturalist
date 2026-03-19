#!/usr/bin/env python3
import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import rasterio


# py repair_files.py --root D:\soilgrids\data --replace --props phh2o
# py repair_files.py --root D:\soilgrids\data --replace --props phh2o --depths 0-5cm 5-15cm
# py repair_files.py --root D:\soilgrids\data --replace --backup-bad --skip-depths 100-200cm

BASE_URL_DEFAULT = "https://files.isric.org/soilgrids/latest/data"


def log(msg):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", file=sys.stderr, flush=True)


def fmt_seconds(seconds):
    return f"{seconds:.1f}s"


def normalize_url(base_url, rel_path):
    rel = str(rel_path).replace("\\", "/").lstrip("/")
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", rel)


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_name_list(values):
    out = set()
    for v in values or []:
        for part in str(v).split(","):
            part = part.strip().lower()
            if part:
                out.add(part)
    return out


def normalize_prop_list(values):
    return normalize_name_list(values)


def normalize_depth_list(values):
    return normalize_name_list(values)


def rel_parts_lower(path, root=None):
    try:
        p = path.relative_to(root) if root is not None else path
    except Exception:
        p = path
    return [part.lower() for part in p.parts]


def top_level_prop_name(path, root=None):
    parts = rel_parts_lower(path, root=root)
    return parts[0] if parts else ""


def path_matches_depth_filters(path, include_depths=None, exclude_depths=None, root=None):
    include_depths = include_depths or set()
    exclude_depths = exclude_depths or set()

    parts = rel_parts_lower(path, root=root)
    stem = path.stem.lower()
    haystacks = parts + [stem]

    found_depths = set()
    for text in haystacks:
        for token in include_depths | exclude_depths:
            if token and token in text:
                found_depths.add(token)

    if include_depths and not (found_depths & include_depths):
        return False

    if exclude_depths and (found_depths & exclude_depths):
        return False

    return True


def discover_vrts(
    root,
    include_props=None,
    exclude_props=None,
    include_depths=None,
    exclude_depths=None,
    progress_every=25,
):
    root = Path(root).resolve()
    include_props = normalize_prop_list(include_props)
    exclude_props = normalize_prop_list(exclude_props)
    include_depths = normalize_depth_list(include_depths)
    exclude_depths = normalize_depth_list(exclude_depths)

    files = []
    start = time.time()

    log(f"Discovering VRTs under {root}")

    prop_dirs = []
    try:
        for p in root.iterdir():
            if p.is_dir():
                prop_dirs.append(p.resolve())
    except Exception as exc:
        raise RuntimeError(f"Failed to iterate root for VRT discovery: {root}: {exc}")

    prop_dirs.sort()

    if include_props:
        log(f"Restricting to properties: {', '.join(sorted(include_props))}")
    if exclude_props:
        log(f"Skipping properties: {', '.join(sorted(exclude_props))}")
    if include_depths:
        log(f"Restricting to depths: {', '.join(sorted(include_depths))}")
    if exclude_depths:
        log(f"Skipping depths: {', '.join(sorted(exclude_depths))}")

    filtered_prop_dirs = []
    skipped_nonselected = 0
    skipped_excluded = 0

    for p in prop_dirs:
        prop_name = p.name.lower()

        if include_props and prop_name not in include_props:
            skipped_nonselected += 1
            continue

        if prop_name in exclude_props:
            skipped_excluded += 1
            continue

        filtered_prop_dirs.append(p)

    log(
        f"Found {len(prop_dirs):,} top-level folder(s), "
        f"selected {len(filtered_prop_dirs):,}, "
        f"skipped_nonselected={skipped_nonselected:,}, "
        f"skipped_excluded={skipped_excluded:,}"
    )

    scanned_dirs = 0
    for prop_dir in filtered_prop_dirs:
        scanned_dirs += 1
        prop_name = prop_dir.name

        try:
            vrts_here = [p.resolve() for p in prop_dir.glob("*.vrt") if p.is_file()]
        except Exception as exc:
            log(f"WARNING: failed to scan {prop_dir}: {exc}")
            continue

        if include_depths or exclude_depths:
            vrts_here = [
                p for p in vrts_here
                if path_matches_depth_filters(
                    p,
                    include_depths=include_depths,
                    exclude_depths=exclude_depths,
                    root=root,
                )
            ]

        vrts_here.sort()
        files.extend(vrts_here)

        log(
            f"[discover_vrts] dir {scanned_dirs}/{len(filtered_prop_dirs)} "
            f"{prop_name}: found {len(vrts_here):,} VRT(s), total={len(files):,}"
        )

        if progress_every > 0 and scanned_dirs % progress_every == 0:
            elapsed = time.time() - start
            log(
                f"[discover_vrts] progress: scanned {scanned_dirs:,}/{len(filtered_prop_dirs):,} dirs "
                f"in {fmt_seconds(elapsed)}"
            )

    elapsed = time.time() - start
    log(f"Finished VRT discovery: {len(files):,} VRT(s) in {fmt_seconds(elapsed)}")
    return files


def discover_tifs(
    root,
    include_props=None,
    exclude_props=None,
    include_depths=None,
    exclude_depths=None,
    progress_every=5000,
):
    root = Path(root).resolve()
    include_props = normalize_prop_list(include_props)
    exclude_props = normalize_prop_list(exclude_props)
    include_depths = normalize_depth_list(include_depths)
    exclude_depths = normalize_depth_list(exclude_depths)

    files = []
    start = time.time()

    log(f"Discovering TIFFs under {root}")

    for i, p in enumerate(root.rglob("*.tif"), start=1):
        if not p.is_file():
            continue

        top = top_level_prop_name(p, root=root)

        if include_props and top not in include_props:
            if progress_every > 0 and i % progress_every == 0:
                elapsed = time.time() - start
                log(f"[discover_tifs] visited {i:,} matches, kept {len(files):,} TIFF(s) in {fmt_seconds(elapsed)}")
            continue

        if top in exclude_props:
            if progress_every > 0 and i % progress_every == 0:
                elapsed = time.time() - start
                log(f"[discover_tifs] visited {i:,} matches, kept {len(files):,} TIFF(s) in {fmt_seconds(elapsed)}")
            continue

        if not path_matches_depth_filters(
            p,
            include_depths=include_depths,
            exclude_depths=exclude_depths,
            root=root,
        ):
            if progress_every > 0 and i % progress_every == 0:
                elapsed = time.time() - start
                log(f"[discover_tifs] visited {i:,} matches, kept {len(files):,} TIFF(s) in {fmt_seconds(elapsed)}")
            continue

        files.append(p.resolve())

        if progress_every > 0 and i % progress_every == 0:
            elapsed = time.time() - start
            log(f"[discover_tifs] visited {i:,} matches, kept {len(files):,} TIFF(s) in {fmt_seconds(elapsed)}")

    files.sort()
    elapsed = time.time() - start
    log(f"Finished TIFF discovery: {len(files):,} TIFF(s) in {fmt_seconds(elapsed)}")
    return files


def parse_vrt_sources(vrt_path):
    vrt_path = Path(vrt_path).resolve()
    out = []

    try:
        tree = ET.parse(vrt_path)
        root = tree.getroot()
    except Exception as exc:
        raise RuntimeError(f"failed_to_parse_vrt: {vrt_path}: {exc}")

    for elem in root.iter():
        if elem.tag.endswith("SourceFilename") and elem.text:
            raw = elem.text.strip()
            relative = elem.attrib.get("relativeToVRT", "0") == "1"

            if relative:
                src = (vrt_path.parent / raw).resolve()
            else:
                src = Path(raw).resolve()

            if src.suffix.lower() == ".tif":
                out.append(src)

    return out


def get_read_windows(ds):
    try:
        windows = list(ds.block_windows(1))
        if windows:
            return [w for _, w in windows]
    except Exception:
        pass

    return [((0, 0), (ds.height, ds.width))]


def read_window(ds, window):
    if hasattr(window, "col_off"):
        return ds.read(1, window=window, masked=False)

    (row_off, col_off), (height, width) = window
    return ds.read(
        1,
        window=((row_off, row_off + height), (col_off, col_off + width)),
        masked=False,
    )


def validate_tif(path, fail_fast=False):
    path = Path(path)
    try:
        with rasterio.open(path) as ds:
            if ds.count < 1:
                return False, "dataset_has_no_bands"

            windows = get_read_windows(ds)
            for idx, window in enumerate(windows, start=1):
                try:
                    read_window(ds, window)
                except Exception as exc:
                    return False, f"read_failed_window_{idx}: {exc}"

                if fail_fast and idx >= 1:
                    break

        return True, ""
    except Exception as exc:
        return False, f"open_failed: {exc}"


def download_file(url, dest_path, retries=5, timeout=120, user_agent="Mozilla/5.0"):
    ensure_parent(dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            if dest_path.exists():
                dest_path.unlink()
            tmp_path.replace(dest_path)
            return
        except Exception as exc:
            last_err = exc
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"Failed to download {url} -> {dest_path}: {last_err}")


def collect_tifs_from_vrts(
    root,
    include_props=None,
    exclude_props=None,
    include_depths=None,
    exclude_depths=None,
    progress_every=25,
):
    start = time.time()
    root = Path(root).resolve()

    vrts = discover_vrts(
        root,
        include_props=include_props,
        exclude_props=exclude_props,
        include_depths=include_depths,
        exclude_depths=exclude_depths,
        progress_every=progress_every,
    )

    if not vrts:
        return [], []

    include_depths = normalize_depth_list(include_depths)
    exclude_depths = normalize_depth_list(exclude_depths)

    tif_set = set()
    bad_vrts = []

    log(f"Parsing {len(vrts):,} VRT(s) to collect referenced TIFFs")

    for i, vrt in enumerate(vrts, start=1):
        vrt_start = time.time()
        try:
            tifs = parse_vrt_sources(vrt)

            if include_depths or exclude_depths:
                tifs = [
                    tif for tif in tifs
                    if path_matches_depth_filters(
                        tif,
                        include_depths=include_depths,
                        exclude_depths=exclude_depths,
                        root=root,
                    )
                ]

            before = len(tif_set)
            for tif in tifs:
                tif_set.add(tif)
            added = len(tif_set) - before

            log(
                f"[collect_tifs] {i}/{len(vrts)} "
                f"{vrt.relative_to(root)}: refs={len(tifs):,}, new={added:,}, unique_total={len(tif_set):,}"
            )
        except Exception as exc:
            bad_vrts.append((vrt, str(exc)))
            log(f"[collect_tifs] {i}/{len(vrts)} FAILED {vrt.relative_to(root)} :: {exc}")

        if progress_every > 0 and i % progress_every == 0:
            elapsed = time.time() - start
            log(
                f"[collect_tifs] progress: parsed {i:,}/{len(vrts):,} VRT(s), "
                f"unique TIFFs={len(tif_set):,}, bad_vrts={len(bad_vrts):,}, elapsed={fmt_seconds(elapsed)}"
            )

        vrt_elapsed = time.time() - vrt_start
        if vrt_elapsed >= 2.0:
            log(f"[collect_tifs] slow VRT parse: {vrt.relative_to(root)} took {fmt_seconds(vrt_elapsed)}")

    files = sorted(tif_set)
    elapsed = time.time() - start
    log(
        f"Finished collecting TIFFs from VRTs: {len(files):,} unique TIFF(s), "
        f"{len(bad_vrts):,} bad VRT(s), {fmt_seconds(elapsed)}"
    )
    return files, bad_vrts


def main():
    ap = argparse.ArgumentParser(
        description="Use SoilGrids VRTs to find referenced TIFFs, validate them, and immediately replace broken ones in place"
    )
    ap.add_argument("--root", required=True, help="Local SoilGrids root, e.g. D:\\soilgrids\\data")
    ap.add_argument(
        "--base-url",
        default=BASE_URL_DEFAULT,
        help="Remote SoilGrids base URL",
    )
    ap.add_argument(
        "--props",
        nargs="+",
        default=[],
        help="Only scan these top-level SoilGrids properties, e.g. --props phh2o soc clay",
    )
    ap.add_argument(
        "--skip-props",
        nargs="+",
        default=[],
        help="Skip these top-level SoilGrids properties, e.g. --skip-props wrb",
    )
    ap.add_argument(
        "--depths",
        nargs="+",
        default=[],
        help="Only scan these depths, e.g. --depths 0-5cm 5-15cm 15-30cm",
    )
    ap.add_argument(
        "--skip-depths",
        nargs="+",
        default=[],
        help="Skip these depths, e.g. --skip-depths 100-200cm 200-300cm",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Replace broken files in place",
    )
    ap.add_argument(
        "--backup-bad",
        action="store_true",
        help="Rename each broken file to .bad before replacing it",
    )
    ap.add_argument(
        "--fail-fast-read",
        action="store_true",
        help="Only test the first readable block/window of each TIFF instead of all blocks",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retry count for downloads",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds",
    )
    ap.add_argument(
        "--scan-all-tifs-if-no-vrts",
        action="store_true",
        help="Fall back to scanning all TIFFs under root if no VRTs are found",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emit progress during discovery/parsing every N items",
    )

    args = ap.parse_args()

    total_start = time.time()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    include_props = normalize_prop_list(args.props)
    exclude_props = normalize_prop_list(args.skip_props)
    include_depths = normalize_depth_list(args.depths)
    exclude_depths = normalize_depth_list(args.skip_depths)

    overlap_props = include_props & exclude_props
    if overlap_props:
        raise ValueError(f"Properties cannot be both included and skipped: {', '.join(sorted(overlap_props))}")

    overlap_depths = include_depths & exclude_depths
    if overlap_depths:
        raise ValueError(f"Depths cannot be both included and skipped: {', '.join(sorted(overlap_depths))}")

    log(f"Starting repair scan under {root}")

    files, bad_vrts = collect_tifs_from_vrts(
        root,
        include_props=include_props,
        exclude_props=exclude_props,
        include_depths=include_depths,
        exclude_depths=exclude_depths,
        progress_every=max(1, args.progress_every),
    )

    if bad_vrts:
        log(f"Encountered {len(bad_vrts):,} VRT parse failure(s)")
        for vrt, err in bad_vrts:
            print(f"[VRT PARSE FAILED] {vrt}", file=sys.stderr, flush=True)
            print(f"  {err}", file=sys.stderr, flush=True)

    if not files and args.scan_all_tifs_if_no_vrts:
        log("No VRT-referenced TIFFs found, falling back to scanning all TIFFs under root")
        files = discover_tifs(
            root,
            include_props=include_props,
            exclude_props=exclude_props,
            include_depths=include_depths,
            exclude_depths=exclude_depths,
        )

    if not files:
        raise RuntimeError(f"No TIFFs found from VRTs under: {root}")

    log(f"Checking {len(files):,} TIFF file(s) for readability")

    broken_count = 0
    repaired_count = 0
    checked_start = time.time()

    for i, path in enumerate(files, start=1):
        try:
            rel_path = path.relative_to(root)
        except Exception:
            print(f"[{i}/{len(files)}] SKIP  outside root: {path}", file=sys.stderr, flush=True)
            continue

        ok, error = validate_tif(path, fail_fast=args.fail_fast_read)

        if ok:
            print(f"[{i}/{len(files)}] OK    {rel_path}", file=sys.stderr, flush=True)
        else:
            broken_count += 1
            print(f"[{i}/{len(files)}] BAD   {rel_path}", file=sys.stderr, flush=True)
            print(f"  {error}", file=sys.stderr, flush=True)

            if args.replace:
                url = normalize_url(args.base_url, rel_path)

                try:
                    print(f"  downloading {url}", file=sys.stderr, flush=True)

                    if args.backup_bad and path.exists():
                        backup_path = path.with_suffix(path.suffix + ".bad")
                        if backup_path.exists():
                            backup_path.unlink()
                        path.replace(backup_path)
                        print(f"  backed up to {backup_path.name}", file=sys.stderr, flush=True)

                    download_file(
                        url=url,
                        dest_path=path,
                        retries=max(1, args.retries),
                        timeout=max(1, args.timeout),
                    )

                    ok2, error2 = validate_tif(path, fail_fast=args.fail_fast_read)
                    if ok2:
                        repaired_count += 1
                        print(f"  repaired {rel_path}", file=sys.stderr, flush=True)
                    else:
                        print(f"  replacement still bad: {rel_path}", file=sys.stderr, flush=True)
                        print(f"  {error2}", file=sys.stderr, flush=True)
                except Exception as exc:
                    print(f"  replace failed: {rel_path}", file=sys.stderr, flush=True)
                    print(f"  {exc}", file=sys.stderr, flush=True)

        if args.progress_every > 0 and i % args.progress_every == 0:
            elapsed = time.time() - checked_start
            log(
                f"[validate] progress: {i:,}/{len(files):,} checked, "
                f"broken={broken_count:,}, repaired={repaired_count:,}, elapsed={fmt_seconds(elapsed)}"
            )

    total_elapsed = time.time() - total_start
    log(
        f"Done. checked={len(files):,} broken={broken_count:,} repaired={repaired_count:,} "
        f"total_elapsed={fmt_seconds(total_elapsed)}"
    )


if __name__ == "__main__":
    main()