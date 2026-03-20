#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio


RESUME_VERSION = 1


def parse_csv_list(value):
    if not value:
        return []
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def discover_vrts(root):
    root = Path(root)
    datasets = {}
    for prop_dir in sorted(root.iterdir()):
        if not prop_dir.is_dir():
            continue
        prop_name = prop_dir.name.strip().lower()
        for vrt in sorted(prop_dir.glob("*.vrt")):
            stem = vrt.stem.strip()
            dataset_name = f"wrb_{stem}" if prop_name == "wrb" else stem
            datasets[dataset_name] = {
                "path": vrt.resolve(),
                "prop": prop_name,
                "stem": stem,
            }
    return datasets


def discover_existing_single_depth_tifs(root):
    root = Path(root)
    datasets = {}
    if not root.exists():
        return datasets

    for tif in sorted(root.rglob("*.tif")):
        if not tif.is_file():
            continue
        name_l = tif.name.lower()
        if name_l.endswith(".part"):
            continue
        if tif.stem.lower().endswith("_depthstack"):
            continue
        dataset_name = tif.stem.strip()
        parent_name = tif.parent.name.strip().lower()
        prop_name = "wrb" if dataset_name.lower().startswith("wrb_") else parent_name
        datasets[dataset_name] = {
            "path": tif.resolve(),
            "prop": prop_name,
            "stem": tif.stem.strip(),
        }
    return datasets


def split_dataset_name(name):
    parts = str(name).split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], "_".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return name, "", ""


def depth_sort_key(depth):
    text = str(depth or "")
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if m:
        return (float(m.group(1)), float(m.group(2)), text)
    return (float("inf"), float("inf"), text)


def is_wrb_meta(meta):
    return str(meta.get("prop", "")).lower() == "wrb"


def wrb_dataset_names(all_datasets):
    names = [name for name, meta in all_datasets.items() if is_wrb_meta(meta)]
    names.sort()
    return names


def resolve_explicit_dataset_names(all_datasets, explicit_datasets):
    if not explicit_datasets:
        return []

    if len(explicit_datasets) == 1 and explicit_datasets[0].lower() == "all":
        return sorted(all_datasets.keys())

    lower_name_map = {name.lower(): name for name in all_datasets.keys()}
    requested = []
    missing = []

    for token in explicit_datasets:
        token = token.strip()
        token_l = token.lower()
        matches = []

        if token_l == "wrb":
            matches = wrb_dataset_names(all_datasets)
        elif token in all_datasets:
            matches = [token]
        elif token_l in lower_name_map:
            matches = [lower_name_map[token_l]]
        else:
            for name, meta in all_datasets.items():
                if meta["stem"].lower() == token_l:
                    matches.append(name)

        if not matches:
            missing.append(token)
            continue

        requested.extend(sorted(matches))

    if missing:
        raise KeyError(f"Requested dataset(s) not found: {', '.join(missing)}")

    return sorted(dict.fromkeys(requested))


def build_requested_dataset_names(all_datasets, props, depths, stat, explicit_datasets):
    if explicit_datasets:
        return resolve_explicit_dataset_names(all_datasets, explicit_datasets)

    if not props and not depths:
        return sorted(all_datasets.keys())

    available_by_prop = defaultdict(set)
    for name, meta in all_datasets.items():
        if is_wrb_meta(meta):
            available_by_prop["wrb"].add((meta["stem"], ""))
            continue
        prop, depth, ds_stat = split_dataset_name(name)
        available_by_prop[prop].add((depth, ds_stat))

    if not props:
        props = sorted(available_by_prop.keys())

    requested = []
    for prop in props:
        prop_l = prop.strip().lower()
        if prop_l not in available_by_prop:
            raise KeyError(f"Property not found: {prop}")

        if prop_l == "wrb":
            requested.extend(wrb_dataset_names(all_datasets))
            continue

        if depths:
            for depth in depths:
                candidate = f"{prop_l}_{depth}_{stat}" if stat else f"{prop_l}_{depth}"
                if candidate in all_datasets:
                    requested.append(candidate)
        else:
            for depth, ds_stat in sorted(available_by_prop[prop_l], key=lambda pair: (depth_sort_key(pair[0]), pair[1])):
                if stat and ds_stat != stat:
                    continue
                if depth:
                    requested.append(f"{prop_l}_{depth}_{ds_stat}" if ds_stat else f"{prop_l}_{depth}")
                else:
                    requested.append(prop_l)

    requested = sorted(dict.fromkeys(requested))
    missing = [name for name in requested if name not in all_datasets]
    if missing:
        raise KeyError(f"Requested dataset(s) not found: {', '.join(missing)}")
    return requested


def get_block_shape(ds, fallback=512):
    if getattr(ds, "block_shapes", None):
        shape = ds.block_shapes[0]
        if shape and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return int(fallback), int(fallback)


def format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600 + minutes * 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:04.1f}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def format_rate(count, seconds, unit_name):
    seconds = max(float(seconds), 1e-9)
    rate = float(count) / seconds
    if rate >= 10:
        return f"{rate:,.0f} {unit_name}/s"
    if rate >= 1:
        return f"{rate:,.1f} {unit_name}/s"
    return f"{rate:,.2f} {unit_name}/s"


def emit_progress_line(phase_label, completed, total, started_at, unit_name, extra=""):
    elapsed = time.time() - float(started_at)
    pct = 100.0 if total <= 0 else (100.0 * float(completed) / float(total))
    msg = (
        f"[progress] {phase_label} {completed:,}/{total:,} "
        f"({pct:.1f}%) elapsed={format_elapsed(elapsed)} "
        f"rate={format_rate(completed, elapsed, unit_name)}"
    )
    if extra:
        msg += f" {extra}"
    print(msg, file=sys.stderr, flush=True)


def make_progress_state(phase_label, total, unit_name, report_every_seconds=10.0, started_at=None):
    started_at = time.time() if started_at is None else float(started_at)
    return {
        "phase_label": str(phase_label),
        "total": int(total),
        "unit_name": str(unit_name),
        "started_at": started_at,
        "report_every_seconds": max(1.0, float(report_every_seconds)),
        "last_report_time": started_at,
        "last_report_completed": 0,
    }


def maybe_emit_progress(progress_state, completed, extra="", force=False):
    if not progress_state:
        return
    completed = int(completed)
    total = int(progress_state["total"])
    now = time.time()
    time_due = (now - float(progress_state["last_report_time"])) >= float(progress_state["report_every_seconds"])
    advanced = completed > int(progress_state["last_report_completed"])
    done = total > 0 and completed >= total
    if not (force or done or (time_due and advanced)):
        return
    emit_progress_line(
        phase_label=progress_state["phase_label"],
        completed=completed,
        total=total,
        started_at=progress_state["started_at"],
        unit_name=progress_state["unit_name"],
        extra=extra,
    )
    progress_state["last_report_time"] = now
    progress_state["last_report_completed"] = completed


def predictor_for_dtype(dtype_name):
    kind = np.dtype(dtype_name).kind
    if kind == "f":
        return 3
    return 2


def default_fill_value(dtype_name, nodata):
    if nodata is not None:
        return nodata
    dt = np.dtype(dtype_name)
    if dt.kind == "f":
        return np.nan
    return 0


def build_output_path(out_root, dataset_name, meta, flat=False):
    out_root = Path(out_root)
    if flat:
        return out_root / f"{dataset_name}.tif"
    return out_root / str(meta.get("prop", "misc")) / f"{dataset_name}.tif"


def build_depth_stack_name(prop, stat):
    return f"{prop}_{stat}_depthstack" if stat else f"{prop}_depthstack"


def build_depth_stack_path(out_root, prop, stat, flat=False):
    out_root = Path(out_root)
    name = build_depth_stack_name(prop, stat)
    if flat:
        return out_root / f"{name}.tif"
    return out_root / prop / f"{name}.tif"


def output_is_fresh(out_path, src_path):
    out_path = Path(out_path)
    src_path = Path(src_path)
    if not out_path.exists():
        return False
    try:
        return out_path.stat().st_mtime_ns >= src_path.stat().st_mtime_ns
    except Exception:
        return False


def output_is_fresh_against_many(out_path, src_paths):
    out_path = Path(out_path)
    if not out_path.exists():
        return False
    try:
        out_mtime = out_path.stat().st_mtime_ns
        for src_path in src_paths:
            if out_mtime < Path(src_path).stat().st_mtime_ns:
                return False
        return True
    except Exception:
        return False


def build_overviews(ds_path, levels, resampling):
    ds_path = Path(ds_path)
    if not levels:
        return
    with rasterio.open(ds_path, "r+") as ds:
        ds.build_overviews(list(levels), resampling=resampling)
        ds.update_tags(ns="rio_overview", resampling=str(resampling))


def resume_state_path_for(tmp_path):
    tmp_path = Path(tmp_path)
    return tmp_path.parent / f"{tmp_path.name}.resume.json"


def path_stat_token(path):
    path = Path(path)
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_unlink(path):
    path = Path(path)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def build_single_resume_signature(src_path, out_path, profile, total_blocks, on_read_error):
    src_info = path_stat_token(src_path)
    return {
        "version": RESUME_VERSION,
        "kind": "single",
        "src": src_info,
        "out_path": str(Path(out_path).resolve()),
        "total_blocks": int(total_blocks),
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "count": int(profile["count"]),
        "dtype": str(profile["dtype"]),
        "nodata": profile.get("nodata"),
        "transform": tuple(profile["transform"]) if profile.get("transform") is not None else None,
        "crs": profile.get("crs").to_string() if profile.get("crs") is not None else None,
        "blockxsize": int(profile["blockxsize"]),
        "blockysize": int(profile["blockysize"]),
        "compress": str(profile.get("compress", "")),
        "predictor": int(profile.get("predictor", 0)),
        "interleave": str(profile.get("interleave", "")),
        "on_read_error": str(on_read_error),
    }


def build_stack_resume_signature(src_items, out_path, profile, total_blocks, on_read_error):
    src_infos = [path_stat_token(item["path"]) for item in src_items]
    band_names = [str(item["name"]) for item in src_items]
    return {
        "version": RESUME_VERSION,
        "kind": "depth-stack",
        "srcs": src_infos,
        "band_names": band_names,
        "out_path": str(Path(out_path).resolve()),
        "total_blocks": int(total_blocks),
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "count": int(profile["count"]),
        "dtype": str(profile["dtype"]),
        "nodata": profile.get("nodata"),
        "transform": tuple(profile["transform"]) if profile.get("transform") is not None else None,
        "crs": profile.get("crs").to_string() if profile.get("crs") is not None else None,
        "blockxsize": int(profile["blockxsize"]),
        "blockysize": int(profile["blockysize"]),
        "compress": str(profile.get("compress", "")),
        "predictor": int(profile.get("predictor", 0)),
        "interleave": str(profile.get("interleave", "")),
        "on_read_error": str(on_read_error),
    }


def prepare_resume(tmp_path, signature, allow_resume):
    tmp_path = Path(tmp_path)
    state_path = resume_state_path_for(tmp_path)
    state = load_json(state_path)

    if not allow_resume:
        safe_unlink(tmp_path)
        safe_unlink(state_path)
        return False, 0, None

    if not tmp_path.exists() or not state:
        if not tmp_path.exists() and state_path.exists():
            safe_unlink(state_path)
        return False, 0, state_path

    if state.get("signature") != signature:
        print(
            f"[resume] discarding incompatible partial output: {tmp_path.name}",
            file=sys.stderr,
            flush=True,
        )
        safe_unlink(tmp_path)
        safe_unlink(state_path)
        return False, 0, state_path

    completed_blocks = int(state.get("completed_blocks", 0))
    total_blocks = int(signature["total_blocks"])
    completed_blocks = max(0, min(completed_blocks, total_blocks))
    print(
        f"[resume] resuming {tmp_path.name} from block {completed_blocks:,}/{total_blocks:,}",
        file=sys.stderr,
        flush=True,
    )
    return True, completed_blocks, state_path


def write_resume_state(state_path, signature, completed_blocks, started_at, read_error_blocks):
    if state_path is None:
        return
    payload = {
        "signature": signature,
        "completed_blocks": int(completed_blocks),
        "started_at": float(started_at),
        "updated_at": time.time(),
        "read_error_blocks": int(read_error_blocks),
    }
    write_json(state_path, payload)


def maybe_log_read_error(prefix, src_name, window, exc, block_idx, total_blocks):
    print(
        f"[warn] {prefix} read error block={block_idx:,}/{total_blocks:,} "
        f"window=row_off:{int(window.row_off)},col_off:{int(window.col_off)},height:{int(window.height)},width:{int(window.width)} "
        f"src={src_name} err={exc}",
        file=sys.stderr,
        flush=True,
    )


def merge_vrt_to_tif(
    src_path,
    out_path,
    gdal_cache_mb,
    blocksize,
    compress,
    bigtiff,
    progress_every_seconds,
    on_read_error,
    num_threads,
    resume_partial,
):
    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    state_path = resume_state_path_for(tmp_path)

    with rasterio.Env(GDAL_CACHEMAX=int(gdal_cache_mb)):
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise RuntimeError(f"Dataset has no CRS: {src_path}")
            if src.count < 1:
                raise RuntimeError(f"Dataset has no bands: {src_path}")

            block_h_src, block_w_src = get_block_shape(src, fallback=blocksize)
            block_h = int(blocksize or block_h_src)
            block_w = int(blocksize or block_w_src)
            predictor = predictor_for_dtype(src.dtypes[0])

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                tiled=True,
                compress=str(compress),
                predictor=int(predictor),
                BIGTIFF=str(bigtiff),
                blockxsize=int(block_w),
                blockysize=int(block_h),
                NUM_THREADS=str(num_threads),
                interleave="band",
            )

            width = int(src.width)
            height = int(src.height)
            total_blocks = int(math.ceil(height / block_h) * math.ceil(width / block_w))
            signature = build_single_resume_signature(src_path, out_path, profile, total_blocks, on_read_error)
            resume_ok, completed_blocks, state_path = prepare_resume(tmp_path, signature, resume_partial)
            started_at = time.time()
            read_error_blocks = 0
            progress_state = make_progress_state(
                phase_label=f"merge {src_path.stem}",
                total=total_blocks,
                unit_name="blocks",
                report_every_seconds=progress_every_seconds,
                started_at=started_at,
            )

            if resume_ok:
                progress_state["last_report_completed"] = completed_blocks
                maybe_emit_progress(
                    progress_state,
                    completed_blocks,
                    extra=f"file={src_path.name} resumed=1",
                    force=True,
                )
                open_mode = "r+"
            else:
                safe_unlink(tmp_path)
                safe_unlink(state_path)
                open_mode = "w"

            if open_mode == "w":
                write_resume_state(state_path, signature, 0, started_at, 0)

            try:
                with rasterio.open(tmp_path, open_mode, **({} if open_mode == "r+" else profile)) as dst:
                    for block_idx, (_, window) in enumerate(dst.block_windows(1), start=1):
                        if block_idx <= completed_blocks:
                            continue
                        try:
                            arr = src.read(window=window, masked=False)
                        except Exception as exc:
                            if on_read_error == "raise":
                                write_resume_state(state_path, signature, block_idx - 1, started_at, read_error_blocks)
                                raise
                            maybe_log_read_error("merge", src_path.name, window, exc, block_idx, total_blocks)
                            fill_value = default_fill_value(src.dtypes[0], src.nodata)
                            arr = np.full(
                                (src.count, int(window.height), int(window.width)),
                                fill_value,
                                dtype=np.dtype(src.dtypes[0]),
                            )
                            read_error_blocks += 1

                        dst.write(arr, window=window)
                        completed_blocks = block_idx
                        if (
                            completed_blocks == total_blocks
                            or completed_blocks == 1
                            or completed_blocks % 32 == 0
                        ):
                            write_resume_state(state_path, signature, completed_blocks, started_at, read_error_blocks)
                        maybe_emit_progress(
                            progress_state,
                            completed_blocks,
                            extra=(
                                f"file={src_path.name}"
                                if read_error_blocks == 0
                                else f"file={src_path.name} read_error_blocks={read_error_blocks:,}"
                            ),
                            force=(completed_blocks == total_blocks),
                        )
            except Exception:
                write_resume_state(state_path, signature, completed_blocks, started_at, read_error_blocks)
                raise

    if out_path.exists():
        out_path.unlink()
    tmp_path.replace(out_path)
    safe_unlink(state_path)
    return out_path, read_error_blocks


def collect_depth_stack_sources(existing_merged, requested_names):
    groups = defaultdict(list)
    for name in requested_names:
        meta = existing_merged.get(name)
        if meta is None:
            continue
        prop, depth, stat = split_dataset_name(name)
        if not depth or prop == "wrb":
            continue
        groups[(prop, stat)].append({
            "name": name,
            "depth": depth,
            "stat": stat,
            "path": Path(meta["path"]).resolve(),
            "prop": prop,
        })

    out = {}
    for key, items in groups.items():
        out[key] = sorted(items, key=lambda item: depth_sort_key(item["depth"]))
    return out


def build_depth_stack_from_merged(
    src_items,
    out_path,
    gdal_cache_mb,
    blocksize,
    compress,
    bigtiff,
    progress_every_seconds,
    on_read_error,
    num_threads,
    resume_partial,
):
    if not src_items:
        raise ValueError("src_items must not be empty")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    state_path = resume_state_path_for(tmp_path)

    src_paths = [Path(item["path"]).resolve() for item in src_items]

    with rasterio.Env(GDAL_CACHEMAX=int(gdal_cache_mb)):
        src_datasets = [rasterio.open(path) for path in src_paths]
        try:
            ref = src_datasets[0]
            if ref.crs is None:
                raise RuntimeError(f"Dataset has no CRS: {src_paths[0]}")
            if ref.count != 1:
                raise RuntimeError(f"Expected single-band merged TIFF for stack source: {src_paths[0]}")

            dtype_name = np.result_type(*[np.dtype(ds.dtypes[0]) for ds in src_datasets]).name
            nodata_values = [ds.nodata for ds in src_datasets]
            common_nodata = nodata_values[0]
            if any(val != common_nodata for val in nodata_values[1:]):
                common_nodata = None

            for idx, ds in enumerate(src_datasets[1:], start=1):
                if ds.count != 1:
                    raise RuntimeError(f"Expected single-band merged TIFF for stack source: {src_paths[idx]}")
                if ds.crs != ref.crs:
                    raise RuntimeError(f"CRS mismatch between {src_paths[0]} and {src_paths[idx]}")
                if ds.width != ref.width or ds.height != ref.height:
                    raise RuntimeError(f"Shape mismatch between {src_paths[0]} and {src_paths[idx]}")
                if ds.transform != ref.transform:
                    raise RuntimeError(f"Transform mismatch between {src_paths[0]} and {src_paths[idx]}")

            block_h_src, block_w_src = get_block_shape(ref, fallback=blocksize)
            block_h = int(blocksize or block_h_src)
            block_w = int(blocksize or block_w_src)
            predictor = predictor_for_dtype(dtype_name)

            profile = ref.profile.copy()
            profile.update(
                driver="GTiff",
                count=len(src_datasets),
                dtype=dtype_name,
                nodata=common_nodata,
                tiled=True,
                compress=str(compress),
                predictor=int(predictor),
                BIGTIFF=str(bigtiff),
                blockxsize=int(block_w),
                blockysize=int(block_h),
                NUM_THREADS=str(num_threads),
                interleave="band",
            )

            width = int(ref.width)
            height = int(ref.height)
            total_blocks = int(math.ceil(height / block_h) * math.ceil(width / block_w))
            signature = build_stack_resume_signature(src_items, out_path, profile, total_blocks, on_read_error)
            resume_ok, completed_blocks, state_path = prepare_resume(tmp_path, signature, resume_partial)
            started_at = time.time()
            read_error_blocks = 0
            progress_state = make_progress_state(
                phase_label=f"stack {out_path.stem}",
                total=total_blocks,
                unit_name="blocks",
                report_every_seconds=progress_every_seconds,
                started_at=started_at,
            )

            if resume_ok:
                progress_state["last_report_completed"] = completed_blocks
                maybe_emit_progress(
                    progress_state,
                    completed_blocks,
                    extra=f"file={out_path.name} resumed=1",
                    force=True,
                )
                open_mode = "r+"
            else:
                safe_unlink(tmp_path)
                safe_unlink(state_path)
                open_mode = "w"

            if open_mode == "w":
                write_resume_state(state_path, signature, 0, started_at, 0)

            try:
                with rasterio.open(tmp_path, open_mode, **({} if open_mode == "r+" else profile)) as dst:
                    if open_mode == "w":
                        dst.update_tags(stack_kind="soilgrids_depthstack")
                        for band_idx, item in enumerate(src_items, start=1):
                            dst.set_band_description(band_idx, str(item["name"]))
                            dst.update_tags(
                                band_idx,
                                logical_name=str(item["name"]),
                                prop=str(item["prop"]),
                                depth=str(item["depth"]),
                                stat=str(item["stat"]),
                            )

                    for block_idx, (_, window) in enumerate(dst.block_windows(1), start=1):
                        if block_idx <= completed_blocks:
                            continue
                        block_arrays = []
                        for ds in src_datasets:
                            try:
                                arr = ds.read(1, window=window, masked=False)
                            except Exception as exc:
                                if on_read_error == "raise":
                                    write_resume_state(state_path, signature, block_idx - 1, started_at, read_error_blocks)
                                    raise
                                maybe_log_read_error("stack", ds.name, window, exc, block_idx, total_blocks)
                                fill_value = default_fill_value(dtype_name, common_nodata)
                                arr = np.full(
                                    (int(window.height), int(window.width)),
                                    fill_value,
                                    dtype=np.dtype(dtype_name),
                                )
                                read_error_blocks += 1
                            if arr.dtype != np.dtype(dtype_name):
                                arr = arr.astype(dtype_name, copy=False)
                            block_arrays.append(arr)
                        dst.write(np.stack(block_arrays, axis=0), window=window)
                        completed_blocks = block_idx
                        if (
                            completed_blocks == total_blocks
                            or completed_blocks == 1
                            or completed_blocks % 32 == 0
                        ):
                            write_resume_state(state_path, signature, completed_blocks, started_at, read_error_blocks)
                        maybe_emit_progress(
                            progress_state,
                            completed_blocks,
                            extra=(
                                f"file={out_path.name}"
                                if read_error_blocks == 0
                                else f"file={out_path.name} read_error_blocks={read_error_blocks:,}"
                            ),
                            force=(completed_blocks == total_blocks),
                        )
            except Exception:
                write_resume_state(state_path, signature, completed_blocks, started_at, read_error_blocks)
                raise
        finally:
            for ds in src_datasets:
                ds.close()

    if out_path.exists():
        out_path.unlink()
    tmp_path.replace(out_path)
    safe_unlink(state_path)
    return out_path, read_error_blocks


def parse_overview_levels(value):
    levels = []
    for part in parse_csv_list(value):
        level = int(part)
        if level > 1:
            levels.append(level)
    return sorted(dict.fromkeys(levels))


def main():
    ap = argparse.ArgumentParser(
        description="Materialize SoilGrids VRT mosaics into real tiled GeoTIFFs"
    )
    ap.add_argument("--root", required=True, help="Path to SoilGrids data folder containing property subfolders and VRTs")
    ap.add_argument("--out-root", required=True, help="Output folder for merged GeoTIFFs")
    ap.add_argument(
        "--props",
        default="",
        help="Comma-separated properties, e.g. bdod,cec,clay,sand,silt,soc,phh2o,nitrogen,cfvo,wrb",
    )
    ap.add_argument(
        "--depths",
        default="",
        help="Comma-separated depths, e.g. 0-5cm,5-15cm,15-30cm",
    )
    ap.add_argument(
        "--datasets",
        default="",
        help="Explicit comma-separated dataset names, e.g. bdod_0-5cm_mean,clay_15-30cm_mean,wrb or all",
    )
    ap.add_argument(
        "--stat",
        default="mean",
        help="Suffix after depth when composing names from --props and --depths",
    )
    ap.add_argument(
        "--merge-mode",
        default="single",
        choices=["single", "depth-stacks", "all"],
        help="What to build: only single-depth TIFFs, only depth stacks from existing single-depth TIFFs, or both",
    )
    ap.add_argument("--list", action="store_true", help="List discovered VRT datasets and exit")
    ap.add_argument("--flat", action="store_true", help="Write all output TIFFs directly into --out-root")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild selected outputs even if they already exist")
    ap.add_argument("--gdal-cache-mb", type=int, default=512, help="GDAL cache size in MB")
    ap.add_argument("--blocksize", type=int, default=512, help="Output GeoTIFF block size")
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["ZSTD", "DEFLATE", "LZW", "NONE"],
        help="GeoTIFF compression",
    )
    ap.add_argument(
        "--bigtiff",
        default="IF_SAFER",
        choices=["YES", "NO", "IF_NEEDED", "IF_SAFER"],
        help="BIGTIFF creation option",
    )
    ap.add_argument(
        "--num-threads",
        default="ALL_CPUS",
        help="GTiff NUM_THREADS creation option",
    )
    ap.add_argument(
        "--on-read-error",
        choices=["raise", "nodata"],
        default="nodata",
        help="On source block read failure, either raise or write nodata/default-fill into that block",
    )
    ap.add_argument(
        "--no-resume-partial",
        action="store_true",
        help="Disable reuse of existing .part outputs and restart partial merges from scratch",
    )
    ap.add_argument(
        "--build-overviews",
        action="store_true",
        help="Build internal overviews after creating each output TIFF",
    )
    ap.add_argument(
        "--overview-levels",
        default="2,4,8,16",
        help="Comma-separated overview decimation levels used with --build-overviews",
    )
    ap.add_argument(
        "--overview-resampling",
        default="nearest",
        choices=["nearest", "average", "bilinear", "mode"],
        help="Overview resampling method",
    )
    ap.add_argument(
        "--progress-every-seconds",
        type=float,
        default=10.0,
        help="Emit progress at least this often while merging",
    )

    args = ap.parse_args()

    if args.blocksize < 64:
        raise ValueError("--blocksize must be >= 64")
    if args.progress_every_seconds <= 0:
        raise ValueError("--progress-every-seconds must be > 0")

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    all_datasets = discover_vrts(root)
    if not all_datasets:
        raise RuntimeError(f"No VRTs found under: {root}")

    if args.list:
        for name in sorted(all_datasets.keys()):
            print(name)
        return

    props = parse_csv_list(args.props)
    depths = parse_csv_list(args.depths)
    explicit_datasets = parse_csv_list(args.datasets)

    requested_names = build_requested_dataset_names(
        all_datasets=all_datasets,
        props=props,
        depths=depths,
        stat=args.stat,
        explicit_datasets=explicit_datasets,
    )
    if not requested_names:
        raise RuntimeError("No datasets selected")

    overview_levels = parse_overview_levels(args.overview_levels)
    try:
        overview_resampling = getattr(rasterio.enums.Resampling, str(args.overview_resampling))
    except AttributeError as exc:
        raise ValueError(f"Unsupported overview resampling: {args.overview_resampling}") from exc

    print(f"Preparing {len(requested_names)} dataset(s)...", file=sys.stderr)
    phase_started = time.time()
    rebuilt = 0
    reused = 0
    stack_rebuilt = 0
    stack_reused = 0
    merge_read_error_blocks = 0
    stack_read_error_blocks = 0

    if args.merge_mode in {"single", "all"}:
        for idx, name in enumerate(requested_names, start=1):
            meta = all_datasets[name]
            src_path = Path(meta["path"]).resolve()
            out_path = build_output_path(out_root, name, meta, flat=args.flat)

            if not args.overwrite and output_is_fresh(out_path, src_path):
                reused += 1
                emit_progress_line(
                    phase_label="merge-datasets",
                    completed=idx,
                    total=len(requested_names),
                    started_at=phase_started,
                    unit_name="datasets",
                    extra=f"reused={reused:,} rebuilt={rebuilt:,} current={name}",
                )
                continue

            print(f"[merge] dataset={name} src={src_path} out={out_path}", file=sys.stderr, flush=True)
            dataset_started = time.time()
            _, read_error_blocks = merge_vrt_to_tif(
                src_path=src_path,
                out_path=out_path,
                gdal_cache_mb=args.gdal_cache_mb,
                blocksize=args.blocksize,
                compress=args.compress,
                bigtiff=args.bigtiff,
                progress_every_seconds=args.progress_every_seconds,
                on_read_error=args.on_read_error,
                num_threads=args.num_threads,
                resume_partial=not args.no_resume_partial,
            )
            merge_read_error_blocks += int(read_error_blocks)
            if args.build_overviews and overview_levels:
                build_overviews(out_path, overview_levels, overview_resampling)
            rebuilt += 1
            print(
                f"[merge] done dataset={name} elapsed={format_elapsed(time.time() - dataset_started)} read_error_blocks={read_error_blocks:,}",
                file=sys.stderr,
                flush=True,
            )
            emit_progress_line(
                phase_label="merge-datasets",
                completed=idx,
                total=len(requested_names),
                started_at=phase_started,
                unit_name="datasets",
                extra=(
                    f"reused={reused:,} rebuilt={rebuilt:,} current={name} "
                    f"read_error_blocks={merge_read_error_blocks:,}"
                ),
            )

    if args.merge_mode in {"depth-stacks", "all"}:
        existing_merged = discover_existing_single_depth_tifs(out_root)
        stack_groups = collect_depth_stack_sources(existing_merged, requested_names)
        if not stack_groups:
            print(
                "[stack] no eligible existing single-depth merged TIFFs found for selected datasets",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"[stack] preparing {len(stack_groups)} depth stack(s)", file=sys.stderr, flush=True)

        stack_items = [item for item in sorted(stack_groups.items()) if len(item[1]) >= 2]
        for idx, ((prop, stat), src_items) in enumerate(stack_items, start=1):
            out_path = build_depth_stack_path(out_root, prop, stat, flat=args.flat)
            src_paths = [item["path"] for item in src_items]

            if not args.overwrite and output_is_fresh_against_many(out_path, src_paths):
                stack_reused += 1
                emit_progress_line(
                    phase_label="merge-depth-stacks",
                    completed=idx,
                    total=len(stack_items),
                    started_at=phase_started,
                    unit_name="stacks",
                    extra=f"reused={stack_reused:,} rebuilt={stack_rebuilt:,} current={out_path.stem}",
                )
                continue

            print(
                f"[stack] dataset={out_path.stem} bands={len(src_items)} srcs={','.join(item['name'] for item in src_items)} out={out_path}",
                file=sys.stderr,
                flush=True,
            )
            stack_started = time.time()
            _, read_error_blocks = build_depth_stack_from_merged(
                src_items=src_items,
                out_path=out_path,
                gdal_cache_mb=args.gdal_cache_mb,
                blocksize=args.blocksize,
                compress=args.compress,
                bigtiff=args.bigtiff,
                progress_every_seconds=args.progress_every_seconds,
                on_read_error=args.on_read_error,
                num_threads=args.num_threads,
                resume_partial=not args.no_resume_partial,
            )
            stack_read_error_blocks += int(read_error_blocks)
            if args.build_overviews and overview_levels:
                build_overviews(out_path, overview_levels, overview_resampling)
            stack_rebuilt += 1
            print(
                f"[stack] done dataset={out_path.stem} elapsed={format_elapsed(time.time() - stack_started)} read_error_blocks={read_error_blocks:,}",
                file=sys.stderr,
                flush=True,
            )
            emit_progress_line(
                phase_label="merge-depth-stacks",
                completed=idx,
                total=len(stack_items),
                started_at=phase_started,
                unit_name="stacks",
                extra=(
                    f"reused={stack_reused:,} rebuilt={stack_rebuilt:,} current={out_path.stem} "
                    f"read_error_blocks={stack_read_error_blocks:,}"
                ),
            )

    print(
        "Finished. "
        f"single_rebuilt={rebuilt:,} single_reused={reused:,} "
        f"stack_rebuilt={stack_rebuilt:,} stack_reused={stack_reused:,} "
        f"merge_read_error_blocks={merge_read_error_blocks:,} stack_read_error_blocks={stack_read_error_blocks:,} "
        f"elapsed={format_elapsed(time.time() - phase_started)}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
