#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
import rasterio


def abspath(p: str) -> str:
  return os.path.abspath(os.path.expanduser(p))


def valid_ratio(path: str, max_blocks: int = 16, early_keep: float = 0.02) -> float:
  # Samples up to max_blocks blocks. Early exits if we already have enough valid pixels.
  with rasterio.open(path) as ds:
    nod = ds.nodata
    valid = 0
    total = 0
    blocks = 0

    for _, win in ds.block_windows(1):
      a = ds.read(1, window=win)

      if nod is None:
        m = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, dtype=bool)
      else:
        m = (a != nod)
        if a.dtype.kind == "f":
          m &= np.isfinite(a)

      valid += int(m.sum())
      total += int(m.size)
      blocks += 1

      if total and (valid / total) >= early_keep:
        return valid / total

      if blocks >= max_blocks:
        break

    return (valid / total) if total else 0.0


def tile_base_from_probe(path: str) -> str:
  for suf in ["_slope_deg.tif", "_aspect_deg.tif", "_dem.tif", "_northness.tif", "_eastness.tif", "_twi.tif"]:
    if path.endswith(suf):
      return path[: -len(suf)]
  root, _ = os.path.splitext(path)
  for suf in ["_slope_deg", "_aspect_deg", "_dem", "_northness", "_eastness", "_twi"]:
    if root.endswith(suf):
      return root[: -len(suf)]
  return root


def collect_siblings(base: str) -> list:
  pats = [
    base + "_*.tif",
    base + "_*.tiff",
    base + "_*.tif.aux.xml",
    base + "_*.tiff.aux.xml",
    base + "*.empty",
  ]
  out = []
  for p in pats:
    out.extend(glob.glob(p))
  return sorted(set(out))


def find_probe_dirs(root: str, probe_glob: str, subfolder_suffix: str = "") -> list:
  root = abspath(root)
  suffix = (subfolder_suffix or "").lower()
  out = []

  for dirpath, dirnames, _ in os.walk(root):
    dirnames[:] = sorted(dirnames)

    if suffix and not os.path.basename(dirpath).lower().endswith(suffix):
      continue

    if glob.glob(os.path.join(dirpath, probe_glob)):
      out.append(dirpath)

  return sorted(set(out))


def prune_folder(folder: str, probe_glob: str, threshold: float, max_blocks: int, dry_run: bool, verbose: bool):
  folder = abspath(folder)
  probes = sorted(glob.glob(os.path.join(folder, probe_glob)))
  if not probes:
    return {"folder": folder, "kept": 0, "pruned": 0, "probes": 0}

  kept = 0
  pruned = 0

  for probe in probes:
    try:
      r = valid_ratio(probe, max_blocks=max_blocks)
    except Exception as e:
      if verbose:
        print(f"[skip] {probe} (read error: {e})")
      continue

    if r >= threshold:
      kept += 1
      continue

    base = tile_base_from_probe(probe)
    siblings = collect_siblings(base)

    print(f"[prune] {folder} :: {os.path.basename(base)} valid={r*100:.3f}% files={len(siblings)}")
    if verbose:
      for f in siblings:
        print("  ", f)

    if not dry_run:
      for f in siblings:
        try:
          os.remove(f)
        except FileNotFoundError:
          pass
        except PermissionError:
          print("  [warn] permission:", f)

      try:
        with open(base + ".empty", "w", encoding="utf-8") as fp:
          fp.write("empty\n")
      except Exception:
        pass

    pruned += 1

  return {"folder": folder, "kept": kept, "pruned": pruned, "probes": len(probes)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", required=True, help=r'Root folder containing derived rasters (e.g. D:\DEM\derived_terrain_tiles)')
  ap.add_argument("--probe", default="*_slope_deg.tif", help="Which file pattern to test per tile base (slope is a good proxy).")
  ap.add_argument("--subfolder-suffix", default="", help="Optional folder-name suffix filter. Leave empty to scan every nested subfolder.")
  ap.add_argument("--threshold", type=float, default=0.0005, help="Valid pixel ratio cutoff. 0.005=0.5% valid pixels.")
  ap.add_argument("--max-blocks", type=int, default=128, help="Max raster blocks sampled per file (speed vs accuracy).")
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--verbose", action="store_true")
  args = ap.parse_args()

  root = abspath(args.root)
  if not os.path.isdir(root):
    raise SystemExit(f"Not a folder: {root}")

  subdirs = find_probe_dirs(
    root=root,
    probe_glob=args.probe,
    subfolder_suffix=args.subfolder_suffix,
  )

  if not subdirs:
    where = f" ending with {args.subfolder_suffix}" if args.subfolder_suffix else ""
    raise SystemExit(f"No matching probe folders found under {root}{where}")

  total_kept = 0
  total_pruned = 0
  total_probes = 0

  print("Scanning:")
  for d in subdirs:
    print("  -", d)

  for d in subdirs:
    stats = prune_folder(
      folder=d,
      probe_glob=args.probe,
      threshold=float(args.threshold),
      max_blocks=int(args.max_blocks),
      dry_run=bool(args.dry_run),
      verbose=bool(args.verbose),
    )
    total_kept += stats["kept"]
    total_pruned += stats["pruned"]
    total_probes += stats["probes"]

  print(f"\ndone. probes={total_probes} kept={total_kept} pruned={total_pruned} dry_run={args.dry_run}")


if __name__ == "__main__":
  main()