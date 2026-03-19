#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

#test run: python prune_non_cog.py --root "D:\DEM_derived_w_flow"
#deletes if cog equivalents are OK: python prune_non_cog.py --root "D:\DEM_derived_w_flow" --verify --delete --remove-aux

def run_ok(cmd):
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0

def find_cog_for(src: Path) -> Path | None:
    # your cogify writes: <stem>.cog.<suffix>
    if src.name.lower().endswith(".cog.tif") or src.name.lower().endswith(".cog.tiff"):
        return None

    cog1 = src.with_name(src.stem + ".cog" + src.suffix)  # foo.tif -> foo.cog.tif
    if cog1.exists() and cog1.stat().st_size > 0:
        return cog1

    # handle .tif vs .tiff mismatch
    if src.suffix.lower() == ".tif":
        cog2 = src.with_name(src.stem + ".cog.tiff")
        if cog2.exists() and cog2.stat().st_size > 0:
            return cog2
    if src.suffix.lower() == ".tiff":
        cog3 = src.with_name(src.stem + ".cog.tif")
        if cog3.exists() and cog3.stat().st_size > 0:
            return cog3

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--delete", action="store_true", help="Actually delete. Otherwise dry-run.")
    ap.add_argument("--remove-aux", action="store_true", help="Also delete *.aux.xml siblings for deleted files.")
    ap.add_argument("--verify", action="store_true", help="Run gdalinfo on the COG before deleting the source.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    deleted = 0
    skipped = 0
    missing = 0
    errors = 0

    for src in root.rglob("*"):
        if not src.is_file():
            continue
        nl = src.name.lower()
        if not (nl.endswith(".tif") or nl.endswith(".tiff")):
            continue
        if nl.endswith(".aux.xml"):
            continue
        if nl.endswith(".cog.tif") or nl.endswith(".cog.tiff"):
            continue

        cog = find_cog_for(src)
        if cog is None:
            missing += 1
            continue

        if args.verify:
            if not run_ok(["gdalinfo", str(cog)]):
                print(f"[bad-cog] {cog}")
                errors += 1
                continue

        if args.delete:
            try:
                src.unlink()
                deleted += 1
                if args.remove_aux:
                    aux = Path(str(src) + ".aux.xml")
                    if aux.exists():
                        aux.unlink()
                print(f"[deleted] {src}")
            except Exception as e:
                print(f"[error] {src} :: {e}")
                errors += 1
        else:
            skipped += 1
            print(f"[would-delete] {src}  (has {cog.name})")

    print(f"done: would_delete={skipped} deleted={deleted} missing_cog={missing} errors={errors}")

if __name__ == "__main__":
    main()