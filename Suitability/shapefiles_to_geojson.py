#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# python shapefiles_to_geojson.py "D:/your_folder"
# python shapefiles_to_geojson.py "D:/your_folder" --recursive
# python shapefiles_to_geojson.py "D:/your_folder" --outdir "D:/geojson_out"
# python shapefiles_to_geojson.py "D:/your_folder" --overwrite

# Keep the original CRS instead of converting to WGS84:
# python shapefiles_to_geojson.py "D:/your_folder" --keep-crs

def find_shapefiles(folder: Path, recursive: bool):
    pattern = "**/*.shp" if recursive else "*.shp"
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def build_ogr2ogr_command(src: Path, dst: Path, overwrite: bool, keep_crs: bool):
    cmd = ["ogr2ogr"]

    if overwrite:
        cmd.append("-overwrite")

    cmd += [
        "-f", "GeoJSON",
        str(dst),
        str(src),
    ]

    if not keep_crs:
        cmd += ["-t_srs", "EPSG:4326", "-lco", "RFC7946=YES"]

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Convert shapefiles in a folder to GeoJSON using GDAL ogr2ogr."
    )
    parser.add_argument("input_folder", help="Folder containing shapefile sets")
    parser.add_argument(
        "--outdir",
        help="Output folder for GeoJSON files. Defaults to input folder.",
        default=None,
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders recursively",
    )
    parser.add_argument(
        "--keep-crs",
        action="store_true",
        help="Keep source CRS instead of reprojecting to EPSG:4326",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GeoJSON files",
    )

    args = parser.parse_args()

    if shutil.which("ogr2ogr") is None:
        print("Error: ogr2ogr was not found on PATH.", file=sys.stderr)
        print("Make sure GDAL is installed and ogr2ogr is available in your terminal.", file=sys.stderr)
        sys.exit(1)

    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Error: input folder does not exist or is not a directory: {input_folder}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else input_folder
    outdir.mkdir(parents=True, exist_ok=True)

    shapefiles = find_shapefiles(input_folder, args.recursive)
    if not shapefiles:
        print(f"No .shp files found in {input_folder}")
        return

    print(f"Found {len(shapefiles)} shapefile(s)")

    converted = 0
    skipped = 0
    failed = 0

    for src in shapefiles:
        if args.recursive:
            rel_parent = src.parent.relative_to(input_folder)
            dst_folder = outdir / rel_parent
        else:
            dst_folder = outdir

        dst_folder.mkdir(parents=True, exist_ok=True)
        dst = dst_folder / f"{src.stem}.geojson"

        if dst.exists() and not args.overwrite:
            print(f"[skip] {dst} already exists")
            skipped += 1
            continue

        cmd = build_ogr2ogr_command(src, dst, args.overwrite, args.keep_crs)

        print(f"[convert] {src} -> {dst}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            converted += 1
        else:
            failed += 1
            print(f"[failed] {src}", file=sys.stderr)
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)

    print()
    print(f"Done. converted={converted} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()