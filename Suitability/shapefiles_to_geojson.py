#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# python shapefiles_to_geojson.py "D:/your_folder"
# python shapefiles_to_geojson.py "D:/your_folder" --recursive
# python shapefiles_to_geojson.py "D:/your_folder" --recursive --tiger-layer county
# python shapefiles_to_geojson.py "D:/your_folder" --recursive --tiger-layer cousub
# python shapefiles_to_geojson.py "D:/your_folder" --recursive --tigeredges
# python shapefiles_to_geojson.py "D:/your_folder" --recursive --path-contains "_edges"
# python shapefiles_to_geojson.py "D:/your_folder" --path-contains "_edges"
# python shapefiles_to_geojson.py "D:/your_folder" --path-contains "interiorAK" --path-contains "_edges"
# python shapefiles_to_geojson.py "D:/your_folder" --outdir "D:/geojson_out" --overwrite
# python shapefiles_to_geojson.py "D:/your_folder" --keep-crs

TIGER_XML_SUFFIXES = (
    ".shp.ea.iso.xml",
    ".shp.iso.xml",
    ".shp.xml",
    ".iso.xml",
    ".xml",
)

TIGER_LAYER_ALIASES = {
    "countyline": "county",
    "countylines": "county",
    "countyboundary": "county",
    "countyboundaries": "county",
    "countyequivalent": "county",
    "countyandequivalent": "county",
    "countyandequivalentfeature": "county",
    "countyandequivalentfeatures": "county",
    "countyorborough": "county",
    "countyborough": "county",
    "edges": "edges",
    "edge": "edges",
    "faces": "faces",
    "face": "faces",
}


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def normalize_tiger_layer(value: str | None) -> str:
    token = normalize_token(value)
    return TIGER_LAYER_ALIASES.get(token, token)


def parse_path_contains(values) -> list[str]:
    parts = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parts.append(item.lower())
    return parts


def find_shapefiles(folder: Path, recursive: bool):
    pattern = "**/*.shp" if recursive else "*.shp"
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def infer_tiger_layer_from_stem(stem: str) -> str:
    parts = stem.lower().split("_")
    if len(parts) >= 4 and parts[0] == "tl":
        return normalize_tiger_layer(parts[-1])
    return ""


def iter_tiger_xml_sidecars(src: Path):
    seen = set()
    for suffix in TIGER_XML_SUFFIXES:
        if suffix.startswith(".shp"):
            candidate = src.parent / f"{src.name}{suffix}"
        else:
            candidate = src.parent / f"{src.stem}{suffix}"
        if candidate not in seen and candidate.exists() and candidate.is_file():
            seen.add(candidate)
            yield candidate
    explicit = [
        src.parent / f"{src.name}.ea.iso.xml",
        src.parent / f"{src.name}.iso.xml",
        src.parent / f"{src.name}.xml",
        src.with_suffix(".shp.xml"),
        src.with_suffix(".xml"),
    ]
    for candidate in explicit:
        if candidate not in seen and candidate.exists() and candidate.is_file():
            seen.add(candidate)
            yield candidate


def tiger_xml_mentions_layer(src: Path, requested_layer: str) -> bool:
    requested_norm = normalize_tiger_layer(requested_layer)
    if not requested_norm:
        return False
    for xml_path in iter_tiger_xml_sidecars(src):
        try:
            text = xml_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        haystack = normalize_token(text)
        if requested_norm and requested_norm in haystack:
            return True
    return False


def matches_tiger_layer(src: Path, requested_layer: str | None) -> bool:
    requested_norm = normalize_tiger_layer(requested_layer)
    if not requested_norm:
        return True
    inferred = infer_tiger_layer_from_stem(src.stem)
    if inferred:
        return inferred == requested_norm
    return tiger_xml_mentions_layer(src, requested_norm)


def matches_path_contains(src: Path, fragments: list[str]) -> bool:
    if not fragments:
        return True
    haystack = str(src).lower().replace("\\", "/")
    return all(fragment in haystack for fragment in fragments)


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


def resolve_tiger_layer_arg(args) -> str | None:
    tiger_layer = normalize_tiger_layer(args.tiger_layer)
    if args.tigeredges:
        if tiger_layer and tiger_layer != "edges":
            raise ValueError("--tigeredges cannot be combined with a different --tiger-layer value")
        tiger_layer = "edges"
    return tiger_layer or None


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
    parser.add_argument(
        "--tiger-layer",
        default=None,
        help="Only convert TIGER/Line shapefiles matching this layer key, such as county, cousub, place, roads, edges, or faces",
    )
    parser.add_argument(
        "--tigeredges",
        action="store_true",
        help="Only convert TIGER/Line edges shapefiles",
    )
    parser.add_argument(
        "--path-contains",
        action="append",
        default=[],
        help="Only convert shapefiles whose full path contains this text, case-insensitive. Repeat the flag to require multiple fragments. Automatically searches subfolders when used.",
    )

    args = parser.parse_args()

    if shutil.which("ogr2ogr") is None:
        print("Error: ogr2ogr was not found on PATH.", file=sys.stderr)
        print("Make sure GDAL is installed and ogr2ogr is available in your terminal.", file=sys.stderr)
        sys.exit(1)

    try:
        tiger_layer = resolve_tiger_layer_arg(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    path_contains = parse_path_contains(args.path_contains)

    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Error: input folder does not exist or is not a directory: {input_folder}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else input_folder
    outdir.mkdir(parents=True, exist_ok=True)

    search_recursive = args.recursive or bool(path_contains)

    shapefiles = find_shapefiles(input_folder, search_recursive)
    total_discovered = len(shapefiles)

    if path_contains:
        shapefiles = [p for p in shapefiles if matches_path_contains(p, path_contains)]
    if tiger_layer:
        shapefiles = [p for p in shapefiles if matches_tiger_layer(p, tiger_layer)]
    if not shapefiles:
        if tiger_layer and path_contains:
            print(
                f"No .shp files matched TIGER layer '{tiger_layer}' and path fragments {path_contains} in {input_folder}"
            )
        elif tiger_layer:
            print(f"No .shp files matched TIGER layer '{tiger_layer}' in {input_folder}")
        elif path_contains:
            print(f"No .shp files matched path fragments {path_contains} in {input_folder}")
        else:
            print(f"No .shp files found in {input_folder}")
        if path_contains and not args.recursive:
            print("Note: --path-contains now searches subfolders automatically.")
        return

    print(f"Discovered {total_discovered} shapefile(s){' recursively' if search_recursive else ''}")
    if path_contains:
        print(f"Filtering to path fragments: {path_contains}")
    if tiger_layer:
        print(f"Filtering to TIGER layer: {tiger_layer}")
    print(f"Matched {len(shapefiles)} shapefile(s)")

    converted = 0
    skipped = 0
    failed = 0

    for src in shapefiles:
        if search_recursive:
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
