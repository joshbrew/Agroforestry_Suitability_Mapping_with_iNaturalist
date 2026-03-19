#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print("RUN:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Convert an existing GLiM raster GeoTIFF into a Cloud Optimized GeoTIFF."
    )
    ap.add_argument("src_tif", help="Input GeoTIFF, for example glim_id_1km.tif")
    ap.add_argument(
        "dst_cog",
        nargs="?",
        help="Output COG path. Defaults to input name with _cog suffix.",
    )
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["ZSTD", "DEFLATE", "LZW", "LERC", "NONE"],
        help="COG compression method.",
    )
    ap.add_argument(
        "--blocksize",
        default="512",
        help="Internal tile size for the COG.",
    )
    ap.add_argument(
        "--resampling",
        default="NEAREST",
        choices=["NEAREST", "BILINEAR", "CUBIC", "CUBICSPLINE", "LANCZOS", "AVERAGE", "MODE"],
        help="Overview resampling. For categorical geology IDs, use NEAREST.",
    )
    ap.add_argument(
        "--overview-levels",
        nargs="*",
        default=["2", "4", "8", "16", "32", "64", "128"],
        help="Overview levels to build before COG creation.",
    )
    args = ap.parse_args()

    src = Path(args.src_tif)
    if not src.exists():
        raise SystemExit(f"Input raster not found: {src}")

    if args.dst_cog:
        dst = Path(args.dst_cog)
    else:
        dst = src.with_name(f"{src.stem}_cog.tif")

    temp = dst.with_name(f"{dst.stem}__temp.tif")

    run([
        "gdal_translate",
        str(src),
        str(temp),
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", f"COMPRESS={args.compress}",
        "-co", "BIGTIFF=IF_SAFER",
        "-co", f"BLOCKXSIZE={args.blocksize}",
        "-co", f"BLOCKYSIZE={args.blocksize}",
    ])

    if args.overview_levels:
        run([
            "gdaladdo",
            "-r", args.resampling.lower(),
            str(temp),
            *args.overview_levels,
        ])

    run([
        "gdal_translate",
        str(temp),
        str(dst),
        "-of", "COG",
        "-co", f"COMPRESS={args.compress}",
        "-co", "BIGTIFF=IF_SAFER",
        "-co", f"BLOCKSIZE={args.blocksize}",
        "-co", f"RESAMPLING={args.resampling}",
    ])

    try:
        temp.unlink()
    except OSError:
        pass

    run(["gdalinfo", "-json", str(dst)])
    print(f"\nWrote COG: {dst}")


if __name__ == "__main__":
    main()
