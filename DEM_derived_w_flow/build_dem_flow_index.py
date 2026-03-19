#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


# python build_dem_flow_index.py --root "D:\DEM_derived_w_flow" --prefer-cog --out "D:\DEM_derived_w_flow\dem_flow_index.json"


CONT_CODES = ["af", "as", "au", "eu", "na", "sa"]
BIG_LAYOUT = {
    "dem": ("DEM", "_dem_3s"),
    "flowacc": ("FlowAccumulation", "_acc_3s"),
    "flowdir": ("FlowDir", "_dir_3s"),
}
DERIVED_FOLDER_DEFAULT = "Derived_Aspect_Slope_Eastness_Northness"
DERIVED_LAYERS = {
    "slope_deg": "_slope_deg",
    "aspect_deg": "_aspect_deg",
    "northness": "_northness",
    "eastness": "_eastness",
}
RX_DERIVED = re.compile(
    r"^(x-?\d+_y-?\d+)(?P<suffix>_slope_deg|_aspect_deg|_northness|_eastness)(?:\.cog)?\.(?:tif|tiff)$",
    re.IGNORECASE,
)


def wgs84_bounds(path: Path):
    with rasterio.open(path) as ds:
        b = ds.bounds
        if ds.crs is None or str(ds.crs).upper() == "EPSG:4326":
            return [float(b.left), float(b.bottom), float(b.right), float(b.top)]
        bb = transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
        return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]


def choose_big_raster(folder: Path, stem: str, prefer_cog: bool):
    if not folder.exists():
        return None
    exact_order = []
    if prefer_cog:
        exact_order.extend([
            folder / f"{stem}.cog.tif",
            folder / f"{stem}.cog.tiff",
            folder / f"{stem}.tif",
            folder / f"{stem}.tiff",
        ])
    else:
        exact_order.extend([
            folder / f"{stem}.tif",
            folder / f"{stem}.tiff",
            folder / f"{stem}.cog.tif",
            folder / f"{stem}.cog.tiff",
        ])
    for p in exact_order:
        if p.exists() and p.is_file():
            return p

    cands = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        nl = p.name.lower()
        if nl.endswith(".aux.xml") or nl.endswith(".ovr") or nl.endswith(".ovr.tmp"):
            continue
        if nl.endswith(".tif") or nl.endswith(".tiff"):
            cands.append(p)
    if not cands:
        return None
    if prefer_cog:
        cogs = [p for p in cands if ".cog." in p.name.lower()]
        if cogs:
            cogs.sort(key=lambda p: p.stat().st_size, reverse=True)
            return cogs[0]
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description="Build exact manifest for DEM/FlowAcc/FlowDir + derived aspect/slope/northness/eastness files")
    ap.add_argument("--root", required=True)
    ap.add_argument("--derived-folder", default=DERIVED_FOLDER_DEFAULT)
    ap.add_argument("--tile-deg", type=float, default=5.0)
    ap.add_argument("--key-scale", type=int, default=100)
    ap.add_argument("--prefer-cog", action="store_true")
    ap.add_argument("--out", default="dem_flow_index.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    derived_root = root / args.derived_folder

    index = {
        "root": str(root),
        "tile_deg": float(args.tile_deg),
        "key_scale": int(args.key_scale),
        "big_layers": list(BIG_LAYOUT.keys()),
        "derived_layers": list(DERIVED_LAYERS.keys()),
        "continents": {},
    }

    total_tiles = 0
    for cc in CONT_CODES:
        cont = {
            "big": {},
            "derived_folder": str(derived_root / f"{cc}_dem_3s"),
            "tiles": {},
        }

        for layer, (group, suffix) in BIG_LAYOUT.items():
            stem = f"{cc}{suffix}"
            folder = root / group / stem
            p = choose_big_raster(folder, stem, prefer_cog=bool(args.prefer_cog))
            if p is None:
                continue
            cont["big"][layer] = {
                "path": str(p),
                "bounds_wgs84": wgs84_bounds(p),
            }

        ddir = derived_root / f"{cc}_dem_3s"
        if ddir.exists():
            for p in ddir.iterdir():
                if not p.is_file():
                    continue
                m = RX_DERIVED.match(p.name)
                if not m:
                    continue
                key = m.group(1)
                suffix = m.group("suffix").lower()
                layer = None
                for layer_name, wanted_suffix in DERIVED_LAYERS.items():
                    if suffix == wanted_suffix:
                        layer = layer_name
                        break
                if layer is None:
                    continue
                entry = cont["tiles"].setdefault(key, {})
                prev = entry.get(layer)
                if prev is None:
                    entry[layer] = str(p)
                else:
                    prev_is_cog = ".cog." in Path(prev).name.lower()
                    cur_is_cog = ".cog." in p.name.lower()
                    if bool(args.prefer_cog):
                        if cur_is_cog and not prev_is_cog:
                            entry[layer] = str(p)
                    else:
                        if (not cur_is_cog) and prev_is_cog:
                            entry[layer] = str(p)
            total_tiles += len(cont["tiles"])

        index["continents"][cc] = cont

    out = Path(args.out).resolve()
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Indexed continents={len(index['continents'])} tile_keys={total_tiles}")


if __name__ == "__main__":
    raise SystemExit(main())
