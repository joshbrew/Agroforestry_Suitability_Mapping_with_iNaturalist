#!/usr/bin/env python3
import subprocess
from pathlib import Path


GDB = Path(r".\LiMW_GIS.gdb")
OUT = Path(r".\glim_rasters")
OUT.mkdir(exist_ok=True)

GPKG = OUT / "glim.gpkg"
LOOKUP_CSV = OUT / "glim_lookup.csv"
TIF = OUT / "glim_id_1km.tif"
VRT = OUT / "glim_id_1km.vrt"

LAYER_NAME = "GLiM_export"

EXTENT = (
    -16653453.7035,
    -8460600.9615,
    16653453.7035,
    8376733.0557,
)

RES = 1000


def run(cmd):
    print("RUN:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


run([
    "ogr2ogr",
    "-overwrite",
    "-f", "GPKG",
    str(GPKG),
    str(GDB),
    LAYER_NAME,
    "-dialect", "OGRSQL",
    "-sql",
    f"SELECT fid AS glim_id, IDENTITY_, Litho, xx, Shape FROM {LAYER_NAME}",
    "-nln", LAYER_NAME,
])

run([
    "ogr2ogr",
    "-overwrite",
    "-f", "CSV",
    str(LOOKUP_CSV),
    str(GPKG),
    "-dialect", "OGRSQL",
    "-sql",
    f"SELECT glim_id, IDENTITY_, Litho, xx FROM {LAYER_NAME}",
])

run([
    "gdal_rasterize",
    "-l", LAYER_NAME,
    "-a", "glim_id",
    "-a_nodata", "0",
    "-init", "0",
    "-ot", "UInt32",
    "-of", "GTiff",
    "-tr", str(RES), str(RES),
    "-te",
    str(EXTENT[0]), str(EXTENT[1]),
    str(EXTENT[2]), str(EXTENT[3]),
    "-co", "TILED=YES",
    "-co", "COMPRESS=LZW",
    "-co", "BIGTIFF=YES",
    str(GPKG),
    str(TIF),
])

run([
    "gdaladdo",
    "-r", "nearest",
    str(TIF),
    "2", "4", "8", "16", "32",
])

run([
    "gdalbuildvrt",
    str(VRT),
    str(TIF),
])