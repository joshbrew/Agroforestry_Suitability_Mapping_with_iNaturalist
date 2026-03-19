#!/usr/bin/env bash
set -euo pipefail

BASE="https://files.isric.org/soilgrids/latest/data"
OUT="soilgrids_mirror"
STAT="mean"
DEPTHS=(0-5cm 5-15cm 15-30cm)

# Agroforestry bundle + WRB class layer, recommend you actually run multiple terminals for like groups of 2 or 3 of these so you dont spend a week downloading all of the files.
PROPS=(clay phh2o soc nitrogen cec clay silt sand bdod cfvo wrb) # phh2o soc nitrogen cec clay silt sand bdod cfvo wrb

mkdir -p "$OUT"

# Optional root metadata
wget -c -N -nH -x -P "$OUT" -e robots=off \
  "$BASE/README.md" \
  "$BASE/checksum.sha256.txt" 2>/dev/null || true

WGET_COMMON=(
  -c
  -nc
  -e robots=off
  --retry-connrefused
  --waitretry=2
  --tries=10
)

for p in "${PROPS[@]}"; do
  if [[ "$p" == "wrb" ]]; then
    echo "== ${p} (mirror folder) =="
    wget "${WGET_COMMON[@]}" -r -np -nH -P "$OUT" \
      -A "*.vrt,*.vrt.ovr,*.tif,*.txt,*.md" -R "index.html*" \
      "$BASE/$p/"
    continue
  fi

  for d in "${DEPTHS[@]}"; do
    layer="${p}_${d}_${STAT}"
    echo "== ${layer} =="

    # VRTs (and overview if present)
    wget "${WGET_COMMON[@]}" -nH -x -P "$OUT" \
      "$BASE/$p/${layer}.vrt" \
      "$BASE/$p/${layer}.vrt.ovr" 2>/dev/null || true

    # Tile GeoTIFF chunks
    wget "${WGET_COMMON[@]}" -r -np -nH -P "$OUT" \
      -A "*.tif" -R "index.html*" \
      "$BASE/$p/$layer/"
  done
done

echo "Done."
echo "Example local VRT:"
echo "  $OUT/soilgrids/latest/data/phh2o/phh2o_0-5cm_mean.vrt"