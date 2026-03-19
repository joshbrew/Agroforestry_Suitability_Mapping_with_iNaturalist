Resolutions:
- discrete iNaturalist occurrences (with error)
- DEM + flow + derived: 3 arcsec (30.87m)
- Terraclimate: 4km
- MCD12Q1 2024 landcover: 500m




iNaturalist Research-Grade Observations

- D:\iNaturalistOccurrenceData
    - - global coordinate-defined species observations
    - - contains a UI viewer
    - - source: https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7

- D:\DEM_derived_w_flow 
    - - HydroSheds 3s DEM + Flow accumulation and direction + derived slope, aspect, northness, and eastness
    - - Use sample_dem_flow_coords_fast.py to pull data via csv (id, lat, lon)
        - - - samples derived COGs
    - - Source: https://www.hydrosheds.org/products/hydrosheds (use the void-filled DEM)

- D:\wetness
    - - 120m global wetness
    - - Source: https://zenodo.org/records/14920387

- D:\soilgrids
    - - 250m .tifs, includes VRTs for quick lookups.
    - - 0-5cm,5-15cm,15-30cm global estimates of 
        phh2o soc nitrogen cec clay silt sand bdod cfvo wrb (wrb is the USDA soil group e.g. vertisols)
    - - Source: https://isric.org/explore/soilgrids pulled via a get.sh script

- D:\GLiM
    - - 1:3,750,000 scale lithological map 
    - - coarse geologic feature set, SoilGrids used it as a training layer.
    - - Source: https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glim.html

- D:\MCD12Q1_landcover
    - - 2024 500m landcover map.
    - - Legend in section 5.1: https://lpdaac.usgs.gov/documents/101/MCD12_User_Guide_V6.pdf
    - - Source: https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MCD12Q1/2024/001/


- D:\terraclimate
    - - 4km global land aet, def, pet, ppt, q, soil, srad, swe, tmax, tmin, vap, ws, vpd, pdsi, 1958-2024
    - - terraclimate_from_coords.py to pull data via csv (id, lat, lon)
        - - - samples derived COGs 
    - - Source: https://www.climatologylab.org/terraclimate.html

aet (Actual Evapotranspiration)
def (Climate Water Deficit)
pet (Potential evapotranspiration)
ppt (Precipitation)
q (Runoff)
soil (Soil Moisture)
srad (Downward surface shortwave radiation)
swe (Snow water equivalent - at end of month)
tmax (Max Temperature)
tmin (Min Temperature)
vap (Vapor pressure)
ws (Wind speed)
vpd (Vapor Pressure Deficit)
pdsi (Palmer Drought Severity Index)

- D:\GlobalCropland30m
    - - 30m cropland TIFs
    - - sample_croplands_coords.py to pull data via csv (id, lat, lon)
    - - Source: https://pubs.usgs.gov/publication/pp1868


Sampling:

```ts
Sample input
 id,lon,lat
 1,-123.1,44.05
 2,-122.68,45.52
 3,-120.5,43.7
```

Get DEM + Derived:

```bash
python D:/DEM_Derived_w_flow/sample_coords.py --coords coords.csv --index "D:/DEM_derived_w_flow/dem_flow_index.json" --layers all --out sampled.csv
```

Get TerraClimate:

```bash
python D:/terraclimate/sample_cogs_from_coords.py --cog-root D:/terraclimate/terraclimate_cogs_global --coords coords.csv --vars all --year latest --out samples.csv
```
Note this samples every variable for all 12 months of the selected year

Get TWI:

```bash
python D:/wetness/sample_twi_coords.py --tif D:/wetness/twi_edtm_120m.tif --input path/to/coords.csv --output path/to/coords_with_twi.csv --lon-col lon --lat-col lat --chunk-size 250000
```

Get SoilGrids:
```bash
python D:/soilgrids/sample_soilgrids_coords.py --root "path/to/data" --coords "path/to/points.csv" --out "path/to/points_soilgrids.csv" --props "bdod,cec,clay,sand,silt,soc,phh2o,nitrogen,cfvo" --depths "0-5cm,5-15cm,15-30cm" --chunk-size 200000 --gdal-cache-mb 2048
```

Get GLiM (from rasterized cogs)
```bash
py D:/GLiM/sample_glim_coords.py D:/GLiM/glim_rasters/glim_id_1km_cog.tif dummycoords.csv dummycoords_with_glim.csv --lookup-csv D:/GLiM/glim_rasters/glim_lookup.csv --x-col lon --y-col lat --input-crs EPSG:4326
```

Get MCD12Q1
```pbashy
py D:/MCD12Q1_landcover/sample_coords.py MCD12Q1_landcover/cogs/mcd12q1_lc_type1.vrt path/to/coords.csv path/to/coords_sampled.csv --x-col lon --y-col lat --value-col mcd12q1
```

Get Cropland (optional):
```bash
python sample_global_cropland.py sample --raster D:/GlobalCropland30m/data/global_cropland.vrt --input path/to/coords.csv --output path/to/coords_with_cropland.csv --lon-col lon --lat-col lat --chunk-size 250000 --value-field cropland30m
```

Get planted forest (optional), 30m sampled grid. Does not cover every country.
```bash
python D:/PFTD30m/sample_pfdt_coords.py --input D:/coords.csv --output D:/coords_with_pftd.csv
```


Run on iNaturalistOccurrences in bulk

```bash
node collect_taxa_env.js occurrences.csv --phase all --include-taxa "species:Daucus pusillus" --out-root D:/envpull_daucus --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 --cleanup-enriched invalid+soilgrids --cleanup
```

```bash
 node collect_taxa_env.js occurrences.csv --phase all --include-taxa "species:Daucus pusillus" --out-root D:/envpull_daucus --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 --dem-script D:/DEM_Derived_w_flow/sample_coords.py --dem-index D:/DEM_derived_w_flow/dem_flow_index.json --terraclimate-script D:/terraclimate/sample_cogs_from_coords.py --terraclimate-root D:/terraclimate/terraclimate_cogs_global --twi-script D:/wetness/sample_twi_coords.py --twi-tif D:/wetness/twi_edtm_120m.tif --soilgrids-script D:/soilgrids/sample_soilgrids_coords.py --soilgrids-root D:/soilgrids/data --glim-script D:/GLiM/sample_glim_coords.py --glim-tif D:/GLiM/glim_rasters/glim_id_1km_cog.tif --glim-lookup-csv D:/GLiM/glim_rasters/glim_lookup.csv --mcd12q1-script D:/MCD12Q1_landcover/sample_coords.py --mcd12q1-vrt D:/MCD12Q1_landcover/cogs/mcd12q1_lc_type1.vrt --mcd12q1-value-col mcd12q1 --cleanup-enriched invalid+soilgrids --cleanup 

```


