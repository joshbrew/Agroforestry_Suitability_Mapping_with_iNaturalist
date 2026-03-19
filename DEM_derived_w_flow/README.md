This dataset is derived with the HydroSheds 3 arcsecond DEM + 3 arcsecond (~30m) Flow Accumulation and Flow Direction. You can remove the original .tifs after deriving the COGs

Download: https://www.hydrosheds.org/products/hydrosheds

-> processdem.py (build derived tiles)
-> prunedem.py (prune empty tiles)
-> cogify_geotiffs.py (create cloud-optimized-geotiffs for quicker lookup)
-> prune_non_cog.py (delete non .cog.tif files to save half the memory)
-> build_dem_flow_index.py 
(make a file index for O(1) coordinate lookups)
-> sample_dem_coords.py with a coords.csv 
    (id, lat, lon)


With that ready, do:
`python sample_coords.py --coords coords.csv --index "D:\DEM_derived_w_flow\dem_flow_index.json" --layers all --out sampled.csv`

#expected format:
#id,lon,lat
#p0,-122.6765,45.5231
#p1,151.2093,-33.8688
