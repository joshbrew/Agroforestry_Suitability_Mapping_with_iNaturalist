EDL_TOKEN="PASTE_YOUR_EARTHDATA_DOWNLOAD_TOKEN_HERE" # get from LAADS DAAC EarthData ladsweb.modaps.eosdis.nasa.gov 
YEAR=2024

wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=6 \
  "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MCD12Q1/${YEAR}/001/" \
  --header "Authorization: Bearer ${EDL_TOKEN}" \
  -P "./MCD12Q1_${YEAR}"