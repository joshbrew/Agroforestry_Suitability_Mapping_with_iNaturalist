#!/usr/bin/env node
"use strict";

/*
node collect_grid_env.js D:/Oregon_Suitability/oregon_grid_1000m.csv --out-root D:/Oregon_Suitability/oregon_grid_1000m_env --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 --cleanup-enriched invalid+soilgrids --cleanup

node collect_grid_env.js D:/grid/oregon_grid_5000m.csv ^
  --out-root D:/grid/oregon_grid_env ^
  --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 ^
  --terraclimate-year 2018-2024 ^
  --cleanup-enriched invalid+soilgrids ^
  --cleanup

node collect_grid_env.js --phase cleanup --out-root D:/Oregon_Suitability/oregon_grid_1000m_env --cleanup-enriched invalid+soilgrids --cleanup-enriched-backup

node collect_grid_env.js --phase cleanup --cleanup-target D:/Oregon_Suitability/oregon_grid_1000m_env/grid_with_env.csv --cleanup-enriched invalid+soilgrids
*/

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { spawn } = require("child_process");

const DEFAULT_RUN_DATASETS = Object.freeze([
  "dem",
  "terraclimate",
  "twi",
  "soilgrids",
  "glim",
  "mcd12q1",
]);

const DEFAULT_PATHS = Object.freeze({
  demScript: "D:/DEM_Derived_w_flow/sample_coords.py",
  demIndex: "D:/DEM_derived_w_flow/dem_flow_index.json",
  terraclimateScript: "D:/terraclimate/sample_cogs_from_coords.py",
  terraclimateRoot: "D:/terraclimate/terraclimate_cogs_global",
  twiScript: "D:/wetness/sample_twi_coords.py",
  twiTif: "D:/wetness/twi_edtm_120m.tif",
  soilgridsScript: "D:/soilgrids/sample_soilgrids_coords.py",
  soilgridsRoot: "D:/soilgrids/data",
  glimScript: "D:/GLiM/sample_glim_coords.py",
  glimTif: "D:/GLiM/glim_rasters/glim_id_1km_cog.tif",
  glimLookupCsv: "D:/GLiM/glim_rasters/glim_lookup.csv",
  mcd12q1Script: "D:/MCD12Q1_landcover/sample_coords.py",
  mcd12q1Vrt: "D:/MCD12Q1_landcover/cogs/mcd12q1_lc_type1.vrt",
  cleanupEnrichedScript: path.resolve(__dirname, "cleanup_occurrences_enriched.js"),
});

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function nowIso() {
  return new Date().toISOString();
}

function pathExists(filePath) {
  return fs.promises
    .stat(filePath)
    .then((st) => st)
    .catch(() => null);
}

async function fileExistsNonEmpty(filePath) {
  const st = await pathExists(filePath);
  return !!(st && st.isFile() && st.size > 0);
}

function fmtGiB(bytes) {
  const n = Number(bytes) || 0;
  return (n / (1024 * 1024 * 1024)).toFixed(3);
}

function csvEscapeValue(value) {
  const s = value == null ? "" : String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function writeStreamChunk(ws, chunk) {
  return new Promise((resolve, reject) => {
    function onError(err) {
      cleanup();
      reject(err);
    }
    function onDrain() {
      cleanup();
      resolve();
    }
    function cleanup() {
      ws.off("error", onError);
      ws.off("drain", onDrain);
    }
    ws.on("error", onError);
    if (ws.write(chunk)) {
      cleanup();
      resolve();
      return;
    }
    ws.on("drain", onDrain);
  });
}

function parseCsvList(raw) {
  return String(raw || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  for (let i = 0; i < values.length; i++) {
    const raw = String(values[i] || "").trim();
    if (!raw) continue;
    const key = raw.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(raw);
  }
  return out;
}

function normalizeTerraclimateVarsSpec(raw) {
  const text = String(raw == null ? "" : raw).trim().toLowerCase();
  if (!text || text === "all") return "all";
  const vals = uniqueStrings(
    text
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  );
  return vals.length ? vals.join(",") : "all";
}

function normalizeTerraclimateYearSpec(raw) {
  const text = String(raw == null ? "" : raw).trim().toLowerCase();
  if (!text) return "latest";

  const out = [];
  const seen = new Set();

  function addYearToken(token) {
    const key = String(token || "").trim().toLowerCase();
    if (!key || seen.has(key)) return;
    seen.add(key);
    out.push(key);
  }

  for (const part of text.split(",")) {
    const token = String(part || "").trim().toLowerCase();
    if (!token) continue;

    if (token === "latest") {
      addYearToken("latest");
      continue;
    }

    const m = token.match(/^(\d{4})\s*-\s*(\d{4})$/);
    if (m) {
      const a = Number(m[1]);
      const b = Number(m[2]);
      const step = b >= a ? 1 : -1;
      for (let y = a; step > 0 ? y <= b : y >= b; y += step) {
        addYearToken(String(y));
      }
      continue;
    }

    if (!/^\d{4}$/.test(token)) {
      throw new Error(
        `Bad --terraclimate-year token '${token}'. Use latest, a year, a comma list, or a range like 2018-2020.`,
      );
    }
    addYearToken(token);
  }

  return out.length ? out.join(",") : "latest";
}

function parseCsvLine(line) {
  const out = [];
  let field = "";
  let inQuotes = false;
  let fieldStart = true;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        const next = line[i + 1];
        if (next === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
          fieldStart = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (fieldStart && ch === '"') {
      inQuotes = true;
      fieldStart = false;
      continue;
    }

    if (ch === ",") {
      out.push(field);
      field = "";
      fieldStart = true;
      continue;
    }

    field += ch;
    fieldStart = false;
  }

  out.push(field);
  return out;
}

function isBlankLike(v) {
  const s = String(v ?? "").trim();
  if (!s) return true;
  const t = s.toLowerCase();
  return t === "nan" || t === "null" || t === "undefined";
}

function parseFiniteNumber(v) {
  const s = String(v ?? "").trim();
  if (!s) return NaN;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}

function isValidLatLon(latRaw, lonRaw) {
  const lat = parseFiniteNumber(latRaw);
  const lon = parseFiniteNumber(lonRaw);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false;
  if (lat < -90 || lat > 90) return false;
  if (lon < -180 || lon > 180) return false;
  return true;
}

function findFirstHeaderIndex(header, names) {
  for (let i = 0; i < names.length; i++) {
    const idx = header.indexOf(names[i]);
    if (idx >= 0) return idx;
  }
  return -1;
}

function detectCoordColumns(header) {
  const latIdx = findFirstHeaderIndex(header, [
    "decimalLatitude",
    "latitude",
    "lat",
    "Latitude",
    "Lat",
  ]);
  const lonIdx = findFirstHeaderIndex(header, [
    "decimalLongitude",
    "longitude",
    "lon",
    "lng",
    "Longitude",
    "Lon",
    "Lng",
  ]);
  return { latIdx, lonIdx };
}

function detectSoilgridsValueColumns(header) {
  const keep = [];
  for (let i = 0; i < header.length; i++) {
    const h = String(header[i] || "").trim();
    if (!h.startsWith("soilgrids_")) continue;
    if (h === "soilgrids_lon" || h === "soilgrids_lat") continue;
    keep.push(i);
  }
  return keep;
}

function makeCleanupPredicate(header, mode) {
  const { latIdx, lonIdx } = detectCoordColumns(header);
  const soilgridsIdxs = detectSoilgridsValueColumns(header);

  function invalidRow(vals) {
    if (latIdx < 0 || lonIdx < 0) return false;
    return !isValidLatLon(vals[latIdx], vals[lonIdx]);
  }

  function missingSoilgrids(vals) {
    if (!soilgridsIdxs.length) return false;
    for (let i = 0; i < soilgridsIdxs.length; i++) {
      if (!isBlankLike(vals[soilgridsIdxs[i]])) return false;
    }
    return true;
  }

  function shouldDrop(vals) {
    if (mode === "none") return false;
    if (mode === "invalid") return invalidRow(vals);
    if (mode === "soilgrids") return missingSoilgrids(vals);
    if (mode === "invalid+soilgrids") return invalidRow(vals) || missingSoilgrids(vals);
    return false;
  }

  return {
    latIdx,
    lonIdx,
    soilgridsIdxs,
    invalidRow,
    missingSoilgrids,
    shouldDrop,
  };
}

function resolveCleanupInputCsv(target, outRoot) {
  const raw = String(target || "").trim();
  if (raw) {
    const p = path.resolve(raw);
    const st = fs.existsSync(p) ? fs.statSync(p) : null;
    if (st && st.isDirectory()) {
      const gridCsv = path.join(p, "grid_with_env.csv");
      const occCsv = path.join(p, "occurrences_enriched.csv");
      if (fs.existsSync(gridCsv)) return gridCsv;
      if (fs.existsSync(occCsv)) return occCsv;
      return gridCsv;
    }
    return p;
  }
  return path.resolve(inputOutputPaths(outRoot).mergedPath);
}

function defaultCleanupOutputPath(inCsvPath, outPath, inplace) {
  if (outPath) return path.resolve(outPath);
  if (inplace) return inCsvPath + ".cleaning.tmp";
  const dir = path.dirname(inCsvPath);
  const ext = path.extname(inCsvPath);
  const base = path.basename(inCsvPath, ext);
  return path.join(dir, `${base}.cleaned${ext || ".csv"}`);
}

async function cleanupCsvFile({
  inCsvPath,
  outCsvPath,
  mode,
  progressEvery,
}) {
  const rs = fs.createReadStream(inCsvPath, { encoding: "utf8", highWaterMark: 4 << 20 });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  const ws = fs.createWriteStream(outCsvPath, { flags: "w", highWaterMark: 4 << 20 });

  let header = null;
  let predicate = null;
  let rowsRead = 0;
  let rowsWritten = 0;
  let droppedInvalid = 0;
  let droppedSoilgrids = 0;
  let droppedBoth = 0;
  let lastLog = 0;

  try {
    for await (const line of rl) {
      if (header == null) {
        header = parseCsvLine(line);
        predicate = makeCleanupPredicate(header, mode);
        ws.write(header.map(csvEscapeValue).join(",") + "\n");
        console.log(
          `[cleanup] header columns=${header.length} lat_idx=${predicate.latIdx} lon_idx=${predicate.lonIdx} soilgrids_value_columns=${predicate.soilgridsIdxs.length}`,
        );
        continue;
      }

      if (!line) continue;
      rowsRead++;
      const vals = parseCsvLine(line);
      const isInvalid = predicate.invalidRow(vals);
      const isMissingSoilgrids = predicate.missingSoilgrids(vals);
      const shouldDrop = predicate.shouldDrop(vals);

      if (shouldDrop) {
        if (isInvalid && isMissingSoilgrids) droppedBoth++;
        else if (isInvalid) droppedInvalid++;
        else if (isMissingSoilgrids) droppedSoilgrids++;
      } else {
        rowsWritten++;
        ws.write(line + "\n");
      }

      if (rowsRead - lastLog >= progressEvery) {
        lastLog = rowsRead;
        console.log(
          `[cleanup] rows_read=${rowsRead.toLocaleString()} rows_written=${rowsWritten.toLocaleString()} dropped_invalid=${droppedInvalid.toLocaleString()} dropped_soilgrids=${droppedSoilgrids.toLocaleString()} dropped_both=${droppedBoth.toLocaleString()}`,
        );
      }
    }
  } finally {
    rl.close();
    rs.destroy();
    await new Promise((resolve, reject) => {
      ws.on("finish", resolve);
      ws.on("error", reject);
      ws.end();
    });
  }

  return {
    rowsRead,
    rowsWritten,
    droppedInvalid,
    droppedSoilgrids,
    droppedBoth,
  };
}

async function finalizeCleanupInplace({
  inCsvPath,
  tempOutPath,
  backup,
  backupSuffix,
}) {
  const backupPath = inCsvPath + backupSuffix;
  if (backup) {
    await fs.promises.rm(backupPath, { force: true }).catch(() => {});
    await fs.promises.rename(inCsvPath, backupPath);
    await fs.promises.rename(tempOutPath, inCsvPath);
    return backupPath;
  }

  await fs.promises.rm(inCsvPath, { force: true }).catch(() => {});
  await fs.promises.rename(tempOutPath, inCsvPath);
  return "";
}

async function readCsvHeader(filePath) {
  const rs = fs.createReadStream(filePath, { encoding: "utf8" });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  try {
    for await (const line of rl) {
      if (line == null) continue;
      return parseCsvLine(line);
    }
    return [];
  } finally {
    rl.close();
    rs.destroy();
  }
}

function fnv1a32(text) {
  let h = 0x811c9dc5;
  const s = String(text || "");
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i) & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function shardIndexForId(idValue, shardCount) {
  return fnv1a32(String(idValue || "")) % (shardCount >>> 0);
}

class LruWriterCache {
  constructor(limit) {
    this.limit = Math.max(1, Number(limit) | 0);
    this.map = new Map();
  }

  async get(filePath) {
    if (this.map.has(filePath)) {
      const value = this.map.get(filePath);
      this.map.delete(filePath);
      this.map.set(filePath, value);
      return value;
    }

    if (this.map.size >= this.limit) {
      const oldestKey = this.map.keys().next().value;
      const oldestWs = this.map.get(oldestKey);
      this.map.delete(oldestKey);
      await this.closeOne(oldestWs);
    }

    const ws = fs.createWriteStream(filePath, {
      flags: "a",
      highWaterMark: 4 << 20,
    });
    this.map.set(filePath, ws);
    return ws;
  }

  async append(filePath, chunk) {
    if (!chunk || chunk.length === 0) return;
    const ws = await this.get(filePath);
    await writeStreamChunk(ws, chunk);
  }

  async closeOne(ws) {
    if (!ws || ws.destroyed) return;
    await new Promise((resolve, reject) => {
      let done = false;
      function ok() {
        if (done) return;
        done = true;
        cleanup();
        resolve();
      }
      function bad(err) {
        if (done) return;
        done = true;
        cleanup();
        reject(err);
      }
      function cleanup() {
        ws.off("finish", ok);
        ws.off("close", ok);
        ws.off("error", bad);
      }
      ws.on("finish", ok);
      ws.on("close", ok);
      ws.on("error", bad);
      ws.end();
    });
  }

  async closeAll() {
    for (const ws of this.map.values()) {
      await this.closeOne(ws);
    }
    this.map.clear();
  }
}

function inputOutputPaths(outRoot) {
  return {
    basePath: path.join(outRoot, "grid_points.csv"),
    coordsPath: path.join(outRoot, "coords.csv"),
    mergedPath: path.join(outRoot, "grid_with_env.csv"),
    manifestPath: path.join(outRoot, "manifest.json"),
  };
}

function datasetOutputPaths(outRoot) {
  return {
    dem: path.join(outRoot, "sample_dem.csv"),
    terraclimate: path.join(outRoot, "sample_terraclimate.csv"),
    twi: path.join(outRoot, "sample_twi.csv"),
    soilgrids: path.join(outRoot, "sample_soilgrids.csv"),
    glim: path.join(outRoot, "sample_glim.csv"),
    mcd12q1: path.join(outRoot, "sample_mcd12q1.csv"),
  };
}

function parseArgs(argv) {
  const out = {
    inputCsv: "",
    phase: "all",
    outRoot: "",
    pythonBin: process.platform === "win32" ? "py" : "python3",
    runDatasets: DEFAULT_RUN_DATASETS.slice(),
    idCol: "id",
    xCol: "lon",
    yCol: "lat",
    inputCrs: "EPSG:4326",
    idPrefix: "g",
    joinShards: 256,
    joinMinRows: 50000,
    openWriters: 64,
    keepTemps: false,
    cleanup: false,
    cleanupTarget: "",
    cleanupEnrichedOutPath: "",
    cleanupEnrichedMode: "none",
    cleanupEnrichedProgressEvery: 250000,
    cleanupEnrichedBackup: false,
    forceRerun: false,
    demScript: DEFAULT_PATHS.demScript,
    demIndex: DEFAULT_PATHS.demIndex,
    demLayers: "all",
    terraclimateScript: DEFAULT_PATHS.terraclimateScript,
    terraclimateRoot: DEFAULT_PATHS.terraclimateRoot,
    terraclimateVars: "all",
    terraclimateYear: "latest",
    terraclimateAggregate: "none",
    twiScript: DEFAULT_PATHS.twiScript,
    twiTif: DEFAULT_PATHS.twiTif,
    twiChunkSize: 250000,
    soilgridsScript: DEFAULT_PATHS.soilgridsScript,
    soilgridsRoot: DEFAULT_PATHS.soilgridsRoot,
    soilgridsProps: "bdod,cec,clay,sand,silt,soc,phh2o,nitrogen,cfvo",
    soilgridsDepths: "0-5cm,5-15cm,15-30cm",
    soilgridsChunkSize: 200000,
    soilgridsGdalCacheMb: 2048,
    glimScript: DEFAULT_PATHS.glimScript,
    glimTif: DEFAULT_PATHS.glimTif,
    glimLookupCsv: DEFAULT_PATHS.glimLookupCsv,
    mcd12q1Script: DEFAULT_PATHS.mcd12q1Script,
    mcd12q1Vrt: DEFAULT_PATHS.mcd12q1Vrt,
    mcd12q1ValueCol: "mcd12q1",
    cleanupEnrichedScript: DEFAULT_PATHS.cleanupEnrichedScript,
  };

  function needValue(i, flag) {
    if (i + 1 >= argv.length) throw new Error(`Missing value for ${flag}`);
    return argv[i + 1];
  }

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a) continue;
    if (!a.startsWith("-")) {
      out.inputCsv = String(a);
      continue;
    }

    if (a === "--help" || a === "-h") {
      console.log(
        [
          "Usage:",
          "  node collect_grid_env.js <grid.csv> [options]",
          "",
          "Main:",
          "  --phase prepare|sample|merge|cleanup|all",
          "  --out-root D:/grid/oregon_grid_env",
          "  --run dem,terraclimate,twi,soilgrids,glim,mcd12q1",
          "  --python py",
          "  --force-rerun",
          "  --cleanup",
          "  --cleanup-enriched invalid|soilgrids|invalid+soilgrids|none",
          "  --cleanup-target D:/grid/oregon_grid_env/grid_with_env.csv",
          "  --cleanup-enriched-out D:/grid/oregon_grid_env/grid_with_env.cleaned.csv",
          "  --cleanup-enriched-progress-every 250000",
          "  --cleanup-enriched-backup",
          "  --keep-temps",
          "",
          "Input grid columns:",
          "  --id-col id",
          "  --x-col lon",
          "  --y-col lat",
          "  --input-crs EPSG:4326",
          "  --id-prefix g",
          "",
          "Merge:",
          "  --join-shards 256",
          "  --join-min-rows 50000",
          "  --open-writers 64",
          "",
          "TerraClimate:",
          "  --terraclimate-script D:/terraclimate/sample_cogs_from_coords.py",
          "  --terraclimate-root D:/terraclimate/terraclimate_cogs_global",
          "  --terraclimate-vars all",
          "  --terraclimate-year latest",
          "  --terraclimate-year 2024",
          "  --terraclimate-year 2018,2019,2020",
          "  --terraclimate-year 2018-2020",
          "  --terraclimate-aggregate none|mean|sum|min|max",
          "",
          "DEM:",
          "  --dem-script D:/DEM_Derived_w_flow/sample_coords.py",
          "  --dem-index D:/DEM_derived_w_flow/dem_flow_index.json",
          "  --dem-layers all",
          "",
          "TWI:",
          "  --twi-script D:/wetness/sample_twi_coords.py",
          "  --twi-tif D:/wetness/twi_edtm_120m.tif",
          "  --twi-chunk-size 250000",
          "",
          "SoilGrids:",
          "  --soilgrids-script D:/soilgrids/sample_soilgrids_coords.py",
          "  --soilgrids-root D:/soilgrids/data",
          "  --soilgrids-props bdod,cec,clay,sand,silt,soc,phh2o,nitrogen,cfvo",
          "  --soilgrids-depths 0-5cm,5-15cm,15-30cm",
          "  --soilgrids-chunk-size 200000",
          "  --soilgrids-gdal-cache-mb 2048",
          "",
          "GLiM:",
          "  --glim-script D:/GLiM/sample_glim_coords.py",
          "  --glim-tif D:/GLiM/glim_rasters/glim_id_1km_cog.tif",
          "  --glim-lookup-csv D:/GLiM/glim_rasters/glim_lookup.csv",
          "",
          "MCD12Q1:",
          "  --mcd12q1-script D:/MCD12Q1_landcover/sample_coords.py",
          "  --mcd12q1-vrt D:/MCD12Q1_landcover/cogs/mcd12q1_lc_type1.vrt",
          "  --mcd12q1-value-col mcd12q1",
        ].join("\n"),
      );
      process.exit(0);
    } else if (a === "--phase") {
      out.phase = String(needValue(i, a)).toLowerCase();
      i++;
    } else if (a === "--out-root") {
      out.outRoot = String(needValue(i, a));
      i++;
    } else if (a === "--python" || a === "--python-bin") {
      out.pythonBin = String(needValue(i, a));
      i++;
    } else if (a === "--run") {
      out.runDatasets = uniqueStrings(parseCsvList(needValue(i, a))).map((s) => s.toLowerCase());
      i++;
    } else if (a === "--id-col") {
      out.idCol = String(needValue(i, a));
      i++;
    } else if (a === "--x-col") {
      out.xCol = String(needValue(i, a));
      i++;
    } else if (a === "--y-col") {
      out.yCol = String(needValue(i, a));
      i++;
    } else if (a === "--input-crs") {
      out.inputCrs = String(needValue(i, a));
      i++;
    } else if (a === "--id-prefix") {
      out.idPrefix = String(needValue(i, a));
      i++;
    } else if (a === "--join-shards") {
      out.joinShards = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--join-min-rows") {
      out.joinMinRows = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--open-writers") {
      out.openWriters = Math.max(4, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--keep-temps") {
      out.keepTemps = true;
    } else if (a === "--cleanup") {
      out.cleanup = true;
    } else if (a === "--cleanup-target") {
      out.cleanupTarget = String(needValue(i, a));
      i++;
    } else if (a === "--cleanup-enriched-out") {
      out.cleanupEnrichedOutPath = String(needValue(i, a));
      i++;
    } else if (a === "--cleanup-enriched") {
      out.cleanupEnrichedMode = String(needValue(i, a)).toLowerCase();
      i++;
    } else if (a === "--cleanup-enriched-script") {
      out.cleanupEnrichedScript = String(needValue(i, a));
      i++;
    } else if (a === "--cleanup-enriched-progress-every") {
      out.cleanupEnrichedProgressEvery = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--cleanup-enriched-backup") {
      out.cleanupEnrichedBackup = true;
    } else if (a === "--force-rerun") {
      out.forceRerun = true;
    } else if (a === "--dem-script") {
      out.demScript = String(needValue(i, a));
      i++;
    } else if (a === "--dem-index") {
      out.demIndex = String(needValue(i, a));
      i++;
    } else if (a === "--dem-layers") {
      out.demLayers = String(needValue(i, a));
      i++;
    } else if (a === "--terraclimate-script") {
      out.terraclimateScript = String(needValue(i, a));
      i++;
    } else if (a === "--terraclimate-root") {
      out.terraclimateRoot = String(needValue(i, a));
      i++;
    } else if (a === "--terraclimate-vars") {
      out.terraclimateVars = normalizeTerraclimateVarsSpec(String(needValue(i, a)));
      i++;
    } else if (a === "--terraclimate-year") {
      out.terraclimateYear = normalizeTerraclimateYearSpec(String(needValue(i, a)));
      i++;
    } else if (a === "--terraclimate-aggregate") {
      out.terraclimateAggregate = String(needValue(i, a)).toLowerCase();
      i++;
    } else if (a === "--twi-script") {
      out.twiScript = String(needValue(i, a));
      i++;
    } else if (a === "--twi-tif") {
      out.twiTif = String(needValue(i, a));
      i++;
    } else if (a === "--twi-chunk-size") {
      out.twiChunkSize = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--soilgrids-script") {
      out.soilgridsScript = String(needValue(i, a));
      i++;
    } else if (a === "--soilgrids-root") {
      out.soilgridsRoot = String(needValue(i, a));
      i++;
    } else if (a === "--soilgrids-props") {
      out.soilgridsProps = String(needValue(i, a));
      i++;
    } else if (a === "--soilgrids-depths") {
      out.soilgridsDepths = String(needValue(i, a));
      i++;
    } else if (a === "--soilgrids-chunk-size") {
      out.soilgridsChunkSize = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--soilgrids-gdal-cache-mb") {
      out.soilgridsGdalCacheMb = Math.max(64, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--glim-script") {
      out.glimScript = String(needValue(i, a));
      i++;
    } else if (a === "--glim-tif") {
      out.glimTif = String(needValue(i, a));
      i++;
    } else if (a === "--glim-lookup-csv") {
      out.glimLookupCsv = String(needValue(i, a));
      i++;
    } else if (a === "--mcd12q1-script") {
      out.mcd12q1Script = String(needValue(i, a));
      i++;
    } else if (a === "--mcd12q1-vrt") {
      out.mcd12q1Vrt = String(needValue(i, a));
      i++;
    } else if (a === "--mcd12q1-value-col") {
      out.mcd12q1ValueCol = String(needValue(i, a));
      i++;
    } else {
      throw new Error(`Unknown option: ${a}`);
    }
  }

  if (!out.outRoot && out.inputCsv) out.outRoot = `${out.inputCsv}.grid_env`;
  if (!["prepare", "sample", "merge", "cleanup", "all"].includes(out.phase)) {
    throw new Error(`Bad --phase value: ${out.phase}`);
  }
  if (!out.inputCsv && out.phase !== "cleanup") {
    throw new Error("Missing input grid CSV");
  }
  if (!out.inputCsv && out.phase === "cleanup" && !out.cleanupTarget && !out.outRoot) {
    throw new Error("Cleanup phase needs --out-root or --cleanup-target");
  }
  if (!["none", "mean", "sum", "min", "max"].includes(out.terraclimateAggregate)) {
    throw new Error(`Bad --terraclimate-aggregate value: ${out.terraclimateAggregate}`);
  }
  if (!["none", "invalid", "soilgrids", "invalid+soilgrids"].includes(out.cleanupEnrichedMode)) {
    throw new Error(`Bad --cleanup-enriched value: ${out.cleanupEnrichedMode}`);
  }

  out.terraclimateVars = normalizeTerraclimateVarsSpec(out.terraclimateVars);
  out.terraclimateYear = normalizeTerraclimateYearSpec(out.terraclimateYear);
  return out;
}

function buildRunPlan(opts, outRoot) {
  const outputs = datasetOutputPaths(outRoot);
  const wanted = new Set((opts.runDatasets || []).map((s) => s.toLowerCase()));
  if (wanted.has("all")) {
    wanted.clear();
    ["dem", "terraclimate", "twi", "soilgrids", "glim", "mcd12q1"].forEach((k) => wanted.add(k));
  }
  const coordsPath = inputOutputPaths(outRoot).coordsPath;
  const coordsXCol = "lon";
  const coordsYCol = "lat";
  const commands = [];

  if (wanted.has("dem") && opts.demScript && opts.demIndex) {
    commands.push({
      key: "dem",
      outPath: outputs.dem,
      cmd: opts.pythonBin,
      args: [
        opts.demScript,
        "--coords",
        coordsPath,
        "--index",
        opts.demIndex,
        "--layers",
        opts.demLayers || "all",
        "--out",
        outputs.dem,
      ],
    });
  }

  if (wanted.has("terraclimate") && opts.terraclimateScript && opts.terraclimateRoot) {
    const args = [
      opts.terraclimateScript,
      "--cog-root",
      opts.terraclimateRoot,
      "--coords",
      coordsPath,
      "--vars",
      normalizeTerraclimateVarsSpec(opts.terraclimateVars || "all"),
      "--year",
      normalizeTerraclimateYearSpec(opts.terraclimateYear || "latest"),
      "--out",
      outputs.terraclimate,
      "--fallback-radius-pixels",
      "3",
    ];
    if ((opts.terraclimateAggregate || "none") !== "none") {
      args.push("--aggregate", opts.terraclimateAggregate);
    }
    commands.push({
      key: "terraclimate",
      outPath: outputs.terraclimate,
      cmd: opts.pythonBin,
      args,
    });
  }

  if (wanted.has("twi") && opts.twiScript && opts.twiTif) {
    commands.push({
      key: "twi",
      outPath: outputs.twi,
      cmd: opts.pythonBin,
      args: [
        opts.twiScript,
        "--tif",
        opts.twiTif,
        "--input",
        coordsPath,
        "--output",
        outputs.twi,
        "--lon-col",
        coordsXCol,
        "--lat-col",
        coordsYCol,
        "--chunk-size",
        String(opts.twiChunkSize || 250000),
      ],
    });
  }

  if (wanted.has("soilgrids") && opts.soilgridsScript && opts.soilgridsRoot) {
    commands.push({
      key: "soilgrids",
      outPath: outputs.soilgrids,
      cmd: opts.pythonBin,
      args: [
        opts.soilgridsScript,
        "--root",
        opts.soilgridsRoot,
        "--coords",
        coordsPath,
        "--out",
        outputs.soilgrids,
        "--input-crs",
        opts.inputCrs,
        "--lon-col",
        coordsXCol,
        "--lat-col",
        coordsYCol,
        "--id-col",
        "id",
        "--props",
        opts.soilgridsProps,
        "--depths",
        opts.soilgridsDepths,
        "--workers",
        "1",
        "--fallback-radius-pixels",
        "3",
        "--chunk-size",
        String(opts.soilgridsChunkSize || 200000),
        "--gdal-cache-mb",
        String(opts.soilgridsGdalCacheMb || 2048),
      ],
    });
  }

  if (wanted.has("glim") && opts.glimScript && opts.glimTif) {
    const args = [
      opts.glimScript,
      opts.glimTif,
      coordsPath,
      outputs.glim,
      "--x-col",
      coordsXCol,
      "--y-col",
      coordsYCol,
      "--input-crs",
      opts.inputCrs,
    ];
    if (opts.glimLookupCsv) args.push("--lookup-csv", opts.glimLookupCsv);
    commands.push({
      key: "glim",
      outPath: outputs.glim,
      cmd: opts.pythonBin,
      args,
    });
  }

  if (wanted.has("mcd12q1") && opts.mcd12q1Script && opts.mcd12q1Vrt) {
    commands.push({
      key: "mcd12q1",
      outPath: outputs.mcd12q1,
      cmd: opts.pythonBin,
      args: [
        opts.mcd12q1Script,
        opts.mcd12q1Vrt,
        coordsPath,
        outputs.mcd12q1,
        "--x-col",
        coordsXCol,
        "--y-col",
        coordsYCol,
        "--value-col",
        opts.mcd12q1ValueCol || "mcd12q1",
      ],
    });
  }

  return { commands, outputs };
}

async function runCommandLive(cmd, args) {
  await new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: "inherit", shell: false });
    p.on("error", reject);
    p.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Command failed with exit ${code}: ${cmd} ${args.join(" ")}`));
    });
  });
}

async function writeJson(filePath, obj) {
  await fs.promises.writeFile(filePath, JSON.stringify(obj, null, 2));
}

async function readJson(filePath) {
  return JSON.parse(await fs.promises.readFile(filePath, "utf8"));
}

async function updateManifestFinalPath(outRoot, updates) {
  const manifestPath = inputOutputPaths(outRoot).manifestPath;
  const manifest = await readJson(manifestPath).catch(() => ({}));
  Object.assign(manifest, updates || {});
  await writeJson(manifestPath, manifest);
  return manifest;
}

async function runCleanupEnriched({ opts, outRoot, mergedPath }) {
  const mode = String(opts.cleanupEnrichedMode || "none").toLowerCase();
  const targetPath = resolveCleanupInputCsv(
    opts.cleanupTarget || mergedPath || inputOutputPaths(outRoot).mergedPath,
    outRoot,
  );
  const inplace = !String(opts.cleanupEnrichedOutPath || "").trim();
  const outCsvPath = defaultCleanupOutputPath(
    targetPath,
    opts.cleanupEnrichedOutPath || "",
    inplace,
  );

  if (mode === "none") {
    await updateManifestFinalPath(outRoot, {
      finalPath: targetPath,
      cleanupEnriched: {
        enabled: false,
        mode,
      },
    }).catch(() => {});
    return {
      finalPath: targetPath,
      cleanupEnriched: false,
      cleanupMode: mode,
    };
  }

  if (!(await fileExistsNonEmpty(targetPath))) {
    throw new Error(`Cleanup target not found: ${targetPath}`);
  }

  await fs.promises.mkdir(path.dirname(outCsvPath), { recursive: true });

  console.log(
    `[cleanup-enriched] starting ${nowIso()} mode=${mode} target=${targetPath} inplace=${inplace}`,
  );

  const stats = await cleanupCsvFile({
    inCsvPath: targetPath,
    outCsvPath,
    mode,
    progressEvery: Math.max(1, Number(opts.cleanupEnrichedProgressEvery) || 250000),
  });

  let backupPath = "";
  let finalPath = outCsvPath;
  if (inplace) {
    backupPath = await finalizeCleanupInplace({
      inCsvPath: targetPath,
      tempOutPath: outCsvPath,
      backup: !!opts.cleanupEnrichedBackup,
      backupSuffix: ".bak",
    });
    finalPath = targetPath;
  }

  console.log(`[cleanup-enriched] done ${nowIso()} target=${finalPath}`);
  console.log(`[cleanup-enriched] rows_read=${stats.rowsRead.toLocaleString()}`);
  console.log(`[cleanup-enriched] rows_written=${stats.rowsWritten.toLocaleString()}`);
  console.log(`[cleanup-enriched] dropped_invalid=${stats.droppedInvalid.toLocaleString()}`);
  console.log(`[cleanup-enriched] dropped_soilgrids=${stats.droppedSoilgrids.toLocaleString()}`);
  console.log(`[cleanup-enriched] dropped_both=${stats.droppedBoth.toLocaleString()}`);
  if (backupPath) console.log(`[cleanup-enriched] backup=${backupPath}`);

  await updateManifestFinalPath(outRoot, {
    finalPath,
    cleanupEnriched: {
      enabled: true,
      mode,
      backup: !!opts.cleanupEnrichedBackup,
      inplace,
      outPath: inplace ? "" : finalPath,
      targetPath,
      progressEvery: Math.max(
        1,
        Number(opts.cleanupEnrichedProgressEvery) || 250000,
      ),
      rowsRead: stats.rowsRead,
      rowsWritten: stats.rowsWritten,
      droppedInvalid: stats.droppedInvalid,
      droppedSoilgrids: stats.droppedSoilgrids,
      droppedBoth: stats.droppedBoth,
      completedAt: nowIso(),
    },
  }).catch(() => {});

  return {
    finalPath,
    cleanupEnriched: true,
    cleanupMode: mode,
    rowsRead: stats.rowsRead,
    rowsWritten: stats.rowsWritten,
    droppedInvalid: stats.droppedInvalid,
    droppedSoilgrids: stats.droppedSoilgrids,
    droppedBoth: stats.droppedBoth,
    backupPath,
  };
}

async function prepareGridInputs({ inputCsv, outRoot, opts }) {
  console.log(`[prepare] starting ${nowIso()}`);
  const { basePath, coordsPath, manifestPath } = inputOutputPaths(outRoot);

  if (!opts.forceRerun && (await fileExistsNonEmpty(basePath)) && (await fileExistsNonEmpty(coordsPath)) && (await fileExistsNonEmpty(manifestPath))) {
    console.log(`[prepare] reusing existing base/coords in ${outRoot}`);
    return await readJson(manifestPath);
  }

  ensureDir(outRoot);

  const rs = fs.createReadStream(inputCsv, {
    encoding: "utf8",
    highWaterMark: 4 << 20,
  });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  const baseWs = fs.createWriteStream(basePath, {
    flags: "w",
    highWaterMark: 4 << 20,
  });
  const coordsWs = fs.createWriteStream(coordsPath, {
    flags: "w",
    highWaterMark: 4 << 20,
  });

  let inputHeader = null;
  let baseHeader = null;
  let idIdx = -1;
  let xIdx = -1;
  let yIdx = -1;
  let rowsIn = 0;
  let rowsOut = 0;
  let syntheticCount = 0;

  try {
    for await (const line of rl) {
      if (inputHeader == null) {
        inputHeader = parseCsvLine(line || "");
        if (!inputHeader.length) throw new Error(`Missing header in ${inputCsv}`);
        idIdx = inputHeader.indexOf(opts.idCol);
        xIdx = inputHeader.indexOf(opts.xCol);
        yIdx = inputHeader.indexOf(opts.yCol);
        if (xIdx < 0 || yIdx < 0) {
          throw new Error(`Missing coordinate columns '${opts.xCol}' and/or '${opts.yCol}' in ${inputCsv}`);
        }

        baseHeader = idIdx >= 0 ? inputHeader.slice() : ["id"].concat(inputHeader);
        await writeStreamChunk(
          baseWs,
          Buffer.from(baseHeader.map(csvEscapeValue).join(",") + "\n", "utf8"),
        );
        await writeStreamChunk(coordsWs, Buffer.from("id,lon,lat\n", "utf8"));
        continue;
      }

      if (!line) continue;
      rowsIn++;
      const vals = parseCsvLine(line);
      const xRaw = vals[xIdx];
      const yRaw = vals[yIdx];
      const x = Number(xRaw);
      const y = Number(yRaw);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

      let idValue = idIdx >= 0 ? String(vals[idIdx] == null ? "" : vals[idIdx]).trim() : "";
      if (!idValue) {
        idValue = `${opts.idPrefix}${rowsIn - 1}`;
        syntheticCount++;
      }

      const baseRow = idIdx >= 0 ? vals.slice() : [idValue].concat(vals);
      if (idIdx >= 0) baseRow[idIdx] = idValue;

      await writeStreamChunk(
        baseWs,
        Buffer.from(baseRow.map(csvEscapeValue).join(",") + "\n", "utf8"),
      );
      await writeStreamChunk(
        coordsWs,
        Buffer.from([idValue, xRaw, yRaw].map(csvEscapeValue).join(",") + "\n", "utf8"),
      );
      rowsOut++;
    }
  } finally {
    rl.close();
    rs.destroy();
    await Promise.all([
      new Promise((resolve, reject) => {
        baseWs.on("finish", resolve);
        baseWs.on("error", reject);
        baseWs.end();
      }),
      new Promise((resolve, reject) => {
        coordsWs.on("finish", resolve);
        coordsWs.on("error", reject);
        coordsWs.end();
      }),
    ]);
  }

  const manifest = {
    createdAt: nowIso(),
    sourceCsv: path.resolve(inputCsv),
    outRoot: path.resolve(outRoot),
    basePath,
    coordsPath,
    mergedPath: inputOutputPaths(outRoot).mergedPath,
    rowCount: rowsOut,
    inputRowsSeen: rowsIn,
    xCol: opts.xCol,
    yCol: opts.yCol,
    idCol: opts.idCol,
    inputCrs: opts.inputCrs,
    syntheticIdsAssigned: syntheticCount,
    sampleConfig: {
      terraclimate: {
        enabled: (opts.runDatasets || []).includes("all") || (opts.runDatasets || []).includes("terraclimate"),
        vars: normalizeTerraclimateVarsSpec(opts.terraclimateVars || "all"),
        year: normalizeTerraclimateYearSpec(opts.terraclimateYear || "latest"),
        aggregate: opts.terraclimateAggregate || "none",
      },
    },
    cleanupEnrichedRequested: {
      mode: String(opts.cleanupEnrichedMode || "none").toLowerCase(),
      backup: !!opts.cleanupEnrichedBackup,
      progressEvery: Math.max(1, Number(opts.cleanupEnrichedProgressEvery) || 250000),
    },
  };
  await writeJson(manifestPath, manifest);

  console.log(
    `[prepare] done ${nowIso()} rows=${rowsOut.toLocaleString()} source_rows=${rowsIn.toLocaleString()} synthetic_ids=${syntheticCount.toLocaleString()} out=${outRoot}`,
  );
  return manifest;
}

async function runSamplers({ opts, outRoot }) {
  const { commands, outputs } = buildRunPlan(opts, outRoot);
  if (!commands.length) {
    console.log("[sample] nothing to run");
    return { commands: [], outputs };
  }

  const coordsPath = inputOutputPaths(outRoot).coordsPath;
  if (!(await fileExistsNonEmpty(coordsPath))) {
    throw new Error(`Missing coords file: ${coordsPath}`);
  }

  const runnable = [];
  for (let i = 0; i < commands.length; i++) {
    const job = commands[i];
    const exists = await fileExistsNonEmpty(job.outPath);
    if (exists && !opts.forceRerun) {
      console.log(`[sample] skipping ${job.key}, reusing ${job.outPath}`);
      continue;
    }
    runnable.push(job);
  }

  if (!runnable.length) {
    console.log("[sample] all requested outputs already exist, nothing to run");
    return { commands, outputs };
  }

  console.log(`[sample] starting ${nowIso()} jobs=${runnable.length}/${commands.length}`);
  for (let i = 0; i < runnable.length; i++) {
    const job = runnable[i];
    console.log(`[sample] ${i + 1}/${runnable.length} ${job.key} -> ${job.outPath}`);
    await runCommandLive(job.cmd, job.args);
  }
  console.log(`[sample] done ${nowIso()}`);
  return { commands, outputs };
}

function chooseEffectiveJoinShards(totalRows, maxShards, minRowsPerShard) {
  const rows = Math.max(0, Number(totalRows) || 0);
  const maxCount = Math.max(1, Number(maxShards) | 0);
  const minRows = Math.max(1, Number(minRowsPerShard) | 0);
  if (rows <= 0) return 1;
  return Math.max(1, Math.min(maxCount, Math.ceil(rows / minRows)));
}

async function countCsvDataRows(filePath) {
  const rs = fs.createReadStream(filePath, {
    encoding: "utf8",
    highWaterMark: 4 << 20,
  });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  let rows = 0;
  let first = true;
  try {
    for await (const line of rl) {
      if (first) {
        first = false;
        continue;
      }
      if (!line) continue;
      rows++;
    }
  } finally {
    rl.close();
    rs.destroy();
  }
  return rows >>> 0;
}

async function shardCsvById({
  filePath,
  shardDir,
  shardCount,
  openWriters,
  idColumnName = "id",
}) {
  ensureDir(shardDir);
  const writerCache = new LruWriterCache(openWriters);
  const rs = fs.createReadStream(filePath, {
    encoding: "utf8",
    highWaterMark: 4 << 20,
  });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });

  let header = null;
  let idIdx = -1;
  let rows = 0;

  try {
    for await (const line of rl) {
      if (header == null) {
        header = parseCsvLine(line || "");
        idIdx = header.indexOf(idColumnName);
        if (idIdx < 0) throw new Error(`Missing ${idColumnName} in ${filePath}`);
        continue;
      }
      if (!line) continue;
      const vals = parseCsvLine(line);
      const id = String(vals[idIdx] == null ? "" : vals[idIdx]);
      if (!id) continue;
      const shardId = shardIndexForId(id, shardCount);
      const shardPath = path.join(shardDir, `shard_${String(shardId).padStart(4, "0")}.csv`);
      await writerCache.append(shardPath, Buffer.from(line + "\n", "utf8"));
      rows++;
    }
  } finally {
    rl.close();
    rs.destroy();
    await writerCache.closeAll();
  }

  return { header: header || [], rows };
}

function prefixedAddonHeader(datasetKey, header) {
  const out = [];
  for (let i = 0; i < header.length; i++) {
    const h = String(header[i] || "");
    if (i === 0 && h === "id") continue;
    out.push(`${datasetKey}_${h}`);
  }
  return out;
}

async function loadAddonShardMap(filePath) {
  const rs = fs.createReadStream(filePath, {
    encoding: "utf8",
    highWaterMark: 4 << 20,
  });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  const map = new Map();
  try {
    for await (const line of rl) {
      if (!line) continue;
      const vals = parseCsvLine(line);
      const id = String(vals[0] == null ? "" : vals[0]);
      if (!id) continue;
      map.set(id, vals.slice(1));
    }
  } finally {
    rl.close();
    rs.destroy();
  }
  return map;
}

async function mergeSampleOutputs({ outRoot, opts }) {
  console.log(`[merge] starting ${nowIso()}`);

  const { basePath, coordsPath, mergedPath, manifestPath } = inputOutputPaths(outRoot);
  const outputs = datasetOutputPaths(outRoot);
  const manifest = await readJson(manifestPath).catch(() => ({}));
  const baseRowCount = Math.max(0, Number(manifest.rowCount) || 0) || (await countCsvDataRows(basePath));

  if (!opts.forceRerun && (await fileExistsNonEmpty(mergedPath))) {
    console.log(`[merge] reusing existing merged file ${mergedPath}`);
    const addonEntriesReuse = [];
    for (const key of ["dem", "terraclimate", "twi", "soilgrids", "glim", "mcd12q1"]) {
      if (await fileExistsNonEmpty(outputs[key])) addonEntriesReuse.push([key, outputs[key]]);
    }
    manifest.mergedPath = mergedPath;
    if (!manifest.finalPath) manifest.finalPath = mergedPath;
    manifest.sampleOutputs = Object.fromEntries(addonEntriesReuse);
    await writeJson(manifestPath, manifest);
    if (opts.cleanup) {
      await fs.promises.rm(coordsPath, { force: true }).catch(() => {});
      for (let i = 0; i < addonEntriesReuse.length; i++) {
        await fs.promises.rm(addonEntriesReuse[i][1], { force: true }).catch(() => {});
      }
    }
    return { mergedPath, mergedRows: manifest.mergedRows || 0 };
  }

  const addonEntries = [];
  for (const key of ["dem", "terraclimate", "twi", "soilgrids", "glim", "mcd12q1"]) {
    try {
      const st = await fs.promises.stat(outputs[key]);
      if (st.size > 0) addonEntries.push([key, outputs[key]]);
    } catch {}
  }

  if (!(await fs.promises.stat(basePath).catch(() => null))) {
    throw new Error(`Missing base file: ${basePath}`);
  }

  const effectiveJoinShards = chooseEffectiveJoinShards(baseRowCount, opts.joinShards, opts.joinMinRows);
  console.log(
    `[merge] base_rows=${baseRowCount.toLocaleString()} max_shards=${opts.joinShards.toLocaleString()} min_rows_per_shard=${opts.joinMinRows.toLocaleString()} effective_shards=${effectiveJoinShards.toLocaleString()}`,
  );

  const shardRoot = path.join(outRoot, `.join_tmp_${Date.now()}_${process.pid}`);
  const baseShardDir = path.join(shardRoot, "base");
  const addonShardDirs = {};
  ensureDir(shardRoot);

  const baseShardInfo = await shardCsvById({
    filePath: basePath,
    shardDir: baseShardDir,
    shardCount: effectiveJoinShards,
    openWriters: opts.openWriters,
    idColumnName: "id",
  });

  const addonHeaders = new Map();
  for (let i = 0; i < addonEntries.length; i++) {
    const [key, filePath] = addonEntries[i];
    const shardDir = path.join(shardRoot, key);
    addonShardDirs[key] = shardDir;
    const info = await shardCsvById({
      filePath,
      shardDir,
      shardCount: effectiveJoinShards,
      openWriters: opts.openWriters,
      idColumnName: "id",
    });
    addonHeaders.set(key, info.header);
  }

  const ws = fs.createWriteStream(mergedPath, {
    flags: "w",
    highWaterMark: 4 << 20,
  });

  const mergedHeader = baseShardInfo.header.slice();
  for (let i = 0; i < addonEntries.length; i++) {
    const [key] = addonEntries[i];
    mergedHeader.push(...prefixedAddonHeader(key, addonHeaders.get(key) || []));
  }
  await writeStreamChunk(ws, Buffer.from(mergedHeader.map(csvEscapeValue).join(",") + "\n", "utf8"));

  let mergedRows = 0;
  try {
    for (let shardId = 0; shardId < effectiveJoinShards; shardId++) {
      const shardName = `shard_${String(shardId).padStart(4, "0")}.csv`;
      const baseShardPath = path.join(baseShardDir, shardName);
      const baseExists = await fs.promises.stat(baseShardPath).catch(() => null);
      if (!baseExists) continue;

      const addonMaps = new Map();
      for (let i = 0; i < addonEntries.length; i++) {
        const [key] = addonEntries[i];
        const addonShardPath = path.join(addonShardDirs[key], shardName);
        const exists = await fs.promises.stat(addonShardPath).catch(() => null);
        if (!exists) {
          addonMaps.set(key, new Map());
          continue;
        }
        addonMaps.set(key, await loadAddonShardMap(addonShardPath));
      }

      const rs = fs.createReadStream(baseShardPath, {
        encoding: "utf8",
        highWaterMark: 4 << 20,
      });
      const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
      const outParts = [];
      try {
        for await (const line of rl) {
          if (!line) continue;
          const vals = parseCsvLine(line);
          const id = String(vals[0] == null ? "" : vals[0]);
          if (!id) continue;
          const row = vals.slice();
          for (let i = 0; i < addonEntries.length; i++) {
            const [key] = addonEntries[i];
            const header = addonHeaders.get(key) || [];
            const wantLen = Math.max(0, header.length - 1);
            const addonVals = addonMaps.get(key).get(id) || new Array(wantLen).fill("");
            if (addonVals.length < wantLen) {
              row.push(...addonVals, ...new Array(wantLen - addonVals.length).fill(""));
            } else {
              row.push(...addonVals.slice(0, wantLen));
            }
          }
          outParts.push(row.map((v) => csvEscapeValue(v || "")).join(",") + "\n");
          mergedRows++;
          if (outParts.length >= 4096) {
            await writeStreamChunk(ws, Buffer.from(outParts.join(""), "utf8"));
            outParts.length = 0;
          }
        }
        if (outParts.length) {
          await writeStreamChunk(ws, Buffer.from(outParts.join(""), "utf8"));
        }
      } finally {
        rl.close();
        rs.destroy();
      }

      console.log(`[merge] shard ${shardId + 1}/${effectiveJoinShards} rows=${mergedRows.toLocaleString()}`);
    }
  } finally {
    await new Promise((resolve, reject) => {
      ws.on("finish", resolve);
      ws.on("error", reject);
      ws.end();
    });
  }

  manifest.mergedPath = mergedPath;
  manifest.finalPath = mergedPath;
  manifest.mergedRows = mergedRows;
  manifest.sampleOutputs = Object.fromEntries(addonEntries);
  manifest.joinShardsRequested = opts.joinShards;
  manifest.joinMinRows = opts.joinMinRows;
  manifest.joinShardsEffective = effectiveJoinShards;
  await writeJson(manifestPath, manifest);

  if (!opts.keepTemps) {
    await fs.promises.rm(shardRoot, { recursive: true, force: true }).catch(() => {});
  }
  if (opts.cleanup) {
    await fs.promises.rm(coordsPath, { force: true }).catch(() => {});
    for (let i = 0; i < addonEntries.length; i++) {
      await fs.promises.rm(addonEntries[i][1], { force: true }).catch(() => {});
    }
  }

  console.log(`[merge] done ${nowIso()} rows=${mergedRows.toLocaleString()} out=${mergedPath}`);
  return { mergedPath, mergedRows };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const inputCsv = opts.inputCsv ? path.resolve(opts.inputCsv) : "";
  const outRoot = path.resolve(opts.outRoot || (opts.cleanupTarget ? path.dirname(path.resolve(opts.cleanupTarget)) : process.cwd()));

  let st = null;
  if (inputCsv) {
    st = await pathExists(inputCsv);
    if (!st || !st.isFile()) {
      throw new Error(`Input CSV not found: ${inputCsv}`);
    }
  }

  if (inputCsv) {
    console.log(`[grid-env] csv=${inputCsv}`);
    console.log(`[grid-env] size=${fmtGiB(st.size)} GiB`);
  } else {
    console.log(`[grid-env] csv=<not required for cleanup phase>`);
  }
  console.log(`[grid-env] phase=${opts.phase} outRoot=${outRoot}`);
  console.log(`[grid-env] input_cols id=${opts.idCol} x=${opts.xCol} y=${opts.yCol} input_crs=${opts.inputCrs}`);

  ensureDir(outRoot);

  if (opts.phase === "cleanup") {
    await runCleanupEnriched({
      opts,
      outRoot,
      mergedPath: opts.cleanupTarget || inputOutputPaths(outRoot).mergedPath,
    });
    console.log(`[grid-env] done ${nowIso()}`);
    return;
  }

  if (opts.phase === "prepare" || opts.phase === "all") {
    await prepareGridInputs({ inputCsv, outRoot, opts });
  }

  if (opts.phase === "sample" || opts.phase === "all") {
    if (!(await fileExistsNonEmpty(inputOutputPaths(outRoot).coordsPath))) {
      await prepareGridInputs({ inputCsv, outRoot, opts });
    }
    await runSamplers({ opts, outRoot });
  }

  if (opts.phase === "merge" || opts.phase === "all") {
    if (!(await fileExistsNonEmpty(inputOutputPaths(outRoot).basePath))) {
      await prepareGridInputs({ inputCsv, outRoot, opts });
    }
    const mergeInfo = await mergeSampleOutputs({ outRoot, opts });
    await runCleanupEnriched({
      opts,
      outRoot,
      mergedPath: mergeInfo && mergeInfo.mergedPath,
    });
  }

  console.log(`[grid-env] done ${nowIso()}`);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
