#!/usr/bin/env node
"use strict";

/*
node collect_taxa_env.js occurrences.csv ^
  --phase all ^
  --include-taxa "species:Daucus pusillus" ^
  --out-root D:/envpull_daucus ^
  --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 ^
  --cleanup-enriched invalid+soilgrids ^
  --cleanup
  
*/

/*
node collect_taxa_env.js occurrences.csv --phase all --include-taxa "species:Epifagus virginiana,species:Fagus grandifolia,species:Acer rubrum" --out-root D:/envpull_epifagus_beech_test --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 --cleanup-enriched invalid+soilgrids --cleanup

node collect_taxa_env.js occurrences.csv --phase all --include-taxa "species:Epifagus virginiana,species:Fagus grandifolia,species:Acer rubrum,species:Boschniakia strobilacea,species:Alnus rubra,species:Pseudotsuga menziesii" --out-root D:/envpull_association_test --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 --cleanup-enriched invalid+soilgrids --cleanup



then run 
python aggregate_occurrence_trends.py D:/envpull_association_test/occurrences_enriched.cleaned.csv --group-by matched_species_name
*/

/* append more taxa then delete trends and rerun. be sure to check that the iNaturalist website actually has the species you want or they will be skipped. You want at least a few observations but any can be valid as long as they're outdoors in some way.
node collect_taxa_env.js occurrences.csv --phase all --append --include-taxa "species:Vaccinium ovatum,species:Ribes sanguineum,species:Echinacea purpurea,species:Achillea millefolium,species:Aquilegia formosa,species:Iris tenax,species:Oemleria cerasiformis,species:Holodiscus discolor,species:Physocarpus capitatus,species:Vitis vinifera,species:Lonicera ciliosa,species:Lathyrus nevadensis,species:Solidago canadensis,species:Camassia leichtlinii,species:Dodecatheon hendersonii,species:Malus domestica,species:Crataegus douglasii,species:Amelanchier alnifolia,species:Corylus cornuta,species:Rubus parviflorus,species:Chamaenerion angustifolium,species:Symphytum uplandicum,species:Asclepias speciosa,species:Taraxacum officinale,species:Tropaeolum majus,species:Rubus pedatus,species:Fragaria vesca,species:Trifolium repens,species:Stellaria media" --out-root D:/envpull_association_test --run dem,terraclimate,twi,soilgrids,glim,mcd12q1 --cleanup-enriched invalid+soilgrids --cleanup
*/


if (!process.env.CSV_READ_BYTES) 
  process.env.CSV_READ_BYTES = "8388608";

if (!process.env.POSTINGS_READ_BYTES)
  process.env.POSTINGS_READ_BYTES = "8388608";

if (!process.env.COOCCUR_SPOOL_SHARDS)
  process.env.COOCCUR_SPOOL_SHARDS = "512";

if (!process.env.COOCCUR_SPOOL_MIN_ROWS_PER_SHARD)
  process.env.COOCCUR_SPOOL_MIN_ROWS_PER_SHARD = "500000";

if (!process.env.COOCCUR_SPOOL_BATCH_TRIPLES)
  process.env.COOCCUR_SPOOL_BATCH_TRIPLES = "524288";

if (!process.env.COOCCUR_EXACT_SPOOL_STRIDE)
  process.env.COOCCUR_EXACT_SPOOL_STRIDE = "131072";

const readline = require("readline");
const { spawn } = require("child_process");

const {
  fs,
  path,
  FILE,
  INDEX_PATH,
  TAXA_DIR,
  ensureDir,
  maybeGC,
  statFile,
  fmtGiB,
  csvEscapeValue,
  writeStreamChunk,
} = require("./server/csvserver.utils.js");

const {
  loadOrBuildIndex,
  chooseHeader,
  findColIndex,
  getBestLookupStride,
  groupRowIdsByBaseRow,
  getLookupWindowForBaseRow,
  readRowsByRowIdsSelectiveRecordsFd,
} = require("./server/csvserver.core.js");

const {
  loadTaxaIndex,
  streamSpeciesRowIds,
} = require("./server/csvserver.taxa.js");

const CSV_QUOTE = 0x22;
const TAXON_RANK_NAME_TO_ID = Object.freeze({
  root: 0,
  kingdom: 1,
  phylum: 2,
  class: 3,
  order: 4,
  family: 5,
  genus: 6,
  species: 7,
});

const DEFAULT_INPUT_CSV = "occurrences.csv";
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
  soilgridsMergedRoot: "D:/soilgrids/merged",
  glimScript: "D:/GLiM/sample_glim_coords.py",
  glimTif: "D:/GLiM/glim_rasters/glim_id_1km_cog.tif",
  glimLookupCsv: "D:/GLiM/glim_rasters/glim_lookup.csv",
  mcd12q1Script: "D:/MCD12Q1_landcover/sample_coords.py",
  mcd12q1Vrt: "D:/MCD12Q1_landcover/cogs/mcd12q1_lc_type1.vrt",
  cleanupEnrichedScript: path.resolve(
    __dirname,
    "cleanup_occurrences_enriched.js",
  ),
});

function normalizeTaxonText(s) {
  return String(s || "")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function formatRatePerSec(count, elapsedMs) {
  const ms = Math.max(1, Number(elapsedMs) || 0);
  return Math.round((count * 1000) / ms).toLocaleString();
}

function pathExists(filePath) {
  return fs.promises
    .stat(filePath)
    .then((st) => st)
    .catch(() => null);
}

function pathExistsSync(filePath) {
  try {
    return fs.statSync(filePath);
  } catch {
    return null;
  }
}

function dirExistsSync(dirPath) {
  const st = pathExistsSync(dirPath);
  return !!(st && st.isDirectory());
}

function fileExistsSync(filePath) {
  const st = pathExistsSync(filePath);
  return !!(st && st.isFile());
}

function treeHasTifSync(rootPath) {
  if (!dirExistsSync(rootPath)) return false;

  const pending = [rootPath];
  while (pending.length) {
    const cur = pending.pop();
    let entries = [];
    try {
      entries = fs.readdirSync(cur, { withFileTypes: true });
    } catch {
      continue;
    }

    for (let i = 0; i < entries.length; i++) {
      const ent = entries[i];
      const full = path.join(cur, ent.name);
      if (ent.isDirectory()) {
        pending.push(full);
        continue;
      }
      if (!ent.isFile()) continue;
      const name = String(ent.name || '').toLowerCase();
      if (name.endsWith('.tif') && !name.endsWith('.tif.part')) return true;
    }
  }

  return false;
}

function resolveSoilgridsSampler(opts) {
  const baseScript = String(opts.soilgridsScript || '').trim();
  const baseRoot = String(opts.soilgridsRoot || '').trim();
  const mergedRoot = String(opts.soilgridsMergedRoot || DEFAULT_PATHS.soilgridsMergedRoot || '').trim();
  const mergedScript = baseScript
    ? path.join(path.dirname(baseScript), 'sample_soilgrids_merged.py')
    : '';

  if (mergedRoot && dirExistsSync(mergedRoot) && treeHasTifSync(mergedRoot) && mergedScript && fileExistsSync(mergedScript)) {
    return {
      mode: 'merged',
      script: mergedScript,
      root: mergedRoot,
    };
  }

  return {
    mode: 'base',
    script: baseScript,
    root: baseRoot,
  };
}

async function fileExistsNonEmpty(filePath) {
  const st = await pathExists(filePath);
  return !!(st && st.isFile() && st.size > 0);
}

async function allFilesExistNonEmpty(filePaths) {
  for (let i = 0; i < filePaths.length; i++) {
    if (!(await fileExistsNonEmpty(filePaths[i]))) return false;
  }
  return true;
}

function isCliFlagToken(value) {
  return typeof value === "string" && value.startsWith("-");
}

function normalizePathString(value) {
  return String(value || "").trim();
}

async function resolveInputCsvAndIndexPath(inputCsvRaw) {
  const inputCsv = path.resolve(
    normalizePathString(inputCsvRaw) || DEFAULT_INPUT_CSV,
  );
  const fileFromUtils = normalizePathString(FILE);

  if (fileFromUtils && !isCliFlagToken(fileFromUtils)) {
    try {
      if (
        path.resolve(fileFromUtils) === inputCsv &&
        normalizePathString(INDEX_PATH)
      ) {
        return { inputCsv, indexPath: INDEX_PATH };
      }
    } catch {}
  }

  const csvDir = path.dirname(inputCsv);
  const csvBase = path.basename(inputCsv);
  const csvStem = path.basename(inputCsv, path.extname(inputCsv));
  const candidates = [
    path.join(csvDir, `${csvBase}.index.json`),
    path.join(csvDir, `${csvBase}.idx.json`),
    path.join(csvDir, `${csvBase}.index`),
    path.join(csvDir, `${csvBase}.idx`),
    path.join(csvDir, `${csvStem}.index.json`),
    path.join(csvDir, `${csvStem}.idx.json`),
    path.join(csvDir, `${csvStem}.index`),
    path.join(csvDir, `${csvStem}.idx`),
  ];

  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i];
    const st = await pathExists(candidate);
    if (st && st.isFile()) return { inputCsv, indexPath: candidate };
  }

  return {
    inputCsv,
    indexPath: path.join(csvDir, `${csvBase}.index.json`),
  };
}

function nowIso() {
  return new Date().toISOString();
}

function safeNumber(x, fallback = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
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

function getTerraclimateSampleConfig(opts) {
  const wanted = new Set(
    (opts.runDatasets || []).map((s) => String(s || "").trim().toLowerCase()),
  );
  const enabled = wanted.has("all") || wanted.has("terraclimate");
  return {
    enabled,
    vars: normalizeTerraclimateVarsSpec(opts.terraclimateVars || "all"),
    year: normalizeTerraclimateYearSpec(opts.terraclimateYear || "latest"),
  };
}

function getSampleConfig(opts) {
  return {
    terraclimate: getTerraclimateSampleConfig(opts),
  };
}

function sameTerraclimateConfig(a, b) {
  const aa = a || { enabled: false, vars: "all", year: "latest" };
  const bb = b || { enabled: false, vars: "all", year: "latest" };
  return (
    !!aa.enabled === !!bb.enabled &&
    String(aa.vars || "all") === String(bb.vars || "all") &&
    String(aa.year || "latest") === String(bb.year || "latest")
  );
}

function formatTerraclimateConfig(cfg) {
  const c = cfg || { enabled: false, vars: "all", year: "latest" };
  return `enabled=${c.enabled ? "1" : "0"} vars=${c.vars || "all"} years=${c.year || "latest"}`;
}

function rankIdFromName(name) {
  const key = normalizeTaxonText(name);
  return Object.prototype.hasOwnProperty.call(TAXON_RANK_NAME_TO_ID, key)
    ? TAXON_RANK_NAME_TO_ID[key]
    : null;
}

function parseTaxonSelector(raw) {
  const text = String(raw || "").trim();
  if (!text) return null;

  const m = text.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+)$/);
  if (m) {
    const rankId = rankIdFromName(m[1]);
    if (rankId == null) {
      throw new Error(`Unknown taxon rank in selector: ${text}`);
    }
    const nameText = String(m[2] || "").trim();
    if (!nameText) {
      throw new Error(`Missing taxon name in selector: ${text}`);
    }
    return {
      raw: text,
      rankId: rankId >>> 0,
      name: nameText,
      nameNorm: normalizeTaxonText(nameText),
    };
  }

  return {
    raw: text,
    rankId: null,
    name: text,
    nameNorm: normalizeTaxonText(text),
  };
}

function parseTaxonSelectorList(raw) {
  return uniqueStrings(parseCsvList(raw))
    .map(parseTaxonSelector)
    .filter(Boolean);
}

function makeSpeciesLabel(taxa, speciesNodeId) {
  const sid = speciesNodeId >>> 0;
  const n = taxa.readNode(sid);
  const species = n ? String(taxa.getString(n.nameId) || "").trim() : "";
  if (!n || n.rankId >>> 0 !== TAXON_RANK_NAME_TO_ID.species)
    return species || `node:${sid}`;
  const parentId = n.parentId >>> 0;
  if (parentId === 0 || parentId === sid) return species || `species:${sid}`;
  const parent = taxa.readNode(parentId);
  const genus = parent
    ? String(taxa.getString(parent.nameId) || "").trim()
    : "";
  if (!genus) return species || `species:${sid}`;
  if (!species) return genus;
  if (species.toLowerCase().startsWith(genus.toLowerCase() + " "))
    return species;
  return `${genus} ${species}`;
}

function nodeSearchNames(taxa, nodeId) {
  const node = taxa.readNode(nodeId >>> 0);
  if (!node) return [];

  const names = [];
  const baseName = String(taxa.getString(node.nameId) || "").trim();
  if (baseName) names.push(baseName);

  if (node.rankId >>> 0 === TAXON_RANK_NAME_TO_ID.species) {
    const label = makeSpeciesLabel(taxa, nodeId >>> 0);
    if (label && label !== baseName) names.push(label);
  }

  return names;
}

function resolveTaxonSelectors(taxa, selectors) {
  const resolvedNodeIds = new Set();
  const resolved = [];
  const unresolved = [];

  for (let si = 0; si < selectors.length; si++) {
    const selector = selectors[si];
    const matchedIds = [];

    for (let nodeId = 1; nodeId < taxa.nodeCount >>> 0; nodeId++) {
      const node = taxa.readNode(nodeId);
      if (!node) continue;
      if (
        selector.rankId != null &&
        node.rankId >>> 0 !== selector.rankId >>> 0
      )
        continue;

      const names = nodeSearchNames(taxa, nodeId);
      let hit = false;
      for (let ni = 0; ni < names.length; ni++) {
        if (normalizeTaxonText(names[ni]) === selector.nameNorm) {
          hit = true;
          break;
        }
      }
      if (!hit) continue;

      matchedIds.push(nodeId >>> 0);
      resolvedNodeIds.add(nodeId >>> 0);
    }

    if (matchedIds.length) {
      resolved.push({
        raw: selector.raw,
        rankId: selector.rankId,
        matchCount: matchedIds.length,
      });
    } else {
      unresolved.push(selector.raw);
    }
  }

  return { nodeIds: resolvedNodeIds, resolved, unresolved };
}

function summarizeTaxonSelectors(selectors) {
  if (!selectors || !selectors.length) return "none";
  return selectors.map((s) => s.raw).join("|");
}

function resolveTaxonFilter(taxa, opts) {
  const includeSelectors = Array.isArray(opts.includeTaxaSelectors)
    ? opts.includeTaxaSelectors
    : [];
  const excludeSelectors = Array.isArray(opts.excludeTaxaSelectors)
    ? opts.excludeTaxaSelectors
    : [];

  const includeResolved = resolveTaxonSelectors(taxa, includeSelectors);
  const excludeResolved = resolveTaxonSelectors(taxa, excludeSelectors);

  if (includeSelectors.length && !includeResolved.nodeIds.size) {
    throw new Error(
      `No taxonomy nodes matched --include-taxa ${summarizeTaxonSelectors(includeSelectors)}`,
    );
  }

  if (includeResolved.unresolved.length) {
    console.warn(
      `[taxon-filter] unresolved include selectors: ${includeResolved.unresolved.join(", ")}`,
    );
  }
  if (excludeResolved.unresolved.length) {
    console.warn(
      `[taxon-filter] unresolved exclude selectors: ${excludeResolved.unresolved.join(", ")}`,
    );
  }

  return {
    includeSelectors,
    excludeSelectors,
    includeNodeIds: includeResolved.nodeIds,
    excludeNodeIds: excludeResolved.nodeIds,
    includeResolved,
    excludeResolved,
    memo: new Int8Array(taxa.nodeCount >>> 0),
  };
}

function taxonFilterSummary(filter) {
  if (!filter) return "none";
  const parts = [];
  if (filter.includeSelectors && filter.includeSelectors.length) {
    parts.push(`include=${summarizeTaxonSelectors(filter.includeSelectors)}`);
  }
  if (filter.excludeSelectors && filter.excludeSelectors.length) {
    parts.push(`exclude=${summarizeTaxonSelectors(filter.excludeSelectors)}`);
  }
  return parts.length ? parts.join(" ") : "none";
}

function speciesMatchesTaxonFilter(taxa, speciesNodeId, filter) {
  if (!filter) return true;
  const sid = speciesNodeId >>> 0;
  const memo = filter.memo;
  if (memo && memo[sid] !== 0) return memo[sid] === 1;

  const includeActive = filter.includeNodeIds && filter.includeNodeIds.size > 0;
  const excludeActive = filter.excludeNodeIds && filter.excludeNodeIds.size > 0;

  if (!includeActive && !excludeActive) {
    if (memo) memo[sid] = 1;
    return true;
  }

  let cur = sid;
  let includeHit = !includeActive;

  while (cur > 0 && cur < taxa.nodeCount >>> 0) {
    if (excludeActive && filter.excludeNodeIds.has(cur)) {
      if (memo) memo[sid] = -1;
      return false;
    }
    if (includeActive && filter.includeNodeIds.has(cur)) includeHit = true;

    const node = taxa.readNode(cur);
    if (!node) break;
    const parentId = node.parentId >>> 0;
    if (parentId === 0 || parentId === cur) break;
    cur = parentId;
  }

  if (memo) memo[sid] = includeHit ? 1 : -1;
  return includeHit;
}

function createWantedColMatcher(wantCols) {
  const cols = Array.isArray(wantCols) ? wantCols.slice() : [];
  let max = -1;
  for (let i = 0; i < cols.length; i++) {
    const v = cols[i] | 0;
    if (v > max) max = v;
  }

  if (max >= 0 && max <= 4096) {
    const lut = new Int32Array(max + 1);
    lut.fill(-1);
    for (let i = 0; i < cols.length; i++) lut[cols[i] | 0] = i;
    return { count: cols.length, lut, map: null };
  }

  const map = new Map();
  for (let i = 0; i < cols.length; i++) map.set(cols[i] | 0, i);
  return { count: cols.length, lut: null, map };
}

function wantedSlot(matcher, col) {
  if (matcher.lut) {
    if (col < 0 || col >= matcher.lut.length) return -1;
    return matcher.lut[col] | 0;
  }
  const v = matcher.map.get(col | 0);
  return v == null ? -1 : v | 0;
}

function parseExactCsvRowSelected(rowBuf, delimiter, matcher) {
  const out = new Array(matcher.count);
  for (let i = 0; i < out.length; i++) out[i] = "";

  const delim = (String(delimiter || ",").charCodeAt(0) & 255) >>> 0;
  let inQuotes = false;
  let fieldStart = true;
  let col = 0;
  let segStart = -1;
  const parts = [];
  let totalLen = 0;
  let rowDone = false;

  function finishSeg(end) {
    if (segStart < 0) return;
    if (end > segStart) {
      const part = rowBuf.subarray(segStart, end);
      parts.push(part);
      totalLen += part.length;
    }
    segStart = -1;
  }

  function flushField() {
    const slot = wantedSlot(matcher, col);
    if (slot >= 0) {
      if (totalLen <= 0) {
        out[slot] = "";
      } else if (parts.length === 1) {
        out[slot] = parts[0].toString("utf8");
      } else {
        const b = Buffer.allocUnsafe(totalLen);
        let p = 0;
        for (let i = 0; i < parts.length; i++) {
          const part = parts[i];
          part.copy(b, p);
          p += part.length;
        }
        out[slot] = b.toString("utf8");
      }
    }
    parts.length = 0;
    totalLen = 0;
    segStart = -1;
    col++;
    fieldStart = true;
  }

  for (let i = 0; i < rowBuf.length; i++) {
    const b = rowBuf[i];

    if (inQuotes) {
      if (b === CSV_QUOTE) {
        finishSeg(i);
        const next = rowBuf[i + 1];
        if (next === CSV_QUOTE) {
          const slot = wantedSlot(matcher, col);
          if (slot >= 0) {
            parts.push(rowBuf.subarray(i, i + 1));
            totalLen += 1;
          }
          i++;
          continue;
        }
        inQuotes = false;
        fieldStart = false;
        continue;
      }
      if (wantedSlot(matcher, col) >= 0 && segStart < 0) segStart = i;
      continue;
    }

    if (fieldStart && b === CSV_QUOTE) {
      inQuotes = true;
      fieldStart = false;
      continue;
    }

    if (b === delim) {
      finishSeg(i);
      flushField();
      continue;
    }

    if (b === 0x0a || b === 0x0d) {
      finishSeg(i);
      flushField();
      rowDone = true;
      break;
    }

    if (wantedSlot(matcher, col) >= 0 && segStart < 0) segStart = i;
    fieldStart = false;
  }

  if (!rowDone) {
    finishSeg(rowBuf.length);
    if (col > 0 || totalLen > 0 || parts.length > 0 || rowBuf.length > 0)
      flushField();
  }

  return out;
}

function getExactGroupOptions() {
  return {
    maxGapBytes: Math.max(
      0,
      Math.min(
        Number(process.env.COLLATE_EXACT_GROUP_GAP_BYTES || 256 << 10) | 0,
        64 << 20,
      ),
    ),
    maxGroupBytes: Math.max(
      64 << 10,
      Math.min(
        Number(process.env.COLLATE_EXACT_GROUP_MAX_BYTES || 4 << 20) | 0,
        128 << 20,
      ),
    ),
    maxGroupRows: Math.max(
      1,
      Math.min(
        Number(process.env.COLLATE_EXACT_GROUP_MAX_ROWS || 4096) | 0,
        1_000_000,
      ),
    ),
  };
}

function buildExactRowReadGroups(taxa, rowIdsSorted, opts) {
  const groups = [];
  if (!rowIdsSorted.length) return groups;
  if (!taxa || typeof taxa.rowByteRange !== "function") return groups;

  const maxGapBytes = Math.max(
    0,
    Math.min(Number(opts.maxGapBytes || 256 << 10) | 0, 64 << 20),
  );
  const maxGroupBytes = Math.max(
    64 << 10,
    Math.min(Number(opts.maxGroupBytes || 4 << 20) | 0, 128 << 20),
  );
  const maxGroupRows = Math.max(
    1,
    Math.min(Number(opts.maxGroupRows || 4096) | 0, 1_000_000),
  );

  let cur = null;
  for (let i = 0; i < rowIdsSorted.length; i++) {
    const rowId = rowIdsSorted[i] >>> 0;
    const range = taxa.rowByteRange(rowId);
    if (!range) continue;
    const start = Number(range.start);
    const end = Number(range.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start)
      continue;

    if (!cur) {
      cur = {
        startOffset: start,
        endOffset: end,
        startRowId: rowId,
        endRowId: rowId,
        rowIds: [rowId],
      };
      continue;
    }

    const gap = start - cur.endOffset;
    const nextSpan = end - cur.startOffset;
    if (
      gap <= maxGapBytes &&
      nextSpan <= maxGroupBytes &&
      cur.rowIds.length < maxGroupRows
    ) {
      cur.endOffset = end;
      cur.endRowId = rowId;
      cur.rowIds.push(rowId);
      continue;
    }

    groups.push(cur);
    cur = {
      startOffset: start,
      endOffset: end,
      startRowId: rowId,
      endRowId: rowId,
      rowIds: [rowId],
    };
  }
  if (cur) groups.push(cur);
  return groups;
}

class ExactOrWindowRowReader {
  constructor(idx, taxa, csvFd, wantCols) {
    this.idx = idx;
    this.taxa = taxa;
    this.csvFd = csvFd;
    this.wantCols = wantCols.slice();
    this.lookupStride = getBestLookupStride(idx, taxa) >>> 0;
    this.matcher = createWantedColMatcher(this.wantCols);
    this.exactOpts = getExactGroupOptions();
  }

  async visitRowIds(rowIdsSorted, onRow) {
    if (!rowIdsSorted.length) return 0;

    if (
      this.taxa &&
      this.taxa.hasExactRowStarts &&
      typeof this.taxa.rowByteRange === "function"
    ) {
      const groups = buildExactRowReadGroups(
        this.taxa,
        rowIdsSorted,
        this.exactOpts,
      );
      let count = 0;

      for (let gi = 0; gi < groups.length; gi++) {
        const group = groups[gi];
        const groupLen = Math.max(
          0,
          Number(group.endOffset) - Number(group.startOffset),
        );
        if (!Number.isFinite(groupLen) || groupLen <= 0) continue;

        const groupBuf = Buffer.allocUnsafe(groupLen);
        let off = 0;
        while (off < groupLen) {
          const { bytesRead } = await this.csvFd.read(
            groupBuf,
            off,
            groupLen - off,
            Number(group.startOffset) + off,
          );
          if (bytesRead <= 0) break;
          off += bytesRead;
        }
        const usable = off === groupLen ? groupBuf : groupBuf.subarray(0, off);

        for (let i = 0; i < group.rowIds.length; i++) {
          const rowId = group.rowIds[i] >>> 0;
          const range = this.taxa.rowByteRange(rowId);
          if (!range) continue;

          const relStart = Number(range.start) - Number(group.startOffset);
          const relEnd = Number(range.end) - Number(group.startOffset);
          if (
            !Number.isFinite(relStart) ||
            !Number.isFinite(relEnd) ||
            relStart < 0 ||
            relEnd < relStart ||
            relEnd > usable.length
          ) {
            continue;
          }

          const rowBuf = usable.subarray(relStart, relEnd);
          const values = parseExactCsvRowSelected(
            rowBuf,
            this.idx.delimiter,
            this.matcher,
          );
          await onRow({ rowId, values });
          count++;
        }
      }

      return count >>> 0;
    }

    const groups = groupRowIdsByBaseRow(rowIdsSorted, this.lookupStride);
    let count = 0;

    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi];
      const win = getLookupWindowForBaseRow(this.idx, this.taxa, group.baseRow);
      const recs = await readRowsByRowIdsSelectiveRecordsFd({
        fd: this.csvFd,
        delimiter: this.idx.delimiter,
        startOffset: win.startOffset,
        startRowId: win.startRowId,
        stopRowIdExclusive: win.stopRowIdExclusive,
        rowIdsSorted: group.rowIds,
        wantCols: this.wantCols,
        limit: group.rowIds.length,
      });

      for (let i = 0; i < recs.length; i++) {
        await onRow(recs[i]);
        count++;
      }
    }

    return count >>> 0;
  }
}

class LruWriterCache {
  constructor(limit) {
    this._limit = Math.max(1, limit | 0);
    this._map = new Map();
  }

  async get(filePath) {
    if (this._map.has(filePath)) {
      const value = this._map.get(filePath);
      this._map.delete(filePath);
      this._map.set(filePath, value);
      return value;
    }

    if (this._map.size >= this._limit) {
      const oldestKey = this._map.keys().next().value;
      const oldestWs = this._map.get(oldestKey);
      this._map.delete(oldestKey);
      await this._close(oldestWs);
    }

    const ws = fs.createWriteStream(filePath, {
      flags: "a",
      highWaterMark: 4 << 20,
    });
    this._map.set(filePath, ws);
    return ws;
  }

  async append(filePath, buf) {
    if (!buf || buf.length === 0) return;
    const ws = await this.get(filePath);
    await writeStreamChunk(ws, buf);
  }

  async _close(ws) {
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
    for (const ws of this._map.values()) await this._close(ws);
    this._map.clear();
  }
}

function parseArgs(argv) {
  const out = {
    inputCsv: DEFAULT_INPUT_CSV,
    phase: "all",
    outRoot: `${DEFAULT_INPUT_CSV}.taxa_env`,
    pythonBin: process.platform === "win32" ? "py" : "python3",
    runDatasets: DEFAULT_RUN_DATASETS.slice(),
    includeTaxaSelectors: [],
    excludeTaxaSelectors: [],
    extraCols: [],
    speciesMinObs: 1,
    maxSpecies: 0,
    rowBatch: 8192,
    progressEveryMs: 10000,
    xCol: "lon",
    yCol: "lat",
    inputCrs: "EPSG:4326",
    joinShards: 100,
    joinMinRows: 100000,
    openWriters: 64,
    keepTemps: false,
    cleanup: false,
    cleanupEnrichedMode: "none",
    cleanupEnrichedProgressEvery: 250000,
    cleanupEnrichedBackup: false,
    append: false,
    forceRerun: false,
    demScript: DEFAULT_PATHS.demScript,
    demIndex: DEFAULT_PATHS.demIndex,
    demLayers: "all",
    terraclimateScript: DEFAULT_PATHS.terraclimateScript,
    terraclimateRoot: DEFAULT_PATHS.terraclimateRoot,
    terraclimateVars: "all",
    terraclimateYear: "latest",
    twiScript: DEFAULT_PATHS.twiScript,
    twiTif: DEFAULT_PATHS.twiTif,
    twiChunkSize: 250000,
    soilgridsScript: DEFAULT_PATHS.soilgridsScript,
    soilgridsRoot: DEFAULT_PATHS.soilgridsRoot,
    soilgridsMergedRoot: DEFAULT_PATHS.soilgridsMergedRoot,
    soilgridsProps: "bdod,cec,clay,sand,silt,soc,phh2o,nitrogen,cfvo",
    soilgridsDepths: "0-5cm,5-15cm,15-30cm",
    soilgridsChunkSize: 50000,
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
    if (!isCliFlagToken(a)) {
      out.inputCsv = String(a);
      if (out.outRoot === `${DEFAULT_INPUT_CSV}.taxa_env`) {
        out.outRoot = `${out.inputCsv}.taxa_env`;
      }
      continue;
    }
    if (a === "--help" || a === "-h") {
      console.log(
        [
          "Usage:",
          "  node collect_taxa_env.js [occurrences.csv] [options]",
          `  default input csv: ${DEFAULT_INPUT_CSV}`,
          "",
          "Main modes:",
          "  --phase extract|sample|merge|all",
          "",
          "Resume / overwrite:",
          "  existing extract/sample/merge outputs are reused by default",
          "  --force-rerun                  overwrite and rebuild existing outputs",
          "",
          "Append mode:",
          "  --append                       append only new species into an existing out-root",
          "  requires --phase all and compares requested species against species_manifest.csv",
          "",
          "Selectors:",
          "  --include-taxa species:Daucus pusillus",
          "  --exclude-taxa kingdom:Animalia",
          "  --extra-cols country,habitat,stateProvince",
          "",
          "Sampling:",
          `  default --run: ${DEFAULT_RUN_DATASETS.join(",")}`,
          "  --run dem,terraclimate,twi,soilgrids,glim,mcd12q1",
          "  --python py",
          "",
          "DEM:",
          "  --dem-script D:/DEM_Derived_w_flow/sample_coords.py",
          "  --dem-index D:/DEM_derived_w_flow/dem_flow_index.json",
          "  --dem-layers all",
          "",
          "TerraClimate:",
          "  --terraclimate-script D:/terraclimate/sample_cogs_from_coords.py",
          "  --terraclimate-root D:/terraclimate/terraclimate_cogs_global",
          "  --terraclimate-vars all",
          "  --terraclimate-year latest",
          "  --terraclimate-year 2020",
          "  --terraclimate-year 2018,2019,2020",
          "  --terraclimate-year 2018-2020",
          "",
          "TWI:",
          "  --twi-script D:/wetness/sample_twi_coords.py",
          "  --twi-tif D:/wetness/twi_edtm_120m.tif",
          "  --twi-chunk-size 250000",
          "",
          "SoilGrids:",
          "  --soilgrids-script D:/soilgrids/sample_soilgrids_coords.py",
          "  --soilgrids-root D:/soilgrids/data",
          "  --soilgrids-merged-root D:/soilgrids/merged",
          "  auto-detects sample_soilgrids_merged.py in the same folder as --soilgrids-script",
          "  and prefers merged sampling when the merged root contains .tif files",
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
          "",
          `  default DEM script: ${DEFAULT_PATHS.demScript}`,
          `  default DEM index: ${DEFAULT_PATHS.demIndex}`,
          `  default TerraClimate script: ${DEFAULT_PATHS.terraclimateScript}`,
          `  default TerraClimate root: ${DEFAULT_PATHS.terraclimateRoot}`,
          `  default TWI script: ${DEFAULT_PATHS.twiScript}`,
          `  default TWI tif: ${DEFAULT_PATHS.twiTif}`,
          `  default SoilGrids script: ${DEFAULT_PATHS.soilgridsScript}`,
          `  default SoilGrids root: ${DEFAULT_PATHS.soilgridsRoot}`,
          `  default SoilGrids merged root: ${DEFAULT_PATHS.soilgridsMergedRoot}`,
          `  default GLiM script: ${DEFAULT_PATHS.glimScript}`,
          `  default GLiM tif: ${DEFAULT_PATHS.glimTif}`,
          `  default GLiM lookup: ${DEFAULT_PATHS.glimLookupCsv}`,
          `  default MCD12Q1 script: ${DEFAULT_PATHS.mcd12q1Script}`,
          `  default MCD12Q1 vrt: ${DEFAULT_PATHS.mcd12q1Vrt}`,
          "Cleanup enriched:",
          "  --cleanup-enriched invalid|soilgrids|invalid+soilgrids|none",
          "  --cleanup-enriched-script ./cleanup_occurrences_enriched.js",
          "  --cleanup-enriched-progress-every 250000",
          "  --cleanup-enriched-backup",
          "",
          "Other:",
          "  --out-root D:/envpull_daucus",
          "  --join-shards 256",
          "  --join-min-rows 2000",
          "  --cleanup",
          "  --keep-temps",
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
      out.runDatasets = uniqueStrings(parseCsvList(needValue(i, a))).map((s) =>
        s.toLowerCase(),
      );
      i++;
    } else if (a === "--include-taxa") {
      out.includeTaxaSelectors = parseTaxonSelectorList(needValue(i, a));
      i++;
    } else if (a === "--exclude-taxa") {
      out.excludeTaxaSelectors = parseTaxonSelectorList(needValue(i, a));
      i++;
    } else if (a === "--extra-cols") {
      out.extraCols = uniqueStrings(parseCsvList(needValue(i, a)));
      i++;
    } else if (a === "--species-min-obs") {
      out.speciesMinObs = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--max-species") {
      out.maxSpecies = Math.max(0, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--row-batch") {
      out.rowBatch = Math.max(256, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--progress-every-ms") {
      out.progressEveryMs = Math.max(1000, Number(needValue(i, a)) | 0);
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
    } else if (a === "--cleanup-enriched") {
      out.cleanupEnrichedMode = String(needValue(i, a)).toLowerCase();
      i++;
    } else if (a === "--cleanup-enriched-script") {
      out.cleanupEnrichedScript = String(needValue(i, a));
      i++;
    } else if (a === "--cleanup-enriched-progress-every") {
      out.cleanupEnrichedProgressEvery = Math.max(
        1,
        Number(needValue(i, a)) | 0,
      );
      i++;
    } else if (a === "--cleanup-enriched-backup") {
      out.cleanupEnrichedBackup = true;
    } else if (a === "--append") {
      out.append = true;
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
      out.terraclimateVars = normalizeTerraclimateVarsSpec(
        String(needValue(i, a)),
      );
      i++;
    } else if (a === "--terraclimate-year") {
      out.terraclimateYear = normalizeTerraclimateYearSpec(
        String(needValue(i, a)),
      );
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
    } else if (a === "--soilgrids-merged-root") {
      out.soilgridsMergedRoot = String(needValue(i, a));
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

  if (!["extract", "sample", "merge", "all"].includes(out.phase)) {
    throw new Error(`Bad --phase value: ${out.phase}`);
  }
  if (
    !["none", "invalid", "soilgrids", "invalid+soilgrids"].includes(
      out.cleanupEnrichedMode,
    )
  ) {
    throw new Error(`Bad --cleanup-enriched value: ${out.cleanupEnrichedMode}`);
  }
  if (out.append && out.phase !== "all") {
    throw new Error(`--append currently requires --phase all`);
  }

  out.terraclimateVars = normalizeTerraclimateVarsSpec(out.terraclimateVars);
  out.terraclimateYear = normalizeTerraclimateYearSpec(out.terraclimateYear);

  return out;
}

async function enumerateSpecies(taxa, opts, taxonFilter) {
  const out = [];
  let filteredOut = 0;

  for (let nodeId = 1; nodeId < taxa.nodeCount >>> 0; nodeId++) {
    const n = taxa.readNode(nodeId);
    if (!n || n.rankId >>> 0 !== TAXON_RANK_NAME_TO_ID.species) continue;
    const count = n.count >>> 0;
    if (count < opts.speciesMinObs >>> 0) continue;
    if (!speciesMatchesTaxonFilter(taxa, nodeId >>> 0, taxonFilter)) {
      filteredOut++;
      continue;
    }
    out.push({ speciesNodeId: nodeId >>> 0, obsCount: count });
  }

  out.sort((a, b) => {
    const c = (b.obsCount >>> 0) - (a.obsCount >>> 0);
    if (c) return c;
    return (a.speciesNodeId >>> 0) - (b.speciesNodeId >>> 0);
  });
  if (opts.maxSpecies > 0 && out.length > opts.maxSpecies)
    out.length = opts.maxSpecies;

  console.log(
    `[taxon-filter] kept species=${out.length.toLocaleString()} filtered_out=${filteredOut.toLocaleString()} summary=${taxonFilterSummary(taxonFilter)}`,
  );
  return out;
}

function defaultOccurrenceCols() {
  return [
    "gbifID",
    "occurrenceID",
    "scientificName",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "country",
    "stateProvince",
    "eventDate",
    "year",
    "month",
    "day",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "basisOfRecord",
  ];
}

async function extractOccurrencesAndCoords({
  idx,
  taxa,
  opts,
  taxonFilter,
  outRoot,
  inputCsv,
}) {
  console.log(`[extract] starting ${nowIso()}`);

  const occurrencesPath = path.join(outRoot, "occurrences_selected.csv");
  const coordsPath = path.join(outRoot, "coords.csv");
  const speciesManifestPath = path.join(outRoot, "species_manifest.csv");
  const manifestPath = path.join(outRoot, "manifest.json");

  if (
    !opts.forceRerun &&
    (await allFilesExistNonEmpty([
      occurrencesPath,
      coordsPath,
      speciesManifestPath,
      manifestPath,
    ]))
  ) {
    console.log(`[extract] reusing existing files in ${outRoot}`);
    return await readJson(manifestPath);
  }

  const speciesPlans = await enumerateSpecies(taxa, opts, taxonFilter);
  if (!speciesPlans.length)
    throw new Error("No species matched the requested selectors");

  return await writeExtractOutputs({
    idx,
    taxa,
    opts,
    speciesPlans,
    outRoot,
    inputCsv,
    includeSelectors: taxonFilter.includeSelectors,
    excludeSelectors: taxonFilter.excludeSelectors,
    startId: 1,
  });
}

function writeJson(filePath, obj) {
  return fs.promises.writeFile(filePath, JSON.stringify(obj, null, 2));
}

async function readJson(filePath) {
  return JSON.parse(await fs.promises.readFile(filePath, "utf8"));
}


async function readSpeciesManifestEntries(filePath) {
  const rows = [];
  const nodeIds = new Set();
  if (!(await fileExistsNonEmpty(filePath))) return { rows, nodeIds };

  const rs = fs.createReadStream(filePath, {
    encoding: "utf8",
    highWaterMark: 1 << 20,
  });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });

  let first = true;
  try {
    for await (const line of rl) {
      if (line == null || line === "") continue;
      if (first) {
        first = false;
        continue;
      }
      const vals = parseCsvLine(line);
      const speciesNodeId = Number(vals[0]);
      const speciesName = String(vals[1] || "");
      const obsCount = Number(vals[2]);
      const row = {
        speciesNodeId: Number.isFinite(speciesNodeId) ? speciesNodeId >>> 0 : vals[0],
        speciesName,
        obsCount: Number.isFinite(obsCount) ? obsCount >>> 0 : 0,
      };
      rows.push(row);
      if (Number.isFinite(speciesNodeId)) nodeIds.add(speciesNodeId >>> 0);
    }
  } finally {
    rl.close();
    rs.destroy();
  }

  return { rows, nodeIds };
}

async function appendCsvDataRows({ srcPath, dstPath, label }) {
  if (!(await fileExistsNonEmpty(srcPath))) return 0;
  if (!(await fileExistsNonEmpty(dstPath))) {
    throw new Error(`Missing append destination for ${label}: ${dstPath}`);
  }

  const srcHeader = await readCsvHeader(srcPath);
  const dstHeader = await readCsvHeader(dstPath);
  if (srcHeader.join("\x1f") !== dstHeader.join("\x1f")) {
    throw new Error(`Header mismatch while appending ${label}: ${srcPath} -> ${dstPath}`);
  }

  const rs = fs.createReadStream(srcPath, {
    encoding: "utf8",
    highWaterMark: 4 << 20,
  });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  const ws = fs.createWriteStream(dstPath, {
    flags: "a",
    highWaterMark: 4 << 20,
  });

  let first = true;
  let rows = 0;

  try {
    for await (const line of rl) {
      if (first) {
        first = false;
        continue;
      }
      if (line == null || line === "") continue;
      await writeStreamChunk(ws, Buffer.from(line + "\n", "utf8"));
      rows++;
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

  return rows >>> 0;
}

function selectorRawList(selectors) {
  return (selectors || []).map((s) => (s && s.raw ? s.raw : String(s || "").trim())).filter(Boolean);
}

function mergeSelectorRawLists(existingList, selectors) {
  return uniqueStrings([].concat(existingList || [], selectorRawList(selectors)));
}


function normalizedSelectorList(selectors) {
  return uniqueStrings(
    (selectors || [])
      .map((s) => String(s || "").trim())
      .filter(Boolean),
  ).sort();
}

function sameStringSets(a, b) {
  const aa = normalizedSelectorList(a);
  const bb = normalizedSelectorList(b);
  if (aa.length !== bb.length) return false;
  for (let i = 0; i < aa.length; i++) {
    if (aa[i] !== bb[i]) return false;
  }
  return true;
}

function sameNumberSets(a, b) {
  const aa = Array.from(a || []).map((v) => Number(v) >>> 0).sort((x, y) => x - y);
  const bb = Array.from(b || []).map((v) => Number(v) >>> 0).sort((x, y) => x - y);
  if (aa.length !== bb.length) return false;
  for (let i = 0; i < aa.length; i++) {
    if (aa[i] !== bb[i]) return false;
  }
  return true;
}

async function listAppendStageDirs(outRoot) {
  const out = [];
  const entries = await fs.promises.readdir(outRoot, { withFileTypes: true }).catch(() => []);
  for (let i = 0; i < entries.length; i++) {
    const d = entries[i];
    if (!d || !d.isDirectory()) continue;
    if (!/^\.append_stage_/.test(d.name)) continue;
    const stageRoot = path.join(outRoot, d.name);
    const st = await fs.promises.stat(stageRoot).catch(() => null);
    out.push({ stageRoot, mtimeMs: st && Number.isFinite(st.mtimeMs) ? st.mtimeMs : 0 });
  }
  out.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return out;
}

async function findResumableAppendStage({
  outRoot,
  inputCsv,
  baseRowCount,
  deltaSpeciesPlans,
  includeSelectors,
  excludeSelectors,
  requestedSampleConfig,
}) {
  const wantedSpeciesIds = new Set(
    (deltaSpeciesPlans || []).map((plan) => Number(plan.speciesNodeId) >>> 0),
  );
  const wantedIdStart = (Number(baseRowCount) >>> 0) + 1;
  const wantedInclude = selectorRawList(includeSelectors);
  const wantedExclude = selectorRawList(excludeSelectors);
  const candidates = await listAppendStageDirs(outRoot);

  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i];
    const stageRoot = candidate.stageRoot;
    const stageManifestPath = path.join(stageRoot, 'manifest.json');
    const stageOccurrencesPath = path.join(stageRoot, 'occurrences_selected.csv');
    const stageCoordsPath = path.join(stageRoot, 'coords.csv');
    const stageSpeciesManifestPath = path.join(stageRoot, 'species_manifest.csv');

    if (!(await allFilesExistNonEmpty([
      stageManifestPath,
      stageOccurrencesPath,
      stageCoordsPath,
      stageSpeciesManifestPath,
    ]))) {
      continue;
    }

    const manifest = await readJson(stageManifestPath).catch(() => null);
    if (!manifest || typeof manifest !== 'object') continue;

    if (manifest.csv && path.resolve(String(manifest.csv)) !== path.resolve(inputCsv)) {
      continue;
    }

    if (Number(manifest.idStart) !== wantedIdStart) {
      continue;
    }

    if (manifest.taxonFilter) {
      if (!sameStringSets(manifest.taxonFilter.include, wantedInclude)) continue;
      if (!sameStringSets(manifest.taxonFilter.exclude, wantedExclude)) continue;
    }

    if (
      manifest.sampleConfig &&
      !sameTerraclimateConfig(
        manifest.sampleConfig.terraclimate,
        requestedSampleConfig && requestedSampleConfig.terraclimate,
      )
    ) {
      continue;
    }

    const speciesInfo = await readSpeciesManifestEntries(stageSpeciesManifestPath).catch(() => null);
    if (!speciesInfo || !sameNumberSets(speciesInfo.nodeIds, wantedSpeciesIds)) {
      continue;
    }

    return {
      stageRoot,
      manifest,
      speciesInfo,
    };
  }

  return null;
}

async function writeExtractOutputs({
  idx,
  taxa,
  opts,
  speciesPlans,
  outRoot,
  inputCsv,
  includeSelectors,
  excludeSelectors,
  startId = 1,
}) {
  const occurrencesPath = path.join(outRoot, "occurrences_selected.csv");
  const coordsPath = path.join(outRoot, "coords.csv");
  const speciesManifestPath = path.join(outRoot, "species_manifest.csv");
  const manifestPath = path.join(outRoot, "manifest.json");

  const header = chooseHeader(idx, "guess") || idx.header || [];
  const wantNames = uniqueStrings(
    defaultOccurrenceCols().concat(opts.extraCols || []),
  );
  const wantCols = [];
  const outHeader = [];
  const seenCols = new Set();
  for (let i = 0; i < wantNames.length; i++) {
    const colIndex = findColIndex(header, wantNames[i]);
    if (colIndex < 0) continue;
    if (seenCols.has(colIndex)) continue;
    seenCols.add(colIndex);
    wantCols.push(colIndex);
    outHeader.push(header[colIndex]);
  }

  const lonIx = outHeader.indexOf("decimalLongitude");
  const latIx = outHeader.indexOf("decimalLatitude");
  if (lonIx < 0 || latIx < 0) {
    throw new Error(
      "Missing decimalLongitude and/or decimalLatitude in projected columns",
    );
  }

  ensureDir(outRoot);
  const occWs = fs.createWriteStream(occurrencesPath, {
    flags: "w",
    highWaterMark: 4 << 20,
  });
  const coordsWs = fs.createWriteStream(coordsPath, {
    flags: "w",
    highWaterMark: 4 << 20,
  });
  const speciesWs = fs.createWriteStream(speciesManifestPath, {
    flags: "w",
    highWaterMark: 1 << 20,
  });

  const reader = new ExactOrWindowRowReader(
    idx,
    taxa,
    idx._sharedCsvFd,
    wantCols,
  );
  const startedAt = Date.now();
  let totalRowsSeen = 0;
  let totalRowsWritten = 0;
  let nextId = Math.max(1, Number(startId) | 0);

  await writeStreamChunk(
    occWs,
    Buffer.from(
      ["id", "row_id", "species_node_id", "matched_species_name"]
        .concat(outHeader)
        .join(",") + "\n",
    ),
  );
  await writeStreamChunk(coordsWs, Buffer.from("id,lon,lat\n"));
  await writeStreamChunk(
    speciesWs,
    Buffer.from("species_node_id,species_name,obs_count\n"),
  );

  try {
    for (let si = 0; si < speciesPlans.length; si++) {
      const plan = speciesPlans[si];
      const sid = plan.speciesNodeId >>> 0;
      const speciesName = makeSpeciesLabel(taxa, sid);
      await writeStreamChunk(
        speciesWs,
        Buffer.from(
          `${sid},${csvEscapeValue(speciesName)},${plan.obsCount >>> 0}\n`,
        ),
      );

      let rowBatch = [];
      let rowsSeenSpecies = 0;
      let rowsWrittenSpecies = 0;
      let lastHeartbeatAt = Date.now();
      const speciesStartedAt = Date.now();

      const flushBatch = async (batch) => {
        if (!batch.length) return;
        const sorted = batch.slice().sort((a, b) => a - b);
        const occParts = [];
        const coordParts = [];

        await reader.visitRowIds(sorted, async (rec) => {
          rowsSeenSpecies++;
          totalRowsSeen++;

          const vals = rec.values;
          const lon = safeNumber(vals[lonIx], NaN);
          const lat = safeNumber(vals[latIx], NaN);
          if (!Number.isFinite(lon) || !Number.isFinite(lat)) return;
          if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;

          const sampleId = nextId++;
          rowsWrittenSpecies++;
          totalRowsWritten++;

          const occLine =
            [
              sampleId,
              rec.rowId >>> 0,
              sid,
              csvEscapeValue(speciesName),
              ...vals.map((v) => csvEscapeValue(v || "")),
            ].join(",") + "\n";
          occParts.push(occLine);
          coordParts.push(`${sampleId},${lon},${lat}\n`);

          const now = Date.now();
          if (now - lastHeartbeatAt >= opts.progressEveryMs) {
            lastHeartbeatAt = now;
            const speciesElapsedMs = Math.max(1, now - speciesStartedAt);
            const totalElapsedMs = Math.max(1, now - startedAt);
            console.log(
              `[extract] species ${si + 1}/${speciesPlans.length} | ${speciesName} | obs=${plan.obsCount.toLocaleString()} rows_seen_species=${rowsSeenSpecies.toLocaleString()} rows_written_species=${rowsWrittenSpecies.toLocaleString()} rows_seen_total=${totalRowsSeen.toLocaleString()} rows_written_total=${totalRowsWritten.toLocaleString()} rate_species=${formatRatePerSec(rowsSeenSpecies, speciesElapsedMs)}/s rate_total=${formatRatePerSec(totalRowsWritten, totalElapsedMs)}/s`,
            );
          }
        });

        if (occParts.length)
          await writeStreamChunk(occWs, Buffer.from(occParts.join("")));
        if (coordParts.length)
          await writeStreamChunk(coordsWs, Buffer.from(coordParts.join("")));
      };

      await streamSpeciesRowIds(taxa, sid, async (rowId) => {
        rowBatch.push(rowId >>> 0);
        if (rowBatch.length >= opts.rowBatch) {
          const batch = rowBatch;
          rowBatch = [];
          await flushBatch(batch);
        }
      });

      if (rowBatch.length) {
        const batch = rowBatch;
        rowBatch = [];
        await flushBatch(batch);
      }

      console.log(
        `[extract] finished species ${si + 1}/${speciesPlans.length} | ${speciesName} | rows_written_species=${rowsWrittenSpecies.toLocaleString()}`,
      );
      maybeGC("collect-taxa-env-species");
    }
  } finally {
    await Promise.all([
      new Promise((resolve, reject) => {
        occWs.on("finish", resolve);
        occWs.on("error", reject);
        occWs.end();
      }),
      new Promise((resolve, reject) => {
        coordsWs.on("finish", resolve);
        coordsWs.on("error", reject);
        coordsWs.end();
      }),
      new Promise((resolve, reject) => {
        speciesWs.on("finish", resolve);
        speciesWs.on("error", reject);
        speciesWs.end();
      }),
    ]);
  }

  const manifest = {
    createdAt: nowIso(),
    csv: path.resolve(inputCsv),
    outRoot,
    occurrencesPath,
    coordsPath,
    speciesManifestPath,
    taxonFilter: {
      include: selectorRawList(includeSelectors),
      exclude: selectorRawList(excludeSelectors),
    },
    speciesCount: speciesPlans.length,
    rowCount: totalRowsWritten,
    projectedColumns: outHeader,
    idStart: Math.max(1, Number(startId) | 0),
    idEnd: totalRowsWritten > 0 ? (nextId - 1) >>> 0 : ((Math.max(1, Number(startId) | 0) - 1) >>> 0),
  };
  await writeJson(manifestPath, manifest);

  console.log(
    `[extract] done ${nowIso()} rows=${totalRowsWritten.toLocaleString()} out=${outRoot}`,
  );
  return manifest;
}

function extractOutputPaths(outRoot) {
  return {
    occurrencesPath: path.join(outRoot, "occurrences_selected.csv"),
    coordsPath: path.join(outRoot, "coords.csv"),
    speciesManifestPath: path.join(outRoot, "species_manifest.csv"),
    manifestPath: path.join(outRoot, "manifest.json"),
  };
}

async function rebuildExtractArtifactsFromSelected({
  outRoot,
  inputCsv,
  includeSelectors,
  excludeSelectors,
}) {
  const {
    occurrencesPath,
    coordsPath,
    speciesManifestPath,
    manifestPath,
  } = extractOutputPaths(outRoot);

  if (!(await fileExistsNonEmpty(occurrencesPath))) return null;

  const haveCoords = await fileExistsNonEmpty(coordsPath);
  const haveSpeciesManifest = await fileExistsNonEmpty(speciesManifestPath);
  const existingManifest = await readJson(manifestPath).catch(() => null);

  if (haveCoords && haveSpeciesManifest && existingManifest) {
    console.log(`[extract] reusing existing files in ${outRoot}`);
    return existingManifest;
  }

  console.log(
    `[extract] reusing existing occurrences_selected.csv and reconstructing missing extract artifacts in ${outRoot}`,
  );

  const rs = fs.createReadStream(occurrencesPath, {
    encoding: "utf8",
    highWaterMark: 4 << 20,
  });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });

  let coordsWs = null;
  let speciesWs = null;
  let header = null;
  let idIdx = -1;
  let lonIdx = -1;
  let latIdx = -1;
  let speciesNodeIdx = -1;
  let speciesNameIdx = -1;
  let rowCount = 0;
  const speciesCounts = new Map();

  try {
    if (!haveCoords) {
      coordsWs = fs.createWriteStream(coordsPath, {
        flags: "w",
        highWaterMark: 4 << 20,
      });
      await writeStreamChunk(coordsWs, Buffer.from("id,lon,lat\n", "utf8"));
    }

    for await (const line of rl) {
      if (line == null || line === "") continue;

      if (header == null) {
        header = parseCsvLine(line);
        idIdx = header.indexOf("id");
        lonIdx = header.indexOf("decimalLongitude");
        latIdx = header.indexOf("decimalLatitude");
        speciesNodeIdx = header.indexOf("species_node_id");
        speciesNameIdx = header.indexOf("matched_species_name");

        if (idIdx < 0)
          throw new Error(`Missing id in ${occurrencesPath}`);
        if (lonIdx < 0 || latIdx < 0) {
          throw new Error(
            `Missing decimalLongitude and/or decimalLatitude in ${occurrencesPath}`,
          );
        }
        if (speciesNodeIdx < 0 || speciesNameIdx < 0) {
          throw new Error(
            `Missing species_node_id and/or matched_species_name in ${occurrencesPath}`,
          );
        }
        continue;
      }

      const vals = parseCsvLine(line);
      rowCount++;

      if (coordsWs) {
        await writeStreamChunk(
          coordsWs,
          Buffer.from(
            `${vals[idIdx] || ""},${vals[lonIdx] || ""},${vals[latIdx] || ""}\n`,
            "utf8",
          ),
        );
      }

      const sidRaw = String(vals[speciesNodeIdx] || "").trim();
      const sid = Number(sidRaw);
      const speciesName = String(vals[speciesNameIdx] || "").trim();
      const key = Number.isFinite(sid) ? String(sid >>> 0) : sidRaw;
      const cur = speciesCounts.get(key) || {
        speciesNodeId: Number.isFinite(sid) ? sid >>> 0 : sidRaw,
        speciesName,
        obsCount: 0,
      };
      cur.obsCount++;
      if (!cur.speciesName && speciesName) cur.speciesName = speciesName;
      speciesCounts.set(key, cur);
    }
  } finally {
    rl.close();
    rs.destroy();

    if (coordsWs) {
      await new Promise((resolve, reject) => {
        coordsWs.on("finish", resolve);
        coordsWs.on("error", reject);
        coordsWs.end();
      });
    }
  }

  if (!header || !header.length) {
    throw new Error(`No header found in ${occurrencesPath}`);
  }

  if (!haveSpeciesManifest) {
    speciesWs = fs.createWriteStream(speciesManifestPath, {
      flags: "w",
      highWaterMark: 1 << 20,
    });
    await writeStreamChunk(
      speciesWs,
      Buffer.from("species_node_id,species_name,obs_count\n", "utf8"),
    );

    const rows = Array.from(speciesCounts.values()).sort((a, b) => {
      const an = Number(a.speciesNodeId);
      const bn = Number(b.speciesNodeId);
      if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn;
      return String(a.speciesNodeId).localeCompare(String(b.speciesNodeId));
    });

    try {
      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        await writeStreamChunk(
          speciesWs,
          Buffer.from(
            `${row.speciesNodeId},${csvEscapeValue(row.speciesName || "")},${row.obsCount >>> 0}\n`,
            "utf8",
          ),
        );
      }
    } finally {
      await new Promise((resolve, reject) => {
        speciesWs.on("finish", resolve);
        speciesWs.on("error", reject);
        speciesWs.end();
      });
    }
  }

  const manifest = Object.assign({}, existingManifest || {}, {
    createdAt: existingManifest && existingManifest.createdAt ? existingManifest.createdAt : nowIso(),
    csv: path.resolve(inputCsv),
    outRoot,
    occurrencesPath,
    coordsPath,
    speciesManifestPath,
    taxonFilter: {
      include: (includeSelectors || []).map((s) => s.raw),
      exclude: (excludeSelectors || []).map((s) => s.raw),
    },
    speciesCount: speciesCounts.size,
    rowCount,
    projectedColumns: header.slice(4),
    reconstructedFromOccurrencesSelected: true,
  });
  await writeJson(manifestPath, manifest);

  console.log(
    `[extract] reconstructed artifacts rows=${rowCount.toLocaleString()} species=${speciesCounts.size.toLocaleString()} out=${outRoot}`,
  );
  return manifest;
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

function buildRunPlan(opts, outRoot) {
  const outputs = datasetOutputPaths(outRoot);
  const wanted = new Set((opts.runDatasets || []).map((s) => s.toLowerCase()));
  if (wanted.has("all")) {
    wanted.clear();
    ["dem", "terraclimate", "twi", "soilgrids", "glim", "mcd12q1"].forEach(
      (k) => wanted.add(k),
    );
  }
  const coordsPath = path.join(outRoot, "coords.csv");
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

  if (
    wanted.has("terraclimate") &&
    opts.terraclimateScript &&
    opts.terraclimateRoot
  ) {
    commands.push({
      key: "terraclimate",
      outPath: outputs.terraclimate,
      cmd: opts.pythonBin,
      args: [
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
        3, //nearest neighbor search e.g. for shoreline species that might miss the soilgrids raster edges, not bulletproof
      ],
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
        opts.xCol,
        "--lat-col",
        opts.yCol,
        "--chunk-size",
        String(opts.twiChunkSize || 250000),
      ],
    });
  }

  const soilgridsSampler = resolveSoilgridsSampler(opts);
  if (wanted.has("soilgrids") && soilgridsSampler.script && soilgridsSampler.root) {
    console.log(
      `[sample-plan] soilgrids sampler=${soilgridsSampler.mode} script=${soilgridsSampler.script} root=${soilgridsSampler.root}`,
    );
    commands.push({
      key: "soilgrids",
      outPath: outputs.soilgrids,
      cmd: opts.pythonBin,
      args: [
        soilgridsSampler.script,
        "--root",
        soilgridsSampler.root,
        "--coords",
        coordsPath,
        "--out",
        outputs.soilgrids,
        "--input-crs",
        opts.inputCrs,
        "--lon-col",
        opts.xCol,
        "--lat-col",
        opts.yCol,
        "--id-col",
        "id",
        "--props",
        opts.soilgridsProps,
        "--depths",
        opts.soilgridsDepths,
        "--workers",
        1,
        "--fallback-radius-pixels",
        3, //nearest neighbor search e.g. for shoreline species that might miss the soilgrids raster edges, not bulletproof
        "--chunk-size",
        String(opts.soilgridsChunkSize || 100000),
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
      opts.xCol,
      "--y-col",
      opts.yCol,
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
        opts.xCol,
        "--y-col",
        opts.yCol,
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
      else
        reject(
          new Error(
            `Command failed with exit ${code}: ${cmd} ${args.join(" ")}`,
          ),
        );
    });
  });
}

async function updateManifestFinalPath(outRoot, updates) {
  const manifestPath = path.join(outRoot, "manifest.json");
  const manifest = await readJson(manifestPath).catch(() => ({}));
  Object.assign(manifest, updates || {});
  await writeJson(manifestPath, manifest);
  return manifest;
}

async function runCleanupEnriched({ opts, outRoot, mergedPath }) {
  const mode = String(opts.cleanupEnrichedMode || "none").toLowerCase();
  const targetPath = path.resolve(
    mergedPath || path.join(outRoot, "occurrences_enriched.csv"),
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

  const manifestPath = path.join(outRoot, "manifest.json");
  const manifest = await readJson(manifestPath).catch(() => ({}));
  const existingCleanup = manifest.cleanupEnriched || null;
  const existingFinalPath = manifest.finalPath
    ? path.resolve(String(manifest.finalPath))
    : "";

  if (
    !opts.forceRerun &&
    existingCleanup &&
    existingCleanup.enabled &&
    String(existingCleanup.mode || "").toLowerCase() === mode &&
    existingFinalPath === targetPath &&
    (await fileExistsNonEmpty(targetPath))
  ) {
    console.log(
      `[cleanup-enriched] reusing existing cleaned file ${targetPath} mode=${mode}`,
    );
    return {
      finalPath: targetPath,
      cleanupEnriched: true,
      cleanupMode: mode,
    };
  }

  const scriptPath = path.resolve(
    String(opts.cleanupEnrichedScript || DEFAULT_PATHS.cleanupEnrichedScript),
  );
  const scriptStat = await pathExists(scriptPath);
  if (!scriptStat || !scriptStat.isFile()) {
    throw new Error(`Cleanup script not found: ${scriptPath}`);
  }
  if (!(await fileExistsNonEmpty(targetPath))) {
    throw new Error(`Missing merged file for cleanup: ${targetPath}`);
  }

  const args = [
    scriptPath,
    targetPath,
    "--mode",
    mode,
    "--inplace",
    "--progress-every",
    String(Math.max(1, Number(opts.cleanupEnrichedProgressEvery) || 250000)),
  ];
  if (!opts.cleanupEnrichedBackup) args.push("--no-backup");

  console.log(
    `[cleanup-enriched] starting ${nowIso()} mode=${mode} target=${targetPath}`,
  );
  await runCommandLive(process.execPath, args);
  console.log(`[cleanup-enriched] done ${nowIso()} target=${targetPath}`);

  await updateManifestFinalPath(outRoot, {
    finalPath: targetPath,
    cleanupEnriched: {
      enabled: true,
      mode,
      scriptPath,
      backup: !!opts.cleanupEnrichedBackup,
      progressEvery: Math.max(
        1,
        Number(opts.cleanupEnrichedProgressEvery) || 250000,
      ),
      completedAt: nowIso(),
    },
  }).catch(() => {});

  return {
    finalPath: targetPath,
    cleanupEnriched: true,
    cleanupMode: mode,
  };
}

async function runSamplers({ opts, outRoot }) {
  const { commands, outputs } = buildRunPlan(opts, outRoot);
  if (!commands.length) {
    console.log("[sample] nothing to run");
    return { commands: [], outputs };
  }

  const coordsPath = path.join(outRoot, "coords.csv");
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

  console.log(
    `[sample] starting ${nowIso()} jobs=${runnable.length}/${commands.length}`,
  );
  for (let i = 0; i < runnable.length; i++) {
    const job = runnable[i];
    console.log(
      `[sample] ${i + 1}/${runnable.length} ${job.key} -> ${job.outPath}`,
    );
    await runCommandLive(job.cmd, job.args);
  }
  console.log(`[sample] done ${nowIso()}`);
  return { commands, outputs };
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

async function assertAppendableCsvHeaders({ srcPath, dstPath, label }) {
  const srcHeader = await readCsvHeader(srcPath);
  const dstHeader = await readCsvHeader(dstPath);

  if (!srcHeader.length) {
    throw new Error(`Missing header in append source for ${label}: ${srcPath}`);
  }
  if (!dstHeader.length) {
    throw new Error(`Missing header in append destination for ${label}: ${dstPath}`);
  }

  if (srcHeader.join("\x1f") !== dstHeader.join("\x1f")) {
    const srcOnly = srcHeader.filter((h) => !dstHeader.includes(h));
    const dstOnly = dstHeader.filter((h) => !srcHeader.includes(h));
    throw new Error(
      [
        `Header mismatch while appending ${label}: ${srcPath} -> ${dstPath}`,
        srcOnly.length ? `source-only columns: ${srcOnly.join(", ")}` : null,
        dstOnly.length ? `destination-only columns: ${dstOnly.join(", ")}` : null,
      ]
        .filter(Boolean)
        .join("\n"),
    );
  }

  return { header: srcHeader };
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

async function getBaseRowCountAndManifest(outRoot, basePath) {
  const manifestPath = path.join(outRoot, "manifest.json");
  const manifest = await readJson(manifestPath).catch(() => ({}));

  const manifestRows = safeNumber(manifest.rowCount, 0);
  if (manifestRows > 0) {
    return {
      manifestPath,
      manifest,
      rowCount: manifestRows >>> 0,
    };
  }

  return {
    manifestPath,
    manifest,
    rowCount: await countCsvDataRows(basePath),
  };
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
        header = parseCsvLine(line);
        idIdx = header.indexOf(idColumnName);
        if (idIdx < 0)
          throw new Error(`Missing ${idColumnName} in ${filePath}`);
        continue;
      }
      if (!line) continue;
      const vals = parseCsvLine(line);
      const id = Number(vals[idIdx]);
      if (!Number.isFinite(id)) continue;
      const shardId = (id >>> 0) % (shardCount >>> 0);
      const shardPath = path.join(
        shardDir,
        `shard_${String(shardId).padStart(4, "0")}.csv`,
      );
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

async function loadAddonShardMap(filePath, header) {
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
      const id = Number(vals[0]);
      if (!Number.isFinite(id)) continue;
      map.set(id >>> 0, vals.slice(1));
    }
  } finally {
    rl.close();
    rs.destroy();
  }
  return map;
}

async function mergeSampleOutputs({ outRoot, opts }) {
  console.log(`[merge] starting ${nowIso()}`);

  const basePath = path.join(outRoot, "occurrences_selected.csv");
  const coordsPath = path.join(outRoot, "coords.csv");
  const mergedPath = path.join(outRoot, "occurrences_enriched.csv");
  const outputs = datasetOutputPaths(outRoot);

  const {
    manifestPath,
    manifest,
    rowCount: baseRowCount,
  } = await getBaseRowCountAndManifest(outRoot, basePath);

  if (!opts.forceRerun && (await fileExistsNonEmpty(mergedPath))) {
    console.log(`[merge] reusing existing merged file ${mergedPath}`);

    const addonEntriesReuse = [];
    for (const key of [
      "dem",
      "terraclimate",
      "twi",
      "soilgrids",
      "glim",
      "mcd12q1",
    ]) {
      if (await fileExistsNonEmpty(outputs[key])) {
        addonEntriesReuse.push([key, outputs[key]]);
      }
    }

    manifest.mergedPath = mergedPath;
    manifest.sampleOutputs = Object.fromEntries(addonEntriesReuse);
    await writeJson(manifestPath, manifest);

    if (opts.cleanup) {
      await fs.promises.rm(coordsPath, { force: true }).catch(() => {});
      for (let i = 0; i < addonEntriesReuse.length; i++) {
        await fs.promises
          .rm(addonEntriesReuse[i][1], { force: true })
          .catch(() => {});
      }
    }

    return { mergedPath, mergedRows: manifest.mergedRows || 0 };
  }

  const addonEntries = [];
  for (const key of [
    "dem",
    "terraclimate",
    "twi",
    "soilgrids",
    "glim",
    "mcd12q1",
  ]) {
    try {
      const st = await fs.promises.stat(outputs[key]);
      if (st.size > 0) addonEntries.push([key, outputs[key]]);
    } catch {}
  }

  if (!(await fs.promises.stat(basePath).catch(() => null))) {
    throw new Error(`Missing base file: ${basePath}`);
  }

  const effectiveJoinShards = chooseEffectiveJoinShards(
    baseRowCount,
    opts.joinShards,
    opts.joinMinRows,
  );

  console.log(
    `[merge] base_rows=${baseRowCount.toLocaleString()} max_shards=${opts.joinShards.toLocaleString()} min_rows_per_shard=${opts.joinMinRows.toLocaleString()} effective_shards=${effectiveJoinShards.toLocaleString()}`,
  );

  const shardRoot = path.join(
    outRoot,
    `.join_tmp_${Date.now()}_${process.pid}`,
  );
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
  await writeStreamChunk(
    ws,
    Buffer.from(mergedHeader.map(csvEscapeValue).join(",") + "\n", "utf8"),
  );

  let mergedRows = 0;
  try {
    for (let shardId = 0; shardId < effectiveJoinShards; shardId++) {
      const shardName = `shard_${String(shardId).padStart(4, "0")}.csv`;
      const baseShardPath = path.join(baseShardDir, shardName);
      const baseExists = await fs.promises
        .stat(baseShardPath)
        .catch(() => null);
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
        addonMaps.set(
          key,
          await loadAddonShardMap(addonShardPath, addonHeaders.get(key) || []),
        );
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
          const id = Number(vals[0]);
          if (!Number.isFinite(id)) continue;
          const row = vals.slice();
          for (let i = 0; i < addonEntries.length; i++) {
            const [key] = addonEntries[i];
            const header = addonHeaders.get(key) || [];
            const wantLen = Math.max(0, header.length - 1);
            const addonVals =
              addonMaps.get(key).get(id >>> 0) || new Array(wantLen).fill("");
            if (addonVals.length < wantLen) {
              row.push(
                ...addonVals,
                ...new Array(wantLen - addonVals.length).fill(""),
              );
            } else {
              row.push(...addonVals.slice(0, wantLen));
            }
          }
          outParts.push(
            row.map((v) => csvEscapeValue(v || "")).join(",") + "\n",
          );
          mergedRows++;
          if (outParts.length >= 4096) {
            await writeStreamChunk(ws, Buffer.from(outParts.join(""), "utf8"));
            outParts.length = 0;
          }
        }
        if (outParts.length)
          await writeStreamChunk(ws, Buffer.from(outParts.join(""), "utf8"));
      } finally {
        rl.close();
        rs.destroy();
      }

      console.log(
        `[merge] shard ${shardId + 1}/${effectiveJoinShards} rows=${mergedRows.toLocaleString()}`,
      );
      maybeGC("collect-taxa-env-merge-shard");
    }
  } finally {
    await new Promise((resolve, reject) => {
      ws.on("finish", resolve);
      ws.on("error", reject);
      ws.end();
    });
  }

  manifest.mergedPath = mergedPath;
  manifest.mergedRows = mergedRows;
  manifest.sampleOutputs = Object.fromEntries(addonEntries);
  manifest.joinShardsRequested = opts.joinShards;
  manifest.joinMinRows = opts.joinMinRows;
  manifest.joinShardsEffective = effectiveJoinShards;
  await writeJson(manifestPath, manifest);

  if (!opts.keepTemps) {
    await fs.promises
      .rm(shardRoot, { recursive: true, force: true })
      .catch(() => {});
  }
  if (opts.cleanup) {
    await fs.promises.rm(coordsPath, { force: true }).catch(() => {});
    for (let i = 0; i < addonEntries.length; i++) {
      await fs.promises.rm(addonEntries[i][1], { force: true }).catch(() => {});
    }
  }

  console.log(
    `[merge] done ${nowIso()} rows=${mergedRows.toLocaleString()} out=${mergedPath}`,
  );
  return { mergedPath, mergedRows };
}

async function runAppendPipeline({ opts, inputCsv, indexPath, outRoot }) {
  const {
    occurrencesPath,
    coordsPath,
    speciesManifestPath,
    manifestPath,
  } = extractOutputPaths(outRoot);

  if (!(await fileExistsNonEmpty(occurrencesPath))) {
    return false;
  }
  if (!opts.includeTaxaSelectors || !opts.includeTaxaSelectors.length) {
    throw new Error(`--append requires --include-taxa with one or more selectors`);
  }

  let idx = null;
  let taxa = null;
  let stageRoot = "";
  let appendCompletedSuccessfully = false;

  try {
    const baseManifest = await rebuildExtractArtifactsFromSelected({
      outRoot,
      inputCsv,
      includeSelectors: opts.includeTaxaSelectors,
      excludeSelectors: opts.excludeTaxaSelectors,
    });
    const baseManifestObj =
      baseManifest || (await readJson(manifestPath).catch(() => ({})));
    const existingSpeciesInfo = await readSpeciesManifestEntries(
      speciesManifestPath,
    );
    const existingSpeciesNodeIds = existingSpeciesInfo.nodeIds;
    const baseRowCount = await countCsvDataRows(occurrencesPath);

    const requestedSampleConfig = getSampleConfig(opts);
    const existingSampleConfig =
      baseManifestObj && baseManifestObj.sampleConfig
        ? baseManifestObj.sampleConfig
        : null;

    if (
      existingSampleConfig &&
      !sameTerraclimateConfig(
        existingSampleConfig.terraclimate,
        requestedSampleConfig.terraclimate,
      )
    ) {
      throw new Error(
        [
          `Append TerraClimate config mismatch for ${outRoot}`,
          `existing: ${formatTerraclimateConfig(existingSampleConfig.terraclimate)}`,
          `requested: ${formatTerraclimateConfig(requestedSampleConfig.terraclimate)}`,
          `Use the same --terraclimate-vars and --terraclimate-year that built the existing out-root, or rebuild the root output with --force-rerun.`,
        ].join("\n"),
      );
    }

    console.log(
      `[append] existing rows=${baseRowCount.toLocaleString()} species=${existingSpeciesNodeIds.size.toLocaleString()} out=${outRoot}`,
    );

    idx = await loadOrBuildIndex(inputCsv, indexPath);
    idx._sharedCsvFd = await fs.promises.open(idx.file, "r");

    taxa = await loadTaxaIndex().catch((e) => {
      console.error(e);
      return null;
    });
    if (!taxa) {
      throw new Error(
        `Taxa index not loaded from ${TAXA_DIR}. Run your taxonomy build first.`,
      );
    }
    if (!taxa._sharedPostingsFd) {
      taxa._sharedPostingsFd = await fs.promises.open(taxa.postingsPath, "r");
    }

    const taxonFilter = resolveTaxonFilter(taxa, opts);
    console.log(`[append] resolved_taxon_filter=${taxonFilterSummary(taxonFilter)}`);

    const requestedSpeciesPlans = await enumerateSpecies(taxa, opts, taxonFilter);
    const deltaSpeciesPlans = requestedSpeciesPlans.filter(
      (plan) => !existingSpeciesNodeIds.has(plan.speciesNodeId >>> 0),
    );

    if (!deltaSpeciesPlans.length) {
      const manifest = Object.assign({}, baseManifestObj || {});
      manifest.lastRequestedTaxonFilter = {
        include: selectorRawList(opts.includeTaxaSelectors),
        exclude: selectorRawList(opts.excludeTaxaSelectors),
      };
      manifest.taxonFilter = {
        include: mergeSelectorRawLists(
          manifest.taxonFilter && manifest.taxonFilter.include,
          opts.includeTaxaSelectors,
        ),
        exclude: mergeSelectorRawLists(
          manifest.taxonFilter && manifest.taxonFilter.exclude,
          opts.excludeTaxaSelectors,
        ),
      };
      manifest.sampleConfig = existingSampleConfig || requestedSampleConfig;
      manifest.appendHistory = Array.isArray(manifest.appendHistory)
        ? manifest.appendHistory
        : [];
      manifest.appendHistory.push({
        appendedAt: nowIso(),
        requestedInclude: selectorRawList(opts.includeTaxaSelectors),
        requestedExclude: selectorRawList(opts.excludeTaxaSelectors),
        appendedSpeciesCount: 0,
        appendedRowCount: 0,
        skippedExistingSpeciesCount: requestedSpeciesPlans.length,
      });
      await writeJson(manifestPath, manifest);
      console.log(`[append] no new species to append`);
      appendCompletedSuccessfully = true;
      return true;
    }

    console.log(
      `[append] new species=${deltaSpeciesPlans.length.toLocaleString()} requested_species=${requestedSpeciesPlans.length.toLocaleString()} start_id=${(baseRowCount + 1).toLocaleString()}`,
    );

    const resumed = !opts.forceRerun
      ? await findResumableAppendStage({
          outRoot,
          inputCsv,
          baseRowCount,
          deltaSpeciesPlans,
          includeSelectors: opts.includeTaxaSelectors,
          excludeSelectors: opts.excludeTaxaSelectors,
          requestedSampleConfig,
        })
      : null;

    if (resumed) {
      stageRoot = resumed.stageRoot;
      console.log(`[append] reusing existing stage ${stageRoot}`);
    } else {
      stageRoot = path.join(
        outRoot,
        `.append_stage_${Date.now()}_${process.pid}`,
      );
      ensureDir(stageRoot);
      console.log(`[append] created stage ${stageRoot}`);
    }

    const stageManifestPath = path.join(stageRoot, "manifest.json");
    const stageOccurrencesPath = path.join(stageRoot, "occurrences_selected.csv");
    const stageCoordsPath = path.join(stageRoot, "coords.csv");
    const stageSpeciesManifestPath = path.join(stageRoot, "species_manifest.csv");
    const stageMergedPath = path.join(stageRoot, "occurrences_enriched.csv");

    let deltaManifest = null;
    const reusedExtract = !opts.forceRerun
      && (await allFilesExistNonEmpty([
        stageManifestPath,
        stageOccurrencesPath,
        stageCoordsPath,
        stageSpeciesManifestPath,
      ]));

    if (reusedExtract) {
      deltaManifest = await readJson(stageManifestPath).catch(() => null);
      console.log(`[append] reusing staged extract ${stageRoot}`);
    } else {
      ensureDir(stageRoot);
      deltaManifest = await writeExtractOutputs({
        idx,
        taxa,
        opts,
        speciesPlans: deltaSpeciesPlans,
        outRoot: stageRoot,
        inputCsv,
        includeSelectors: opts.includeTaxaSelectors,
        excludeSelectors: opts.excludeTaxaSelectors,
        startId: baseRowCount + 1,
      });
    }

    const stageOpts = Object.assign({}, opts, {
      append: false,
      cleanup: false,
      keepTemps: true,
      forceRerun: false,
    });

    await runSamplers({ opts: stageOpts, outRoot: stageRoot });
    const mergeInfo = await mergeSampleOutputs({ outRoot: stageRoot, opts: stageOpts });

    if (opts.cleanupEnrichedMode !== "none") {
      await runCleanupEnriched({
        opts: Object.assign({}, stageOpts, { forceRerun: true }),
        outRoot: stageRoot,
        mergedPath: mergeInfo && mergeInfo.mergedPath,
      });
    }

    const rootManifest = Object.assign(
      {},
      await readJson(manifestPath).catch(() => baseManifestObj || {}),
    );
    const targetMergedPath = path.resolve(
      String(
        (rootManifest && (rootManifest.finalPath || rootManifest.mergedPath)) ||
          path.join(outRoot, "occurrences_enriched.csv"),
      ),
    );
    if (!(await fileExistsNonEmpty(targetMergedPath))) {
      throw new Error(
        `Append target missing existing enriched file: ${targetMergedPath}`,
      );
    }

    await assertAppendableCsvHeaders({
      srcPath: stageMergedPath,
      dstPath: targetMergedPath,
      label: path.basename(targetMergedPath),
    });

    const appendedSelectedRows = await appendCsvDataRows({
      srcPath: stageOccurrencesPath,
      dstPath: occurrencesPath,
      label: "occurrences_selected.csv",
    });
    const appendedCoordsRows = await appendCsvDataRows({
      srcPath: stageCoordsPath,
      dstPath: coordsPath,
      label: "coords.csv",
    });
    const appendedSpeciesRows = await appendCsvDataRows({
      srcPath: stageSpeciesManifestPath,
      dstPath: speciesManifestPath,
      label: "species_manifest.csv",
    });
    const appendedMergedRows = await appendCsvDataRows({
      srcPath: stageMergedPath,
      dstPath: targetMergedPath,
      label: path.basename(targetMergedPath),
    });

    const mergedRowCount = await countCsvDataRows(targetMergedPath);
    const selectedRowCount = await countCsvDataRows(occurrencesPath);
    const speciesInfoAfter = await readSpeciesManifestEntries(speciesManifestPath);

    rootManifest.csv = path.resolve(inputCsv);
    rootManifest.outRoot = outRoot;
    rootManifest.occurrencesPath = occurrencesPath;
    rootManifest.coordsPath = coordsPath;
    rootManifest.speciesManifestPath = speciesManifestPath;
    rootManifest.mergedPath = targetMergedPath;
    rootManifest.finalPath = targetMergedPath;
    rootManifest.rowCount = selectedRowCount;
    rootManifest.mergedRows = mergedRowCount;
    rootManifest.speciesCount = speciesInfoAfter.rows.length;
    rootManifest.projectedColumns =
      rootManifest.projectedColumns || (deltaManifest && deltaManifest.projectedColumns) || [];
    rootManifest.sampleConfig = requestedSampleConfig;
    rootManifest.taxonFilter = {
      include: mergeSelectorRawLists(
        rootManifest.taxonFilter && rootManifest.taxonFilter.include,
        opts.includeTaxaSelectors,
      ),
      exclude: mergeSelectorRawLists(
        rootManifest.taxonFilter && rootManifest.taxonFilter.exclude,
        opts.excludeTaxaSelectors,
      ),
    };
    rootManifest.lastRequestedTaxonFilter = {
      include: selectorRawList(opts.includeTaxaSelectors),
      exclude: selectorRawList(opts.excludeTaxaSelectors),
    };
    rootManifest.appendHistory = Array.isArray(rootManifest.appendHistory)
      ? rootManifest.appendHistory
      : [];
    rootManifest.appendHistory.push({
      appendedAt: nowIso(),
      requestedInclude: selectorRawList(opts.includeTaxaSelectors),
      requestedExclude: selectorRawList(opts.excludeTaxaSelectors),
      appendedSpeciesCount: deltaSpeciesPlans.length,
      appendedRowCount: (deltaManifest && deltaManifest.rowCount ? deltaManifest.rowCount : 0) >>> 0,
      appendedSelectedRows: appendedSelectedRows >>> 0,
      appendedCoordsRows: appendedCoordsRows >>> 0,
      appendedSpeciesRows: appendedSpeciesRows >>> 0,
      appendedMergedRows: appendedMergedRows >>> 0,
      stageRoot,
      resumedStage: !!resumed,
    });

    if (opts.cleanupEnrichedMode !== "none") {
      rootManifest.cleanupEnriched = {
        enabled: true,
        mode: String(opts.cleanupEnrichedMode || "none").toLowerCase(),
        scriptPath: path.resolve(
          String(
            opts.cleanupEnrichedScript || DEFAULT_PATHS.cleanupEnrichedScript,
          ),
        ),
        backup: !!opts.cleanupEnrichedBackup,
        progressEvery: Math.max(
          1,
          Number(opts.cleanupEnrichedProgressEvery) || 250000,
        ),
        completedAt: nowIso(),
        appendMode: true,
      };
    }

    await writeJson(manifestPath, rootManifest);

    console.log(
      `[append] appended rows=${((deltaManifest && deltaManifest.rowCount) || 0 >>> 0).toLocaleString()} species=${deltaSpeciesPlans.length.toLocaleString()} merged=${targetMergedPath}`,
    );
    appendCompletedSuccessfully = true;
    return true;
  } finally {
    if (idx && idx._sharedCsvFd) await idx._sharedCsvFd.close().catch(() => {});
    if (taxa && taxa._sharedPostingsFd)
      await taxa._sharedPostingsFd.close().catch(() => {});
    if (!opts.keepTemps && appendCompletedSuccessfully && stageRoot) {
      await fs.promises.rm(stageRoot, { recursive: true, force: true }).catch(() => {});
    } else if (!appendCompletedSuccessfully && stageRoot) {
      console.log(`[append] preserving stage for resume: ${stageRoot}`);
    }
  }
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const { inputCsv, indexPath } = await resolveInputCsvAndIndexPath(
    opts.inputCsv,
  );
  const st = await statFile(inputCsv).catch((e) => {
    console.error(e);
    return null;
  });
  if (!st || !st.isFile()) throw new Error(`File not found: ${inputCsv}`);

  console.log(`[collect] csv=${inputCsv}`);
  console.log(`[collect] index=${indexPath}`);
  console.log(`[collect] size=${fmtGiB(st.size)} GiB`);
  console.log(`[collect] phase=${opts.phase} outRoot=${opts.outRoot}`);
  console.log(
    `[collect] taxon_filter=${summarizeTaxonSelectors(opts.includeTaxaSelectors)} exclude=${summarizeTaxonSelectors(opts.excludeTaxaSelectors)}`,
  );

  const outRoot = path.resolve(opts.outRoot);
  ensureDir(outRoot);

  let idx = null;
  let taxa = null;

  if (opts.append) {
    const handledAppend = await runAppendPipeline({
      opts,
      inputCsv,
      indexPath,
      outRoot,
    });
    if (handledAppend) {
      console.log(`[collect] done ${nowIso()}`);
      return;
    }
    console.log(
      `[append] no existing occurrences_selected.csv in ${outRoot}, falling back to normal full pipeline`,
    );
  }

  if (opts.phase === "extract" || opts.phase === "all") {
    const reusedExtract = !opts.forceRerun
      ? await rebuildExtractArtifactsFromSelected({
          outRoot,
          inputCsv,
          includeSelectors: opts.includeTaxaSelectors,
          excludeSelectors: opts.excludeTaxaSelectors,
        })
      : null;

    if (!reusedExtract) {
      idx = await loadOrBuildIndex(inputCsv, indexPath);
      idx._sharedCsvFd = await fs.promises.open(idx.file, "r");

      taxa = await loadTaxaIndex().catch((e) => {
        console.error(e);
        return null;
      });
      if (!taxa)
        throw new Error(
          `Taxa index not loaded from ${TAXA_DIR}. Run your taxonomy build first.`,
        );
      if (!taxa._sharedPostingsFd)
        taxa._sharedPostingsFd = await fs.promises.open(taxa.postingsPath, "r");

      const taxonFilter = resolveTaxonFilter(taxa, opts);
      console.log(
        `[collect] resolved_taxon_filter=${taxonFilterSummary(taxonFilter)}`,
      );
      await extractOccurrencesAndCoords({
        idx,
        taxa,
        opts,
        taxonFilter,
        outRoot,
        inputCsv,
      });
    } else {
      console.log(
        `[collect] extract already available in ${outRoot}, skipped input index/taxa load`,
      );
    }
  }

  if (opts.phase === "sample" || opts.phase === "all") {
    await runSamplers({ opts, outRoot });
  }

  let mergeInfo = null;
  if (opts.phase === "merge" || opts.phase === "all") {
    mergeInfo = await mergeSampleOutputs({ outRoot, opts });
    if (opts.cleanupEnrichedMode !== "none") {
      await runCleanupEnriched({
        opts,
        outRoot,
        mergedPath: mergeInfo && mergeInfo.mergedPath,
      });
    } else if (mergeInfo && mergeInfo.mergedPath) {
      await updateManifestFinalPath(outRoot, {
        finalPath: mergeInfo.mergedPath,
        cleanupEnriched: {
          enabled: false,
          mode: "none",
        },
      }).catch(() => {});
    }
  }

  if (idx && idx._sharedCsvFd) await idx._sharedCsvFd.close().catch(() => {});
  if (taxa && taxa._sharedPostingsFd)
    await taxa._sharedPostingsFd.close().catch(() => {});
  console.log(`[collect] done ${nowIso()}`);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});