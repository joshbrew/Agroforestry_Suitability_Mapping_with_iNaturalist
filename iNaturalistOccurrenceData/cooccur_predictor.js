#!/usr/bin/env node
"use strict";

/*
Indexed co-occurrence predictor pipeline for very large opportunistic occurrence tables.

Example:

First, indexing for speed:
node cooccur_predictor.js occurrences.csv --phase index --scales 250,1000,5000 --include-taxa kingdom:Plantae,kingdom:Fungi --sort-workers 2 --out-root occurrences.csv.cooccur_pf --progress-every-ms 10000 

Then run the predictor classification
node cooccur_predictor.js occurrences.csv --phase pairs --scales 250,1000,5000 --max-cell-species 120 --include-taxa kingdom:Plantae,kingdom:Fungi --sort-workers 2 --out-root occurrences.csv.cooccur_pf --progress-every-ms 10000 

What this script estimates:
- a directional, multi-scale shared-habitat / shared-observation prior
- a reusable species-by-anchor-cell incidence index per scale / phase / adjacency mode
- top associates per focal species, per grid variant
- a collapsed predictor set across variants for later environmental modelling

What this script does not estimate by itself:
- direct interaction, causation, facilitation, competition, or dependency
- an unbiased occupancy model
- a substitute for climate / soil / land-use modelling

Why this structure is a reasonable first-pass predictor:
1) Presence-only records are spatially biased toward where people go. The outputs here should therefore be
   treated as candidate shared-habitat priors, not proof of ecological interaction.
   - Fithian et al. 2015, Methods in Ecology and Evolution, doi:10.1111/2041-210X.12242
   - Cretois et al. 2021, Ecology and Evolution, doi:10.1002/ece3.8187
   - Blanchet et al. 2020, Ecology Letters, doi:10.1111/ele.13525

2) Repeated observations of the same species in the same grid cell are collapsed to one presence.
   That keeps hotspots, parks, trailheads, and heavily photographed sites from overwhelming the result.
   - MacKenzie et al. 2002, Ecology, doi:10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2

3) Co-occurrence structure is scale dependent, so the script keeps multiple grid sizes instead of forcing
   one cell size to represent both narrow local habitat and broader regional similarity.
   - Hart et al. 2017, Nature Ecology & Evolution, doi:10.1038/s41559-017-0277-5
   - Kraan et al. 2020, BMC Ecology, doi:10.1186/s12898-020-00308-4

4) Jaccard is reported because co-absent cells are usually not informative in sparse presence settings.
   - Mainali et al. 2017, Annals of Translational Medicine, doi:10.21037/atm.2017.10.51

5) Reusing a compact site-incidence representation is usually far more efficient than repeatedly re-reading
   a full flat occurrence table, especially when downstream tasks depend only on taxon id and site occupancy.
   This script therefore treats sorted presence shards as a reusable incidence index that later pair/rank runs
   can consume without touching the raw CSV again.

6) Phase-shifted grids and neighborhood-expanded anchor cells are practical ways to reduce hard boundary
   artifacts from strict raster binning. In this script, --phase-offsets reruns the same scale on shifted
   grid origins, and --adjacency moore1 expands each occupied anchor cell to its 8 touching neighbors plus
   itself. That should be interpreted as an anchor-neighborhood co-occurrence prior rather than a strict
   same-cell count.
5) Ranking blends specificity beyond independence, directional usefulness for the focal species, and a
   shrinkage term that downweights tiny support counts.
   - Bouma 2009, Normalized (Pointwise) Mutual Information in Collocation Extraction

Taxon selector syntax:
- bare name, matched anywhere in the taxonomy tree: Plantae, Fungi, Rosales, Salix
- explicit rank:name selectors: kingdom:Plantae, order:Rosales, genus:Salix, species:Salix sitchensis
- selectors are unioned within --include-taxa and within --exclude-taxa
- species are kept when they descend from any included selector and from no excluded selector
- if no include selectors are given, all species are eligible except those removed by --exclude-taxa

Typical uses:
- build only the reusable incidence index:
  --phase index
- reuse the incidence index later for pair/rank passes:
  --phase pairs
  --phase rank
- plants and fungi only:
  --include-taxa kingdom:Plantae,kingdom:Fungi
- one genus plus one order:
  --include-taxa genus:Salix,order:Rosales
- a hand-curated species set:
  --include-taxa species:Alnus rubra,species:Salix sitchensis,species:Carex obnupta

Interpretation rule of thumb:
- use these associations as candidate ecological context and environmental proxy signals
- for strict same-cell support, use --adjacency exact and base phase offsets
- for boundary-robust local context, add --phase-offsets quad and/or --adjacency moore1
- then join them to climate, soil, geology, hydrology, and land-use covariates for downstream modelling
*/

if (!process.env.CSV_READ_BYTES) process.env.CSV_READ_BYTES = "8388608";
if (!process.env.POSTINGS_READ_BYTES) process.env.POSTINGS_READ_BYTES = "8388608";
if (!process.env.COOCCUR_SPOOL_SHARDS) process.env.COOCCUR_SPOOL_SHARDS = "512";
if (!process.env.COOCCUR_SPOOL_MIN_ROWS_PER_SHARD) process.env.COOCCUR_SPOOL_MIN_ROWS_PER_SHARD = "1000000";
if (!process.env.COOCCUR_SPOOL_BATCH_TRIPLES) process.env.COOCCUR_SPOOL_BATCH_TRIPLES = "524288";
if (!process.env.COOCCUR_EXACT_SPOOL_STRIDE) process.env.COOCCUR_EXACT_SPOOL_STRIDE = "262144";
if (!process.env.COOCCUR_EXACT_GROUP_GAP_BYTES) process.env.COOCCUR_EXACT_GROUP_GAP_BYTES = "1048576";
if (!process.env.COOCCUR_EXACT_GROUP_MAX_BYTES) process.env.COOCCUR_EXACT_GROUP_MAX_BYTES = "16777216";
if (!process.env.COOCCUR_EXACT_GROUP_MAX_ROWS) process.env.COOCCUR_EXACT_GROUP_MAX_ROWS = "16384";
if (!process.env.COOCCUR_PRESENCE_BUFFER_RECORDS) process.env.COOCCUR_PRESENCE_BUFFER_RECORDS = "4096";
if (!process.env.COOCCUR_PRESENCE_GRID_READ_BYTES) process.env.COOCCUR_PRESENCE_GRID_READ_BYTES = "16777216";
if (!process.env.COOCCUR_SORT_WORKERS && process.env.EXPORT_SORT_WORKERS) {
  process.env.COOCCUR_SORT_WORKERS = process.env.EXPORT_SORT_WORKERS;
}

const os = require("os");
const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");

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
  radixSortPairsInPlace,
} = require("./server/csvserver.taxa.js");

const EARTH_RADIUS_M = 6378137;
const MAX_MERCATOR_LAT = 85.05112878;
const RAD = Math.PI / 180;
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

function normalizeTaxonText(s) {
  return String(s || "").trim().replace(/\s+/g, " ").toLowerCase();
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

function nodeSearchNames(taxa, nodeId) {
  const node = taxa.readNode(nodeId >>> 0);
  if (!node) return [];

  const names = [];
  const baseName = String(taxa.getString(node.nameId) || "").trim();
  if (baseName) names.push(baseName);

  if ((node.rankId >>> 0) === TAXON_RANK_NAME_TO_ID.species) {
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

    for (let nodeId = 1; nodeId < (taxa.nodeCount >>> 0); nodeId++) {
      const node = taxa.readNode(nodeId);
      if (!node) continue;
      if (selector.rankId != null && (node.rankId >>> 0) !== (selector.rankId >>> 0)) {
        continue;
      }

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

  return {
    nodeIds: resolvedNodeIds,
    resolved,
    unresolved,
  };
}

function summarizeTaxonSelectors(selectors) {
  if (!selectors || !selectors.length) return "none";
  return selectors.map((s) => s.raw).join("|");
}

function resolveTaxonFilter(taxa, opts) {
  const includeSelectors = Array.isArray(opts.includeTaxaSelectors) ? opts.includeTaxaSelectors : [];
  const excludeSelectors = Array.isArray(opts.excludeTaxaSelectors) ? opts.excludeTaxaSelectors : [];

  const includeResolved = resolveTaxonSelectors(taxa, includeSelectors);
  const excludeResolved = resolveTaxonSelectors(taxa, excludeSelectors);

  if (includeSelectors.length && !includeResolved.nodeIds.size) {
    throw new Error(`No taxonomy nodes matched --include-taxa ${summarizeTaxonSelectors(includeSelectors)}`);
  }

  if (includeResolved.unresolved.length) {
    console.warn(`[taxon-filter] unresolved include selectors: ${includeResolved.unresolved.join(", ")}`);
  }
  if (excludeResolved.unresolved.length) {
    console.warn(`[taxon-filter] unresolved exclude selectors: ${excludeResolved.unresolved.join(", ")}`);
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
  if (memo && memo[sid] !== 0) {
    return memo[sid] === 1;
  }

  const includeActive = filter.includeNodeIds && filter.includeNodeIds.size > 0;
  const excludeActive = filter.excludeNodeIds && filter.excludeNodeIds.size > 0;

  if (!includeActive && !excludeActive) {
    if (memo) memo[sid] = 1;
    return true;
  }

  let cur = sid;
  let includeHit = !includeActive;

  while (cur > 0 && cur < (taxa.nodeCount >>> 0)) {
    if (excludeActive && filter.excludeNodeIds.has(cur)) {
      if (memo) memo[sid] = -1;
      return false;
    }

    if (includeActive && filter.includeNodeIds.has(cur)) {
      includeHit = true;
    }

    const node = taxa.readNode(cur);
    if (!node) break;
    const parentId = node.parentId >>> 0;
    if (parentId === 0 || parentId === cur) break;
    cur = parentId;
  }

  if (memo) memo[sid] = includeHit ? 1 : -1;
  return includeHit;
}

function mixU32(x) {
  let v = x >>> 0;
  v ^= v >>> 16;
  v = Math.imul(v, 0x7feb352d) >>> 0;
  v ^= v >>> 15;
  v = Math.imul(v, 0x846ca68b) >>> 0;
  v ^= v >>> 16;
  return v >>> 0;
}

function rotl32(x, r) {
  const v = x >>> 0;
  const s = r & 31;
  return ((v << s) | (v >>> (32 - s))) >>> 0;
}

function buildRowRunMap(rowIds, speciesIds, count) {
  const rowRunMap = new Map();
  const uniqueRowIds = new Uint32Array(Math.max(1, count | 0));
  if (!count) return { rowRunMap, uniqueRowIds: uniqueRowIds.subarray(0, 0), uniqueCount: 0 };

  let uniqueCount = 0;
  let start = 0;
  let prevRowId = rowIds[0] >>> 0;

  for (let i = 1; i < count; i++) {
    const rowId = rowIds[i] >>> 0;
    if (rowId === prevRowId) continue;
    rowRunMap.set(prevRowId, { start, end: i });
    uniqueRowIds[uniqueCount++] = prevRowId;
    prevRowId = rowId;
    start = i;
  }

  rowRunMap.set(prevRowId, { start, end: count });
  uniqueRowIds[uniqueCount++] = prevRowId;

  return {
    rowRunMap,
    uniqueRowIds: uniqueRowIds.subarray(0, uniqueCount),
    uniqueCount: uniqueCount >>> 0,
  };
}

function radixSortTriplesByOrderInPlace(triplesU32, orderMostToLeast, work) {
  const n = (triplesU32.length / 3) | 0;
  if (n <= 1) return;

  const wantLen = triplesU32.length | 0;
  let tmp = work && work.tmp && work.tmp.length >= wantLen ? work.tmp : new Uint32Array(wantLen);
  let counts = work && work.counts && work.counts.length === 1 << 16 ? work.counts : new Uint32Array(1 << 16);

  function pass16(src, dst, which, shift) {
    counts.fill(0);
    for (let i = 0; i < n; i++) counts[(src[i * 3 + which] >>> shift) & 0xffff]++;
    let sum = 0;
    for (let i = 0; i < counts.length; i++) {
      const c = counts[i] >>> 0;
      counts[i] = sum >>> 0;
      sum = (sum + c) >>> 0;
    }
    for (let i = 0; i < n; i++) {
      const a = src[i * 3] >>> 0;
      const b = src[i * 3 + 1] >>> 0;
      const c = src[i * 3 + 2] >>> 0;
      const k = (src[i * 3 + which] >>> shift) & 0xffff;
      const pos = (counts[k]++ >>> 0) * 3;
      dst[pos] = a;
      dst[pos + 1] = b;
      dst[pos + 2] = c;
    }
  }

  let src = triplesU32;
  let dst = tmp;

  for (let oi = orderMostToLeast.length - 1; oi >= 0; oi--) {
    const which = orderMostToLeast[oi] | 0;
    pass16(src, dst, which, 0);
    {
      const t = src;
      src = dst;
      dst = t;
    }
    pass16(src, dst, which, 16);
    {
      const t = src;
      src = dst;
      dst = t;
    }
  }

  if (src !== triplesU32) triplesU32.set(src);
  if (work) {
    work.tmp = tmp;
    work.counts = counts;
  }
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
    if (col > 0 || totalLen > 0 || parts.length > 0 || rowBuf.length > 0) flushField();
  }

  return out;
}

function getExactGroupOptions() {
  return {
    maxGapBytes: Math.max(0, Math.min(Number(process.env.COOCCUR_EXACT_GROUP_GAP_BYTES || 1 << 20) | 0, 64 << 20)),
    maxGroupBytes: Math.max(64 << 10, Math.min(Number(process.env.COOCCUR_EXACT_GROUP_MAX_BYTES || 16 << 20) | 0, 128 << 20)),
    maxGroupRows: Math.max(1, Math.min(Number(process.env.COOCCUR_EXACT_GROUP_MAX_ROWS || 16384) | 0, 1_000_000)),
  };
}

function buildExactRowReadGroups(taxa, rowIdsSorted, opts) {
  const groups = [];
  if (!rowIdsSorted.length) return groups;
  if (!taxa || typeof taxa.rowByteRange !== "function") return groups;

  const maxGapBytes = Math.max(0, Math.min(Number(opts.maxGapBytes || 1 << 20) | 0, 64 << 20));
  const maxGroupBytes = Math.max(64 << 10, Math.min(Number(opts.maxGroupBytes || 16 << 20) | 0, 128 << 20));
  const maxGroupRows = Math.max(1, Math.min(Number(opts.maxGroupRows || 16384) | 0, 1_000_000));

  let cur = null;
  for (let i = 0; i < rowIdsSorted.length; i++) {
    const rowId = rowIdsSorted[i] >>> 0;
    const range = taxa.rowByteRange(rowId);
    if (!range) continue;
    const start = Number(range.start);
    const end = Number(range.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) continue;

    if (!cur) {
      cur = { startOffset: start, endOffset: end, startRowId: rowId, endRowId: rowId, rowIds: [rowId] };
      continue;
    }

    const gap = start - cur.endOffset;
    const nextSpan = end - cur.startOffset;
    if (gap <= maxGapBytes && nextSpan <= maxGroupBytes && cur.rowIds.length < maxGroupRows) {
      cur.endOffset = end;
      cur.endRowId = rowId;
      cur.rowIds.push(rowId);
      continue;
    }

    groups.push(cur);
    cur = { startOffset: start, endOffset: end, startRowId: rowId, endRowId: rowId, rowIds: [rowId] };
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

    if (this.taxa && this.taxa.hasExactRowStarts && typeof this.taxa.rowByteRange === "function") {
      const groups = buildExactRowReadGroups(this.taxa, rowIdsSorted, this.exactOpts);
      let count = 0;

      for (let gi = 0; gi < groups.length; gi++) {
        const group = groups[gi];
        const groupLen = Math.max(0, Number(group.endOffset) - Number(group.startOffset));
        if (!Number.isFinite(groupLen) || groupLen <= 0) continue;

        const groupBuf = Buffer.allocUnsafe(groupLen);
        let off = 0;
        while (off < groupLen) {
          const { bytesRead } = await this.csvFd.read(groupBuf, off, groupLen - off, Number(group.startOffset) + off);
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
          if (!Number.isFinite(relStart) || !Number.isFinite(relEnd) || relStart < 0 || relEnd < relStart || relEnd > usable.length) {
            continue;
          }

          const rowBuf = usable.subarray(relStart, relEnd);
          const values = parseExactCsvRowSelected(rowBuf, this.idx.delimiter, this.matcher);
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

class TripleShardAppender {
  constructor({ dir, shardCount, openWriters, bufferRecords, shardNamer, shardChooser }) {
    this.dir = dir;
    this.shardCount = Math.max(1, shardCount | 0);
    this.bufferRecords = Math.max(16, bufferRecords | 0);
    this.shardNamer = shardNamer;
    this.shardChooser = shardChooser;
    this.writerCache = new LruWriterCache(openWriters);
    this.buffers = new Map();
    this.recordCount = 0;
  }

  _filePathForShard(shardId) {
    return path.join(this.dir, this.shardNamer(shardId >>> 0));
  }

  _stateForShard(shardId) {
    const sid = shardId >>> 0;
    let st = this.buffers.get(sid);
    if (!st) {
      st = {
        path: this._filePathForShard(sid),
        data: new Uint32Array(this.bufferRecords * 3),
        count: 0,
      };
      this.buffers.set(sid, st);
    }
    return st;
  }

  async append(a, b, c) {
    const shardId = this.shardChooser(a >>> 0, b >>> 0, this.shardCount);
    const st = this._stateForShard(shardId);
    const off = st.count * 3;
    st.data[off] = a >>> 0;
    st.data[off + 1] = b >>> 0;
    st.data[off + 2] = c >>> 0;
    st.count++;
    this.recordCount++;

    if (st.count >= this.bufferRecords) await this.flushShard(shardId);
  }

  async flushShard(shardId) {
    const st = this.buffers.get(shardId >>> 0);
    if (!st || st.count <= 0) return;
    const buf = Buffer.from(st.data.buffer, st.data.byteOffset, st.count * 12);
    await this.writerCache.append(st.path, buf);
    st.count = 0;
  }

  async flushAll() {
    for (const shardId of this.buffers.keys()) await this.flushShard(shardId);
  }

  async close() {
    await this.flushAll();
    await this.writerCache.closeAll();
  }
}


class TextShardAppender {
  constructor({ dir, shardCount, openWriters, bufferBytes, shardNamer, shardChooser }) {
    this.dir = dir;
    this.shardCount = Math.max(1, shardCount | 0);
    this.bufferBytes = Math.max(4096, bufferBytes | 0);
    this.shardNamer = shardNamer;
    this.shardChooser = shardChooser;
    this.writerCache = new LruWriterCache(openWriters);
    this.buffers = new Map();
    this.lineCount = 0;
  }

  _filePathForShard(shardId) {
    return path.join(this.dir, this.shardNamer(shardId >>> 0));
  }

  _stateForShard(shardId) {
    const sid = shardId >>> 0;
    let st = this.buffers.get(sid);
    if (!st) {
      st = {
        path: this._filePathForShard(sid),
        parts: [],
        bytes: 0,
      };
      this.buffers.set(sid, st);
    }
    return st;
  }

  async appendLine(a, b, line) {
    const shardId = this.shardChooser(a >>> 0, b >>> 0, this.shardCount);
    const st = this._stateForShard(shardId);
    st.parts.push(line);
    st.bytes += Buffer.byteLength(line);
    this.lineCount++;
    if (st.bytes >= this.bufferBytes) await this.flushShard(shardId);
  }

  async flushShard(shardId) {
    const st = this.buffers.get(shardId >>> 0);
    if (!st || st.bytes <= 0 || !st.parts.length) return;
    const buf = Buffer.from(st.parts.join(""), "utf8");
    st.parts.length = 0;
    st.bytes = 0;
    await this.writerCache.append(st.path, buf);
  }

  async flushAll() {
    for (const shardId of this.buffers.keys()) await this.flushShard(shardId);
  }

  async close() {
    await this.flushAll();
    await this.writerCache.closeAll();
  }
}

function clampLatForMercator(lat) {
  if (lat > MAX_MERCATOR_LAT) return MAX_MERCATOR_LAT;
  if (lat < -MAX_MERCATOR_LAT) return -MAX_MERCATOR_LAT;
  return lat;
}

function lonLatToMercatorCellBias(lon, lat, scaleM) {
  const clampedLat = clampLatForMercator(lat);
  const mx = EARTH_RADIUS_M * lon * RAD;
  const my = EARTH_RADIUS_M * Math.log(Math.tan(Math.PI * 0.25 + (clampedLat * RAD) * 0.5));
  const cx = Math.floor(mx / scaleM) | 0;
  const cy = Math.floor(my / scaleM) | 0;
  return {
    xBias: (cx ^ 0x80000000) >>> 0,
    yBias: (cy ^ 0x80000000) >>> 0,
  };
}

function lonLatToMercatorCellBiasPhase(lon, lat, scaleM, phaseXFrac, phaseYFrac) {
  const clampedLat = clampLatForMercator(lat);
  const mx = EARTH_RADIUS_M * lon * RAD;
  const my = EARTH_RADIUS_M * Math.log(Math.tan(Math.PI * 0.25 + (clampedLat * RAD) * 0.5));

  const shiftX = safeNumber(phaseXFrac, 0) * scaleM;
  const shiftY = safeNumber(phaseYFrac, 0) * scaleM;

  const cx = Math.floor((mx - shiftX) / scaleM) | 0;
  const cy = Math.floor((my - shiftY) / scaleM) | 0;

  return {
    xBias: (cx ^ 0x80000000) >>> 0,
    yBias: (cy ^ 0x80000000) >>> 0,
  };
}

function choosePresenceShard(xBias, yBias, shardCount) {
  const count = Math.max(1, shardCount >>> 0);
  const h = ((mixU32(xBias) ^ rotl32(mixU32(yBias), 13)) >>> 0);
  return h % count;
}

function choosePairShard(a, b, shardCount) {
  const count = Math.max(1, shardCount >>> 0);
  const h = ((mixU32(a) ^ rotl32(mixU32(b), 11)) >>> 0);
  return h % count;
}

function chooseCandidateShard(focalSid, _other, shardCount) {
  const count = Math.max(1, shardCount >>> 0);
  return (mixU32(focalSid) >>> 0) % count;
}

function chooseCollapseShard(focalSid, _other, shardCount) {
  const count = Math.max(1, shardCount >>> 0);
  return (mixU32(focalSid) >>> 0) % count;
}

function nowIso() {
  return new Date().toISOString();
}

function safeNumber(x, fallback = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
}

function makeScaleDirName(scaleM) {
  return `scale_${scaleM}m`;
}

function formatPhasePart(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "0";
  const s = String(Math.round(n * 1000) / 1000);
  return s.replace(/\./g, "_").replace(/-+/g, "m");
}

function makePhaseLabel(xFrac, yFrac) {
  return `p${formatPhasePart(xFrac)}_${formatPhasePart(yFrac)}`;
}

function parseSinglePhaseOffsetToken(raw) {
  const text = String(raw || "").trim();
  if (!text) return null;

  const m = text.match(/^\s*([-+]?\d*\.?\d+)\s*[:/]\s*([-+]?\d*\.?\d+)\s*$/);
  if (!m) {
    throw new Error(`Bad phase offset token: ${text}. Use x:y such as 0:0 or 0.5:0.5`);
  }

  const xFrac = Number(m[1]);
  const yFrac = Number(m[2]);
  if (!Number.isFinite(xFrac) || !Number.isFinite(yFrac)) {
    throw new Error(`Bad numeric phase offset token: ${text}`);
  }

  return {
    xFrac,
    yFrac,
    label: makePhaseLabel(xFrac, yFrac),
  };
}

function parsePhaseOffsets(raw) {
  const text = String(raw || "").trim().toLowerCase();

  if (!text || text === "base" || text === "none" || text === "0:0") {
    return [{ xFrac: 0, yFrac: 0, label: makePhaseLabel(0, 0) }];
  }

  if (text === "quad") {
    return [
      { xFrac: 0, yFrac: 0, label: makePhaseLabel(0, 0) },
      { xFrac: 0.5, yFrac: 0, label: makePhaseLabel(0.5, 0) },
      { xFrac: 0, yFrac: 0.5, label: makePhaseLabel(0, 0.5) },
      { xFrac: 0.5, yFrac: 0.5, label: makePhaseLabel(0.5, 0.5) },
    ];
  }

  const parts = String(raw || "")
    .split(/[;|]/)
    .map((s) => s.trim())
    .filter(Boolean);

  if (!parts.length) {
    return [{ xFrac: 0, yFrac: 0, label: makePhaseLabel(0, 0) }];
  }

  const seen = new Set();
  const out = [];

  for (let i = 0; i < parts.length; i++) {
    const item = parseSinglePhaseOffsetToken(parts[i]);
    const key = `${item.xFrac}:${item.yFrac}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }

  return out;
}

function resolveAdjacencyOffsets(mode) {
  const text = String(mode || "exact").trim().toLowerCase();
  if (text === "exact") return [{ dx: 0, dy: 0 }];

  if (text === "moore1" || text === "neighbors" || text === "exact+neighbors") {
    const out = [];
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        out.push({ dx, dy });
      }
    }
    return out;
  }

  throw new Error(`Unknown adjacency mode: ${mode}`);
}

function variantKeyForScaleContext(scaleCtx) {
  return `${scaleCtx.scaleM}m_${scaleCtx.phaseLabel}_${scaleCtx.adjacency}`;
}

function makeSpeciesLabel(taxa, speciesNodeId) {
  const sid = speciesNodeId >>> 0;
  const n = taxa.readNode(sid);
  const species = n ? String(taxa.getString(n.nameId) || "").trim() : "";
  if (!n || (n.rankId >>> 0) !== 7) return species || `node:${sid}`;
  const parentId = n.parentId >>> 0;
  if (parentId === 0 || parentId === sid) return species || `species:${sid}`;
  const parent = taxa.readNode(parentId);
  const genus = parent ? String(taxa.getString(parent.nameId) || "").trim() : "";
  if (!genus) return species || `species:${sid}`;
  if (!species) return genus;
  if (species.toLowerCase().startsWith(genus.toLowerCase() + " ")) return species;
  return `${genus} ${species}`;
}

function insertTopK(list, item, topK) {
  if (!Array.isArray(list)) list = [];
  if (list.length < topK) {
    list.push(item);
    return list;
  }

  let worstIdx = 0;
  for (let i = 1; i < list.length; i++) {
    const a = list[i];
    const b = list[worstIdx];
    if (
      a.score < b.score ||
      (a.score === b.score && a.shared_cells < b.shared_cells) ||
      (a.score === b.score && a.shared_cells === b.shared_cells && a.lift < b.lift) ||
      (a.score === b.score && a.shared_cells === b.shared_cells && a.lift === b.lift && a.associate_id > b.associate_id)
    ) {
      worstIdx = i;
    }
  }

  const worst = list[worstIdx];
  const better =
    item.score > worst.score ||
    (item.score === worst.score && item.shared_cells > worst.shared_cells) ||
    (item.score === worst.score && item.shared_cells === worst.shared_cells && item.lift > worst.lift) ||
    (item.score === worst.score && item.shared_cells === worst.shared_cells && item.lift === worst.lift && item.associate_id < worst.associate_id);

  if (better) list[worstIdx] = item;
  return list;
}

function makeAssocMetrics(shared, aCells, bCells, totalCells) {
  const s = safeNumber(shared, 0);
  const a = safeNumber(aCells, 0);
  const b = safeNumber(bCells, 0);
  const t = safeNumber(totalCells, 0);

  const pBgivenA = a > 0 ? s / a : 0;
  const pAgivenB = b > 0 ? s / b : 0;
  const condPostMean = (s + 1) / (a + 2);
  const revCondPostMean = (s + 1) / (b + 2);
  const union = Math.max(0, a + b - s);
  const jaccard = union > 0 ? s / union : 0;
  const overlap = Math.min(a, b) > 0 ? s / Math.min(a, b) : 0;

  let lift = 0;
  let npmi = 0;
  if (t > 0 && s > 0 && a > 0 && b > 0) {
    const pxy = s / t;
    const pa = a / t;
    const pb = b / t;
    const denom = pa * pb;
    if (denom > 0) {
      lift = pxy / denom;
      const pmi = Math.log(pxy / denom);
      const norm = -Math.log(pxy);
      npmi = norm > 0 ? pmi / norm : 0;
      if (!Number.isFinite(lift)) lift = 0;
      if (!Number.isFinite(npmi)) npmi = 0;
    }
  }

  const npmiPos = Math.max(0, npmi);
  const supportShrink = s / (s + 5);
  const liftStrength = lift > 1 ? Math.log(lift) / (1 + Math.log(lift)) : 0;

  // Heuristic predictor score, not a causal interaction score.
  // It balances three things:
  // 1) specificity beyond independence (NPMI),
  // 2) directional usefulness for the focal species (posterior mean P(B|A)),
  // 3) evidence strength via support shrinkage.
  const score = supportShrink * ((0.55 * npmiPos) + (0.30 * condPostMean) + (0.15 * liftStrength));

  return {
    shared_cells: s,
    p_associate_given_species: pBgivenA,
    p_species_given_associate: pAgivenB,
    cond_post_mean: condPostMean,
    reverse_cond_post_mean: revCondPostMean,
    lift,
    jaccard,
    overlap,
    npmi,
    support_shrink: supportShrink,
    score,
  };
}

async function listNonEmptyFiles(dir, suffix) {
  const names = await fs.promises.readdir(dir).catch(() => []);
  const out = [];
  for (let i = 0; i < names.length; i++) {
    const full = path.join(dir, names[i]);
    if (suffix && !names[i].endsWith(suffix)) continue;
    try {
      const st = await fs.promises.stat(full);
      if (st.isFile() && st.size > 0) out.push(full);
    } catch {}
  }
  out.sort();
  return out;
}

function logMem(prefix) {
  const mu = process.memoryUsage();
  console.log(`${prefix} rss=${((mu.rss || 0) / 1024 ** 2).toFixed(1)}MB heap=${((mu.heapUsed || 0) / 1024 ** 2).toFixed(1)}MB`);
}

async function writeJson(filePath, obj) {
  await fs.promises.writeFile(filePath, JSON.stringify(obj, null, 2));
}

async function readJson(filePath) {
  return JSON.parse(await fs.promises.readFile(filePath, "utf8"));
}

function buildScaleContexts(outRoot, opts) {
  const scales = [];
  for (let i = 0; i < opts.scales.length; i++) {
    const scaleM = opts.scales[i] | 0;

    for (let pi = 0; pi < opts.phaseOffsets.length; pi++) {
      const phase = opts.phaseOffsets[pi];
      const phaseLabel = phase.label || makePhaseLabel(phase.xFrac, phase.yFrac);
      const baseDir = path.join(
        outRoot,
        `${makeScaleDirName(scaleM)}__${phaseLabel}__${opts.adjacency}`,
      );

      const presenceDir = path.join(baseDir, "presence");
      const presenceSortedDir = path.join(baseDir, "presence_sorted");
      const pairDir = path.join(baseDir, "pairs");
      const pairSortedDir = path.join(baseDir, "pairs_sorted");
      const candidateDir = path.join(baseDir, "candidates");

      ensureDir(baseDir);
      ensureDir(presenceDir);
      ensureDir(presenceSortedDir);
      ensureDir(pairDir);
      ensureDir(pairSortedDir);
      ensureDir(candidateDir);

      const scaleCtx = {
        scaleM,
        phaseXFrac: phase.xFrac,
        phaseYFrac: phase.yFrac,
        phaseLabel,
        adjacency: opts.adjacency,
        anchorOffsets: resolveAdjacencyOffsets(opts.adjacency),
        baseDir,
        presenceDir,
        presenceSortedDir,
        pairDir,
        pairSortedDir,
        candidateDir,
        cellCountsPath: path.join(baseDir, "species_cell_counts.u32.bin"),
        cellCountsCsvPath: path.join(baseDir, "species_cell_counts.csv"),
        statsPath: path.join(baseDir, "scale_stats.json"),
        outputCsvPath: path.join(baseDir, "top_associations.csv"),
        presenceShardNamer: (shardId) => `presence_${String(shardId).padStart(5, "0")}.bin`,
        pairShardNamer: (shardId) => `pair_${String(shardId).padStart(5, "0")}.bin`,
        candidateShardNamer: (shardId) => `cand_${String(shardId).padStart(5, "0")}.bin`,
      };

      scaleCtx.variantKey = variantKeyForScaleContext(scaleCtx);
      scales.push(scaleCtx);
    }
  }
  return scales;
}

function buildCollapseContext(outRoot, opts) {
  const dir = path.join(outRoot, "collapsed_sources");
  ensureDir(dir);
  return {
    dir,
    outputCsvPath: path.join(outRoot, "cooccur_predictor_set.csv"),
    shardNamer: (shardId) => `collapse_${String(shardId).padStart(5, "0")}.tsv`,
    shardCount: opts.collapseShards,
  };
}

function formatRatePerSec(count, elapsedMs) {
  const ms = Math.max(1, Number(elapsedMs) || 0);
  return Math.round((count * 1000) / ms).toLocaleString();
}


function parseArgs(argv) {
  const out = {
    phase: "all",
    outRoot: `${FILE}.cooccur`,
    scales: [250, 1000, 5000],
    phaseOffsetsRaw: "base",
    phaseOffsets: [{ xFrac: 0, yFrac: 0, label: makePhaseLabel(0, 0) }],
    adjacency: "exact",
    topK: 20,
    minShared: 2,
    minSpeciesCells: 2,
    speciesMinObs: 1,
    maxCellSpecies: 150,
    uncFactor: 1.0,
    rowBatch: 8192,
    progressEveryMs: 10000,
    presenceShards: 2048,
    pairShards: 8192,
    candidateShards: 1024,
    openWriters: 64,
    sortWorkers: Math.max(1, Math.min(Number(process.env.COOCCUR_SORT_WORKERS || process.env.EXPORT_SORT_WORKERS || 2) | 0, 8)),
    maxSpecies: 0,
    minLift: 1,
    minConditional: 0,
    minNpmi: 0,
    collapseShards: 1024,
    keepTemp: false,
    includeTaxaSelectors: [],
    excludeTaxaSelectors: [],
  };

  function needValue(i, flag) {
    if (i + 1 >= argv.length) throw new Error(`Missing value for ${flag}`);
    return argv[i + 1];
  }

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") {
      console.log([
        "Usage:",
        "  node cooccur_predictor_indexed.js occurrences.csv [options]",
        "",
        "Main modes:",
        "  --phase all|index|pairs|rank",
        "",
        "Index / boundary options:",
        "  --scales 250,1000,5000",
        "  --phase-offsets base",
        "  --phase-offsets quad",
        "  --phase-offsets 0:0;0.5:0;0:0.5;0.5:0.5",
        "  --adjacency exact|moore1",
        "",
        "Ranking options:",
        "  --topk 20",
        "  --min-shared 2",
        "  --min-species-cells 2",
        "  --min-lift 1",
        "  --min-conditional 0",
        "  --min-npmi 0",
        "",
        "Performance / IO:",
        "  --out-root PATH",
        "  --species-min-obs 1",
        "  --max-cell-species 150",
        "  --unc-factor 1.0",
        "  --row-batch 8192",
        "  --progress-every-ms 10000",
        "  --presence-shards 2048",
        "  --pair-shards 8192",
        "  --candidate-shards 1024",
        "  --open-writers 64",
        "  --sort-workers 2",
        "  --collapse-shards 1024",
        "  --max-species 0",
        "  --keep-temp",
        "",
        "Taxon filters:",
        "  --include-taxa kingdom:Plantae,kingdom:Fungi",
        "  --exclude-taxa kingdom:Animalia",
      ].join("\n"));
      process.exit(0);
    } else if (a === "--phase") {
      out.phase = String(needValue(i, a)).toLowerCase();
      i++;
    } else if (a === "--out-root") {
      out.outRoot = String(needValue(i, a));
      i++;
    } else if (a === "--scales") {
      out.scales = String(needValue(i, a)).split(",").map((v) => Number(v.trim())).filter((v) => Number.isFinite(v) && v > 0).map((v) => Math.round(v));
      i++;
    } else if (a === "--phase-offsets") {
      out.phaseOffsetsRaw = String(needValue(i, a));
      out.phaseOffsets = parsePhaseOffsets(out.phaseOffsetsRaw);
      i++;
    } else if (a === "--adjacency") {
      out.adjacency = String(needValue(i, a)).trim().toLowerCase();
      resolveAdjacencyOffsets(out.adjacency);
      i++;
    } else if (a === "--topk") {
      out.topK = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--min-shared") {
      out.minShared = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--min-species-cells") {
      out.minSpeciesCells = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--species-min-obs") {
      out.speciesMinObs = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--max-cell-species") {
      out.maxCellSpecies = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--unc-factor") {
      out.uncFactor = Math.max(0, Number(needValue(i, a)));
      i++;
    } else if (a === "--row-batch") {
      out.rowBatch = Math.max(256, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--progress-every-ms") {
      out.progressEveryMs = Math.max(1000, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--presence-shards") {
      out.presenceShards = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--pair-shards") {
      out.pairShards = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--candidate-shards") {
      out.candidateShards = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--open-writers") {
      out.openWriters = Math.max(4, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--sort-workers") {
      out.sortWorkers = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--max-species") {
      out.maxSpecies = Math.max(0, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--min-lift") {
      out.minLift = Math.max(0, Number(needValue(i, a)));
      i++;
    } else if (a === "--min-conditional") {
      out.minConditional = Math.max(0, Number(needValue(i, a)));
      i++;
    } else if (a === "--min-npmi") {
      out.minNpmi = Math.max(-1, Math.min(1, Number(needValue(i, a))));
      i++;
    } else if (a === "--collapse-shards") {
      out.collapseShards = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (a === "--include-taxa") {
      out.includeTaxaSelectors = parseTaxonSelectorList(needValue(i, a));
      i++;
    } else if (a === "--exclude-taxa") {
      out.excludeTaxaSelectors = parseTaxonSelectorList(needValue(i, a));
      i++;
    } else if (a === "--keep-temp") {
      out.keepTemp = true;
    } else {
      throw new Error(`Unknown option: ${a}`);
    }
  }

  if (!out.scales.length) throw new Error("No valid scales were provided");

  if (out.phase === "presence") out.phase = "index";
  if (out.phase !== "all" && out.phase !== "index" && out.phase !== "pairs" && out.phase !== "rank") {
    throw new Error(`Bad --phase value: ${out.phase}`);
  }

  out.scales.sort((a, b) => a - b);
  return out;
}


async function sortShardWorkerMain() {
  const shardPath = String(workerData.shardPath || "");
  const sortedPath = String(workerData.sortedPath || "");
  const order = Array.isArray(workerData.order) ? workerData.order.map((v) => v | 0) : [0, 1, 2];

  const raw = await fs.promises.readFile(shardPath);
  if (raw.length === 0) {
    await fs.promises.writeFile(sortedPath, raw);
    parentPort.postMessage({ ok: true, records: 0 });
    return;
  }
  if (raw.length % 12 !== 0) throw new Error(`Shard byte length must be a multiple of 12: ${path.basename(shardPath)} size=${raw.length}`);

  const view = new Uint32Array(raw.buffer, raw.byteOffset, raw.length / 4);
  radixSortTriplesByOrderInPlace(view, order, {});
  await fs.promises.writeFile(sortedPath, raw);
  parentPort.postMessage({ ok: true, records: raw.length / 12 });
}

async function sortShards({ files, sortedDir, order, workers }) {
  ensureDir(sortedDir);
  const active = new Map();
  const out = [];
  let next = 0;

  function startOne(shardPath) {
    return new Promise((resolve, reject) => {
      const sortedPath = path.join(sortedDir, path.basename(shardPath));
      const worker = new Worker(__filename, {
        workerData: { shardPath, sortedPath, order },
      });

      let settled = false;
      function doneOk(msg) {
        if (settled) return;
        settled = true;
        cleanup();
        resolve({ shardPath, sortedPath, msg });
      }
      function doneErr(err) {
        if (settled) return;
        settled = true;
        cleanup();
        reject(err);
      }
      function cleanup() {
        worker.off("message", onMsg);
        worker.off("error", onErr);
        worker.off("exit", onExit);
      }
      function onMsg(msg) {
        if (!msg || msg.ok !== true) {
          doneErr(new Error(msg && msg.error ? msg.error : `Sort failed for ${shardPath}`));
          return;
        }
        doneOk(msg);
      }
      function onErr(err) { doneErr(err); }
      function onExit(code) {
        if (!settled && code !== 0) doneErr(new Error(`Worker exited ${code} for ${shardPath}`));
      }
      worker.on("message", onMsg);
      worker.on("error", onErr);
      worker.on("exit", onExit);
    });
  }

  function launchMore() {
    while (next < files.length && active.size < workers) {
      const f = files[next++];
      const p = startOne(f);
      active.set(p, f);
    }
  }

  launchMore();
  while (active.size) {
    const wrapped = Array.from(active.keys()).map((p) => p.then((result) => ({ ok: true, p, result })).catch((error) => ({ ok: false, p, error })));
    const settled = await Promise.race(wrapped);
    active.delete(settled.p);
    if (!settled.ok) throw settled.error;
    out.push(settled.result.sortedPath);
    launchMore();
  }

  out.sort();
  return out;
}

async function enumerateSpecies(taxa, opts, taxonFilter) {
  const out = [];
  let filteredOut = 0;

  for (let nodeId = 1; nodeId < (taxa.nodeCount >>> 0); nodeId++) {
    const n = taxa.readNode(nodeId);
    if (!n || (n.rankId >>> 0) !== TAXON_RANK_NAME_TO_ID.species) continue;

    const count = n.count >>> 0;
    if (count < (opts.speciesMinObs >>> 0)) continue;

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

  if (opts.maxSpecies > 0 && out.length > opts.maxSpecies) out.length = opts.maxSpecies;

  console.log(
    `[taxon-filter] kept species=${out.length.toLocaleString()} filtered_out=${filteredOut.toLocaleString()} summary=${taxonFilterSummary(taxonFilter)}`,
  );

  return out;
}


function getCooccurSpoolStride(idx, taxa) {
  const fallback = Math.max(1, getBestLookupStride(idx, taxa) | 0) >>> 0;
  if (!taxa || !taxa.hasExactRowStarts) return fallback;

  const envStride = Number(process.env.COOCCUR_EXACT_SPOOL_STRIDE || 262144);
  const idxStride =
    idx && Number.isFinite(idx.indexStride) ? idx.indexStride >>> 0 : 0;
  const taxaStride =
    taxa && Number.isFinite(taxa.rowStride) ? taxa.rowStride >>> 0 : 0;

  let stride = Math.max(
    fallback,
    idxStride,
    taxaStride,
    Number.isFinite(envStride) ? envStride | 0 : 0,
  );

  if (!Number.isFinite(stride) || stride <= 0) stride = fallback;
  if (stride >= 0xffffffff) stride = 0xfffffffe;

  return stride >>> 0;
}

function chooseEffectivePresenceSpoolShards(totalRowsEstimate, configuredMax) {
  const rows = Math.max(0, Number(totalRowsEstimate) || 0);
  const maxCount = Math.max(1, Number(configuredMax) | 0);
  const minRowsPerShard = Math.max(
    50_000,
    Number(process.env.COOCCUR_SPOOL_MIN_ROWS_PER_SHARD || 1_000_000) | 0,
  );

  if (rows <= 0) return 1;
  return Math.max(1, Math.min(maxCount, Math.ceil(rows / minRowsPerShard)));
}

async function appendTriplesByShard(
  shardPaths,
  triplesU32,
  tripleCount,
  shardCount,
) {
  if (!tripleCount) return;

  const count = Math.max(1, shardCount >>> 0);
  const shardSizes = new Uint32Array(count);

  for (let i = 0; i < tripleCount; i++) {
    const off = i * 3;
    const a = triplesU32[off] >>> 0;
    const b = triplesU32[off + 1] >>> 0;
    const shardId = choosePresenceShard(a, b, count);
    if (!Number.isInteger(shardId) || shardId < 0 || shardId >= count) {
      throw new Error(
        `Bad row spool shard id in size pass: shardId=${shardId} a=${a} b=${b} shardCount=${count}`,
      );
    }
    shardSizes[shardId] = (shardSizes[shardId] + 1) >>> 0;
  }

  const shardBuffers = new Array(count);
  const shardOffsets = new Uint32Array(count);
  for (let shardId = 0; shardId < count; shardId++) {
    const n = shardSizes[shardId] >>> 0;
    if (!n) continue;
    const shardPath = shardPaths[shardId];
    if (!shardPath) {
      throw new Error(
        `Missing row spool shard path for shardId=${shardId} shardCount=${count} paths=${shardPaths.length}`,
      );
    }
    shardBuffers[shardId] = new Uint32Array(n * 3);
  }

  for (let i = 0; i < tripleCount; i++) {
    const off = i * 3;
    const a = triplesU32[off] >>> 0;
    const b = triplesU32[off + 1] >>> 0;
    const c = triplesU32[off + 2] >>> 0;
    const shardId = choosePresenceShard(a, b, count);
    if (!Number.isInteger(shardId) || shardId < 0 || shardId >= count) {
      throw new Error(
        `Bad row spool shard id in write pass: shardId=${shardId} a=${a} b=${b} c=${c} shardCount=${count}`,
      );
    }
    const buf = shardBuffers[shardId];
    if (!buf) {
      throw new Error(
        `Missing row spool shard buffer for shardId=${shardId} a=${a} b=${b} c=${c} shardCount=${count}`,
      );
    }
    const pos = shardOffsets[shardId] * 3;
    buf[pos] = a;
    buf[pos + 1] = b;
    buf[pos + 2] = c;
    shardOffsets[shardId] = (shardOffsets[shardId] + 1) >>> 0;
  }

  for (let shardId = 0; shardId < count; shardId++) {
    const u32 = shardBuffers[shardId];
    if (!u32 || !u32.length) continue;
    const buf = Buffer.from(u32.buffer, u32.byteOffset, u32.byteLength);
    await fs.promises.appendFile(shardPaths[shardId], buf);
  }
}

async function appendSortedDedupedTriplesToAppender(u32, tripleCount, appender) {
  if (!tripleCount) return;

  let prevA = 0xffffffff;
  let prevB = 0xffffffff;
  let prevC = 0xffffffff;

  for (let i = 0; i < tripleCount; i++) {
    const off = i * 3;
    const a = u32[off] >>> 0;
    const b = u32[off + 1] >>> 0;
    const c = u32[off + 2] >>> 0;

    if (i > 0 && a === prevA && b === prevB && c === prevC) continue;

    prevA = a;
    prevB = b;
    prevC = c;

    const shardId = appender.shardChooser(a, b, appender.shardCount);
    let st = appender._stateForShard(shardId);
    if (st.count >= appender.bufferRecords) {
      await appender.flushShard(shardId);
      st = appender._stateForShard(shardId);
    }

    const pos = st.count * 3;
    st.data[pos] = a;
    st.data[pos + 1] = b;
    st.data[pos + 2] = c;
    st.count++;
    appender.recordCount++;
  }
}

function dedupeAndAppendPresenceTriples(triplesU32, tripleCount, appender, sortWork) {
  if (!tripleCount) return Promise.resolve();

  const u32 = triplesU32.subarray(0, tripleCount * 3);
  radixSortTriplesByOrderInPlace(u32, [0, 1, 2], sortWork || {});
  return appendSortedDedupedTriplesToAppender(u32, tripleCount, appender);
}

async function spoolPresenceBaseRowTriples({
  idx,
  taxa,
  speciesPlans,
  spoolDir,
  shardCount,
}) {
  const stride = getCooccurSpoolStride(idx, taxa);
  const shardPaths = new Array(shardCount);

  for (let i = 0; i < shardCount; i++) {
    shardPaths[i] = path.join(
      spoolDir,
      `rowspool_${String(i).padStart(5, "0")}.bin`,
    );
  }

  const SPOOL_BATCH_TRIPLES = Math.max(
    16_384,
    Math.min(
      Number(process.env.COOCCUR_SPOOL_BATCH_TRIPLES || 524_288) | 0,
      2_000_000,
    ),
  );

  let rowsSpooled = 0;
  let speciesDone = 0;
  let lastReportRows = 0;
  let triples = new Uint32Array(SPOOL_BATCH_TRIPLES * 3);
  let tripleCount = 0;

  async function flushTriples() {
    if (!tripleCount) return;
    await appendTriplesByShard(
      shardPaths,
      triples,
      tripleCount,
      shardCount,
    );
    tripleCount = 0;
  }

  for (let i = 0; i < speciesPlans.length; i++) {
    const plan = speciesPlans[i];
    const sid = plan.speciesNodeId >>> 0;

    await streamSpeciesRowIds(taxa, sid, async (rowId) => {
      const rid = rowId >>> 0;
      const baseRow = (Math.floor(rid / stride) * stride) >>> 0;
      const off = tripleCount * 3;

      triples[off] = baseRow >>> 0;
      triples[off + 1] = rid;
      triples[off + 2] = sid;

      tripleCount++;
      rowsSpooled++;

      if (tripleCount >= SPOOL_BATCH_TRIPLES) {
        await flushTriples();
      }
    });

    speciesDone++;

    if (
      rowsSpooled - lastReportRows >= 250000 ||
      speciesDone === 1 ||
      speciesDone === speciesPlans.length ||
      speciesDone % 100 === 0
    ) {
      lastReportRows = rowsSpooled;
      console.log(
        `[presence-spool] species ${speciesDone}/${speciesPlans.length} rows=${rowsSpooled.toLocaleString()} current=${makeSpeciesLabel(taxa, sid)}`,
      );
      logMem("[presence-spool]");
    }

    maybeGC("cooccur-presence-spool-species");
  }

  await flushTriples();

  const existingShardPaths = [];
  for (let i = 0; i < shardPaths.length; i++) {
    try {
      const st = await fs.promises.stat(shardPaths[i]);
      if (st.isFile() && st.size > 0) existingShardPaths.push(shardPaths[i]);
    } catch {}
  }

  return {
    stride,
    spoolDir,
    shardPaths: existingShardPaths,
    shardCount: existingShardPaths.length >>> 0,
    rowsSpooled: rowsSpooled >>> 0,
  };
}

async function processPresenceSortedSpoolShard({
  filePath,
  reader,
  scales,
  opts,
  presenceAppenders,
}) {
  const fd = await fs.promises.open(filePath, "r");

  const scalePlans = scales.map((scaleCtx) => {
    const anchorOffsets = Array.isArray(scaleCtx.anchorOffsets)
      ? scaleCtx.anchorOffsets
      : [{ dx: 0, dy: 0 }];
    const anchorCount = anchorOffsets.length >>> 0;
    const anchorDx = new Int32Array(anchorCount || 1);
    const anchorDy = new Int32Array(anchorCount || 1);
    for (let i = 0; i < anchorCount; i++) {
      anchorDx[i] = anchorOffsets[i].dx | 0;
      anchorDy[i] = anchorOffsets[i].dy | 0;
    }

    return {
      scaleM: scaleCtx.scaleM,
      phaseXFrac: scaleCtx.phaseXFrac,
      phaseYFrac: scaleCtx.phaseYFrac,
      anchorDx,
      anchorDy,
      anchorCount,
      exactOnly: anchorCount === 1 && anchorDx[0] === 0 && anchorDy[0] === 0,
      maxUnc:
        opts.uncFactor > 0 ? safeNumber(scaleCtx.scaleM, 0) * opts.uncFactor : Infinity,
    };
  });

  const scaleTriples = scalePlans.map(
    () => new Uint32Array(Math.max(4096, scales.length * 4096 * 3)),
  );
  const scaleTripleCounts = new Uint32Array(scales.length);
  const scaleSortWork = scalePlans.map(() => ({}));
  const rowCellX = new Uint32Array(Math.max(1, scales.length));
  const rowCellY = new Uint32Array(Math.max(1, scales.length));
  const rowScaleOk = new Uint8Array(Math.max(1, scales.length));

  function ensureScaleTripleCapacity(scaleIdx, tripleCount) {
    const need = Math.max(3, tripleCount * 3);
    if (need <= scaleTriples[scaleIdx].length) return;
    const grown = new Uint32Array(
      Math.max(need, scaleTriples[scaleIdx].length * 2),
    );
    grown.set(scaleTriples[scaleIdx]);
    scaleTriples[scaleIdx] = grown;
  }

  async function flushBaseRowRun(baseRow, rowIds, speciesIds, runCount, stats) {
    if (!runCount) return;

    const { rowRunMap, uniqueRowIds, uniqueCount } = buildRowRunMap(
      rowIds,
      speciesIds,
      runCount,
    );

    scaleTripleCounts.fill(0);
    let recoveredRows = 0;

    await reader.visitRowIds(uniqueRowIds, (rec) => {
      recoveredRows++;
      stats.rowsVisited++;

      const rid = rec.rowId >>> 0;
      const run = rowRunMap.get(rid);
      if (!run) return;

      const vals = rec.values;
      const lon = safeNumber(vals[0], NaN);
      const lat = safeNumber(vals[1], NaN);
      const unc = vals.length > 2 ? safeNumber(vals[2], NaN) : NaN;

      if (!Number.isFinite(lon) || !Number.isFinite(lat)) return;
      if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;

      stats.rowsAccepted++;

      const uncFinite = Number.isFinite(unc);
      for (let k = 0; k < scalePlans.length; k++) {
        const plan = scalePlans[k];
        if (uncFinite && opts.uncFactor > 0 && unc > plan.maxUnc) {
          rowScaleOk[k] = 0;
          continue;
        }

        const baseCell = lonLatToMercatorCellBiasPhase(
          lon,
          lat,
          plan.scaleM,
          plan.phaseXFrac,
          plan.phaseYFrac,
        );

        rowScaleOk[k] = 1;
        rowCellX[k] = baseCell.xBias >>> 0;
        rowCellY[k] = baseCell.yBias >>> 0;
      }

      for (let ri = run.start; ri < run.end; ri++) {
        const sid = speciesIds[ri] >>> 0;

        for (let k = 0; k < scalePlans.length; k++) {
          if (!rowScaleOk[k]) continue;

          const plan = scalePlans[k];
          const x = rowCellX[k] >>> 0;
          const y = rowCellY[k] >>> 0;

          if (plan.exactOnly) {
            const nextCount = (scaleTripleCounts[k] + 1) >>> 0;
            ensureScaleTripleCapacity(k, nextCount);
            const off = scaleTripleCounts[k] * 3;
            scaleTriples[k][off] = x;
            scaleTriples[k][off + 1] = y;
            scaleTriples[k][off + 2] = sid;
            scaleTripleCounts[k] = nextCount;
            continue;
          }

          const nextCount =
            (scaleTripleCounts[k] + plan.anchorCount) >>> 0;
          ensureScaleTripleCapacity(k, nextCount);

          let off = scaleTripleCounts[k] * 3;
          for (let ai = 0; ai < plan.anchorCount; ai++) {
            scaleTriples[k][off] = (x + plan.anchorDx[ai]) >>> 0;
            scaleTriples[k][off + 1] = (y + plan.anchorDy[ai]) >>> 0;
            scaleTriples[k][off + 2] = sid;
            off += 3;
          }

          scaleTripleCounts[k] = nextCount;
        }
      }
    });

    stats.baseRowRuns++;
    stats.uniqueRows += uniqueCount >>> 0;
    stats.recoveredRows += recoveredRows >>> 0;

    for (let k = 0; k < scalePlans.length; k++) {
      await dedupeAndAppendPresenceTriples(
        scaleTriples[k],
        scaleTripleCounts[k] >>> 0,
        presenceAppenders[k],
        scaleSortWork[k],
      );
    }

    const missing = Math.max(0, uniqueCount - recoveredRows);
    if (missing > 0) {
      stats.missingRows += missing >>> 0;
      console.warn(
        `[presence-grid] incomplete readback baseRow=${baseRow >>> 0} recovered=${recoveredRows}/${uniqueCount}`,
      );
    }
  }

  const READ_BYTES = Math.max(
    1 << 20,
    Math.min(
      Number(process.env.COOCCUR_PRESENCE_GRID_READ_BYTES || 16 << 20) | 0,
      64 << 20,
    ),
  );
  const raw = Buffer.allocUnsafe(READ_BYTES);

  let carry = Buffer.alloc(0);
  let curBaseRow = null;
  let rowIds = new Uint32Array(65536);
  let speciesIds = new Uint32Array(65536);
  let runCount = 0;

  function ensureRunCapacity(need) {
    if (need <= rowIds.length) return;
    const nextCap = Math.max(need, rowIds.length * 2);
    const nextRowIds = new Uint32Array(nextCap);
    const nextSpeciesIds = new Uint32Array(nextCap);
    nextRowIds.set(rowIds.subarray(0, runCount));
    nextSpeciesIds.set(speciesIds.subarray(0, runCount));
    rowIds = nextRowIds;
    speciesIds = nextSpeciesIds;
  }

  const stats = {
    baseRowRuns: 0,
    uniqueRows: 0,
    recoveredRows: 0,
    missingRows: 0,
    rowsVisited: 0,
    rowsAccepted: 0,
  };

  try {
    for (;;) {
      const { bytesRead } = await fd.read(raw, 0, raw.length, null);
      if (bytesRead <= 0) break;

      const view = raw.subarray(0, bytesRead);
      const total = carry.length + view.length;
      const usable = total - (total % 12);
      const buf =
        carry.length > 0 ? Buffer.concat([carry, view], total) : view;

      for (let off = 0; off < usable; off += 12) {
        const baseRow = buf.readUInt32LE(off) >>> 0;
        const rowId = buf.readUInt32LE(off + 4) >>> 0;
        const sid = buf.readUInt32LE(off + 8) >>> 0;

        if (curBaseRow === null) {
          curBaseRow = baseRow;
        } else if (baseRow !== (curBaseRow >>> 0)) {
          await flushBaseRowRun(curBaseRow, rowIds, speciesIds, runCount, stats);
          runCount = 0;
          curBaseRow = baseRow;
        }

        ensureRunCapacity(runCount + 1);
        rowIds[runCount] = rowId;
        speciesIds[runCount] = sid;
        runCount++;
      }

      carry = usable < buf.length ? buf.subarray(usable) : Buffer.alloc(0);
    }

    if (carry.length) {
      throw new Error(
        `Trailing bytes not aligned in ${filePath}: carry=${carry.length}`,
      );
    }

    if (curBaseRow !== null && runCount) {
      await flushBaseRowRun(curBaseRow, rowIds, speciesIds, runCount, stats);
    }
  } finally {
    await fd.close().catch(() => {});
  }

  return stats;
}

async function spoolPresencePhase({ idx, taxa, csvFd, scales, opts, taxonFilter }) {
  console.log(`[presence] starting ${nowIso()}`);

  const header = chooseHeader(idx, "guess") || idx.header || [];
  const lonCol = findColIndex(header, "decimalLongitude");
  const latCol = findColIndex(header, "decimalLatitude");
  const uncCol = findColIndex(header, "coordinateUncertaintyInMeters");
  if (lonCol < 0 || latCol < 0) {
    throw new Error("Missing decimalLongitude and/or decimalLatitude");
  }

  const wantCols = uncCol >= 0 ? [lonCol, latCol, uncCol] : [lonCol, latCol];
  const reader = new ExactOrWindowRowReader(idx, taxa, csvFd, wantCols);
  const speciesPlans = await enumerateSpecies(taxa, opts, taxonFilter);
  if (!speciesPlans.length) throw new Error("No species nodes found to process");

  let rowsTotalEstimate = 0;
  for (let i = 0; i < speciesPlans.length; i++) {
    rowsTotalEstimate += Number(speciesPlans[i].obsCount || 0);
  }

  const outRoot = path.dirname(scales[0].baseDir);
  const spoolRoot = path.join(outRoot, ".presence_row_spool");
  const sortedSpoolDir = path.join(outRoot, ".presence_row_spool_sorted");
  ensureDir(spoolRoot);
  ensureDir(sortedSpoolDir);

  const spoolShardCount = chooseEffectivePresenceSpoolShards(
    rowsTotalEstimate,
    Math.max(
      1,
      Number(process.env.COOCCUR_SPOOL_SHARDS || opts.presenceShards) | 0,
    ),
  );

  console.log(
    `[presence] species=${speciesPlans.length.toLocaleString()} row_estimate=${Math.round(rowsTotalEstimate).toLocaleString()} spool_shards=${spoolShardCount.toLocaleString()}`,
  );

  const spool = await spoolPresenceBaseRowTriples({
    idx,
    taxa,
    speciesPlans,
    spoolDir: spoolRoot,
    shardCount: spoolShardCount,
  });

  console.log(
    `[presence] row spool done shards=${spool.shardCount.toLocaleString()} rows=${spool.rowsSpooled.toLocaleString()} stride=${spool.stride.toLocaleString()}`,
  );

  const sortedSpoolPaths = await sortShards({
    files: spool.shardPaths,
    sortedDir: sortedSpoolDir,
    order: [0, 1, 2],
    workers: opts.sortWorkers,
  });

  if (!opts.keepTemp) {
    for (let i = 0; i < spool.shardPaths.length; i++) {
      await fs.promises.unlink(spool.shardPaths[i]).catch(() => {});
    }
  }

  const presenceAppenders = scales.map(
    (scaleCtx) =>
      new TripleShardAppender({
        dir: scaleCtx.presenceDir,
        shardCount: opts.presenceShards,
        openWriters: opts.openWriters,
        bufferRecords: Math.max(512, Number(process.env.COOCCUR_PRESENCE_BUFFER_RECORDS || 4096) | 0),
        shardNamer: scaleCtx.presenceShardNamer,
        shardChooser: choosePresenceShard,
      }),
  );

  const globalStats = {
    baseRowRuns: 0,
    uniqueRows: 0,
    recoveredRows: 0,
    missingRows: 0,
    rowsVisited: 0,
    rowsAccepted: 0,
  };

  try {
    for (let i = 0; i < sortedSpoolPaths.length; i++) {
      const filePath = sortedSpoolPaths[i];
      const stats = await processPresenceSortedSpoolShard({
        filePath,
        reader,
        scales,
        opts,
        presenceAppenders,
      });

      globalStats.baseRowRuns += stats.baseRowRuns >>> 0;
      globalStats.uniqueRows += stats.uniqueRows >>> 0;
      globalStats.recoveredRows += stats.recoveredRows >>> 0;
      globalStats.missingRows += stats.missingRows >>> 0;
      globalStats.rowsVisited += stats.rowsVisited >>> 0;
      globalStats.rowsAccepted += stats.rowsAccepted >>> 0;

      console.log(
        `[presence-grid] shard ${i + 1}/${sortedSpoolPaths.length} base_rows=${stats.baseRowRuns.toLocaleString()} unique_rows=${stats.uniqueRows.toLocaleString()} recovered=${stats.recoveredRows.toLocaleString()} kept=${stats.rowsAccepted.toLocaleString()}`,
      );
      logMem("[presence-grid]");
      maybeGC("cooccur-presence-grid-shard");
    }
  } finally {
    for (let i = 0; i < presenceAppenders.length; i++) {
      await presenceAppenders[i].close();
    }
  }

  if (!opts.keepTemp) {
    for (let i = 0; i < sortedSpoolPaths.length; i++) {
      await fs.promises.unlink(sortedSpoolPaths[i]).catch(() => {});
    }
    await fs.promises.rmdir(spoolRoot).catch(() => {});
    await fs.promises.rmdir(sortedSpoolDir).catch(() => {});
  }

  for (let i = 0; i < scales.length; i++) {
    const files = await listNonEmptyFiles(scales[i].presenceDir, ".bin");
    console.log(
      `[presence] variant=${scales[i].variantKey} shards=${files.length.toLocaleString()}`,
    );
  }

  console.log(
    `[presence] done ${nowIso()} base_rows=${globalStats.baseRowRuns.toLocaleString()} unique_rows=${globalStats.uniqueRows.toLocaleString()} recovered=${globalStats.recoveredRows.toLocaleString()} kept=${globalStats.rowsAccepted.toLocaleString()} missing=${globalStats.missingRows.toLocaleString()}`,
  );
}


async function sortPresencePhase(scales, opts) {
  for (let i = 0; i < scales.length; i++) {
    const scaleCtx = scales[i];
    const files = await listNonEmptyFiles(scaleCtx.presenceDir, ".bin");
    console.log(`[presence-sort] variant=${scaleCtx.variantKey} sorting ${files.length.toLocaleString()} shard(s)`);
    await sortShards({ files, sortedDir: scaleCtx.presenceSortedDir, order: [0, 1, 2], workers: opts.sortWorkers });
    if (!opts.keepTemp) {
      for (let j = 0; j < files.length; j++) await fs.promises.unlink(files[j]).catch(() => {});
    }
  }
}

async function reducePresenceToPairsPhase({ taxa, scales, opts }) {
  console.log(`[pairs] starting ${nowIso()}`);

  for (let i = 0; i < scales.length; i++) {
    const scaleCtx = scales[i];
    const speciesCellCounts = new Uint32Array(taxa.nodeCount >>> 0);
    const pairAppender = new TripleShardAppender({
      dir: scaleCtx.pairDir,
      shardCount: opts.pairShards,
      openWriters: opts.openWriters,
      bufferRecords: 256,
      shardNamer: scaleCtx.pairShardNamer,
      shardChooser: choosePairShard,
    });

    const sortedPresenceFiles = await listNonEmptyFiles(scaleCtx.presenceSortedDir, ".bin");
    let totalCells = 0;
    let skippedRichCells = 0;
    let keptPresenceRecords = 0;
    let emittedPairs = 0;

    try {
      for (let fi = 0; fi < sortedPresenceFiles.length; fi++) {
        const filePath = sortedPresenceFiles[fi];
        const fd = await fs.promises.open(filePath, "r");

        try {
          const READ_BYTES = 8 << 20;
          const raw = Buffer.allocUnsafe(READ_BYTES);
          let carry = Buffer.alloc(0);
          let curX = null;
          let curY = null;
          let curSpecies = [];

          async function flushCell() {
            if (!curSpecies.length) return;
            if (curSpecies.length > opts.maxCellSpecies) {
              skippedRichCells++;
              curSpecies.length = 0;
              return;
            }

            totalCells++;
            for (let s = 0; s < curSpecies.length; s++) {
              const sid = curSpecies[s] >>> 0;
              speciesCellCounts[sid] = (speciesCellCounts[sid] + 1) >>> 0;
              keptPresenceRecords++;
            }

            for (let a = 0; a < curSpecies.length; a++) {
              const sidA = curSpecies[a] >>> 0;
              for (let b = a + 1; b < curSpecies.length; b++) {
                const sidB = curSpecies[b] >>> 0;
                await pairAppender.append(sidA, sidB, 1);
                emittedPairs++;
              }
            }

            curSpecies.length = 0;
          }

          for (;;) {
            const { bytesRead } = await fd.read(raw, 0, raw.length, null);
            if (bytesRead <= 0) break;
            const view = raw.subarray(0, bytesRead);
            const total = carry.length + view.length;
            const usable = total - (total % 12);
            const buf = carry.length > 0 ? Buffer.concat([carry, view], total) : view;

            for (let off = 0; off < usable; off += 12) {
              const x = buf.readUInt32LE(off);
              const y = buf.readUInt32LE(off + 4);
              const sid = buf.readUInt32LE(off + 8) >>> 0;

              if (curX === null) {
                curX = x;
                curY = y;
                curSpecies.push(sid);
                continue;
              }

              if (x !== curX || y !== curY) {
                await flushCell();
                curX = x;
                curY = y;
                curSpecies.push(sid);
                continue;
              }

              const last = curSpecies.length ? curSpecies[curSpecies.length - 1] >>> 0 : 0xffffffff;
              if (sid !== last) curSpecies.push(sid);
            }

            carry = usable < buf.length ? buf.subarray(usable) : Buffer.alloc(0);
          }

          if (carry.length) throw new Error(`Trailing bytes not aligned in ${filePath}: carry=${carry.length}`);
          await flushCell();
        } finally {
          await fd.close().catch(() => {});
        }

        console.log(`[pairs] variant=${scaleCtx.variantKey} file ${fi + 1}/${sortedPresenceFiles.length} cells=${totalCells.toLocaleString()} pairs=${emittedPairs.toLocaleString()}`);
        maybeGC("cooccur-presence-reduce-file");
      }
    } finally {
      await pairAppender.close();
    }

    await fs.promises.writeFile(scaleCtx.cellCountsPath, Buffer.from(speciesCellCounts.buffer));
    await writeJson(scaleCtx.statsPath, {
      scaleM: scaleCtx.scaleM,
      phaseXFrac: scaleCtx.phaseXFrac,
      phaseYFrac: scaleCtx.phaseYFrac,
      phaseLabel: scaleCtx.phaseLabel,
      adjacency: scaleCtx.adjacency,
      variantKey: scaleCtx.variantKey,
      totalCells,
      skippedRichCells,
      keptPresenceRecords,
      emittedPairs,
      createdAt: nowIso(),
    });

    const ws = fs.createWriteStream(scaleCtx.cellCountsCsvPath, { flags: "w", highWaterMark: 4 << 20 });
    await writeStreamChunk(ws, Buffer.from("species_id,species_name,species_cells\n"));
    for (let sid = 1; sid < speciesCellCounts.length; sid++) {
      const cells = speciesCellCounts[sid] >>> 0;
      if (!cells) continue;
      const line = `${sid},${csvEscapeValue(makeSpeciesLabel(taxa, sid))},${cells}\n`;
      await writeStreamChunk(ws, Buffer.from(line));
    }
    await new Promise((resolve, reject) => {
      ws.on("finish", resolve);
      ws.on("error", reject);
      ws.end();
    });

    console.log(`[pairs] variant=${scaleCtx.variantKey} totalCells=${totalCells.toLocaleString()} skippedRichCells=${skippedRichCells.toLocaleString()} emittedPairs=${emittedPairs.toLocaleString()}`);
    logMem(`[pairs ${scaleCtx.variantKey}]`);
  }

  console.log(`[pairs] done ${nowIso()}`);
}

async function sortPairsPhase(scales, opts) {
  for (let i = 0; i < scales.length; i++) {
    const scaleCtx = scales[i];
    const files = await listNonEmptyFiles(scaleCtx.pairDir, ".bin");
    console.log(`[pair-sort] variant=${scaleCtx.variantKey} sorting ${files.length.toLocaleString()} shard(s)`);
    await sortShards({ files, sortedDir: scaleCtx.pairSortedDir, order: [0, 1], workers: opts.sortWorkers });
    if (!opts.keepTemp) {
      for (let j = 0; j < files.length; j++) await fs.promises.unlink(files[j]).catch(() => {});
    }
  }
}

async function aggregatePairsToCandidatesPhase({ scales, opts }) {
  for (let i = 0; i < scales.length; i++) {
    const scaleCtx = scales[i];
    const sortedPairFiles = await listNonEmptyFiles(scaleCtx.pairSortedDir, ".bin");
    const candidateAppender = new TripleShardAppender({
      dir: scaleCtx.candidateDir,
      shardCount: opts.candidateShards,
      openWriters: opts.openWriters,
      bufferRecords: 256,
      shardNamer: scaleCtx.candidateShardNamer,
      shardChooser: chooseCandidateShard,
    });

    let uniquePairs = 0;
    let keptPairs = 0;

    try {
      for (let fi = 0; fi < sortedPairFiles.length; fi++) {
        const filePath = sortedPairFiles[fi];
        const fd = await fs.promises.open(filePath, "r");

        try {
          const READ_BYTES = 8 << 20;
          const raw = Buffer.allocUnsafe(READ_BYTES);
          let carry = Buffer.alloc(0);
          let curA = null;
          let curB = null;
          let curCount = 0;

          async function flushPair() {
            if (curA == null || curB == null || curCount <= 0) return;
            uniquePairs++;
            if (curCount >= opts.minShared) {
              await candidateAppender.append(curA >>> 0, curB >>> 0, curCount >>> 0);
              await candidateAppender.append(curB >>> 0, curA >>> 0, curCount >>> 0);
              keptPairs++;
            }
            curA = null;
            curB = null;
            curCount = 0;
          }

          for (;;) {
            const { bytesRead } = await fd.read(raw, 0, raw.length, null);
            if (bytesRead <= 0) break;
            const view = raw.subarray(0, bytesRead);
            const total = carry.length + view.length;
            const usable = total - (total % 12);
            const buf = carry.length > 0 ? Buffer.concat([carry, view], total) : view;

            for (let off = 0; off < usable; off += 12) {
              const a = buf.readUInt32LE(off) >>> 0;
              const b = buf.readUInt32LE(off + 4) >>> 0;
              const c = buf.readUInt32LE(off + 8) >>> 0;
              if (curA === null) {
                curA = a;
                curB = b;
                curCount = c;
                continue;
              }
              if (a === curA && b === curB) {
                curCount = (curCount + c) >>> 0;
                continue;
              }
              await flushPair();
              curA = a;
              curB = b;
              curCount = c;
            }
            carry = usable < buf.length ? buf.subarray(usable) : Buffer.alloc(0);
          }

          if (carry.length) throw new Error(`Trailing bytes not aligned in ${filePath}: carry=${carry.length}`);
          await flushPair();
        } finally {
          await fd.close().catch(() => {});
        }

        console.log(`[rank-agg] variant=${scaleCtx.variantKey} file ${fi + 1}/${sortedPairFiles.length} uniquePairs=${uniquePairs.toLocaleString()} keptPairs=${keptPairs.toLocaleString()}`);
        maybeGC("cooccur-rank-aggregate-file");
      }
    } finally {
      await candidateAppender.close();
    }

    console.log(`[rank-agg] variant=${scaleCtx.variantKey} uniquePairs=${uniquePairs.toLocaleString()} keptPairs=${keptPairs.toLocaleString()}`);
  }
}

async function rankCandidatesPhase({ taxa, scales, opts, collapseAppender }) {
  for (let i = 0; i < scales.length; i++) {
    const scaleCtx = scales[i];
    const stats = await readJson(scaleCtx.statsPath);
    const totalCells = safeNumber(stats.totalCells, 0);
    const cellCountsBuf = await fs.promises.readFile(scaleCtx.cellCountsPath);
    const speciesCellCounts = new Uint32Array(cellCountsBuf.buffer, cellCountsBuf.byteOffset, (cellCountsBuf.length / 4) | 0);
    const candidateFiles = await listNonEmptyFiles(scaleCtx.candidateDir, ".bin");
    const outWs = fs.createWriteStream(scaleCtx.outputCsvPath, { flags: "w", highWaterMark: 4 << 20 });
    const nameCache = new Map();

    function speciesLabelCached(sid) {
      const k = sid >>> 0;
      if (nameCache.has(k)) return nameCache.get(k);
      const v = makeSpeciesLabel(taxa, k);
      nameCache.set(k, v);
      return v;
    }

    await writeStreamChunk(outWs, Buffer.from([
      "scale_m",
      "phase_x_frac",
      "phase_y_frac",
      "adjacency",
      "variant_key",
      "rank",
      "species_id",
      "species_name",
      "species_cells",
      "associate_id",
      "associate_name",
      "associate_cells",
      "shared_cells",
      "p_associate_given_species",
      "p_species_given_associate",
      "cond_post_mean",
      "reverse_cond_post_mean",
      "lift",
      "jaccard",
      "overlap",
      "npmi",
      "support_shrink",
      "score",
    ].join(",") + "\n"));

    let writtenRows = 0;

    try {
      for (let fi = 0; fi < candidateFiles.length; fi++) {
        const filePath = candidateFiles[fi];
        const fd = await fs.promises.open(filePath, "r");
        const topMap = new Map();

        try {
          const READ_BYTES = 8 << 20;
          const raw = Buffer.allocUnsafe(READ_BYTES);
          let carry = Buffer.alloc(0);

          for (;;) {
            const { bytesRead } = await fd.read(raw, 0, raw.length, null);
            if (bytesRead <= 0) break;
            const view = raw.subarray(0, bytesRead);
            const total = carry.length + view.length;
            const usable = total - (total % 12);
            const buf = carry.length > 0 ? Buffer.concat([carry, view], total) : view;

            for (let off = 0; off < usable; off += 12) {
              const focalSid = buf.readUInt32LE(off) >>> 0;
              const otherSid = buf.readUInt32LE(off + 4) >>> 0;
              const shared = buf.readUInt32LE(off + 8) >>> 0;
              const focalCells = speciesCellCounts[focalSid] >>> 0;
              const otherCells = speciesCellCounts[otherSid] >>> 0;
              if (focalCells < opts.minSpeciesCells) continue;
              if (otherCells <= 0) continue;
              if (shared < opts.minShared) continue;

              const metrics = makeAssocMetrics(shared, focalCells, otherCells, totalCells);
              if (opts.minLift > 0 && metrics.lift < opts.minLift) continue;
              if (opts.minConditional > 0 && metrics.p_associate_given_species < opts.minConditional) continue;
              if (metrics.npmi < opts.minNpmi) continue;

              const item = {
                species_id: focalSid,
                species_cells: focalCells,
                associate_id: otherSid,
                associate_cells: otherCells,
                shared_cells: metrics.shared_cells,
                p_associate_given_species: metrics.p_associate_given_species,
                p_species_given_associate: metrics.p_species_given_associate,
                cond_post_mean: metrics.cond_post_mean,
                reverse_cond_post_mean: metrics.reverse_cond_post_mean,
                lift: metrics.lift,
                jaccard: metrics.jaccard,
                overlap: metrics.overlap,
                npmi: metrics.npmi,
                support_shrink: metrics.support_shrink,
                score: metrics.score,
              };

              const cur = topMap.get(focalSid) || [];
              topMap.set(focalSid, insertTopK(cur, item, opts.topK));
            }

            carry = usable < buf.length ? buf.subarray(usable) : Buffer.alloc(0);
          }

          if (carry.length) throw new Error(`Trailing bytes not aligned in ${filePath}: carry=${carry.length}`);
        } finally {
          await fd.close().catch(() => {});
        }

        const speciesIds = Array.from(topMap.keys()).sort((a, b) => a - b);
        for (let s = 0; s < speciesIds.length; s++) {
          const sid = speciesIds[s] >>> 0;
          const arr = topMap.get(sid) || [];
          arr.sort((a, b) => {
            if (b.score !== a.score) return b.score - a.score;
            if (b.shared_cells !== a.shared_cells) return b.shared_cells - a.shared_cells;
            if (b.lift !== a.lift) return b.lift - a.lift;
            return (a.associate_id >>> 0) - (b.associate_id >>> 0);
          });

          const speciesName = speciesLabelCached(sid);
          const csvLines = [];
          for (let rank = 0; rank < arr.length; rank++) {
            const row = arr[rank];
            const associateName = speciesLabelCached(row.associate_id);
            csvLines.push([
              scaleCtx.scaleM,
              scaleCtx.phaseXFrac,
              scaleCtx.phaseYFrac,
              csvEscapeValue(scaleCtx.adjacency),
              csvEscapeValue(scaleCtx.variantKey),
              rank + 1,
              sid,
              csvEscapeValue(speciesName),
              row.species_cells,
              row.associate_id,
              csvEscapeValue(associateName),
              row.associate_cells,
              row.shared_cells,
              row.p_associate_given_species.toFixed(8),
              row.p_species_given_associate.toFixed(8),
              row.cond_post_mean.toFixed(8),
              row.reverse_cond_post_mean.toFixed(8),
              row.lift.toFixed(8),
              row.jaccard.toFixed(8),
              row.overlap.toFixed(8),
              row.npmi.toFixed(8),
              row.support_shrink.toFixed(8),
              row.score.toFixed(8),
            ].join(",") + "\n");

            if (collapseAppender) {
              const compactLine = [
                sid,
                row.associate_id,
                scaleCtx.scaleM,
                scaleCtx.phaseXFrac,
                scaleCtx.phaseYFrac,
                scaleCtx.adjacency,
                scaleCtx.variantKey,
                row.species_cells,
                row.associate_cells,
                row.shared_cells,
                row.p_associate_given_species.toFixed(8),
                row.p_species_given_associate.toFixed(8),
                row.cond_post_mean.toFixed(8),
                row.reverse_cond_post_mean.toFixed(8),
                row.lift.toFixed(8),
                row.jaccard.toFixed(8),
                row.overlap.toFixed(8),
                row.npmi.toFixed(8),
                row.support_shrink.toFixed(8),
                row.score.toFixed(8),
              ].join("	") + "\n";
              await collapseAppender.appendLine(sid, row.associate_id, compactLine);
            }

            writtenRows++;
          }
          if (csvLines.length) await writeStreamChunk(outWs, Buffer.from(csvLines.join("")));
        }

        console.log(`[rank] variant=${scaleCtx.variantKey} file ${fi + 1}/${candidateFiles.length} wrote=${writtenRows.toLocaleString()}`);
        maybeGC("cooccur-rank-file");
      }
    } finally {
      await new Promise((resolve, reject) => {
        outWs.on("finish", resolve);
        outWs.on("error", reject);
        outWs.end();
      });
    }

    console.log(`[rank] variant=${scaleCtx.variantKey} output=${scaleCtx.outputCsvPath} rows=${writtenRows.toLocaleString()}`);
  }
}

async function collapsePredictorSourcesPhase({ taxa, collapseCtx, opts }) {
  const files = await listNonEmptyFiles(collapseCtx.dir, ".tsv");
  const outWs = fs.createWriteStream(collapseCtx.outputCsvPath, { flags: "w", highWaterMark: 4 << 20 });
  const nameCache = new Map();

  function speciesLabelCached(sid) {
    const k = sid >>> 0;
    if (nameCache.has(k)) return nameCache.get(k);
    const v = makeSpeciesLabel(taxa, k);
    nameCache.set(k, v);
    return v;
  }

  await writeStreamChunk(outWs, Buffer.from([
    "rank",
    "species_id",
    "species_name",
    "associate_id",
    "associate_name",
    "variant_keys_supported",
    "scales_supported",
    "n_variants",
    "n_scales",
    "best_variant_key",
    "best_scale_m",
    "best_phase_x_frac",
    "best_phase_y_frac",
    "best_adjacency",
    "best_species_cells",
    "best_associate_cells",
    "best_shared_cells",
    "best_p_associate_given_species",
    "best_cond_post_mean",
    "best_lift",
    "best_jaccard",
    "best_overlap",
    "best_npmi",
    "best_scale_score",
    "mean_scale_score",
    "predictor_score",
  ].join(",") + "\n"));

  let writtenRows = 0;

  try {
    for (let fi = 0; fi < files.length; fi++) {
      const filePath = files[fi];
      const rs = fs.createReadStream(filePath, { encoding: "utf8", highWaterMark: 4 << 20 });
      const pairMap = new Map();
      let tail = "";

      function absorbLine(line) {
        if (!line) return;
        const a = line.split("\t");
        if (a.length < 20) return;
        const focalSid = Number(a[0]) >>> 0;
        const otherSid = Number(a[1]) >>> 0;
        const scaleM = Number(a[2]) | 0;
        const phaseXFrac = safeNumber(a[3], 0);
        const phaseYFrac = safeNumber(a[4], 0);
        const adjacency = String(a[5] || "exact");
        const variantKey = String(a[6] || `${scaleM}m_${makePhaseLabel(phaseXFrac, phaseYFrac)}_${adjacency}`);
        const focalCells = Number(a[7]) >>> 0;
        const otherCells = Number(a[8]) >>> 0;
        const sharedCells = Number(a[9]) >>> 0;
        const pBgivenA = safeNumber(a[10], 0);
        const condPost = safeNumber(a[12], 0);
        const lift = safeNumber(a[14], 0);
        const jaccard = safeNumber(a[15], 0);
        const overlap = safeNumber(a[16], 0);
        const npmi = safeNumber(a[17], 0);
        const scaleScore = safeNumber(a[19], 0);

        const key = `${focalSid}:${otherSid}`;
        let st = pairMap.get(key);
        if (!st) {
          st = {
            species_id: focalSid,
            associate_id: otherSid,
            scales: [],
            scaleSet: new Set(),
            variants: [],
            variantSet: new Set(),
            best_variant_key: variantKey,
            best_scale_m: scaleM,
            best_phase_x_frac: phaseXFrac,
            best_phase_y_frac: phaseYFrac,
            best_adjacency: adjacency,
            best_species_cells: focalCells,
            best_associate_cells: otherCells,
            best_shared_cells: sharedCells,
            best_p_associate_given_species: pBgivenA,
            best_cond_post_mean: condPost,
            best_lift: lift,
            best_jaccard: jaccard,
            best_overlap: overlap,
            best_npmi: npmi,
            best_scale_score: scaleScore,
            score_sum: scaleScore,
            score_count: 1,
          };
          st.scales.push(scaleM);
          st.scaleSet.add(scaleM);
          st.variants.push(variantKey);
          st.variantSet.add(variantKey);
          pairMap.set(key, st);
          return;
        }

        if (!st.scaleSet.has(scaleM)) {
          st.scaleSet.add(scaleM);
          st.scales.push(scaleM);
        }
        if (!st.variantSet.has(variantKey)) {
          st.variantSet.add(variantKey);
          st.variants.push(variantKey);
        }
        st.score_sum += scaleScore;
        st.score_count++;

        if (
          scaleScore > st.best_scale_score ||
          (scaleScore === st.best_scale_score && sharedCells > st.best_shared_cells) ||
          (scaleScore === st.best_scale_score && sharedCells === st.best_shared_cells && scaleM < st.best_scale_m)
        ) {
          st.best_variant_key = variantKey;
          st.best_scale_m = scaleM;
          st.best_phase_x_frac = phaseXFrac;
          st.best_phase_y_frac = phaseYFrac;
          st.best_adjacency = adjacency;
          st.best_species_cells = focalCells;
          st.best_associate_cells = otherCells;
          st.best_shared_cells = sharedCells;
          st.best_p_associate_given_species = pBgivenA;
          st.best_cond_post_mean = condPost;
          st.best_lift = lift;
          st.best_jaccard = jaccard;
          st.best_overlap = overlap;
          st.best_npmi = npmi;
          st.best_scale_score = scaleScore;
        }
      }

      for await (const chunk of rs) {
        tail += chunk;
        const lines = tail.split(/\r?\n/);
        tail = lines.pop() || "";
        for (let i = 0; i < lines.length; i++) absorbLine(lines[i]);
      }
      if (tail) absorbLine(tail);

      const topMap = new Map();
      for (const st of pairMap.values()) {
        st.scales.sort((a, b) => a - b);
        st.variants.sort();
        st.n_scales = st.scales.length;
        st.n_variants = st.variants.length;
        st.scales_supported = st.scales.join("|");
        st.variant_keys_supported = st.variants.join("|");
        st.mean_scale_score = st.score_count > 0 ? st.score_sum / st.score_count : 0;
        const scaleBonus = 1 + Math.min(0.5, Math.max(0, st.n_scales - 1) * 0.15);
        st.predictor_score = ((0.7 * st.best_scale_score) + (0.3 * st.mean_scale_score)) * scaleBonus;
        const cur = topMap.get(st.species_id) || [];
        topMap.set(st.species_id, insertTopK(cur, {
          associate_id: st.associate_id,
          shared_cells: st.best_shared_cells,
          lift: st.best_lift,
          score: st.predictor_score,
          payload: st,
        }, opts.topK));
      }

      const speciesIds = Array.from(topMap.keys()).sort((a, b) => a - b);
      for (let si = 0; si < speciesIds.length; si++) {
        const sid = speciesIds[si] >>> 0;
        const arr = topMap.get(sid) || [];
        arr.sort((a, b) => {
          if (b.score !== a.score) return b.score - a.score;
          if (b.shared_cells !== a.shared_cells) return b.shared_cells - a.shared_cells;
          if (b.lift !== a.lift) return b.lift - a.lift;
          return (a.associate_id >>> 0) - (b.associate_id >>> 0);
        });

        const speciesName = speciesLabelCached(sid);
        const lines = [];
        for (let rank = 0; rank < arr.length; rank++) {
          const st = arr[rank].payload;
          const associateName = speciesLabelCached(st.associate_id);
          lines.push([
            rank + 1,
            sid,
            csvEscapeValue(speciesName),
            st.associate_id,
            csvEscapeValue(associateName),
            csvEscapeValue(st.variant_keys_supported),
            csvEscapeValue(st.scales_supported),
            st.n_variants,
            st.n_scales,
            csvEscapeValue(st.best_variant_key),
            st.best_scale_m,
            st.best_phase_x_frac,
            st.best_phase_y_frac,
            csvEscapeValue(st.best_adjacency),
            st.best_species_cells,
            st.best_associate_cells,
            st.best_shared_cells,
            st.best_p_associate_given_species.toFixed(8),
            st.best_cond_post_mean.toFixed(8),
            st.best_lift.toFixed(8),
            st.best_jaccard.toFixed(8),
            st.best_overlap.toFixed(8),
            st.best_npmi.toFixed(8),
            st.best_scale_score.toFixed(8),
            st.mean_scale_score.toFixed(8),
            st.predictor_score.toFixed(8),
          ].join(",") + "\n");
          writtenRows++;
        }
        if (lines.length) await writeStreamChunk(outWs, Buffer.from(lines.join("")));
      }

      console.log(`[collapse] file ${fi + 1}/${files.length} wrote=${writtenRows.toLocaleString()}`);
      maybeGC("cooccur-collapse-file");
    }
  } finally {
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
        outWs.off("finish", ok);
        outWs.off("close", ok);
        outWs.off("error", bad);
      }
      outWs.on("finish", ok);
      outWs.on("close", ok);
      outWs.on("error", bad);
      outWs.end();
    });
  }

  console.log(`[collapse] output=${collapseCtx.outputCsvPath} rows=${writtenRows.toLocaleString()}`);
}

async function maybeCleanupCollapseSources(collapseCtx, keepTemp) {
  if (keepTemp) return;
  const files = await listNonEmptyFiles(collapseCtx.dir, ".tsv");
  for (let i = 0; i < files.length; i++) await fs.promises.unlink(files[i]).catch(() => {});
}

async function maybeCleanupCandidates(scales, keepTemp) {
  if (keepTemp) return;
  for (let i = 0; i < scales.length; i++) {
    const files = await listNonEmptyFiles(scales[i].candidateDir, ".bin");
    for (let j = 0; j < files.length; j++) await fs.promises.unlink(files[j]).catch(() => {});
  }
}

async function main() {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    parseArgs(["--help"]);
    return;
  }

  const opts = parseArgs(process.argv.slice(3));
  const st = await statFile(FILE).catch((e) => {
    console.error(e);
    return null;
  });
  if (!st || !st.isFile()) throw new Error(`File not found: ${FILE}`);

  console.log(`[cooccur] csv=${FILE}`);
  console.log(`[cooccur] size=${fmtGiB(st.size)} GiB`);
  console.log(`[cooccur] phase=${opts.phase} scales=${opts.scales.join(",")} phaseOffsets=${opts.phaseOffsets.map((p) => `${p.xFrac}:${p.yFrac}`).join("|")} adjacency=${opts.adjacency} topK=${opts.topK} minShared=${opts.minShared} minSpeciesCells=${opts.minSpeciesCells} minLift=${opts.minLift} minNpmi=${opts.minNpmi} maxCellSpecies=${opts.maxCellSpecies}`);
  console.log(`[cooccur] taxon_filter=${summarizeTaxonSelectors(opts.includeTaxaSelectors)} exclude=${summarizeTaxonSelectors(opts.excludeTaxaSelectors)}`);

  const idx = await loadOrBuildIndex(FILE, INDEX_PATH);
  idx._sharedCsvFd = await fs.promises.open(idx.file, "r");

  let taxa = null;
  try {
    taxa = await loadTaxaIndex().catch((e) => {
      console.error(e);
      return null;
    });
    if (!taxa) throw new Error(`Taxa index not loaded from ${TAXA_DIR}. Run your taxonomy build first.`);
    if (!taxa._sharedPostingsFd) taxa._sharedPostingsFd = await fs.promises.open(taxa.postingsPath, "r");

    const taxonFilter = resolveTaxonFilter(taxa, opts);
    console.log(`[cooccur] resolved_taxon_filter=${taxonFilterSummary(taxonFilter)}`);

    const outRoot = path.resolve(opts.outRoot);
    ensureDir(outRoot);
    await writeJson(path.join(outRoot, "run_config.json"), {
      csv: path.resolve(FILE),
      indexPath: path.resolve(INDEX_PATH),
      taxaDir: path.resolve(TAXA_DIR),
      options: opts,
      variantContexts: buildScaleContexts(path.resolve(opts.outRoot), opts).map((s) => ({
        scaleM: s.scaleM,
        phaseXFrac: s.phaseXFrac,
        phaseYFrac: s.phaseYFrac,
        phaseLabel: s.phaseLabel,
        adjacency: s.adjacency,
        variantKey: s.variantKey,
      })),
      taxonFilter: {
        includeSelectors: (taxonFilter.includeSelectors || []).map((s) => s.raw),
        excludeSelectors: (taxonFilter.excludeSelectors || []).map((s) => s.raw),
        includeResolved: taxonFilter.includeResolved,
        excludeResolved: taxonFilter.excludeResolved,
      },
      createdAt: nowIso(),
    });

    const scales = buildScaleContexts(outRoot, opts);
    const collapseCtx = buildCollapseContext(outRoot, opts);
    if (opts.phase === "all" || opts.phase === "index") {
      await spoolPresencePhase({ idx, taxa, csvFd: idx._sharedCsvFd, scales, opts, taxonFilter });
      await sortPresencePhase(scales, opts);
    }
    if (opts.phase === "all" || opts.phase === "pairs") {
      await reducePresenceToPairsPhase({ taxa, scales, opts });
      await sortPairsPhase(scales, opts);
    }
    if (opts.phase === "all" || opts.phase === "rank") {
      await aggregatePairsToCandidatesPhase({ scales, opts });
      const collapseAppender = new TextShardAppender({
        dir: collapseCtx.dir,
        shardCount: collapseCtx.shardCount,
        openWriters: opts.openWriters,
        bufferBytes: 1 << 20,
        shardNamer: collapseCtx.shardNamer,
        shardChooser: chooseCollapseShard,
      });
      try {
        await rankCandidatesPhase({ taxa, scales, opts, collapseAppender });
      } finally {
        await collapseAppender.close();
      }
      await collapsePredictorSourcesPhase({ taxa, collapseCtx, opts });
      await maybeCleanupCandidates(scales, opts.keepTemp);
      await maybeCleanupCollapseSources(collapseCtx, opts.keepTemp);
    }

    console.log(`[cooccur] done ${nowIso()}`);
  } finally {
    if (idx && idx._sharedCsvFd) await idx._sharedCsvFd.close().catch(() => {});
    if (taxa && taxa._sharedPostingsFd) await taxa._sharedPostingsFd.close().catch(() => {});
  }
}

if (!isMainThread) {
  sortShardWorkerMain().catch((err) => {
    if (parentPort) parentPort.postMessage({ ok: false, error: err && err.stack ? err.stack : String(err) });
    process.exit(1);
  });
} else {
  main().catch((err) => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
}
