"use strict";

// ./server/csvserver.export.js

const {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} = require("worker_threads");

const {
  fs,
  path,
  taxonomyRankName,
  clamp01,
  ensureDir,
  maybeGC,
  sanitizePathSegment,
  writeStreamChunk,
  csvEscapeValue,
} = require("./csvserver.utils.js");

const {
  findColIndex,
  getBestLookupStride,
  getLookupWindowForBaseRow,
  readRowsByRowIdsSelectiveRecordsFd,
} = require("./csvserver.core.js");

const {
  streamSpeciesRowIds,
  walkSpeciesTargetsUnderNode,
  countSpeciesTargetsUnderNode,
  dedupeNodeIds,
} = require("./csvserver.taxa.js");

const CSV_QUOTE = 0x22;

function detectExportIdCol(header) {
  let idColIndex = findColIndex(header, "gbifID");
  if (idColIndex < 0) idColIndex = findColIndex(header, "occurrenceID");
  if (idColIndex < 0) {
    throw new Error("Could not find gbifID or occurrenceID column");
  }
  return { idColIndex, idColName: header[idColIndex] };
}

function onceEvent(emitter, okEvent, errEvent) {
  return new Promise((resolve, reject) => {
    let settled = false;

    function doneOk(arg) {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(arg);
    }

    function doneErr(err) {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    }

    function cleanup() {
      emitter.off(okEvent, doneOk);
      emitter.off(errEvent, doneErr);
    }

    emitter.on(okEvent, doneOk);
    emitter.on(errEvent, doneErr);
  });
}

class LruFileWriterCache {
  constructor(limit, openFn, closeFn) {
    this._limit = Math.max(1, limit | 0);
    this._openFn = openFn;
    this._closeFn = closeFn;
    this._map = new Map();
  }

  async get(key) {
    if (this._map.has(key)) {
      const value = this._map.get(key);
      this._map.delete(key);
      this._map.set(key, value);
      return value;
    }

    if (this._map.size >= this._limit) {
      const oldestKey = this._map.keys().next().value;
      const oldestValue = this._map.get(oldestKey);
      this._map.delete(oldestKey);
      await this._closeFn(oldestValue, oldestKey);
    }

    const value = await this._openFn(key);
    this._map.set(key, value);
    return value;
  }

  async closeAll() {
    for (const [key, value] of this._map.entries()) {
      await this._closeFn(value, key);
    }
    this._map.clear();
  }
}

async function closeWriteStream(ws) {
  if (!ws) return;
  if (ws.destroyed) return;
  ws.end();
  await onceEvent(ws, "finish", "error");
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

function pickShardId(baseRow, shardCount) {
  const n = Math.max(1, shardCount | 0) >>> 0;
  return mixU32(baseRow >>> 0) % n;
}

function makeTripleBufferFromU32(u32, tripleCount) {
  return Buffer.from(u32.buffer, u32.byteOffset, tripleCount * 12);
}

async function appendSpoolTriples(spoolPath, triplesU32, tripleCount) {
  if (!tripleCount) return;
  const buf = makeTripleBufferFromU32(triplesU32, tripleCount);
  await fs.promises.appendFile(spoolPath, buf);
}

function radixSortTriplesByKeysInPlace(triplesU32, work) {
  const n = (triplesU32.length / 3) | 0;
  if (n <= 1) return;

  const wantLen = triplesU32.length | 0;

  let tmp =
    work && work.tmp && work.tmp.length >= wantLen
      ? work.tmp
      : new Uint32Array(wantLen);

  let counts =
    work && work.counts && work.counts.length === 1 << 16
      ? work.counts
      : new Uint32Array(1 << 16);

  function pass16(src, dst, which, shift) {
    counts.fill(0);

    for (let i = 0; i < n; i++) {
      counts[(src[i * 3 + which] >>> shift) & 0xffff]++;
    }

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

  pass16(triplesU32, tmp, 2, 0);
  pass16(tmp, triplesU32, 2, 16);
  pass16(triplesU32, tmp, 1, 0);
  pass16(tmp, triplesU32, 1, 16);
  pass16(triplesU32, tmp, 0, 0);
  pass16(tmp, triplesU32, 0, 16);

  if (work) {
    work.tmp = tmp;
    work.counts = counts;
  }
}

function addU32Clamped(a, b) {
  const aa = a >>> 0;
  const bb = b >>> 0;
  const sum = aa + bb;
  if (!Number.isFinite(sum) || sum >= 0xffffffff) return 0xffffffff;
  return sum >>> 0;
}

function countUniqueRowsInSortedTriples(triplesU32, tripleCount) {
  let uniqueRows = 0;
  let prevRowId = 0xffffffff;

  for (let i = 0; i < tripleCount; i++) {
    const rowId = triplesU32[i * 3 + 1] >>> 0;
    if (i === 0 || rowId !== prevRowId) {
      uniqueRows++;
      prevRowId = rowId;
    }
  }

  return uniqueRows >>> 0;
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

    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
      continue;
    }

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

function buildMissingRowSample(uniqueRowIds, recoveredSet, maxItems) {
  const out = [];
  const cap = Math.max(1, maxItems | 0);

  for (let i = 0; i < uniqueRowIds.length; i++) {
    const rowId = uniqueRowIds[i] >>> 0;
    if (!recoveredSet.has(rowId)) {
      out.push(rowId);
      if (out.length >= cap) break;
    }
  }

  return out;
}

function makeReadbackWarning(mode, baseRow, uniqueRowIds, recoveredSet) {
  const missingSample = buildMissingRowSample(uniqueRowIds, recoveredSet, 8);
  let msg =
    `Incomplete ${mode} export readback for baseRow ${baseRow >>> 0}: ` +
    `recovered ${recoveredSet.size} / ${uniqueRowIds.length} unique row(s)`;

  if (missingSample.length) {
    msg += ` | missing sample=${missingSample.join(",")}`;
  }

  return msg;
}

function getExactGroupOptions() {
  return {
    maxGapBytes: Math.max(
      0,
      Math.min(
        Number(process.env.EXPORT_EXACT_GROUP_GAP_BYTES || 256 << 10) | 0,
        64 << 20,
      ),
    ),
    maxGroupBytes: Math.max(
      64 << 10,
      Math.min(
        Number(process.env.EXPORT_EXACT_GROUP_MAX_BYTES || 4 << 20) | 0,
        128 << 20,
      ),
    ),
    maxGroupRows: Math.max(
      1,
      Math.min(
        Number(process.env.EXPORT_EXACT_GROUP_MAX_ROWS || 4096) | 0,
        1_000_000,
      ),
    ),
  };
}

function getExportSpoolStride(idx, taxa) {
  const fallback = Math.max(1, getBestLookupStride(idx, taxa) | 0) >>> 0;
  if (!taxa || !taxa.hasExactRowStarts) return fallback;

  const envStride = Number(process.env.EXPORT_EXACT_SPOOL_STRIDE || 65536);
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

async function readFileRangeFdSafe(fd, start, endExclusive) {
  const startPos = Number(start);
  const endPos = Number(endExclusive);

  if (
    !Number.isFinite(startPos) ||
    !Number.isFinite(endPos) ||
    startPos < 0 ||
    endPos < startPos
  ) {
    throw new Error(`Bad read range ${start}..${endExclusive}`);
  }

  const len = endPos - startPos;
  if (len <= 0) return Buffer.alloc(0);

  const out = Buffer.allocUnsafe(len);
  let off = 0;

  while (off < len) {
    const { bytesRead } = await fd.read(out, off, len - off, startPos + off);
    if (bytesRead <= 0) break;
    off += bytesRead;
  }

  return off === len ? out : out.subarray(0, off);
}

function createCsvScanState(delimChar) {
  return {
    delim: (String(delimChar || ",").charCodeAt(0) & 255) >>> 0,
    inQuotes: false,
    fieldStart: true,
    sawCR: false,
  };
}

async function locateRawRangesDelimitedFd({
  fd,
  delimiter,
  startOffset,
  startRowId,
  stopRowIdExclusive,
  rowIdsSorted,
  chunkBytes,
  onRange,
}) {
  if (!rowIdsSorted.length) return;

  const want = rowIdsSorted.slice();
  let wantIndex = 0;
  let targetRowId = want[0] >>> 0;

  const chunkSize = Math.max(64 << 10, Number(chunkBytes) | 0);
  const buf = Buffer.allocUnsafe(chunkSize);

  let pos = Number(startOffset);
  let rowId = startRowId >>> 0;
  let rowStartAbs = Number(startOffset);

  const state = createCsvScanState(delimiter);

  while (wantIndex < want.length) {
    if (stopRowIdExclusive != null && rowId >>> 0 >= stopRowIdExclusive >>> 0) {
      break;
    }

    const { bytesRead } = await fd.read(buf, 0, buf.length, pos);
    if (bytesRead <= 0) break;

    const chunkStart = pos;
    pos += bytesRead;

    for (let i = 0; i < bytesRead; i++) {
      const b = buf[i];

      if (state.sawCR) {
        state.sawCR = false;
        if (b === 0x0a) {
          rowStartAbs = chunkStart + i + 1;
          continue;
        }
      }

      if (state.inQuotes) {
        if (b === CSV_QUOTE) {
          const next = buf[i + 1];
          if (next === CSV_QUOTE) {
            i++;
          } else {
            state.inQuotes = false;
            state.fieldStart = false;
          }
        }
        continue;
      }

      if (state.fieldStart && b === CSV_QUOTE) {
        state.inQuotes = true;
        state.fieldStart = false;
        continue;
      }

      if (b === state.delim) {
        state.fieldStart = true;
        continue;
      }

      if (b === 0x0a || b === 0x0d) {
        const rowEndAbs = chunkStart + i + 1;
        if (b === 0x0d) state.sawCR = true;

        if (rowId >>> 0 === targetRowId >>> 0) {
          await onRange(rowId >>> 0, rowStartAbs, rowEndAbs);
          wantIndex++;
          if (wantIndex >= want.length) return;
          targetRowId = want[wantIndex] >>> 0;
        }

        rowId = (rowId + 1) >>> 0;
        rowStartAbs = rowEndAbs;
        state.fieldStart = true;
        state.inQuotes = false;
        continue;
      }

      state.fieldStart = false;
    }
  }

  if (
    wantIndex < want.length &&
    (stopRowIdExclusive == null || rowId >>> 0 < stopRowIdExclusive >>> 0) &&
    rowStartAbs < pos &&
    rowId >>> 0 === targetRowId >>> 0
  ) {
    await onRange(rowId >>> 0, rowStartAbs, pos);
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
    for (let i = 0; i < cols.length; i++) {
      lut[cols[i] | 0] = i;
    }
    return { count: cols.length, lut, map: null };
  }

  const map = new Map();
  for (let i = 0; i < cols.length; i++) {
    map.set(cols[i] | 0, i);
  }
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

      if (wantedSlot(matcher, col) >= 0 && segStart < 0) {
        segStart = i;
      }
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

    if (wantedSlot(matcher, col) >= 0 && segStart < 0) {
      segStart = i;
    }

    fieldStart = false;
  }

  if (!rowDone) {
    finishSeg(rowBuf.length);
    if (col > 0 || totalLen > 0 || parts.length > 0 || rowBuf.length > 0) {
      flushField();
    }
  }

  return out;
}

async function flushWriter(writer) {
  if (!writer.parts || !writer.parts.length) return;
  await writeStreamChunk(writer.ws, Buffer.from(writer.parts.join(""), "utf8"));
  writer.parts.length = 0;
  writer.bytes = 0;
}

async function writeCoordsLineForRun(writerCache, triplesU32, run, line) {
  const lineBytes = Buffer.byteLength(line);
  for (let t = run.start; t < run.end; t++) {
    const speciesIndex = triplesU32[t * 3 + 2] >>> 0;
    const writer = await writerCache.get(speciesIndex);
    writer.parts.push(line);
    writer.bytes += lineBytes;
    if (writer.bytes >= writer.flushBytes) {
      await flushWriter(writer);
    }
  }
}

async function writeRawBufferForRun(writerCache, triplesU32, run, rowBuf) {
  for (let t = run.start; t < run.end; t++) {
    const speciesIndex = triplesU32[t * 3 + 2] >>> 0;
    const writer = await writerCache.get(speciesIndex);
    await writeStreamChunk(writer.ws, rowBuf);
  }
}

async function sortShardWorkerMain() {
  const shardPath = String(workerData.shardPath || "");
  const sortedPath = String(workerData.sortedPath || "");

  const raw = await fs.promises.readFile(shardPath);
  const tripleCount = Math.floor(raw.length / 12);

  if (!tripleCount) {
    await fs.promises.writeFile(sortedPath, Buffer.alloc(0));
    parentPort.postMessage({
      ok: true,
      shardPath,
      sortedPath,
      tripleCount: 0,
      baseRowRuns: 0,
    });
    return;
  }

  const triplesU32 = new Uint32Array(
    raw.buffer,
    raw.byteOffset,
    tripleCount * 3,
  );
  const sortWork = {
    tmp: null,
    counts: new Uint32Array(1 << 16),
  };

  radixSortTriplesByKeysInPlace(triplesU32, sortWork);

  let baseRowRuns = 0;
  let prevBaseRow = 0xffffffff;
  for (let i = 0; i < tripleCount; i++) {
    const baseRow = triplesU32[i * 3] >>> 0;
    if (i === 0 || baseRow !== prevBaseRow) {
      baseRowRuns++;
      prevBaseRow = baseRow;
    }
  }

  ensureDir(path.dirname(sortedPath));
  await fs.promises.writeFile(sortedPath, raw);

  parentPort.postMessage({
    ok: true,
    shardPath,
    sortedPath,
    tripleCount,
    baseRowRuns,
  });
}

if (!isMainThread && workerData && workerData.type === "sortShard") {
  sortShardWorkerMain().catch((err) => {
    if (parentPort) {
      parentPort.postMessage({
        ok: false,
        error: String(err && err.message ? err.message : err),
      });
    }
    process.exit(1);
  });
}

async function collectExportPlan({
  taxa,
  targetNodeIds,
  mode,
  outRoot,
  reporter,
}) {
  const report = typeof reporter === "function" ? reporter : null;
  const rootPlans = [];
  let speciesTotal = 0;
  let rowsTotalEstimate = 0;

  for (let i = 0; i < targetNodeIds.length; i++) {
    const rootNodeId = targetNodeIds[i] >>> 0;
    const rootNode = taxa.readNode(rootNodeId);
    if (!rootNode) continue;

    const rootRank = taxonomyRankName(rootNode.rankId >>> 0);
    const rootName =
      String(taxa.getString(rootNode.nameId) || "(missing)").trim() ||
      "(missing)";

    const rootDir = path.join(
      outRoot,
      sanitizePathSegment(rootRank),
      sanitizePathSegment(rootName),
    );

    const counts = await countSpeciesTargetsUnderNode(taxa, rootNodeId);

    rootPlans.push({
      rootNodeId,
      rootRank,
      rootName,
      rootDir,
      speciesCount: counts.speciesCount >>> 0,
      rowEstimate: Number(counts.rowEstimate || 0),
    });

    speciesTotal += counts.speciesCount >>> 0;
    rowsTotalEstimate += Number(counts.rowEstimate || 0);

    if (report) {
      report({
        phase: "plan",
        message: `Scanned ${i + 1} / ${targetNodeIds.length}: ${rootName}`,
        targetNodeCount: targetNodeIds.length,
        plannedSpeciesCount: speciesTotal,
        speciesTotal,
        rowsTotalEstimate,
        progress:
          targetNodeIds.length > 0
            ? 0.08 * ((i + 1) / targetNodeIds.length)
            : 0.08,
      });
    }

    maybeGC("export-plan-root");
  }

  const speciesPlans = [];

  for (let rp = 0; rp < rootPlans.length; rp++) {
    const rootPlan = rootPlans[rp];

    await walkSpeciesTargetsUnderNode(
      taxa,
      rootPlan.rootNodeId,
      async (target) => {
        const speciesNodeId = target.speciesNodeId >>> 0;
        const speciesNode = taxa.readNode(speciesNodeId);
        const speciesName = speciesNode
          ? String(taxa.getString(speciesNode.nameId) || "(missing)").trim() ||
            "(missing)"
          : String(speciesNodeId);

        const outFile = path.join(
          rootPlan.rootDir,
          ...target.relParts,
          mode === "coords" ? "coords.csv" : "rows.csv",
        );

        speciesPlans.push({
          speciesIndex: speciesPlans.length >>> 0,
          speciesNodeId,
          speciesName,
          outFile,
          rowEstimate:
            speciesNode && Number.isFinite(speciesNode.count)
              ? speciesNode.count >>> 0
              : target.rowEstimate >>> 0,
        });
      },
    );
  }

  return {
    rootPlans,
    speciesPlans,
    speciesTotal: speciesPlans.length >>> 0,
    rowsTotalEstimate,
  };
}

function getEffectiveSpoolShardCount(rowsTotalEstimate) {
  const configuredShardCount = Math.max(
    1,
    Math.min(Number(process.env.EXPORT_SPOOL_SHARDS || 512) | 0, 4096),
  );

  const minShardRecords = Math.max(
    1,
    Math.min(
      Number(process.env.EXPORT_MIN_SHARD_RECORDS || 1000) | 0,
      50_000_000,
    ),
  );

  const estimatedRows = Number(rowsTotalEstimate || 0);

  if (!Number.isFinite(estimatedRows) || estimatedRows <= 0) {
    return configuredShardCount;
  }

  if (estimatedRows <= minShardRecords) {
    return 1;
  }

  const maxShardsAllowedByMinSize = Math.max(
    1,
    Math.floor(estimatedRows / minShardRecords),
  );

  return Math.max(1, Math.min(configuredShardCount, maxShardsAllowedByMinSize));
}

async function spoolExportRecords({
  idx,
  taxa,
  speciesPlans,
  rowsTotalEstimate,
  reporter,
  tmpDir,
}) {
  const report = typeof reporter === "function" ? reporter : null;
  const stride = getExportSpoolStride(idx, taxa);
  const spoolDir = path.join(tmpDir, "spool_shards");
  ensureDir(spoolDir);

  const SPOOL_SHARDS = getEffectiveSpoolShardCount(rowsTotalEstimate);

  const SPOOL_BATCH_TRIPLES = Math.max(
    16_384,
    Math.min(
      Number(process.env.EXPORT_SPOOL_BATCH_TRIPLES || 262_144) | 0,
      2_000_000,
    ),
  );

  const shardPaths = new Array(SPOOL_SHARDS);
  for (let i = 0; i < SPOOL_SHARDS; i++) {
    shardPaths[i] = path.join(
      spoolDir,
      `shard_${String(i).padStart(4, "0")}.bin`,
    );
  }

  let rowsSpooled = 0;
  let speciesDone = 0;
  let lastReportRows = 0;

  for (let i = 0; i < speciesPlans.length; i++) {
    const plan = speciesPlans[i];
    const speciesIndex = plan.speciesIndex >>> 0;

    let triples = new Uint32Array(SPOOL_BATCH_TRIPLES * 3);
    let tripleCount = 0;

    async function flushTriples() {
      if (!tripleCount) return;

      const byShard = new Map();

      for (let t = 0; t < tripleCount; t++) {
        const off = t * 3;
        const baseRow = triples[off] >>> 0;
        const rowId = triples[off + 1] >>> 0;
        const spIndex = triples[off + 2] >>> 0;

        const shardId = pickShardId(baseRow, SPOOL_SHARDS);
        let arr = byShard.get(shardId);
        if (!arr) {
          arr = [];
          byShard.set(shardId, arr);
        }

        arr.push(baseRow, rowId, spIndex);
      }

      for (const [shardId, arr] of byShard.entries()) {
        const u32 = Uint32Array.from(arr);
        await appendSpoolTriples(
          shardPaths[shardId],
          u32,
          (u32.length / 3) | 0,
        );
      }

      tripleCount = 0;
    }

    await streamSpeciesRowIds(taxa, plan.speciesNodeId, async (rowId) => {
      const baseRow = (Math.floor((rowId >>> 0) / stride) * stride) >>> 0;
      const off = tripleCount * 3;

      triples[off] = baseRow >>> 0;
      triples[off + 1] = rowId >>> 0;
      triples[off + 2] = speciesIndex >>> 0;

      tripleCount++;
      rowsSpooled++;

      if (tripleCount >= SPOOL_BATCH_TRIPLES) {
        await flushTriples();
      }

      if (report && rowsSpooled - lastReportRows >= 25000) {
        lastReportRows = rowsSpooled;
        report({
          phase: "spool",
          message: `Indexing export rows for ${plan.speciesName}`,
          speciesDone,
          speciesTotal: speciesPlans.length,
          rowsDone: rowsSpooled,
          rowsTotalEstimate,
          fileCount: 0,
          currentSpecies: plan.speciesName,
          currentSpeciesRowsDone: 0,
          currentSpeciesRowsTotal: plan.rowEstimate >>> 0,
          progress:
            0.08 +
            0.28 *
              clamp01(
                rowsTotalEstimate > 0
                  ? rowsSpooled / rowsTotalEstimate
                  : speciesDone / Math.max(1, speciesPlans.length),
              ),
        });
      }
    });

    await flushTriples();

    speciesDone++;
    if (report) {
      report({
        phase: "spool",
        message: `Indexed ${speciesDone} / ${speciesPlans.length}: ${plan.speciesName}`,
        speciesDone,
        speciesTotal: speciesPlans.length,
        rowsDone: rowsSpooled,
        rowsTotalEstimate,
        fileCount: 0,
        currentSpecies: plan.speciesName,
        currentSpeciesRowsDone: plan.rowEstimate >>> 0,
        currentSpeciesRowsTotal: plan.rowEstimate >>> 0,
        progress:
          0.08 +
          0.28 *
            clamp01(
              rowsTotalEstimate > 0
                ? rowsSpooled / rowsTotalEstimate
                : speciesDone / Math.max(1, speciesPlans.length),
            ),
      });
    }

    maybeGC("export-spool-species");
  }

  const existingShardPaths = [];
  for (let i = 0; i < shardPaths.length; i++) {
    try {
      const st = await fs.promises.stat(shardPaths[i]);
      if (st.size > 0) existingShardPaths.push(shardPaths[i]);
    } catch {}
  }

  return {
    stride,
    spoolDir,
    shardPaths: existingShardPaths,
    rowsSpooled,
    shardCount: existingShardPaths.length,
    exactRowStarts: !!(taxa && taxa.hasExactRowStarts),
  };
}

function buildRowRunMapFromSortedTriples(triplesU32, startTriple, endTriple) {
  const rowRunMap = new Map();

  let i = startTriple;
  while (i < endTriple) {
    const rowId = triplesU32[i * 3 + 1] >>> 0;
    const speciesStart = i;

    i++;
    while (i < endTriple && triplesU32[i * 3 + 1] >>> 0 === rowId) i++;

    rowRunMap.set(rowId, { start: speciesStart, end: i });
  }

  return rowRunMap;
}

async function processCoordsBaseRowRun({
  csvFd,
  idx,
  taxa,
  baseRow,
  triplesU32,
  startTriple,
  endTriple,
  writerCache,
  idColIndex,
  lonColIndex,
  latColIndex,
  rowCounter,
}) {
  const rowRunMap = buildRowRunMapFromSortedTriples(
    triplesU32,
    startTriple,
    endTriple,
  );
  const uniqueRowIds = Array.from(rowRunMap.keys()).sort((a, b) => a - b);

  if (!uniqueRowIds.length) {
    return {
      uniqueRowCount: 0,
      recoveredRowCount: 0,
      exportedRows: 0,
      missingUniqueRows: 0,
      warnings: [],
    };
  }

  const beforeRows = rowCounter.count >>> 0;
  const recoveredSet = new Set();

  if (
    taxa &&
    taxa.hasExactRowStarts &&
    typeof taxa.rowByteRange === "function"
  ) {
    const groups = buildExactRowReadGroups(
      taxa,
      uniqueRowIds,
      getExactGroupOptions(),
    );
    const matcher = createWantedColMatcher([
      idColIndex,
      lonColIndex,
      latColIndex,
    ]);

    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi];
      const groupBuf = await readFileRangeFdSafe(
        csvFd,
        group.startOffset,
        group.endOffset,
      );

      for (let i = 0; i < group.rowIds.length; i++) {
        const rowId = group.rowIds[i] >>> 0;
        const run = rowRunMap.get(rowId);
        if (!run) continue;

        const range = taxa.rowByteRange(rowId);
        if (!range) continue;

        const relStart = Number(range.start) - group.startOffset;
        const relEnd = Number(range.end) - group.startOffset;

        if (
          !Number.isFinite(relStart) ||
          !Number.isFinite(relEnd) ||
          relStart < 0 ||
          relEnd < relStart ||
          relEnd > groupBuf.length
        ) {
          continue;
        }

        const rowBuf = groupBuf.subarray(relStart, relEnd);
        const vals = parseExactCsvRowSelected(rowBuf, idx.delimiter, matcher);
        const line = `${csvEscapeValue(vals[0] || "")},${csvEscapeValue(vals[1] || "")},${csvEscapeValue(vals[2] || "")}\n`;

        recoveredSet.add(rowId);
        await writeCoordsLineForRun(writerCache, triplesU32, run, line);
        rowCounter.count += (run.end - run.start) >>> 0;
      }
    }
  } else {
    const loc = getLookupWindowForBaseRow(idx, taxa, baseRow);

    const rows = await readRowsByRowIdsSelectiveRecordsFd({
      fd: csvFd,
      delimiter: idx.delimiter,
      startOffset: loc.startOffset,
      startRowId: loc.startRowId,
      stopRowIdExclusive: loc.stopRowIdExclusive,
      rowIdsSorted: uniqueRowIds,
      wantCols: [idColIndex, lonColIndex, latColIndex],
      limit: uniqueRowIds.length,
    });

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const rowId = row.rowId >>> 0;
      const run = rowRunMap.get(rowId);
      if (!run) continue;

      recoveredSet.add(rowId);

      const vals = row.values;
      const line = `${csvEscapeValue(vals[0] || "")},${csvEscapeValue(vals[1] || "")},${csvEscapeValue(vals[2] || "")}\n`;

      await writeCoordsLineForRun(writerCache, triplesU32, run, line);
      rowCounter.count += (run.end - run.start) >>> 0;
    }
  }

  const warnings = [];
  const missingUniqueRows = Math.max(
    0,
    uniqueRowIds.length - recoveredSet.size,
  );

  if (missingUniqueRows > 0) {
    const msg = makeReadbackWarning(
      "coords",
      baseRow,
      uniqueRowIds,
      recoveredSet,
    );
    console.warn(msg);
    warnings.push(msg);
  }

  return {
    uniqueRowCount: uniqueRowIds.length >>> 0,
    recoveredRowCount: recoveredSet.size >>> 0,
    exportedRows: (rowCounter.count - beforeRows) >>> 0,
    missingUniqueRows: missingUniqueRows >>> 0,
    warnings,
  };
}

async function processRawBaseRowRun({
  csvFd,
  idx,
  taxa,
  baseRow,
  triplesU32,
  startTriple,
  endTriple,
  writerCache,
  rowCounter,
}) {
  const rowRunMap = buildRowRunMapFromSortedTriples(
    triplesU32,
    startTriple,
    endTriple,
  );
  const uniqueRowIds = Array.from(rowRunMap.keys()).sort((a, b) => a - b);

  if (!uniqueRowIds.length) {
    return {
      uniqueRowCount: 0,
      recoveredRowCount: 0,
      exportedRows: 0,
      missingUniqueRows: 0,
      warnings: [],
    };
  }

  const beforeRows = rowCounter.count >>> 0;
  const recoveredSet = new Set();

  if (
    taxa &&
    taxa.hasExactRowStarts &&
    typeof taxa.rowByteRange === "function"
  ) {
    const groups = buildExactRowReadGroups(
      taxa,
      uniqueRowIds,
      getExactGroupOptions(),
    );

    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi];
      const groupBuf = await readFileRangeFdSafe(
        csvFd,
        group.startOffset,
        group.endOffset,
      );

      for (let i = 0; i < group.rowIds.length; i++) {
        const rowId = group.rowIds[i] >>> 0;
        const run = rowRunMap.get(rowId);
        if (!run) continue;

        const range = taxa.rowByteRange(rowId);
        if (!range) continue;

        const relStart = Number(range.start) - group.startOffset;
        const relEnd = Number(range.end) - group.startOffset;

        if (
          !Number.isFinite(relStart) ||
          !Number.isFinite(relEnd) ||
          relStart < 0 ||
          relEnd < relStart ||
          relEnd > groupBuf.length
        ) {
          continue;
        }

        const rowBuf = groupBuf.subarray(relStart, relEnd);
        recoveredSet.add(rowId);
        await writeRawBufferForRun(writerCache, triplesU32, run, rowBuf);
        rowCounter.count += (run.end - run.start) >>> 0;
      }
    }
  } else {
    const loc = getLookupWindowForBaseRow(idx, taxa, baseRow);

    await locateRawRangesDelimitedFd({
      fd: csvFd,
      delimiter: idx.delimiter,
      startOffset: loc.startOffset,
      startRowId: loc.startRowId,
      stopRowIdExclusive: loc.stopRowIdExclusive,
      rowIdsSorted: uniqueRowIds,
      chunkBytes: 4 << 20,
      onRange: async (rowId, startAbs, endAbs) => {
        const run = rowRunMap.get(rowId >>> 0);
        if (!run) return;

        const rowBuf = await readFileRangeFdSafe(
          csvFd,
          Number(startAbs),
          Number(endAbs),
        );

        recoveredSet.add(rowId >>> 0);
        await writeRawBufferForRun(writerCache, triplesU32, run, rowBuf);
        rowCounter.count += (run.end - run.start) >>> 0;
      },
    });
  }

  const warnings = [];
  const missingUniqueRows = Math.max(
    0,
    uniqueRowIds.length - recoveredSet.size,
  );

  if (missingUniqueRows > 0) {
    const msg = makeReadbackWarning("raw", baseRow, uniqueRowIds, recoveredSet);
    console.warn(msg);
    warnings.push(msg);
  }

  return {
    uniqueRowCount: uniqueRowIds.length >>> 0,
    recoveredRowCount: recoveredSet.size >>> 0,
    exportedRows: (rowCounter.count - beforeRows) >>> 0,
    missingUniqueRows: missingUniqueRows >>> 0,
    warnings,
  };
}

async function processSortedShard({
  sortedPath,
  csvFd,
  idx,
  taxa,
  mode,
  writerCache,
  idColIndex,
  lonColIndex,
  latColIndex,
  rowCounter,
  reporter,
  shardIndex,
  shardCount,
}) {
  const raw = await fs.promises.readFile(sortedPath);
  const tripleCount = Math.floor(raw.length / 12);
  if (!tripleCount) {
    return {
      baseRowRuns: 0,
      uniqueRowsDone: 0,
      uniqueRowsTotal: 0,
      recoveredRows: 0,
      exportedRows: 0,
      missingUniqueRows: 0,
      warningCount: 0,
      warnings: [],
    };
  }

  const triplesU32 = new Uint32Array(
    raw.buffer,
    raw.byteOffset,
    tripleCount * 3,
  );

  const uniqueRowsTotal = countUniqueRowsInSortedTriples(
    triplesU32,
    tripleCount,
  );
  const reportEveryRows = Math.max(
    1,
    Number(process.env.EXPORT_SHARD_REPORT_ROWS || 1024) | 0,
  );

  let baseRowRuns = 0;
  let startTriple = 0;
  let uniqueRowsDone = 0;
  let recoveredRows = 0;
  let exportedRows = 0;
  let missingUniqueRows = 0;
  let warningCount = 0;
  const warnings = [];
  let lastReportedUniqueRows = 0;

  while (startTriple < tripleCount) {
    const baseRow = triplesU32[startTriple * 3] >>> 0;
    let endTriple = startTriple + 1;

    while (
      endTriple < tripleCount &&
      triplesU32[endTriple * 3] >>> 0 === baseRow
    ) {
      endTriple++;
    }

    let stats;
    if (mode === "coords") {
      stats = await processCoordsBaseRowRun({
        csvFd,
        idx,
        taxa,
        baseRow,
        triplesU32,
        startTriple,
        endTriple,
        writerCache,
        idColIndex,
        lonColIndex,
        latColIndex,
        rowCounter,
      });
    } else {
      stats = await processRawBaseRowRun({
        csvFd,
        idx,
        taxa,
        baseRow,
        triplesU32,
        startTriple,
        endTriple,
        writerCache,
        rowCounter,
      });
    }

    baseRowRuns++;
    uniqueRowsDone += stats.uniqueRowCount >>> 0;
    recoveredRows += stats.recoveredRowCount >>> 0;
    exportedRows += stats.exportedRows >>> 0;
    missingUniqueRows += stats.missingUniqueRows >>> 0;
    warningCount += stats.warnings.length >>> 0;

    for (let i = 0; i < stats.warnings.length && warnings.length < 25; i++) {
      warnings.push(stats.warnings[i]);
    }

    startTriple = endTriple;

    if (
      reporter &&
      (uniqueRowsDone - lastReportedUniqueRows >= reportEveryRows ||
        startTriple >= tripleCount)
    ) {
      lastReportedUniqueRows = uniqueRowsDone;

      reporter({
        phase: "export",
        message:
          `Reading shard ${shardIndex} / ${shardCount}: ${path.basename(sortedPath)}` +
          ` | unique rows ${uniqueRowsDone}/${uniqueRowsTotal}` +
          ` | recovered ${recoveredRows}/${uniqueRowsDone}` +
          ` | missing ${missingUniqueRows}` +
          ` | exported rows ${exportedRows}`,
        rowsDone: rowCounter.count,
      });
    }
  }

  return {
    baseRowRuns: baseRowRuns >>> 0,
    uniqueRowsDone: uniqueRowsDone >>> 0,
    uniqueRowsTotal: uniqueRowsTotal >>> 0,
    recoveredRows: recoveredRows >>> 0,
    exportedRows: exportedRows >>> 0,
    missingUniqueRows: missingUniqueRows >>> 0,
    warningCount: warningCount >>> 0,
    warnings,
  };
}

function startSortWorker(shardPath, sortedDir) {
  return new Promise((resolve, reject) => {
    const sortedPath = path.join(
      sortedDir,
      path.basename(shardPath).replace(/\.bin$/i, ".sorted.bin"),
    );

    let settled = false;

    const worker = new Worker(__filename, {
      workerData: {
        type: "sortShard",
        shardPath,
        sortedPath,
      },
    });

    function doneOk(msg) {
      if (settled) return;
      settled = true;
      resolve({
        worker,
        shardPath,
        sortedPath,
        ...msg,
      });
    }

    function doneErr(err) {
      if (settled) return;
      settled = true;
      reject(err);
    }

    worker.once("message", (msg) => {
      if (!msg || msg.ok !== true) {
        doneErr(
          new Error(
            msg && msg.error
              ? msg.error
              : `Sort worker failed for ${shardPath}`,
          ),
        );
        return;
      }
      doneOk(msg);
    });

    worker.once("error", doneErr);

    worker.once("exit", (code) => {
      if (!settled && code !== 0) {
        doneErr(new Error(`Sort worker exited with code ${code}`));
      }
    });
  });
}

async function exportTaxaTargets({
  idx,
  taxa,
  nodeIds,
  mode,
  outRoot,
  reporter,
}) {
  const report = typeof reporter === "function" ? reporter : null;
  const targetNodeIds = dedupeNodeIds(nodeIds || [], taxa.nodeCount >>> 0);

  if (!targetNodeIds.length) {
    throw new Error("No valid nodeIds to export");
  }

  const header = idx.header || [];
  const { idColIndex, idColName } = detectExportIdCol(header);
  const lonColIndex = findColIndex(header, "decimalLongitude");
  const latColIndex = findColIndex(header, "decimalLatitude");

  if (mode !== "coords" && mode !== "raw") {
    throw new Error(`Bad export mode ${mode}`);
  }

  if (mode === "coords" && (lonColIndex < 0 || latColIndex < 0)) {
    throw new Error("Missing decimalLongitude and/or decimalLatitude column");
  }

  const rootOut = path.resolve(outRoot);
  ensureDir(rootOut);

  const plan = await collectExportPlan({
    taxa,
    targetNodeIds,
    mode,
    outRoot: rootOut,
    reporter: report,
  });

  if (!plan.speciesPlans.length) {
    throw new Error("No species found under the requested node selection");
  }

  const tmpDir = path.join(
    rootOut,
    `.export_tmp_${Date.now()}_${process.pid}_${Math.random()
      .toString(36)
      .slice(2, 8)}`,
  );
  ensureDir(tmpDir);

  const spool = await spoolExportRecords({
    idx,
    taxa,
    speciesPlans: plan.speciesPlans,
    rowsTotalEstimate: plan.rowsTotalEstimate,
    reporter: report,
    tmpDir,
  });

  const ownCsvFd = !idx._sharedCsvFd;
  const csvFd = idx._sharedCsvFd || (await fs.promises.open(idx.file, "r"));

  const sortedDir = path.join(tmpDir, "sorted_shards");
  ensureDir(sortedDir);

  const shardPaths = spool.shardPaths.slice();
  const sampleFiles = [];
  const rowCounter = { count: 0 };
  let fileCount = 0;
  let totalBaseRowRunsDone = 0;
  let shardsProcessed = 0;
  let totalUniqueRowsRecovered = 0;
  let totalUniqueRowsSeen = 0;
  let totalMissingUniqueRows = 0;
  let totalWarningCount = 0;
  const warningSamples = [];

  const rawHeaderBytes =
    mode === "raw"
      ? await readFileRangeFdSafe(csvFd, 0, idx.firstDataOffset)
      : null;

  const FLUSH_BYTES = Math.max(
    1 << 20,
    Math.min(
      Number(process.env.EXPORT_WRITER_FLUSH_BYTES || 4 << 20) | 0,
      16 << 20,
    ),
  );

  const SORT_WORKERS = Math.max(
    1,
    Math.min(Number(process.env.EXPORT_SORT_WORKERS || 2) | 0, 8),
  );

  const openWriter = async (speciesIndex) => {
    const planItem = plan.speciesPlans[speciesIndex];
    ensureDir(path.dirname(planItem.outFile));

    const ws = fs.createWriteStream(planItem.outFile, {
      highWaterMark: 4 << 20,
    });

    if (mode === "coords") {
      const head = Buffer.from(
        `${csvEscapeValue(idColName)},${csvEscapeValue("decimalLongitude")},${csvEscapeValue("decimalLatitude")}\n`,
        "utf8",
      );
      await writeStreamChunk(ws, head);
    } else {
      await writeStreamChunk(ws, rawHeaderBytes);
    }

    if (sampleFiles.length < 25) sampleFiles.push(planItem.outFile);
    fileCount++;

    return {
      ws,
      parts: [],
      bytes: 0,
      flushBytes: FLUSH_BYTES,
      path: planItem.outFile,
    };
  };

  const closeWriter = async (writer) => {
    if (!writer) return;
    await flushWriter(writer);
    await closeWriteStream(writer.ws);
  };

  const writerCache = new LruFileWriterCache(
    Math.max(4, Number(process.env.EXPORT_OPEN_WRITERS || 64) | 0),
    openWriter,
    closeWriter,
  );

  try {
    const active = new Map();
    let nextShardIndex = 0;

    function launchMoreWorkers() {
      while (nextShardIndex < shardPaths.length && active.size < SORT_WORKERS) {
        const shardPath = shardPaths[nextShardIndex++];
        const promise = startSortWorker(shardPath, sortedDir);
        active.set(promise, shardPath);
      }
    }

    launchMoreWorkers();

    while (active.size) {
      const wrapped = Array.from(active.keys()).map((p) =>
        p
          .then((result) => ({ ok: true, promise: p, result }))
          .catch((error) => ({
            ok: false,
            promise: p,
            error,
          })),
      );

      const settled = await Promise.race(wrapped);
      active.delete(settled.promise);

      if (!settled.ok) throw settled.error;

      launchMoreWorkers();

      const result = settled.result;
      const shardLabel = path.basename(result.shardPath);

      if (report) {
        report({
          phase: "export",
          message: `Exporting sorted shard ${shardsProcessed + 1} / ${shardPaths.length}: ${shardLabel}`,
          targetNodeCount: targetNodeIds.length,
          plannedSpeciesCount: plan.speciesTotal,
          speciesDone: plan.speciesTotal,
          speciesTotal: plan.speciesTotal,
          rowsDone: rowCounter.count,
          rowsTotalEstimate: plan.rowsTotalEstimate,
          fileCount,
          currentSpecies: "",
          currentSpeciesRowsDone: 0,
          currentSpeciesRowsTotal: 0,
          progress:
            0.36 +
            0.64 *
              clamp01(
                shardPaths.length > 0 ? shardsProcessed / shardPaths.length : 1,
              ),
        });
      }

      const shardStats = await processSortedShard({
        sortedPath: result.sortedPath,
        csvFd,
        idx,
        taxa,
        mode,
        writerCache,
        idColIndex,
        lonColIndex,
        latColIndex,
        rowCounter,
        reporter: report,
        shardIndex: shardsProcessed + 1,
        shardCount: shardPaths.length,
      });

      totalBaseRowRunsDone += shardStats.baseRowRuns;
      totalUniqueRowsRecovered += shardStats.recoveredRows;
      totalUniqueRowsSeen += shardStats.uniqueRowsDone;
      totalMissingUniqueRows += shardStats.missingUniqueRows;
      totalWarningCount += shardStats.warningCount;

      for (
        let i = 0;
        i < shardStats.warnings.length && warningSamples.length < 25;
        i++
      ) {
        warningSamples.push(shardStats.warnings[i]);
      }

      shardsProcessed++;

      await fs.promises.unlink(result.shardPath).catch((e) => {
        console.error(e);
      });
      await fs.promises.unlink(result.sortedPath).catch((e) => {
        console.error(e);
      });

      if (report) {
        report({
          phase: "export",
          message:
            `Processed shard ${shardsProcessed} / ${shardPaths.length}: ${shardLabel}` +
            ` (${shardStats.baseRowRuns} base-row runs, unique rows ${shardStats.uniqueRowsDone}/${shardStats.uniqueRowsTotal}, recovered ${shardStats.recoveredRows}/${shardStats.uniqueRowsDone}, missing ${shardStats.missingUniqueRows}, exported rows ${shardStats.exportedRows})`,
          targetNodeCount: targetNodeIds.length,
          plannedSpeciesCount: plan.speciesTotal,
          speciesDone: plan.speciesTotal,
          speciesTotal: plan.speciesTotal,
          rowsDone: rowCounter.count,
          rowsTotalEstimate: plan.rowsTotalEstimate,
          fileCount,
          currentSpecies: "",
          currentSpeciesRowsDone: 0,
          currentSpeciesRowsTotal: 0,
          progress:
            0.36 +
            0.64 *
              clamp01(
                shardPaths.length > 0 ? shardsProcessed / shardPaths.length : 1,
              ),
        });
      }

      maybeGC("export-shard");
    }
  } finally {
    await writerCache.closeAll().catch((e) => {
      console.error(e);
    });

    if (ownCsvFd) {
      await csvFd.close().catch((e) => {
        console.error(e);
      });
    }

    await fs.promises
      .rm(tmpDir, { recursive: true, force: true })
      .catch((e) => {
        console.error(e);
      });
  }

  const missingRows =
    Math.max(0, (spool.rowsSpooled >>> 0) - (rowCounter.count >>> 0)) >>> 0;

  if (missingRows > 0) {
    const msg =
      `Export finished with missing rows: wrote ${rowCounter.count} / ${spool.rowsSpooled}` +
      ` | missing=${missingRows}`;
    console.warn(msg);
    if (warningSamples.length < 25) warningSamples.push(msg);
    totalWarningCount++;
  }

  const summary = {
    outRoot: rootOut,
    mode,
    targetNodeCount: targetNodeIds.length,
    speciesFileCount: fileCount,
    fileCount,
    rowCount: rowCounter.count,
    rowsTotalEstimate: plan.rowsTotalEstimate,
    rowsSpooled: spool.rowsSpooled,
    missingRows,
    speciesTotal: plan.speciesTotal,
    shardCount: shardPaths.length,
    baseRowRuns: totalBaseRowRunsDone,
    sortWorkers: SORT_WORKERS,
    spoolStride: spool.stride >>> 0,
    exactRowStarts: !!spool.exactRowStarts,
    uniqueRowsRecovered: totalUniqueRowsRecovered >>> 0,
    uniqueRowsSeen: totalUniqueRowsSeen >>> 0,
    missingUniqueRows: totalMissingUniqueRows >>> 0,
    warningCount: totalWarningCount >>> 0,
    warnings: warningSamples,
    files: sampleFiles,
  };

  if (report) {
    report({
      phase: "done",
      message:
        totalWarningCount > 0 || missingRows > 0
          ? `Export complete with warnings: ${fileCount} file(s), ${rowCounter.count} row(s), missing ${missingRows}`
          : `Export complete: ${fileCount} file(s), ${rowCounter.count} row(s)`,
      targetNodeCount: targetNodeIds.length,
      plannedSpeciesCount: plan.speciesTotal,
      speciesDone: plan.speciesTotal,
      speciesTotal: plan.speciesTotal,
      rowsDone: rowCounter.count,
      rowsTotalEstimate: plan.rowsTotalEstimate,
      fileCount,
      currentSpecies: "",
      currentSpeciesRowsDone: 0,
      currentSpeciesRowsTotal: 0,
      result: summary,
      progress: 1,
    });
  }

  return summary;
}

module.exports = {
  detectExportIdCol,
  exportTaxaTargets,
};
