"use strict";

// ./server/csvserver.taxa.js

const {
  fs,
  path,
  TAXA_DIR,
  ROW_STRIDE,
  PAIR_CHUNK_ROWS,
  SKIP_EVERY,
  SEGMENT_MAX_ROWS,
  MERGE_READ_BYTES,
  POSTINGS_READ_BYTES,
  fmtGiB,
  statFile,
  ensureDir,
  maybeGC,
  sanitizePathSegment,
} = require("./csvserver.utils.js");

const {
  chooseHeader,
  findColIndex,
  CsvByteParser,
  findNearestOffset,
  readRowsByRowIdsSelectiveFd,
} = require("./csvserver.core.js");

function onceDrain(ws) {
  return new Promise((resolve) => ws.once("drain", resolve));
}

function finishWritable(ws) {
  return new Promise((resolve, reject) => {
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

function addU32Clamped(a, b) {
  const aa = a >>> 0;
  const bb = b >>> 0;
  const sum = aa + bb;
  if (!Number.isFinite(sum) || sum >= 0xffffffff) return 0xffffffff;
  return sum >>> 0;
}

function sortUniqueU32(values) {
  if (!values || !values.length) return [];
  const arr = values.slice().sort((a, b) => a - b);
  const out = [arr[0] >>> 0];

  for (let i = 1; i < arr.length; i++) {
    const v = arr[i] >>> 0;
    if (v !== out[out.length - 1]) out.push(v);
  }

  return out;
}

function buildExactRowReadGroups(taxa, rowIdsSorted, opts) {
  const groups = [];
  if (!rowIdsSorted.length) return groups;

  const maxGapBytes = Math.max(
    0,
    Math.min(Number(opts.maxGapBytes || (256 << 10)) | 0, 64 << 20),
  );

  const maxGroupBytes = Math.max(
    64 << 10,
    Math.min(Number(opts.maxGroupBytes || (4 << 20)) | 0, 128 << 20),
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

    if (!cur) {
      cur = {
        startOffset: Number(range.start),
        endOffset: Number(range.end),
        startRowId: rowId,
        endRowId: rowId,
        rowIds: [rowId],
      };
      continue;
    }

    const gap = Number(range.start) - cur.endOffset;
    const nextSpan = Number(range.end) - cur.startOffset;

    if (
      gap <= maxGapBytes &&
      nextSpan <= maxGroupBytes &&
      cur.rowIds.length < maxGroupRows
    ) {
      cur.endOffset = Number(range.end);
      cur.endRowId = rowId;
      cur.rowIds.push(rowId);
      continue;
    }

    groups.push(cur);
    cur = {
      startOffset: Number(range.start),
      endOffset: Number(range.end),
      startRowId: rowId,
      endRowId: rowId,
      rowIds: [rowId],
    };
  }

  if (cur) groups.push(cur);
  return groups;
}

async function listChildNodeIds(taxa, parentNodeId) {
  const total = taxa.childCount(parentNodeId >>> 0) >>> 0;
  const pageSize = 10000;
  const out = [];

  for (let start = 0; start < total; start += pageSize) {
    const page = taxa.readChildrenPage(
      parentNodeId >>> 0,
      start,
      Math.min(pageSize, total - start),
    );
    for (let i = 0; i < page.ids.length; i++) out.push(page.ids[i] >>> 0);
  }

  return out;
}

function varintEncodeU32ToParts(parts, n) {
  let x = n >>> 0;
  const tmp = [];
  while (x >= 0x80) {
    tmp.push((x & 0x7f) | 0x80);
    x >>>= 7;
  }
  tmp.push(x);
  const b = Buffer.from(tmp);
  parts.push(b);
  return b.length;
}

function varintDecodeU32(buf, pos) {
  let x = 0;
  let shift = 0;
  let p = pos;
  while (p < buf.length) {
    const b = buf[p++];
    x |= (b & 0x7f) << shift;
    if ((b & 0x80) === 0) return { value: x >>> 0, next: p };
    shift += 7;
    if (shift > 35) break;
  }
  return null;
}

class MinHeap {
  constructor(cmp) {
    this._a = [];
    this._cmp = cmp;
  }

  get size() {
    return this._a.length;
  }

  push(x) {
    const a = this._a;
    a.push(x);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this._cmp(a[i], a[p]) >= 0) break;
      const t = a[i];
      a[i] = a[p];
      a[p] = t;
      i = p;
    }
  }

  pop() {
    const a = this._a;
    if (!a.length) return null;
    const out = a[0];
    const last = a.pop();
    if (!a.length) return out;
    a[0] = last;
    let i = 0;
    for (;;) {
      const l = i * 2 + 1;
      const r = l + 1;
      let best = i;
      if (l < a.length && this._cmp(a[l], a[best]) < 0) best = l;
      if (r < a.length && this._cmp(a[r], a[best]) < 0) best = r;
      if (best === i) break;
      const t = a[i];
      a[i] = a[best];
      a[best] = t;
      i = best;
    }
    return out;
  }
}

function radixSortPairsInPlace(pairsU32, work) {
  const nPairs = (pairsU32.length / 2) | 0;
  if (nPairs <= 1) return;

  const wantLen = pairsU32.length | 0;

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

    if (which === 0) {
      for (let i = 0; i < nPairs; i++) {
        counts[(src[i * 2] >>> shift) & 0xffff]++;
      }
    } else {
      for (let i = 0; i < nPairs; i++) {
        counts[(src[i * 2 + 1] >>> shift) & 0xffff]++;
      }
    }

    let sum = 0;
    for (let i = 0; i < counts.length; i++) {
      const c = counts[i] >>> 0;
      counts[i] = sum >>> 0;
      sum = (sum + c) >>> 0;
    }

    if (which === 0) {
      for (let i = 0; i < nPairs; i++) {
        const sid = src[i * 2] >>> 0;
        const rid = src[i * 2 + 1] >>> 0;
        const k = (sid >>> shift) & 0xffff;
        const pos = (counts[k]++ >>> 0) * 2;
        dst[pos] = sid;
        dst[pos + 1] = rid;
      }
    } else {
      for (let i = 0; i < nPairs; i++) {
        const sid = src[i * 2] >>> 0;
        const rid = src[i * 2 + 1] >>> 0;
        const k = (rid >>> shift) & 0xffff;
        const pos = (counts[k]++ >>> 0) * 2;
        dst[pos] = sid;
        dst[pos + 1] = rid;
      }
    }
  }

  pass16(pairsU32, tmp, 1, 0);
  pass16(tmp, pairsU32, 1, 16);
  pass16(pairsU32, tmp, 0, 0);
  pass16(tmp, pairsU32, 0, 16);

  if (work) {
    work.tmp = tmp;
    work.counts = counts;
  }
}

async function readSpeciesRowIds(taxa, speciesNodeId) {
  const sid = speciesNodeId >>> 0;
  const segs = taxa.readSpeciesSegments(sid);
  if (!segs.length) return [];

  let totalCount = 0;
  for (let i = 0; i < segs.length; i++) {
    totalCount += segs[i].count >>> 0;
  }

  const out = new Array(totalCount);
  let outPos = 0;

  const ownPostingsFd = !taxa._sharedPostingsFd;
  const postingsFd =
    taxa._sharedPostingsFd || (await fs.promises.open(taxa.postingsPath, "r"));

  try {
    const pst = await postingsFd.stat();
    const postingsSize = pst.size;

    const MAX_SKIP_BYTES = 64 << 20;
    const MAX_READ = Math.min(POSTINGS_READ_BYTES | 0, 128 << 20);

    for (let si = 0; si < segs.length; si++) {
      const seg = segs[si];

      const segStart = Number(seg.postingsOffset);
      const segBytes = seg.postingsBytes >>> 0;
      const segEnd = segStart + segBytes;

      if (!Number.isSafeInteger(segStart) || segStart < 0) {
        throw new Error("bad postingsOffset");
      }
      if (!Number.isSafeInteger(segEnd) || segEnd <= segStart) {
        throw new Error("bad postingsBytes");
      }
      if (segStart + 12 > postingsSize) {
        throw new Error("postings segment outside file");
      }

      const headBuf = Buffer.allocUnsafe(12);
      await postingsFd.read(headBuf, 0, 12, segStart);

      const countOnDisk = headBuf.readUInt32LE(0) >>> 0;
      const skipEveryOnDisk = headBuf.readUInt32LE(4) >>> 0;
      const nSkipsOnDisk = headBuf.readUInt32LE(8) >>> 0;

      const count = seg.count >>> 0;
      if (countOnDisk !== count) {
        if (countOnDisk === 0 || countOnDisk > count + 16) {
          throw new Error(
            `segment header count mismatch (disk=${countOnDisk}, dict=${count})`,
          );
        }
      }

      if (skipEveryOnDisk === 0 || skipEveryOnDisk > 1 << 24) {
        throw new Error(`bad skipEvery=${skipEveryOnDisk}`);
      }

      const expectedMaxSkips = Math.ceil(count / skipEveryOnDisk) + 2;
      if (nSkipsOnDisk > expectedMaxSkips) {
        throw new Error(
          `bad nSkips=${nSkipsOnDisk} (expected <= ${expectedMaxSkips})`,
        );
      }

      const skipBytes = (nSkipsOnDisk * 12) >>> 0;
      if (skipBytes > MAX_SKIP_BYTES) {
        throw new Error(`skip table too large: ${skipBytes} bytes`);
      }
      if (12 + skipBytes > segBytes) {
        throw new Error("skip table exceeds segment bytes");
      }

      const streamAbsOffset = segStart + 12 + skipBytes;
      if (streamAbsOffset > segEnd) {
        throw new Error("stream offset beyond segment");
      }

      const skipRowIndex = new Uint32Array(nSkipsOnDisk);
      const skipRowId = new Uint32Array(nSkipsOnDisk);

      if (nSkipsOnDisk > 0) {
        const skipBuf = Buffer.allocUnsafe(skipBytes);
        await postingsFd.read(skipBuf, 0, skipBytes, segStart + 12);

        for (let i = 0; i < nSkipsOnDisk; i++) {
          const off = i * 12;
          skipRowIndex[i] = skipBuf.readUInt32LE(off) >>> 0;
          skipRowId[i] = skipBuf.readUInt32LE(off + 4) >>> 0;
        }
      }

      let nextSkipIdx = 0;
      let nextSkipAt = nSkipsOnDisk > 0 ? skipRowIndex[0] >>> 0 : 0xffffffff;
      let nextSkipRid = nSkipsOnDisk > 0 ? skipRowId[0] >>> 0 : 0;

      let filePos = streamAbsOffset;
      let carry = Buffer.alloc(0);
      let rowIndex = 0;
      let prevRid = 0;

      while (rowIndex < count) {
        const remainInSeg = segEnd - filePos;
        if (remainInSeg <= 0) break;

        const chunkLen = Math.min(MAX_READ, remainInSeg) | 0;
        if (chunkLen <= 0 || chunkLen > 0x7fffffff) {
          throw new Error("bad postings read length");
        }

        const raw = Buffer.allocUnsafe(chunkLen);
        const { bytesRead } = await postingsFd.read(raw, 0, chunkLen, filePos);
        if (bytesRead <= 0) break;

        const rawView = raw.subarray(0, bytesRead);
        const prevCarryLen = carry.length;
        const buf = prevCarryLen
          ? Buffer.concat([carry, rawView], prevCarryLen + rawView.length)
          : rawView;

        let pos = 0;

        while (rowIndex < count) {
          const dec = varintDecodeU32(buf, pos);
          if (!dec) break;

          const delta = dec.value >>> 0;
          pos = dec.next;

          let rid;
          if (rowIndex === nextSkipAt) {
            rid = nextSkipRid >>> 0;
            nextSkipIdx++;
            if (nextSkipIdx < nSkipsOnDisk) {
              nextSkipAt = skipRowIndex[nextSkipIdx] >>> 0;
              nextSkipRid = skipRowId[nextSkipIdx] >>> 0;
            } else {
              nextSkipAt = 0xffffffff;
              nextSkipRid = 0;
            }
          } else if (rowIndex === 0) {
            rid = delta >>> 0;
          } else {
            rid = (prevRid + delta) >>> 0;
          }

          prevRid = rid >>> 0;
          out[outPos++] = rid;
          rowIndex++;
        }

        const consumedFromRaw = Math.max(0, pos - prevCarryLen);
        filePos += consumedFromRaw;
        carry = pos < buf.length ? buf.subarray(pos) : Buffer.alloc(0);

        if (pos === 0 && carry.length > 32) {
          throw new Error("stuck decoding varints (corrupt postings)");
        }
      }

      if (outPos > totalCount) {
        throw new Error("decoded more row ids than expected");
      }
    }
  } finally {
    if (ownPostingsFd) {
      await postingsFd.close().catch((e) => {
        console.error(e);
      });
    }
  }

  return outPos === out.length ? out : out.slice(0, outPos);
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
      if (b === 0x22) {
        finishSeg(i);
        const next = rowBuf[i + 1];
        if (next === 0x22) {
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

    if (fieldStart && b === 0x22) {
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

function resolveProjectedHeaderAndCols(header, requestedColNames) {
  const fullHeader = Array.isArray(header) ? header.slice() : [];

  if (!requestedColNames || !requestedColNames.length) {
    const wantCols = new Array(fullHeader.length);
    for (let i = 0; i < fullHeader.length; i++) wantCols[i] = i;
    return {
      header: fullHeader,
      wantCols,
    };
  }

  const outHeader = [];
  const wantCols = [];
  const seen = new Set();

  for (let i = 0; i < requestedColNames.length; i++) {
    const name = String(requestedColNames[i] || "").trim();
    if (!name) continue;

    const colIndex = findColIndex(fullHeader, name);
    if (colIndex < 0) continue;
    if (seen.has(colIndex)) continue;

    seen.add(colIndex);
    outHeader.push(fullHeader[colIndex]);
    wantCols.push(colIndex);
  }

  if (!wantCols.length) {
    const fallbackLen = Math.min(10, fullHeader.length);
    for (let i = 0; i < fallbackLen; i++) {
      outHeader.push(fullHeader[i]);
      wantCols.push(i);
    }
  }

  return {
    header: outHeader,
    wantCols,
  };
}

async function readExactRowsByRowIdsSelected({
  fd,
  taxa,
  delimiter,
  rowIdsSorted,
  wantCols,
  limit,
}) {
  if (!rowIdsSorted.length || limit <= 0) return [];

  const maxGapBytes = Math.max(
    0,
    Math.min(
      Number(process.env.EXACT_ROW_GROUP_GAP_BYTES || (256 << 10)) | 0,
      64 << 20,
    ),
  );

  const maxGroupBytes = Math.max(
    64 << 10,
    Math.min(
      Number(process.env.EXACT_ROW_GROUP_MAX_BYTES || (4 << 20)) | 0,
      128 << 20,
    ),
  );

  const maxGroupRows = Math.max(
    1,
    Math.min(
      Number(process.env.EXACT_ROW_GROUP_MAX_ROWS || 4096) | 0,
      1_000_000,
    ),
  );

  const groups = buildExactRowReadGroups(taxa, rowIdsSorted, {
    maxGapBytes,
    maxGroupBytes,
    maxGroupRows,
  });

  const matcher = createWantedColMatcher(wantCols);
  const out = [];

  for (let gi = 0; gi < groups.length && out.length < limit; gi++) {
    const group = groups[gi];
    const groupBuf = await readFileRangeFdSafe(
      fd,
      group.startOffset,
      group.endOffset,
    );

    for (let i = 0; i < group.rowIds.length && out.length < limit; i++) {
      const rowId = group.rowIds[i] >>> 0;
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
      out.push(parseExactCsvRowSelected(rowBuf, delimiter, matcher));
    }
  }

  return out;
}

async function walkSpeciesTargetsUnderNode(taxa, rootNodeId, onSpecies) {
  const rootId = rootNodeId >>> 0;
  const rootNode = taxa.readNode(rootId);
  if (!rootNode) throw new Error(`Bad nodeId ${rootNodeId}`);

  const visit = typeof onSpecies === "function" ? onSpecies : async () => {};

  const stack = [{ nodeId: rootId, relParts: [] }];

  while (stack.length) {
    const cur = stack.pop();
    const node = taxa.readNode(cur.nodeId >>> 0);
    if (!node) continue;

    if (node.rankId >>> 0 === 7) {
      await visit({
        speciesNodeId: cur.nodeId >>> 0,
        relParts: cur.relParts.slice(),
        rowEstimate: node.count >>> 0,
      });
      continue;
    }

    const childIds = await listChildNodeIds(taxa, cur.nodeId >>> 0);
    for (let i = childIds.length - 1; i >= 0; i--) {
      const childId = childIds[i] >>> 0;
      const childNode = taxa.readNode(childId);
      if (!childNode) continue;

      stack.push({
        nodeId: childId,
        relParts: cur.relParts.concat(
          sanitizePathSegment(taxa.getString(childNode.nameId)),
        ),
      });
    }
  }
}

async function countSpeciesTargetsUnderNode(taxa, rootNodeId) {
  let speciesCount = 0;
  let rowEstimate = 0;

  await walkSpeciesTargetsUnderNode(
    taxa,
    rootNodeId,
    async ({ rowEstimate: n }) => {
      speciesCount++;
      rowEstimate += Number(n || 0);
    },
  );

  return {
    speciesCount,
    rowEstimate,
  };
}

function dedupeNodeIds(nodeIds, maxNodeCount) {
  const seen = new Set();
  const out = [];

  for (let i = 0; i < nodeIds.length; i++) {
    const n = Number(nodeIds[i]);
    if (!Number.isFinite(n)) continue;
    const id = (n | 0) >>> 0;
    if (id >= maxNodeCount >>> 0) continue;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }

  return out;
}

async function buildTaxaIndexPhase1(idx) {
  ensureDir(TAXA_DIR);

  const metaPath = path.join(TAXA_DIR, "meta.json");
  const nodesPath = path.join(TAXA_DIR, "nodes.tsv");
  const edgesPath = path.join(TAXA_DIR, "edges.bin");
  const pairsPath = path.join(TAXA_DIR, "species_pairs.bin");
  const rowOffsetsPath = path.join(TAXA_DIR, "row_offsets.bin");
  const rowStartsFullPath = path.join(TAXA_DIR, "row_starts_full.bin");

  const header = chooseHeader(idx, "guess");
  const ci = {
    kingdom: findColIndex(header, "kingdom"),
    phylum: findColIndex(header, "phylum"),
    class: findColIndex(header, "class"),
    order: findColIndex(header, "order"),
    family: findColIndex(header, "family"),
    genus: findColIndex(header, "genus"),
    species: findColIndex(header, "species"),
  };

  for (const k of Object.keys(ci)) {
    if (ci[k] < 0) throw new Error(`Missing taxonomy column "${k}" in header.`);
  }

  const st = await statFile(idx.file);
  const fileSize = st.size;

  console.log(`[taxa] phase1 scan: ${idx.file}`);
  console.log(`[taxa] ROW_STRIDE=${ROW_STRIDE}`);
  console.log(
    `[taxa] cols: kingdom=${ci.kingdom} phylum=${ci.phylum} class=${ci.class} order=${ci.order} family=${ci.family} genus=${ci.genus} species=${ci.species}`,
  );

  const nodesWs = fs.createWriteStream(nodesPath);
  const edgesWs = fs.createWriteStream(edgesPath);
  const pairsWs = fs.createWriteStream(pairsPath);
  const rowOffsetsWs = fs.createWriteStream(rowOffsetsPath);
  const rowStartsFullWs = fs.createWriteStream(rowStartsFullPath);

  const RANK = {
    root: 0,
    kingdom: 1,
    phylum: 2,
    class: 3,
    order: 4,
    family: 5,
    genus: 6,
    species: 7,
  };

  let nextNodeId = 1;
  const nodeKeyToId = new Map();

  let counts = new Uint32Array(1 << 20);
  function bump(id) {
    if (id >= counts.length) {
      let n = counts.length;
      while (id >= n) n <<= 1;
      const nu = new Uint32Array(n);
      nu.set(counts);
      counts = nu;
    }
    counts[id] = (counts[id] + 1) >>> 0;
  }

  function makeKey(parentId, rankId, name) {
    return parentId + "\t" + rankId + "\t" + name;
  }

  function normName(nameRaw) {
    const s = (nameRaw || "").trim();
    return s.length ? s : "(missing)";
  }

  function getOrCreateNode(parentId, rankId, nameRaw) {
    const name = normName(nameRaw);
    const key = makeKey(parentId, rankId, name);
    const existing = nodeKeyToId.get(key);
    if (existing != null) return existing;

    const id = nextNodeId++;
    nodeKeyToId.set(key, id);

    nodesWs.write(`${id}\t${parentId}\t${rankId}\t${name}\n`);

    const eb = Buffer.allocUnsafe(8);
    eb.writeUInt32LE(parentId >>> 0, 0);
    eb.writeUInt32LE(id >>> 0, 4);
    edgesWs.write(eb);

    return id;
  }

  bump(0);

  const wantCols = [
    ci.kingdom,
    ci.phylum,
    ci.class,
    ci.order,
    ci.family,
    ci.genus,
    ci.species,
  ];
  const parser = new CsvByteParser(idx.delimiter, wantCols);
  parser.resetOffsets(idx.firstDataOffset);

  const rs = fs.createReadStream(idx.file, {
    start: idx.firstDataOffset,
    highWaterMark: 32 << 20,
  });

  let lastLogT = Date.now();
  let absBytes = BigInt(idx.firstDataOffset);

  function logProgress(rowId) {
    const now = Date.now();
    if (now - lastLogT < 1500) return;
    lastLogT = now;
    const pct = ((Number(absBytes) / fileSize) * 100).toFixed(2);
    process.stdout.write(
      `[taxa] ${fmtGiB(Number(absBytes))} / ${fmtGiB(fileSize)} GiB (${pct}%) rows=${rowId} nodes=${nextNodeId}\r`,
    );
  }

  function writeU64(ws, n) {
    const b = Buffer.allocUnsafe(8);
    b.writeBigUInt64LE(BigInt(n), 0);
    ws.write(b);
  }

  function writeRowOffsetIfNeeded(rowId, rowStartAbs) {
    if (rowId % ROW_STRIDE !== 0) return;
    writeU64(rowOffsetsWs, rowStartAbs);
  }

  function processRow(rowId, rowStartAbs, vals) {
    writeRowOffsetIfNeeded(rowId, rowStartAbs);
    writeU64(rowStartsFullWs, rowStartAbs);

    let p = 0;
    const nKingdom = getOrCreateNode(p, RANK.kingdom, vals[0]);
    p = nKingdom;
    const nPhylum = getOrCreateNode(p, RANK.phylum, vals[1]);
    p = nPhylum;
    const nClass = getOrCreateNode(p, RANK.class, vals[2]);
    p = nClass;
    const nOrder = getOrCreateNode(p, RANK.order, vals[3]);
    p = nOrder;
    const nFamily = getOrCreateNode(p, RANK.family, vals[4]);
    p = nFamily;
    const nGenus = getOrCreateNode(p, RANK.genus, vals[5]);
    p = nGenus;
    const nSpecies = getOrCreateNode(p, RANK.species, vals[6]);
    p = nSpecies;

    bump(0);
    bump(nKingdom);
    bump(nPhylum);
    bump(nClass);
    bump(nOrder);
    bump(nFamily);
    bump(nGenus);
    bump(nSpecies);

    const pb = Buffer.allocUnsafe(8);
    pb.writeUInt32LE(nSpecies >>> 0, 0);
    pb.writeUInt32LE(rowId >>> 0, 4);
    pairsWs.write(pb);
  }

  await new Promise((resolve, reject) => {
    rs.on("data", (buf) => {
      const absStart = Number(absBytes);
      absBytes += BigInt(buf.length);

      const rows = parser.push(buf, absStart);
      for (const r of rows) processRow(r.rowId, r.rowStartAbs, r.values);

      logProgress(parser.rowId);
    });

    rs.on("error", reject);

    rs.on("end", () => {
      const tail = parser.finish();
      for (const r of tail) processRow(r.rowId, r.rowStartAbs, r.values);
      resolve();
    });
  });

  await Promise.all([
    finishWritable(nodesWs),
    finishWritable(edgesWs),
    finishWritable(pairsWs),
    finishWritable(rowOffsetsWs),
    finishWritable(rowStartsFullWs),
  ]);

  const nodeCount = nextNodeId;

  const countsPath = path.join(TAXA_DIR, "counts.bin");
  const outCounts = new Uint32Array(nodeCount);
  outCounts[0] = counts[0] >>> 0;
  for (let i = 1; i < nodeCount; i++) outCounts[i] = counts[i] >>> 0;
  await fs.promises.writeFile(countsPath, Buffer.from(outCounts.buffer));

  const meta = {
    version: 5,
    file: idx.file,
    fileSize: idx.fileSize,
    delimiter: idx.delimiter,
    firstDataOffset: idx.firstDataOffset,
    header: chooseHeader(idx, "guess"),
    colIndex: ci,
    rowStride: ROW_STRIDE,
    rowCount: parser.rowId,
    nodeCount,
    files: {
      nodes: "nodes.tsv",
      edges: "edges.bin",
      counts: "counts.bin",
      pairs: "species_pairs.bin",
      rowOffsets: "row_offsets.bin",
      rowStartsFull: "row_starts_full.bin",
    },
    createdAt: new Date().toISOString(),
  };

  await fs.promises.writeFile(metaPath, JSON.stringify(meta, null, 2));
  console.log(`\n[taxa] phase1 wrote ${metaPath}`);
  console.log(`[taxa] rows=${meta.rowCount} nodes=${meta.nodeCount}`);
  console.log(`[taxa] next: run --finalize-taxa`);
  return meta;
}

async function finalizeTaxaIndexPhase2() {
  const metaPath = path.join(TAXA_DIR, "meta.json");
  const meta = JSON.parse(await fs.promises.readFile(metaPath, "utf8"));

  const pairsPath = path.join(TAXA_DIR, meta.files.pairs);
  const pairsStat = await statFile(pairsPath);
  const totalPairs = Math.floor(pairsStat.size / 8);

  console.log(`[taxa] phase2 sorting pairs: ${pairsPath}`);
  console.log(`[taxa] pairs=${totalPairs} (~${fmtGiB(pairsStat.size)} GiB)`);
  console.log(`[taxa] chunkPairs=${PAIR_CHUNK_ROWS} radixSort16`);

  const tmpDir = path.join(TAXA_DIR, "tmp_sort");
  ensureDir(tmpDir);

  const MERGE_FANIN = Math.max(2, Number(process.env.MERGE_FANIN || 8));

  const MERGE_READ_BYTES_CAP = Math.max(
    64 << 10,
    Math.min(Number(process.env.MERGE_READ_BYTES_CAP || 512 << 10), 8 << 20),
  );

  const MERGE_BUF_BYTES =
    Math.max(
      64 << 10,
      Math.min(MERGE_READ_BYTES | 0, MERGE_READ_BYTES_CAP | 0),
    ) & ~7;

  const OUT_BATCH_BYTES =
    Math.max(1 << 20, Number(process.env.MERGE_OUT_BATCH_BYTES || (4 << 20))) &
    ~7;

  console.log(
    `[taxa] merge tuning: MERGE_FANIN=${MERGE_FANIN} MERGE_BUF_BYTES=${MERGE_BUF_BYTES} OUT_BATCH_BYTES=${OUT_BATCH_BYTES}`,
  );

  function makeWriteBuf(ws, outerReject) {
    let errored = false;
    ws.once("error", (e) => {
      errored = true;
      outerReject(e);
    });

    return async function writeBuf(buf) {
      if (errored) return;
      if (!buf || buf.length === 0) return;
      if (!ws.write(buf)) await onceDrain(ws);
    };
  }

  async function mergeSortedPairFiles(
    inFiles,
    outFile,
    expectedPairs,
    logPrefix,
  ) {
    const out = fs.createWriteStream(outFile, { highWaterMark: 4 << 20 });

    let resolveClose = null;
    let rejectClose = null;
    const closed = new Promise((resolve, reject) => {
      resolveClose = resolve;
      rejectClose = reject;
    });

    out.once("close", () => resolveClose());
    out.once("finish", () => resolveClose());
    out.once("error", (e) => rejectClose(e));

    const writeBuf = makeWriteBuf(out, rejectClose);

    const streams = [];
    for (let i = 0; i < inFiles.length; i++) {
      const p = inFiles[i];
      const st = await statFile(p);
      const h = await fs.promises.open(p, "r");
      streams.push({
        path: p,
        fd: h,
        size: st.size,
        pos: 0,
        buf: Buffer.allocUnsafe(MERGE_BUF_BYTES),
        bufLen: 0,
        bufPos: 0,
        sid: 0,
        rid: 0,
        alive: true,
      });
    }

    async function refill(s) {
      if (!s.alive) return false;
      if (s.pos >= s.size) {
        s.alive = false;
        return false;
      }
      const want = Math.min(s.buf.length, s.size - s.pos);
      const { bytesRead } = await s.fd.read(s.buf, 0, want, s.pos);
      if (bytesRead <= 0) {
        s.alive = false;
        return false;
      }
      s.pos += bytesRead;
      s.bufLen = bytesRead - (bytesRead % 8);
      s.bufPos = 0;
      if (s.bufLen === 0) return refill(s);
      return true;
    }

    function pullRecord(s) {
      if (!s.alive) return false;
      if (s.bufPos + 8 > s.bufLen) return false;
      s.sid = s.buf.readUInt32LE(s.bufPos);
      s.rid = s.buf.readUInt32LE(s.bufPos + 4);
      s.bufPos += 8;
      return true;
    }

    const heap = new MinHeap((a, b) => {
      if (a.sid !== b.sid) return a.sid - b.sid;
      return a.rid - b.rid;
    });

    for (const s of streams) {
      await refill(s);
      if (pullRecord(s)) heap.push(s);
      else s.alive = false;
    }

    let merged = 0;

    let batchA = Buffer.allocUnsafe(OUT_BATCH_BYTES);
    let batchB = Buffer.allocUnsafe(OUT_BATCH_BYTES);
    let batch = batchA;
    let batchPos = 0;

    async function flushBatch() {
      if (batchPos <= 0) return;
      const view = batch.subarray(0, batchPos);
      batch = batch === batchA ? batchB : batchA;
      batchPos = 0;
      await writeBuf(view);
    }

    try {
      while (heap.size) {
        const s = heap.pop();
        if (!s) break;

        batch.writeUInt32LE(s.sid >>> 0, batchPos);
        batch.writeUInt32LE(s.rid >>> 0, batchPos + 4);
        batchPos += 8;
        merged++;

        if (batchPos >= OUT_BATCH_BYTES - 8) await flushBatch();

        if (!pullRecord(s)) {
          const ok = await refill(s);
          if (ok && pullRecord(s)) heap.push(s);
          else s.alive = false;
        } else {
          heap.push(s);
        }

        if (merged % 10_000_000 === 0) {
          if (expectedPairs) {
            console.log(
              `${logPrefix} merged ${merged} / ${expectedPairs} (${((merged / expectedPairs) * 100).toFixed(2)}%)`,
            );
          } else {
            console.log(`${logPrefix} merged ${merged}`);
          }
          maybeGC(`${logPrefix}merge${merged}`);
        }
      }

      await flushBatch();
      out.end();
      await closed;
    } finally {
      for (const s of streams) {
        await s.fd.close().catch((e) => {
          console.error(e);
        });
      }
    }

    return merged >>> 0;
  }

  async function multiPassMerge(files, finalOutPath) {
    let cur = files.slice();
    let pass = 0;

    while (cur.length > 1) {
      const next = [];
      for (let i = 0; i < cur.length; i += MERGE_FANIN) {
        const group = cur.slice(i, i + MERGE_FANIN);

        let expectedPairs = 0;
        for (const p of group) {
          const st = await statFile(p);
          expectedPairs += Math.floor(st.size / 8);
        }

        const outPath = path.join(
          tmpDir,
          `merge_p${String(pass).padStart(2, "0")}_${String(next.length).padStart(4, "0")}.bin`,
        );

        await mergeSortedPairFiles(
          group,
          outPath,
          expectedPairs,
          `[taxa] p${pass}:${next.length}`,
        );

        for (const p of group) {
          await fs.promises.unlink(p).catch((e) => {
            console.error(e);
          });
        }

        next.push(outPath);
        maybeGC(`mergePass${pass}`);
      }

      cur = next;
      pass++;
    }

    if (cur.length === 1) {
      await fs.promises.rename(cur[0], finalOutPath).catch(async () => {
        await fs.promises.copyFile(cur[0], finalOutPath);
        await fs.promises.unlink(cur[0]).catch((e) => {
          console.error(e);
        });
      });
    }
  }

  const chunkFiles = [];
  const sortWork = { tmp: null, counts: new Uint32Array(1 << 16) };

  const fd = await fs.promises.open(pairsPath, "r");
  try {
    let readPairs = 0;
    let chunkIdx = 0;

    while (readPairs < totalPairs) {
      const n = Math.min(PAIR_CHUNK_ROWS, totalPairs - readPairs);
      const buf = Buffer.allocUnsafe(n * 8);

      const { bytesRead } = await fd.read(buf, 0, buf.length, readPairs * 8);
      const actualPairs = Math.floor(bytesRead / 8);
      if (actualPairs <= 0) break;

      const view = buf.subarray(0, actualPairs * 8);
      const u32 = new Uint32Array(
        view.buffer,
        view.byteOffset,
        actualPairs * 2,
      );

      radixSortPairsInPlace(u32, sortWork);

      const chunkPath = path.join(
        tmpDir,
        `chunk_${String(chunkIdx).padStart(5, "0")}.bin`,
      );

      await fs.promises.writeFile(chunkPath, view);
      chunkFiles.push(chunkPath);

      readPairs += actualPairs;
      chunkIdx++;

      console.log(
        `[taxa] chunk ${chunkIdx} wrote ${chunkPath} rows=${actualPairs} progress=${((readPairs / totalPairs) * 100).toFixed(2)}%`,
      );

      maybeGC(`chunk${chunkIdx}`);
    }
  } finally {
    await fd.close();
  }

  const sortedPairsPath = path.join(TAXA_DIR, "species_pairs.sorted.bin");
  console.log(
    `[taxa] merging ${chunkFiles.length} chunks -> ${sortedPairsPath}`,
  );

  if (chunkFiles.length === 1) {
    await fs.promises.rename(chunkFiles[0], sortedPairsPath).catch(async () => {
      await fs.promises.copyFile(chunkFiles[0], sortedPairsPath);
      await fs.promises.unlink(chunkFiles[0]).catch((e) => {
        console.error(e);
      });
    });
  } else {
    await multiPassMerge(chunkFiles, sortedPairsPath);
  }

  await fs.promises.rmdir(tmpDir).catch((e) => {
    console.error(e);
  });
  maybeGC("post-merge");

  const nodesTsvPath = path.join(TAXA_DIR, meta.files.nodes);
  const countsPath = path.join(TAXA_DIR, meta.files.counts);
  const edgesPath = path.join(TAXA_DIR, meta.files.edges);

  const stringsPath = path.join(TAXA_DIR, "strings.bin");
  const namesBodyPath = path.join(TAXA_DIR, "names_body.tmp");
  const nodesBinPath = path.join(TAXA_DIR, "nodes.bin");
  const childrenOffsetsPath = path.join(TAXA_DIR, "children_offsets.bin");
  const childrenPath = path.join(TAXA_DIR, "children.bin");

  const nodeCount = meta.nodeCount >>> 0;

  const parentId = new Uint32Array(nodeCount);
  const rankId = new Uint32Array(nodeCount);
  const nameId = new Uint32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) nameId[i] = i >>> 0;

  const countsBufRaw = await fs.promises.readFile(countsPath);
  const counts = new Uint32Array(
    countsBufRaw.buffer,
    countsBufRaw.byteOffset,
    (countsBufRaw.length / 4) | 0,
  );

  parentId[0] = 0;
  rankId[0] = 0;

  const nameOffsets = Buffer.allocUnsafe((nodeCount + 1) * 8);
  let nameBytePos = 0n;
  nameOffsets.writeBigUInt64LE(nameBytePos, 0);

  const namesBodyWs = fs.createWriteStream(namesBodyPath, {
    highWaterMark: 8 << 20,
  });

  function writeNameForNode(id, nameStr) {
    const s = nameStr || "";
    const b = Buffer.from(s, "utf8");
    namesBodyWs.write(b);
    namesBodyWs.write(Buffer.from([0]));
    nameBytePos += BigInt(b.length + 1);
    nameOffsets.writeBigUInt64LE(nameBytePos, (id + 1) * 8);
  }

  writeNameForNode(0, "");

  {
    const rs = fs.createReadStream(nodesTsvPath, {
      encoding: "utf8",
      highWaterMark: 4 << 20,
    });
    let tail = "";
    for await (const chunk of rs) {
      const s = tail + chunk;
      const lines = s.split("\n");
      tail = lines.pop() || "";
      for (const line of lines) {
        if (!line) continue;
        const a = line.split("\t");
        if (a.length < 4) continue;
        const id = Number(a[0]) >>> 0;
        if (id === 0 || id >= nodeCount) continue;
        parentId[id] = Number(a[1]) >>> 0;
        rankId[id] = Number(a[2]) >>> 0;
        const nm = a.slice(3).join("\t");
        writeNameForNode(id, nm);
      }
    }
    if (tail) {
      const a = tail.split("\t");
      if (a.length >= 4) {
        const id = Number(a[0]) >>> 0;
        if (id !== 0 && id < nodeCount) {
          parentId[id] = Number(a[1]) >>> 0;
          rankId[id] = Number(a[2]) >>> 0;
          const nm = a.slice(3).join("\t");
          writeNameForNode(id, nm);
        }
      }
    }
  }

  await new Promise((resolve, reject) => {
    namesBodyWs.end();
    namesBodyWs.once("finish", resolve);
    namesBodyWs.once("error", reject);
  });

  {
    const stringsWs = fs.createWriteStream(stringsPath, {
      highWaterMark: 4 << 20,
    });

    await new Promise((resolve, reject) => {
      stringsWs.once("error", reject);
      const head = Buffer.allocUnsafe(4);
      head.writeUInt32LE(nodeCount >>> 0, 0);

      stringsWs.write(head, (err) => {
        if (err) return reject(err);
        stringsWs.write(nameOffsets, (err2) => {
          if (err2) return reject(err2);

          const bodyRs = fs.createReadStream(namesBodyPath, {
            highWaterMark: 8 << 20,
          });
          bodyRs.once("error", reject);
          bodyRs.pipe(stringsWs, { end: true });
          stringsWs.once("finish", resolve);
        });
      });
    });

    await fs.promises.unlink(namesBodyPath).catch((e) => {
      console.error(e);
    });
  }

  const childCounts = new Uint32Array(nodeCount);

  {
    const rs = fs.createReadStream(edgesPath, { highWaterMark: 64 << 20 });
    let carry = Buffer.alloc(0);

    for await (const chunk of rs) {
      const buf = carry.length
        ? Buffer.concat([carry, chunk], carry.length + chunk.length)
        : chunk;

      const usableBytes = buf.length - (buf.length % 8);

      for (let off = 0; off < usableBytes; off += 8) {
        const p = buf.readUInt32LE(off);
        childCounts[p] = (childCounts[p] + 1) >>> 0;
      }

      carry =
        usableBytes === buf.length
          ? Buffer.alloc(0)
          : buf.subarray(usableBytes);
    }

    if (carry.length) {
      throw new Error(
        `edges.bin truncated: ${carry.length} dangling bytes (not multiple of 8)`,
      );
    }
  }

  const childOffsets = new Uint32Array(nodeCount + 1);
  {
    let sum = 0;
    for (let i = 0; i < nodeCount; i++) {
      childOffsets[i] = sum >>> 0;
      sum += childCounts[i] >>> 0;
    }
    childOffsets[nodeCount] = sum >>> 0;
  }

  const totalEdges = childOffsets[nodeCount] >>> 0;
  const children = new Uint32Array(totalEdges);
  const writeCursor = new Uint32Array(nodeCount);
  writeCursor.set(childOffsets.subarray(0, nodeCount));

  {
    const rs = fs.createReadStream(edgesPath, { highWaterMark: 64 << 20 });
    let carry = Buffer.alloc(0);

    for await (const chunk of rs) {
      const buf = carry.length
        ? Buffer.concat([carry, chunk], carry.length + chunk.length)
        : chunk;

      const usableBytes = buf.length - (buf.length % 8);

      for (let off = 0; off < usableBytes; off += 8) {
        const p = buf.readUInt32LE(off);
        const c = buf.readUInt32LE(off + 4);

        const pos = writeCursor[p] >>> 0;
        children[pos] = c >>> 0;
        writeCursor[p] = (pos + 1) >>> 0;
      }

      carry =
        usableBytes === buf.length
          ? Buffer.alloc(0)
          : buf.subarray(usableBytes);
    }

    if (carry.length) {
      throw new Error(
        `edges.bin truncated: ${carry.length} dangling bytes (not multiple of 8)`,
      );
    }
  }

  for (let id = 0; id < nodeCount; id++) {
    const a = childOffsets[id] >>> 0;
    const b = childOffsets[id + 1] >>> 0;
    if (b - a > 1) children.subarray(a, b).sort();
  }

  {
    const offsWs = fs.createWriteStream(childrenOffsetsPath, {
      highWaterMark: 4 << 20,
    });
    const kidsWs = fs.createWriteStream(childrenPath, {
      highWaterMark: 8 << 20,
    });

    async function writeAll(ws, buf) {
      if (!ws.write(buf)) await onceDrain(ws);
    }

    let bytePos = 0n;

    for (let id = 0; id < nodeCount; id++) {
      const ob = Buffer.allocUnsafe(8);
      ob.writeBigUInt64LE(bytePos, 0);
      await writeAll(offsWs, ob);

      const a = childOffsets[id];
      const b = childOffsets[id + 1];
      const n = (b - a) >>> 0;

      const rec = Buffer.allocUnsafe(4 + n * 4);
      rec.writeUInt32LE(n, 0);
      let p = 4;
      for (let i = 0; i < n; i++) {
        rec.writeUInt32LE(children[a + i] >>> 0, p);
        p += 4;
      }

      await writeAll(kidsWs, rec);
      bytePos += BigInt(rec.length);
    }

    {
      const ob = Buffer.allocUnsafe(8);
      ob.writeBigUInt64LE(bytePos, 0);
      await writeAll(offsWs, ob);
    }

    await Promise.all([finishWritable(offsWs), finishWritable(kidsWs)]);
  }

  {
    const buf = Buffer.allocUnsafe(4 + nodeCount * 16);
    buf.writeUInt32LE(nodeCount >>> 0, 0);
    let p = 4;
    for (let id = 0; id < nodeCount; id++) {
      buf.writeUInt32LE(parentId[id] >>> 0, p);
      p += 4;
      buf.writeUInt32LE(rankId[id] >>> 0, p);
      p += 4;
      buf.writeUInt32LE(nameId[id] >>> 0, p);
      p += 4;
      buf.writeUInt32LE((counts[id] || 0) >>> 0, p);
      p += 4;
    }
    await fs.promises.writeFile(nodesBinPath, buf);
  }

  maybeGC("post-strings-children");

  const postingsPath = path.join(TAXA_DIR, "species_postings.bin");
  const dictPath = path.join(TAXA_DIR, "species_dict.bin");

  const postOut = await fs.promises.open(postingsPath, "w");
  const dictOut = await fs.promises.open(dictPath, "w");

  async function writeAllAt(fdh, buf, pos) {
    let off = 0;
    while (off < buf.length) {
      const { bytesWritten } = await fdh.write(
        buf,
        off,
        buf.length - off,
        pos + off,
      );
      if (bytesWritten <= 0) throw new Error("short write");
      off += bytesWritten;
    }
  }

  class VarintChunkWriter {
    constructor(chunkBytes) {
      const raw = Number(chunkBytes);
      const v = Number.isFinite(raw) ? raw : 1 << 20;
      const clamped = (Math.max(64 << 10, Math.min(v | 0, 8 << 20)) & ~7) >>> 0;

      this._chunkBytes = clamped;
      this._pool = [];
      this._poolCap = 128;

      this._chunks = [];
      this._buf = Buffer.allocUnsafe(this._chunkBytes);
      this._pos = 0;
      this._total = 0;
    }

    reset() {
      for (let i = 0; i < this._chunks.length; i++) {
        const b = this._chunks[i].buf;
        if (this._pool.length < this._poolCap) this._pool.push(b);
      }
      this._chunks.length = 0;
      this._pos = 0;
      this._total = 0;
    }

    get byteLength() {
      return (this._total + this._pos) >>> 0;
    }

    _takeBuf() {
      const b = this._pool.pop();
      return b ? b : Buffer.allocUnsafe(this._chunkBytes);
    }

    _pushFull() {
      if (this._pos <= 0) return;
      this._chunks.push({ buf: this._buf, len: this._pos });
      this._total = (this._total + this._pos) >>> 0;
      this._buf = this._takeBuf();
      this._pos = 0;
    }

    writeVarintU32(n) {
      let x = n >>> 0;
      let wrote = 0;

      if (this._pos + 5 > this._buf.length) this._pushFull();

      while (x >= 0x80) {
        this._buf[this._pos++] = (x & 0x7f) | 0x80;
        wrote++;
        x >>>= 7;
        if (this._pos >= this._buf.length) this._pushFull();
      }

      this._buf[this._pos++] = x & 0xff;
      wrote++;
      if (this._pos >= this._buf.length) this._pushFull();

      return wrote;
    }

    async writeToFd(fdh, pos) {
      let p = pos;

      for (let i = 0; i < this._chunks.length; i++) {
        const c = this._chunks[i];
        const view = c.buf.subarray(0, c.len);
        await writeAllAt(fdh, view, p);
        p += view.length;

        if (this._pool.length < this._poolCap) this._pool.push(c.buf);
      }
      this._chunks.length = 0;

      if (this._pos > 0) {
        const view = this._buf.subarray(0, this._pos);
        await writeAllAt(fdh, view, p);
        p += view.length;
        this._pos = 0;
      }

      this._total = 0;
      return p;
    }
  }

  const POSTINGS_VARINT_CHUNK =
    Math.max(
      64 << 10,
      Math.min(
        Number(process.env.POSTINGS_VARINT_CHUNK || (1 << 20)) | 0,
        8 << 20,
      ),
    ) & ~7;

  const varWriter = new VarintChunkWriter(POSTINGS_VARINT_CHUNK);

  let postPos = 0;
  let dictPos = 0;

  const sortedFd = await fs.promises.open(sortedPairsPath, "r");
  try {
    const READ_CHUNK = (64 << 20) & ~7;
    const readBuf = Buffer.allocUnsafe(READ_CHUNK);

    const carry = Buffer.allocUnsafe(8);
    let carryLen = 0;

    let filePos = 0;

    let curSid = 0xffffffff;
    let segIndex = 0;
    let segCount = 0;
    let prevRid = 0;

    let byteCursor = 0;
    let skips = [];

    async function flushSegment() {
      if (curSid === 0xffffffff || segCount === 0) return;

      const nSkips = skips.length >>> 0;
      const headerBytes = (12 + nSkips * 12) >>> 0;

      const head = Buffer.allocUnsafe(headerBytes);
      head.writeUInt32LE(segCount >>> 0, 0);
      head.writeUInt32LE(SKIP_EVERY >>> 0, 4);
      head.writeUInt32LE(nSkips >>> 0, 8);

      let hp = 12;
      for (let i = 0; i < nSkips; i++) {
        const s = skips[i];
        head.writeUInt32LE(s.rowIndex >>> 0, hp);
        hp += 4;
        head.writeUInt32LE(s.rowId >>> 0, hp);
        hp += 4;
        head.writeUInt32LE(s.byteOffset >>> 0, hp);
        hp += 4;
      }

      const postingsOffset = postPos;
      await writeAllAt(postOut, head, postPos);
      postPos += head.length;

      postPos = await varWriter.writeToFd(postOut, postPos);

      const postingsBytes = (headerBytes + byteCursor) >>> 0;

      const rec = Buffer.allocUnsafe(32);
      rec.writeUInt32LE(curSid >>> 0, 0);
      rec.writeUInt32LE(segIndex >>> 0, 4);
      rec.writeUInt32LE(segCount >>> 0, 8);
      rec.writeUInt32LE(nameId[curSid] >>> 0, 12);
      rec.writeBigUInt64LE(BigInt(postingsOffset), 16);
      rec.writeUInt32LE(postingsBytes >>> 0, 24);
      rec.writeUInt32LE(0, 28);

      await writeAllAt(dictOut, rec, dictPos);
      dictPos += rec.length;

      segIndex++;
      segCount = 0;
      prevRid = 0;
      byteCursor = 0;
      skips = [];
      varWriter.reset();
    }

    async function switchSpecies(nextSid) {
      await flushSegment();
      curSid = nextSid >>> 0;
      segIndex = 0;
      segCount = 0;
      prevRid = 0;
      byteCursor = 0;
      skips = [];
      varWriter.reset();
    }

    while (true) {
      if (carryLen > 0) carry.copy(readBuf, 0, 0, carryLen);

      const { bytesRead } = await sortedFd.read(
        readBuf,
        carryLen,
        readBuf.length - carryLen,
        filePos,
      );

      if (bytesRead <= 0) break;
      filePos += bytesRead;

      const total = (carryLen + bytesRead) | 0;
      const usable = total - (total % 8);
      const rem = total - usable;

      if (rem > 0) {
        readBuf.copy(carry, 0, usable, total);
        carryLen = rem;
      } else {
        carryLen = 0;
      }

      for (let off = 0; off < usable; off += 8) {
        const sid = readBuf.readUInt32LE(off);
        const rid = readBuf.readUInt32LE(off + 4);

        if (sid !== curSid) await switchSpecies(sid);

        if (segCount >= SEGMENT_MAX_ROWS) await flushSegment();

        if (segCount > 0 && rid === prevRid) continue;

        const isSkip = segCount % SKIP_EVERY === 0;
        if (isSkip) {
          skips.push({
            rowIndex: segCount >>> 0,
            rowId: rid >>> 0,
            byteOffset: byteCursor >>> 0,
          });
        }

        const delta = isSkip ? 0 : (rid - prevRid) >>> 0;
        prevRid = rid >>> 0;
        segCount++;

        byteCursor = (byteCursor + varWriter.writeVarintU32(delta)) >>> 0;
      }

      maybeGC("postings-loop");
    }

    if (carryLen !== 0) {
      throw new Error(`sortedPairs tail not aligned: carryLen=${carryLen}`);
    }

    await flushSegment();
  } finally {
    await sortedFd.close().catch((e) => {
      console.error(e);
    });
    await postOut.close().catch((e) => {
      console.error(e);
    });
    await dictOut.close().catch((e) => {
      console.error(e);
    });
  }

  meta.phase2 = {
    sortedPairs: path.basename(sortedPairsPath),
    strings: path.basename(stringsPath),
    nodes: path.basename(nodesBinPath),
    children: path.basename(childrenPath),
    childrenOffsets: path.basename(childrenOffsetsPath),
    postings: path.basename(postingsPath),
    dict: path.basename(dictPath),
    skipEvery: SKIP_EVERY,
    segmentMaxRows: SEGMENT_MAX_ROWS,
    finalizedAt: new Date().toISOString(),
  };

  await fs.promises.writeFile(metaPath, JSON.stringify(meta, null, 2));

  console.log(`[taxa] phase2 wrote:`);
  console.log(`  ${dictPath}`);
  console.log(`  ${postingsPath}`);
  console.log(`  ${nodesBinPath}`);
  console.log(`  ${childrenPath}`);
  console.log(`[taxa] meta updated: ${metaPath}`);
}

async function loadTaxaIndex() {
  const metaPath = path.join(TAXA_DIR, "meta.json");
  const meta = JSON.parse(await fs.promises.readFile(metaPath, "utf8"));
  if (!meta.phase2) return null;

  const nodesBuf = await fs.promises.readFile(
    path.join(TAXA_DIR, meta.phase2.nodes),
  );
  const nodeCount = nodesBuf.readUInt32LE(0);

  function readNode(id) {
    const off = 4 + id * 16;
    if (off + 16 > nodesBuf.length) return null;
    return {
      id,
      parentId: nodesBuf.readUInt32LE(off),
      rankId: nodesBuf.readUInt32LE(off + 4),
      nameId: nodesBuf.readUInt32LE(off + 8),
      count: nodesBuf.readUInt32LE(off + 12),
    };
  }

  const stringsBuf = await fs.promises.readFile(
    path.join(TAXA_DIR, meta.phase2.strings),
  );
  const stringCount = stringsBuf.readUInt32LE(0);

  const offsetsBase = 4;
  const offsetsBytes = (stringCount + 1) * 8;
  const stringBase = offsetsBase + offsetsBytes;

  function getString(id) {
    if (id < 0 || id >= stringCount) return "";
    const a64 = stringsBuf.readBigUInt64LE(offsetsBase + id * 8);
    const b64 = stringsBuf.readBigUInt64LE(offsetsBase + (id + 1) * 8);
    const a = Number(a64);
    const b = Number(b64);
    if (!Number.isSafeInteger(a) || !Number.isSafeInteger(b)) {
      throw new Error("strings.bin offsets exceed JS safe integer range");
    }
    const start = stringBase + a;
    const endRaw = stringBase + b;
    let end = endRaw;
    while (end > start && stringsBuf[end - 1] === 0) end--;
    return stringsBuf.subarray(start, end).toString("utf8");
  }

  const childOffsetsBuf = await fs.promises.readFile(
    path.join(TAXA_DIR, meta.phase2.childrenOffsets),
  );
  const childrenBuf = await fs.promises.readFile(
    path.join(TAXA_DIR, meta.phase2.children),
  );

  function childrenRecordOffset(nodeId) {
    const offA = Number(childOffsetsBuf.readBigUInt64LE(nodeId * 8));
    const offB = Number(childOffsetsBuf.readBigUInt64LE((nodeId + 1) * 8));
    if (!Number.isSafeInteger(offA) || !Number.isSafeInteger(offB))
      return { offA: 0, offB: 0, ok: false };
    if (offA < 0 || offB > childrenBuf.length || offB <= offA)
      return { offA: 0, offB: 0, ok: false };
    return { offA, offB, ok: true };
  }

  function childCount(nodeId) {
    const { offA, ok } = childrenRecordOffset(nodeId);
    if (!ok) return 0;
    return childrenBuf.readUInt32LE(offA) >>> 0;
  }

  function readChildrenPage(nodeId, start, limit) {
    const { offA, ok } = childrenRecordOffset(nodeId);
    if (!ok) return { total: 0, ids: [] };

    const total = childrenBuf.readUInt32LE(offA) >>> 0;
    const safeStart = Math.min(total, Math.max(0, start | 0));
    const safeLimit = Math.max(1, limit | 0);
    const end = Math.min(total, safeStart + safeLimit);
    const n = (end - safeStart) >>> 0;

    const ids = new Array(n);
    let p = offA + 4 + safeStart * 4;
    for (let i = 0; i < n; i++) {
      ids[i] = childrenBuf.readUInt32LE(p) >>> 0;
      p += 4;
    }
    return { total, ids };
  }

  function readChildrenAllBounded(nodeId, maxChildren) {
    const { offA, ok } = childrenRecordOffset(nodeId);
    if (!ok) return { total: 0, ids: [] };

    const total = childrenBuf.readUInt32LE(offA) >>> 0;
    if (total > maxChildren >>> 0) return null;

    const ids = new Array(total);
    let p = offA + 4;
    for (let i = 0; i < total; i++) {
      ids[i] = childrenBuf.readUInt32LE(p) >>> 0;
      p += 4;
    }
    return { total, ids };
  }

  const dictBuf = await fs.promises.readFile(
    path.join(TAXA_DIR, meta.phase2.dict),
  );
  const dictRecBytes = 32;
  const dictRecs = Math.floor(dictBuf.length / dictRecBytes);

  function dictAt(i) {
    const off = i * dictRecBytes;
    return {
      speciesNodeId: dictBuf.readUInt32LE(off),
      segmentIndex: dictBuf.readUInt32LE(off + 4),
      count: dictBuf.readUInt32LE(off + 8),
      nameId: dictBuf.readUInt32LE(off + 12),
      postingsOffset: Number(dictBuf.readBigUInt64LE(off + 16)),
      postingsBytes: dictBuf.readUInt32LE(off + 24),
    };
  }

  function findSpeciesFirst(speciesNodeId) {
    let lo = 0;
    let hi = dictRecs - 1;
    let best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const s = dictBuf.readUInt32LE(mid * dictRecBytes);
      if (s >= speciesNodeId) {
        if (s === speciesNodeId) best = mid;
        hi = mid - 1;
      } else {
        lo = mid + 1;
      }
    }
    return best;
  }

  function readSpeciesSegments(speciesNodeId) {
    const first = findSpeciesFirst(speciesNodeId >>> 0);
    if (first < 0) return [];
    const out = [];
    for (let i = first; i < dictRecs; i++) {
      const off = i * dictRecBytes;
      const s = dictBuf.readUInt32LE(off);
      if (s !== speciesNodeId >>> 0) break;
      out.push(dictAt(i));
    }
    return out;
  }

  const rowOffsetsBuf = await fs.promises.readFile(
    path.join(TAXA_DIR, meta.files.rowOffsets),
  );
  const rowStride = meta.rowStride >>> 0;
  const fileSize = Number(meta.fileSize || 0);

  let rowStartsFullBuf = null;
  if (meta.files && meta.files.rowStartsFull) {
    const rowStartsFullPath = path.join(TAXA_DIR, meta.files.rowStartsFull);
    try {
      rowStartsFullBuf = await fs.promises.readFile(rowStartsFullPath);
      const expectBytes = Number(meta.rowCount || 0) * 8;
      if (
        !Number.isFinite(expectBytes) ||
        expectBytes <= 0 ||
        rowStartsFullBuf.length < expectBytes
      ) {
        rowStartsFullBuf = null;
      }
    } catch {
      rowStartsFullBuf = null;
    }
  }

  function rowStartOffsetExact(rowId) {
    if (!rowStartsFullBuf) return null;
    const rid = Number(rowId);
    if (!Number.isFinite(rid) || rid < 0) return null;
    const safeRowId = rid >>> 0;
    if (safeRowId >= (meta.rowCount >>> 0)) return null;
    const off = safeRowId * 8;
    if (off + 8 > rowStartsFullBuf.length) return null;
    return Number(rowStartsFullBuf.readBigUInt64LE(off));
  }

  function rowEndOffsetExact(rowId) {
    if (!rowStartsFullBuf) return null;
    const rid = Number(rowId);
    if (!Number.isFinite(rid) || rid < 0) return null;
    const safeRowId = rid >>> 0;

    if (safeRowId + 1 < (meta.rowCount >>> 0)) {
      const off = (safeRowId + 1) * 8;
      if (off + 8 > rowStartsFullBuf.length) return null;
      return Number(rowStartsFullBuf.readBigUInt64LE(off));
    }

    return fileSize;
  }

  function rowByteRange(rowId) {
    const start = rowStartOffsetExact(rowId);
    const end = rowEndOffsetExact(rowId);
    if (
      start == null ||
      end == null ||
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      end < start
    ) {
      return null;
    }
    return { start, end };
  }

  function rowStartOffset(rowId) {
    const exact = rowStartOffsetExact(rowId);
    if (exact != null) return BigInt(exact);

    const base = Math.floor((rowId >>> 0) / rowStride);
    const idxOff = base * 8;
    if (idxOff + 8 > rowOffsetsBuf.length)
      return BigInt(meta.firstDataOffset || 0);
    return rowOffsetsBuf.readBigUInt64LE(idxOff);
  }

  return {
    meta,
    nodeCount,
    readNode,
    getString,
    childCount,
    readChildrenPage,
    readChildrenAllBounded,
    postingsPath: path.join(TAXA_DIR, meta.phase2.postings),
    readSpeciesSegments,
    rowStartOffset,
    rowStartOffsetExact,
    rowEndOffsetExact,
    rowByteRange,
    hasExactRowStarts: !!rowStartsFullBuf,
    rowStride,
    skipEvery: meta.phase2.skipEvery >>> 0,
  };
}

async function readSpeciesRows(
  idx,
  taxa,
  speciesNodeId,
  start,
  limit,
  headerOverride,
  requestedColNames,
) {
  const sid = speciesNodeId >>> 0;
  const segs = taxa.readSpeciesSegments(sid);
  const baseHeader = headerOverride || idx.header || [];
  const projected = resolveProjectedHeaderAndCols(baseHeader, requestedColNames);
  const header = projected.header;
  const wantCols = projected.wantCols;

  if (!segs.length) {
    return {
      error: "species not indexed",
      header,
      rows: [],
      returned: 0,
    };
  }

  const wantStart = Math.max(0, start | 0);
  const wantLimit = Math.max(1, limit | 0);
  const wantEnd = wantStart + wantLimit;

  let totalCount = 0;
  for (let i = 0; i < segs.length; i++) totalCount += segs[i].count >>> 0;

  if (wantStart >= totalCount) {
    return {
      speciesNodeId: sid,
      name: taxa.getString(segs[0].nameId),
      header,
      totalCount,
      start: wantStart,
      limit: wantLimit,
      returned: 0,
      rows: [],
    };
  }

  let global = 0;
  const overlaps = [];
  for (let i = 0; i < segs.length; i++) {
    const seg = segs[i];
    const segA = global;
    const segB = global + (seg.count >>> 0);
    if (segB > wantStart && segA < wantEnd) {
      overlaps.push({ seg, segA, segB });
    }
    global = segB;
  }

  function bad(msg) {
    return {
      error: msg,
      speciesNodeId: sid,
      name: taxa.getString(segs[0].nameId),
      header,
      totalCount,
      start: wantStart,
      limit: wantLimit,
      returned: 0,
      rows: [],
    };
  }

  const ownPostingsFd = !taxa._sharedPostingsFd;
  const ownCsvFd = !idx._sharedCsvFd;

  const postingsFd =
    taxa._sharedPostingsFd || (await fs.promises.open(taxa.postingsPath, "r"));
  const csvFd = idx._sharedCsvFd || (await fs.promises.open(idx.file, "r"));

  try {
    const pst = await postingsFd.stat();
    const postingsSize = pst.size;

    const targetRowIds = [];

    const MAX_SKIP_BYTES = 64 << 20;
    const MAX_READ = Math.min(POSTINGS_READ_BYTES | 0, 128 << 20);

    for (let oi = 0; oi < overlaps.length; oi++) {
      const o = overlaps[oi];
      const seg = o.seg;

      const localStart = Math.max(0, wantStart - o.segA);
      const localEnd = Math.min(seg.count >>> 0, wantEnd - o.segA);
      if (localStart >= localEnd) continue;

      const segStart = Number(seg.postingsOffset);
      const segBytes = seg.postingsBytes >>> 0;
      const segEnd = segStart + segBytes;

      if (!Number.isSafeInteger(segStart) || segStart < 0) {
        return bad("bad postingsOffset");
      }
      if (!Number.isSafeInteger(segEnd) || segEnd <= segStart) {
        return bad("bad postingsBytes");
      }
      if (segStart + 12 > postingsSize) {
        return bad("postings segment outside file");
      }

      const headBuf = Buffer.allocUnsafe(12);
      await postingsFd.read(headBuf, 0, 12, segStart);

      const countOnDisk = headBuf.readUInt32LE(0) >>> 0;
      const skipEveryOnDisk = headBuf.readUInt32LE(4) >>> 0;
      const nSkipsOnDisk = headBuf.readUInt32LE(8) >>> 0;

      const count = seg.count >>> 0;
      if (countOnDisk !== count) {
        if (countOnDisk === 0 || countOnDisk > count + 16) {
          return bad(
            `segment header count mismatch (disk=${countOnDisk}, dict=${count})`,
          );
        }
      }

      if (skipEveryOnDisk === 0 || skipEveryOnDisk > 1 << 24) {
        return bad(`bad skipEvery=${skipEveryOnDisk}`);
      }

      const expectedMaxSkips = Math.ceil(count / skipEveryOnDisk) + 2;
      if (nSkipsOnDisk > expectedMaxSkips) {
        return bad(
          `bad nSkips=${nSkipsOnDisk} (expected <= ${expectedMaxSkips})`,
        );
      }

      const skipBytes = (nSkipsOnDisk * 12) >>> 0;
      if (skipBytes > MAX_SKIP_BYTES) {
        return bad(`skip table too large: ${skipBytes} bytes`);
      }
      if (12 + skipBytes > segBytes) {
        return bad("skip table exceeds segment bytes");
      }

      const streamOffsetInBlock = 12 + skipBytes;
      const streamAbsOffset = segStart + streamOffsetInBlock;
      if (streamAbsOffset > segEnd) {
        return bad("stream offset beyond segment");
      }

      let best = { rowIndex: 0, rowId: 0, byteOffset: 0 };

      if (nSkipsOnDisk) {
        const skipBuf = Buffer.allocUnsafe(skipBytes);
        await postingsFd.read(skipBuf, 0, skipBytes, segStart + 12);

        for (let i = 0; i < nSkipsOnDisk; i++) {
          const off = i * 12;
          const rowIndex = skipBuf.readUInt32LE(off) >>> 0;
          if (rowIndex > localStart) break;
          best = {
            rowIndex,
            rowId: skipBuf.readUInt32LE(off + 4) >>> 0,
            byteOffset: skipBuf.readUInt32LE(off + 8) >>> 0,
          };
        }
      }

      if (streamAbsOffset + best.byteOffset >= segEnd) {
        return bad("bad skip byteOffset");
      }

      let filePos = streamAbsOffset + best.byteOffset;
      let carry = Buffer.alloc(0);

      let curIndex = best.rowIndex >>> 0;
      const firstIndex = curIndex;
      let prevRid = best.rowId >>> 0;

      while (curIndex < localEnd) {
        const remainInSeg = segEnd - filePos;
        if (remainInSeg <= 0) break;

        const chunkLen = Math.min(MAX_READ, remainInSeg) | 0;
        if (chunkLen <= 0 || chunkLen > 0x7fffffff) {
          return bad("bad postings read length");
        }

        const raw = Buffer.allocUnsafe(chunkLen);
        const { bytesRead } = await postingsFd.read(raw, 0, chunkLen, filePos);
        if (bytesRead <= 0) break;

        const rawView = raw.subarray(0, bytesRead);
        const prevCarryLen = carry.length;
        const buf = prevCarryLen
          ? Buffer.concat([carry, rawView], prevCarryLen + rawView.length)
          : rawView;

        let pos = 0;

        while (curIndex < localEnd) {
          const dec = varintDecodeU32(buf, pos);
          if (!dec) break;

          const v = dec.value >>> 0;
          pos = dec.next;

          let rid;
          if (curIndex === firstIndex) {
            rid = prevRid >>> 0;
          } else {
            rid = (prevRid + v) >>> 0;
            prevRid = rid;
          }

          if (curIndex >= localStart) {
            targetRowIds.push(rid >>> 0);
          }
          curIndex++;
        }

        const consumedFromRaw = Math.max(0, pos - prevCarryLen);
        filePos += consumedFromRaw;
        carry = pos < buf.length ? buf.subarray(pos) : Buffer.alloc(0);

        if (pos === 0 && carry.length > 32) {
          return bad("stuck decoding varints (corrupt postings)");
        }
      }
    }

    const maxRow =
      taxa.meta && taxa.meta.rowCount != null
        ? taxa.meta.rowCount >>> 0
        : 0xffffffff;

    const cleaned = [];
    for (let i = 0; i < targetRowIds.length; i++) {
      const rid = targetRowIds[i] >>> 0;
      if (rid < maxRow) cleaned.push(rid);
    }

    const uniqueSorted = sortUniqueU32(cleaned);

    if (taxa.hasExactRowStarts && typeof taxa.rowByteRange === "function") {
      const exactRows = await readExactRowsByRowIdsSelected({
        fd: csvFd,
        taxa,
        delimiter: idx.delimiter,
        rowIdsSorted: uniqueSorted,
        wantCols,
        limit: wantLimit,
      });

      const expectedRows = Math.min(wantLimit, uniqueSorted.length);
      if (exactRows.length !== expectedRows) {
        return bad(
          `exact row-start lookup recovered ${exactRows.length} / ${expectedRows} row(s)`,
        );
      }

      return {
        speciesNodeId: sid,
        name: taxa.getString(segs[0].nameId),
        header,
        totalCount,
        start: wantStart,
        limit: wantLimit,
        returned: exactRows.length,
        rows: exactRows,
      };
    }

    const outRows = [];
    const strideA = taxa.rowStride >>> 0 || 0xffffffff;
    const strideB = idx.indexStride >>> 0 || 0xffffffff;
    const stride = Math.min(strideA, strideB) >>> 0;
    const useIdxStride = stride === strideB;

    const groups = new Map();
    for (let i = 0; i < uniqueSorted.length; i++) {
      const rid = uniqueSorted[i] >>> 0;
      const baseRow = (Math.floor(rid / stride) * stride) >>> 0;
      let arr = groups.get(baseRow);
      if (!arr) {
        arr = [];
        groups.set(baseRow, arr);
      }
      arr.push(rid);
    }

    const sortedBases = Array.from(groups.keys()).sort((a, b) => a - b);

    for (let bi = 0; bi < sortedBases.length; bi++) {
      if (outRows.length >= wantLimit) break;

      const baseRow = sortedBases[bi] >>> 0;
      const want = groups.get(baseRow).slice().sort((a, b) => a - b);

      let startRowId = baseRow;
      let startOff;

      if (useIdxStride) {
        const near = findNearestOffset(idx, startRowId);
        startRowId = near.baseRow >>> 0;
        startOff = near.seekOffset;
      } else {
        startOff = Number(taxa.rowStartOffset(startRowId));
      }

      const got = await readRowsByRowIdsSelectiveFd({
        fd: csvFd,
        delimiter: idx.delimiter,
        startOffset: startOff,
        startRowId,
        stopRowIdExclusive: addU32Clamped(startRowId, addU32Clamped(stride, 1)),
        rowIdsSorted: want,
        wantCols,
        limit: wantLimit - outRows.length,
      });

      for (let i = 0; i < got.length; i++) {
        outRows.push(got[i]);
        if (outRows.length >= wantLimit) break;
      }
    }

    return {
      speciesNodeId: sid,
      name: taxa.getString(segs[0].nameId),
      header,
      totalCount,
      start: wantStart,
      limit: wantLimit,
      returned: outRows.length,
      rows: outRows,
    };
  } finally {
    if (ownPostingsFd) {
      await postingsFd.close().catch((e) => {
        console.error(e);
      });
    }
    if (ownCsvFd) {
      await csvFd.close().catch((e) => {
        console.error(e);
      });
    }
  }
}

async function streamSpeciesRowIds(taxa, speciesNodeId, onRowId) {
  const sid = speciesNodeId >>> 0;
  const segs = taxa.readSpeciesSegments(sid);
  if (!segs.length) return 0;

  const visit = typeof onRowId === "function" ? onRowId : async () => {};
  let emitted = 0;

  const ownPostingsFd = !taxa._sharedPostingsFd;
  const postingsFd =
    taxa._sharedPostingsFd || (await fs.promises.open(taxa.postingsPath, "r"));

  try {
    const pst = await postingsFd.stat();
    const postingsSize = pst.size;

    const MAX_SKIP_BYTES = 64 << 20;
    const MAX_READ = Math.min(POSTINGS_READ_BYTES | 0, 128 << 20);

    for (let si = 0; si < segs.length; si++) {
      const seg = segs[si];

      const segStart = Number(seg.postingsOffset);
      const segBytes = seg.postingsBytes >>> 0;
      const segEnd = segStart + segBytes;

      if (!Number.isSafeInteger(segStart) || segStart < 0) {
        throw new Error("bad postingsOffset");
      }
      if (!Number.isSafeInteger(segEnd) || segEnd <= segStart) {
        throw new Error("bad postingsBytes");
      }
      if (segStart + 12 > postingsSize) {
        throw new Error("postings segment outside file");
      }

      const headBuf = Buffer.allocUnsafe(12);
      await postingsFd.read(headBuf, 0, 12, segStart);

      const countOnDisk = headBuf.readUInt32LE(0) >>> 0;
      const skipEveryOnDisk = headBuf.readUInt32LE(4) >>> 0;
      const nSkipsOnDisk = headBuf.readUInt32LE(8) >>> 0;

      const count = seg.count >>> 0;
      if (countOnDisk !== count) {
        if (countOnDisk === 0 || countOnDisk > count + 16) {
          throw new Error(
            `segment header count mismatch (disk=${countOnDisk}, dict=${count})`,
          );
        }
      }

      if (skipEveryOnDisk === 0 || skipEveryOnDisk > 1 << 24) {
        throw new Error(`bad skipEvery=${skipEveryOnDisk}`);
      }

      const expectedMaxSkips = Math.ceil(count / skipEveryOnDisk) + 2;
      if (nSkipsOnDisk > expectedMaxSkips) {
        throw new Error(
          `bad nSkips=${nSkipsOnDisk} (expected <= ${expectedMaxSkips})`,
        );
      }

      const skipBytes = (nSkipsOnDisk * 12) >>> 0;
      if (skipBytes > MAX_SKIP_BYTES) {
        throw new Error(`skip table too large: ${skipBytes} bytes`);
      }
      if (12 + skipBytes > segBytes) {
        throw new Error("skip table exceeds segment bytes");
      }

      const streamAbsOffset = segStart + 12 + skipBytes;
      if (streamAbsOffset > segEnd) {
        throw new Error("stream offset beyond segment");
      }

      const skipRowIndex = new Uint32Array(nSkipsOnDisk);
      const skipRowId = new Uint32Array(nSkipsOnDisk);

      if (nSkipsOnDisk > 0) {
        const skipBuf = Buffer.allocUnsafe(skipBytes);
        await postingsFd.read(skipBuf, 0, skipBytes, segStart + 12);

        for (let i = 0; i < nSkipsOnDisk; i++) {
          const off = i * 12;
          skipRowIndex[i] = skipBuf.readUInt32LE(off) >>> 0;
          skipRowId[i] = skipBuf.readUInt32LE(off + 4) >>> 0;
        }
      }

      let nextSkipIdx = 0;
      let nextSkipAt = nSkipsOnDisk > 0 ? skipRowIndex[0] >>> 0 : 0xffffffff;
      let nextSkipRid = nSkipsOnDisk > 0 ? skipRowId[0] >>> 0 : 0;

      let filePos = streamAbsOffset;
      let carry = Buffer.alloc(0);
      let rowIndex = 0;
      let prevRid = 0;

      while (rowIndex < count) {
        const remainInSeg = segEnd - filePos;
        if (remainInSeg <= 0) break;

        const chunkLen = Math.min(MAX_READ, remainInSeg) | 0;
        if (chunkLen <= 0 || chunkLen > 0x7fffffff) {
          throw new Error("bad postings read length");
        }

        const raw = Buffer.allocUnsafe(chunkLen);
        const { bytesRead } = await postingsFd.read(raw, 0, chunkLen, filePos);
        if (bytesRead <= 0) break;

        const rawView = raw.subarray(0, bytesRead);
        const prevCarryLen = carry.length;
        const buf = prevCarryLen
          ? Buffer.concat([carry, rawView], prevCarryLen + rawView.length)
          : rawView;

        let pos = 0;

        while (rowIndex < count) {
          const dec = varintDecodeU32(buf, pos);
          if (!dec) break;

          const delta = dec.value >>> 0;
          pos = dec.next;

          let rid;
          if (rowIndex === nextSkipAt) {
            rid = nextSkipRid >>> 0;
            nextSkipIdx++;
            if (nextSkipIdx < nSkipsOnDisk) {
              nextSkipAt = skipRowIndex[nextSkipIdx] >>> 0;
              nextSkipRid = skipRowId[nextSkipIdx] >>> 0;
            } else {
              nextSkipAt = 0xffffffff;
              nextSkipRid = 0;
            }
          } else if (rowIndex === 0) {
            rid = delta >>> 0;
          } else {
            rid = (prevRid + delta) >>> 0;
          }

          prevRid = rid >>> 0;
          await visit(rid >>> 0);
          emitted++;
          rowIndex++;
        }

        const consumedFromRaw = Math.max(0, pos - prevCarryLen);
        filePos += consumedFromRaw;
        carry = pos < buf.length ? buf.subarray(pos) : Buffer.alloc(0);

        if (pos === 0 && carry.length > 32) {
          throw new Error("stuck decoding varints (corrupt postings)");
        }
      }
    }
  } finally {
    if (ownPostingsFd) {
      await postingsFd.close().catch((e) => {
        console.error(e);
      });
    }
  }

  return emitted >>> 0;
}

module.exports = {
  listChildNodeIds,
  readSpeciesRowIds,
  walkSpeciesTargetsUnderNode,
  countSpeciesTargetsUnderNode,
  dedupeNodeIds,
  radixSortPairsInPlace,
  MinHeap,
  varintEncodeU32ToParts,
  varintDecodeU32,
  buildTaxaIndexPhase1,
  finalizeTaxaIndexPhase2,
  loadTaxaIndex,
  readSpeciesRows,
  streamSpeciesRowIds,
};