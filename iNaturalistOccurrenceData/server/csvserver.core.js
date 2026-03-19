"use strict";

//server/csvserver.core.js

const {
  fs,
  path,
  os,
  INDEX_STRIDE,
  MAX_LIMIT,
  GUESSED_HEADERS_50,
  statFile,
  fmtGiB,
} = require("./csvserver.utils.js");

const CSV_QUOTE = 0x22;
const CSV_CANDIDATE_DELIMS = [0x2c, 0x09, 0x3b, 0x7c];
const INDEX_VERSION = 5;

function splitHeaderLine(lineStr, delim) {
  const out = [];
  let field = "";
  let inQuotes = false;
  let fieldStart = true;

  for (let i = 0; i < lineStr.length; i++) {
    const ch = lineStr[i];

    if (inQuotes) {
      if (ch === '"') {
        const next = lineStr[i + 1];
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

    if (ch === delim) {
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

function detectDelimiterFromHeaderBytes(headerBytes) {
  const counts = new Map(CSV_CANDIDATE_DELIMS.map((b) => [b, 0]));
  let inQuotes = false;
  let fieldStart = true;

  for (let i = 0; i < headerBytes.length; i++) {
    const b = headerBytes[i];

    if (inQuotes) {
      if (b === CSV_QUOTE) {
        const next = headerBytes[i + 1];
        if (next === CSV_QUOTE) {
          i++;
        } else {
          inQuotes = false;
          fieldStart = false;
        }
      }
      continue;
    }

    if (fieldStart && b === CSV_QUOTE) {
      inQuotes = true;
      fieldStart = false;
      continue;
    }

    if (counts.has(b)) {
      counts.set(b, counts.get(b) + 1);
      fieldStart = true;
      continue;
    }

    fieldStart = false;
  }

  let best = 0x2c;
  let bestCount = 0;
  for (const [b, c] of counts.entries()) {
    if (c > bestCount) {
      best = b;
      bestCount = c;
    }
  }
  if (bestCount === 0) best = 0x2c;
  return best;
}

function findFirstCsvRowBoundary(buf) {
  let inQuotes = false;
  let fieldStart = true;

  for (let i = 0; i < buf.length; i++) {
    const b = buf[i];

    if (inQuotes) {
      if (b === CSV_QUOTE) {
        const next = buf[i + 1];
        if (next === CSV_QUOTE) {
          i++;
        } else {
          inQuotes = false;
          fieldStart = false;
        }
      }
      continue;
    }

    if (fieldStart && b === CSV_QUOTE) {
      inQuotes = true;
      fieldStart = false;
      continue;
    }

    if (
      b === 0x2c ||
      b === 0x09 ||
      b === 0x3b ||
      b === 0x7c
    ) {
      fieldStart = true;
      continue;
    }

    if (b === 0x0a) {
      return {
        headerEndIndex: i,
        firstDataOffset: i + 1,
      };
    }

    if (b === 0x0d) {
      const next = buf[i + 1];
      if (i + 1 >= buf.length) return null;
      if (next === 0x0a) {
        return {
          headerEndIndex: i,
          firstDataOffset: i + 2,
        };
      }
      return {
        headerEndIndex: i,
        firstDataOffset: i + 1,
      };
    }

    fieldStart = false;
  }

  return null;
}

function createCsvByteScanState(delimChar) {
  return {
    delim: (String(delimChar || ",").charCodeAt(0) & 255) >>> 0,
    inQuotes: false,
    fieldStart: true,
    sawCR: false,
  };
}

async function readHeaderAndFirstDataOffset(filePath) {
  return new Promise((resolve, reject) => {
    const rs = fs.createReadStream(filePath, { highWaterMark: 4 << 20 });
    const buffers = [];
    let total = 0;
    let found = false;

    rs.on("data", (buf) => {
      if (found) return;

      buffers.push(buf);
      total += buf.length;

      if (total > 64 * 1024 * 1024) {
        rs.destroy();
        reject(new Error("Header row > 64MB; unexpected format."));
        return;
      }

      const whole = Buffer.concat(buffers, total);
      const boundary = findFirstCsvRowBoundary(whole);
      if (!boundary) return;

      const headerBytes = whole.subarray(0, boundary.headerEndIndex);
      const delimByte = detectDelimiterFromHeaderBytes(headerBytes);
      const delimChar = String.fromCharCode(delimByte);
      const headerLine = headerBytes.toString("utf8");
      const header = splitHeaderLine(headerLine, delimChar);

      found = true;
      rs.destroy();
      resolve({
        header,
        delimiter: delimChar,
        firstDataOffset: boundary.firstDataOffset,
      });
    });

    rs.on("error", reject);
    rs.on("close", () => {
      if (!found) reject(new Error("Could not find header line ending."));
    });
  });
}

function chooseHeader(idx, headerMode) {
  if (headerMode === "guess" && idx.header && idx.header.length === 50) {
    return GUESSED_HEADERS_50;
  }
  if (headerMode === "guess" && (!idx.header || idx.header.length <= 1)) {
    return GUESSED_HEADERS_50;
  }
  return idx.header;
}

function findColIndex(header, name) {
  if (!header) return -1;
  const target = String(name).toLowerCase();

  for (let i = 0; i < header.length; i++) {
    const h = String(header[i] || "").trim().toLowerCase();
    if (h === target) return i;
  }

  return -1;
}

class CsvByteParser {
  constructor(delimChar, wantCols) {
    this._delim = (String(delimChar || ",").charCodeAt(0) & 255) >>> 0;
    this._wantCols = Array.isArray(wantCols) ? wantCols.slice() : [];

    let max = -1;
    for (let i = 0; i < this._wantCols.length; i++) {
      const v = this._wantCols[i] | 0;
      if (v > max) max = v;
    }

    if (max >= 0 && max <= 4096) {
      const lut = new Int32Array(max + 1);
      lut.fill(-1);
      for (let i = 0; i < this._wantCols.length; i++) {
        lut[this._wantCols[i] | 0] = i;
      }
      this._wantLut = lut;
      this._wantMap = null;
    } else {
      const m = new Map();
      for (let i = 0; i < this._wantCols.length; i++) {
        m.set(this._wantCols[i] | 0, i);
      }
      this._wantMap = m;
      this._wantLut = null;
    }

    const n = this._wantCols.length;
    this._vals = new Array(n);
    for (let i = 0; i < n; i++) this._vals[i] = "";
    this._touched = new Uint16Array(Math.max(1, n));
    this._touchedMark = new Uint8Array(Math.max(1, n));
    this._touchedLen = 0;

    this._curFieldParts = [];
    this._curFieldLen = 0;
    this._segActive = false;
    this._segStart = 0;

    this._inQuotes = false;
    this._fieldStart = true;
    this._sawCR = false;

    this._col = 0;
    this._rowId = 0;
    this._rowStartAbs = 0n;
    this._absPos = 0n;

    this._rowKeepFn = null;
    this._keepRow = true;

    this._outRows = [];
  }

  resetOffsets(absStart) {
    this._absPos = BigInt(absStart || 0);
    this._rowStartAbs = BigInt(absStart || 0);
  }

  resetRowId(rowId) {
    this._rowId = rowId >>> 0 || 0;
    this._updateKeepRow();
  }

  setRowKeepFn(fn) {
    this._rowKeepFn = typeof fn === "function" ? fn : null;
    this._updateKeepRow();
  }

  get rowId() {
    return this._rowId >>> 0;
  }

  _updateKeepRow() {
    this._keepRow = this._rowKeepFn
      ? !!this._rowKeepFn(this._rowId >>> 0)
      : true;
  }

  _wantSlot(col) {
    if (!this._keepRow) return -1;

    if (this._wantLut) {
      if (col < 0 || col >= this._wantLut.length) return -1;
      return this._wantLut[col] | 0;
    }

    const v = this._wantMap.get(col | 0);
    return v == null ? -1 : v | 0;
  }

  _finishSeg(buf, i) {
    if (!this._segActive) return;
    if (i > this._segStart) {
      const part = buf.subarray(this._segStart, i);
      this._curFieldParts.push(part);
      this._curFieldLen += part.length;
    }
    this._segActive = false;
  }

  _flushFieldToSlot(slot) {
    if (slot < 0) {
      this._curFieldParts.length = 0;
      this._curFieldLen = 0;
      return;
    }

    let s = "";

    if (this._curFieldLen > 0) {
      if (this._curFieldParts.length === 1) {
        s = this._curFieldParts[0].toString("utf8");
      } else {
        const b = Buffer.allocUnsafe(this._curFieldLen);
        let p = 0;
        for (const part of this._curFieldParts) {
          part.copy(b, p);
          p += part.length;
        }
        s = b.toString("utf8");
      }
    }

    this._curFieldParts.length = 0;
    this._curFieldLen = 0;

    this._vals[slot] = s;
    if (!this._touchedMark[slot]) {
      this._touchedMark[slot] = 1;
      this._touched[this._touchedLen++] = slot;
    }
  }

  _commitField() {
    const slot = this._wantSlot(this._col);
    this._flushFieldToSlot(slot);
    this._col++;
    this._fieldStart = true;
  }

  _clearTouched() {
    for (let i = 0; i < this._touchedLen; i++) {
      const slot = this._touched[i];
      this._vals[slot] = "";
      this._touchedMark[slot] = 0;
    }
    this._touchedLen = 0;
  }

  _commitRow(rowStartAbs) {
    if (this._keepRow) {
      const out = new Array(this._wantCols.length);
      for (let i = 0; i < out.length; i++) out[i] = this._vals[i];
      this._outRows.push({
        rowId: this._rowId >>> 0,
        rowStartAbs,
        values: out,
      });
    }

    this._clearTouched();

    this._rowId = (this._rowId + 1) >>> 0;
    this._updateKeepRow();

    this._col = 0;
    this._fieldStart = true;
    this._inQuotes = false;

    this._curFieldParts.length = 0;
    this._curFieldLen = 0;
    this._segActive = false;
  }

  push(buf, absChunkStart) {
    this._outRows.length = 0;

    const startAbs = BigInt(absChunkStart || 0);
    if (this._absPos !== startAbs) this._absPos = startAbs;

    let i = 0;
    while (i < buf.length) {
      const b = buf[i];

      if (this._sawCR) {
        this._sawCR = false;
        if (b === 0x0a) {
          i++;
          this._absPos++;
          continue;
        }
      }

      if (this._inQuotes) {
        if (b === 0x22) {
          this._finishSeg(buf, i);
          const next = buf[i + 1];
          if (next === 0x22) {
            const slot = this._wantSlot(this._col);
            if (slot >= 0) {
              const q = buf.subarray(i, i + 1);
              this._curFieldParts.push(q);
              this._curFieldLen += 1;
            }
            i += 2;
            this._absPos += 2n;
            continue;
          }

          this._inQuotes = false;
          this._fieldStart = false;
          i++;
          this._absPos++;
          continue;
        }

        const slot = this._wantSlot(this._col);
        if (slot >= 0) {
          if (!this._segActive) {
            this._segActive = true;
            this._segStart = i;
          }
        }

        i++;
        this._absPos++;
        continue;
      }

      if (this._fieldStart && b === 0x22) {
        this._inQuotes = true;
        this._fieldStart = false;
        i++;
        this._absPos++;
        continue;
      }

      if (b === this._delim) {
        this._finishSeg(buf, i);
        this._commitField();
        i++;
        this._absPos++;
        continue;
      }

      if (b === 0x0a || b === 0x0d) {
        if (b === 0x0d) this._sawCR = true;

        this._finishSeg(buf, i);
        this._commitField();

        const rowStartAbs = this._rowStartAbs;
        let nextStart = this._absPos + 1n;

        if (b === 0x0d) {
          const next = buf[i + 1];
          if (next === 0x0a) nextStart += 1n;
        }

        this._rowStartAbs = nextStart;
        this._commitRow(rowStartAbs);

        i++;
        this._absPos++;
        continue;
      }

      const slot = this._wantSlot(this._col);
      if (slot >= 0) {
        if (!this._segActive) {
          this._segActive = true;
          this._segStart = i;
        }
      }

      this._fieldStart = false;
      i++;
      this._absPos++;
    }

    this._finishSeg(buf, buf.length);
    return this._outRows.slice();
  }

  finish() {
    const out = [];

    if (this._segActive) this._segActive = false;
    if (this._inQuotes) this._inQuotes = false;

    if (
      this._col !== 0 ||
      this._curFieldLen !== 0 ||
      this._curFieldParts.length
    ) {
      this._commitField();
      const rowStartAbs = this._rowStartAbs;
      this._commitRow(rowStartAbs);
      out.push(...this._outRows);
      this._outRows.length = 0;
    }

    return out;
  }
}

function getBestLookupStride(idx, taxa) {
  const a = (idx && idx.indexStride ? idx.indexStride : 0xffffffff) >>> 0;
  const b = (taxa && taxa.rowStride ? taxa.rowStride : 0xffffffff) >>> 0;
  return Math.min(a, b) >>> 0;
}

function groupRowIdsByBaseRow(rowIds, stride) {
  const groups = new Map();
  const s = Math.max(1, stride | 0);

  for (let i = 0; i < rowIds.length; i++) {
    const rid = rowIds[i] >>> 0;
    const baseRow = (Math.floor(rid / s) * s) >>> 0;
    let arr = groups.get(baseRow);
    if (!arr) {
      arr = [];
      groups.set(baseRow, arr);
    }
    arr.push(rid);
  }

  return Array.from(groups.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([baseRow, ids]) => {
      ids.sort((x, y) => x - y);
      return { baseRow: baseRow >>> 0, rowIds: ids };
    });
}

function findNearestOffset(idx, startRow) {
  if (startRow <= 0) return { seekOffset: idx.firstDataOffset, baseRow: 0 };

  const stride = idx.indexStride;
  const baseRow = Math.floor(startRow / stride) * stride;

  const arr = idx.index;
  let lo = 0;
  let hi = arr.length - 1;
  let best = null;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const v = arr[mid].row;
    if (v === baseRow) {
      best = arr[mid];
      break;
    }
    if (v < baseRow) {
      best = arr[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  if (!best) return { seekOffset: idx.firstDataOffset, baseRow: 0 };
  return { seekOffset: best.offset, baseRow: best.row };
}

function getLookupWindowForBaseRow(idx, taxa, baseRow) {
  const stride = getBestLookupStride(idx, taxa);
  const useIdxStride = stride === ((idx && idx.indexStride) || 0xffffffff);

  let startRowId = baseRow >>> 0;
  let startOffset = 0;

  if (useIdxStride) {
    const near = findNearestOffset(idx, startRowId);
    startRowId = near.baseRow >>> 0;
    startOffset = Number(near.seekOffset);
  } else {
    startOffset = Number(taxa.rowStartOffset(startRowId));
  }

  return {
    startRowId,
    startOffset,
    stopRowIdExclusive: (startRowId + stride + 1) >>> 0,
  };
}

async function locateRawRangesFd({
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

  const startPos = Number(startOffset);
  if (!Number.isFinite(startPos) || startPos < 0) {
    throw new Error(`Bad startOffset: ${startOffset}`);
  }

  const chunkSize = Math.max(
    64 << 10,
    Math.floor(Number(chunkBytes) || 0),
  );
  const buf = Buffer.allocUnsafe(chunkSize);

  const want = rowIdsSorted.slice();
  let wantIndex = 0;
  let targetRowId = want[0] >>> 0;

  let pos = startPos;
  let rowId = startRowId >>> 0;
  let rowStartAbs = startPos;

  const state = createCsvByteScanState(delimiter || ",");

  while (wantIndex < want.length) {
    if (
      stopRowIdExclusive != null &&
      (rowId >>> 0) >= (stopRowIdExclusive >>> 0)
    ) {
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

        if ((rowId >>> 0) === (targetRowId >>> 0)) {
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
    (stopRowIdExclusive == null || (rowId >>> 0) < (stopRowIdExclusive >>> 0)) &&
    rowStartAbs < pos &&
    (rowId >>> 0) === (targetRowId >>> 0)
  ) {
    await onRange(rowId >>> 0, rowStartAbs, pos);
  }
}

async function buildIndex(filePath, outPath) {
  const st = await statFile(filePath);
  const fileSize = st.size;

  console.log(`[index] building index for ${filePath}`);
  console.log(`[index] file size: ${fmtGiB(fileSize)} GiB`);
  console.log(`[index] stride: every ${INDEX_STRIDE} rows`);

  const hdr = await readHeaderAndFirstDataOffset(filePath);
  const header = hdr.header;
  const delimiter = hdr.delimiter;
  const firstDataOffset = hdr.firstDataOffset;
  const delimByte = (String(delimiter).charCodeAt(0) & 255) >>> 0;

  const delimName =
    delimiter === "\t" ? "\\t (tab)" : JSON.stringify(delimiter);
  console.log(`[index] detected delimiter: ${delimName}`);
  console.log(`[index] header columns: ${header.length}`);

  return new Promise((resolve, reject) => {
    const rs = fs.createReadStream(filePath, {
      start: firstDataOffset,
      highWaterMark: 32 << 20,
    });

    let offset = firstDataOffset;
    let rowCount = 0;
    const index = [];
    let rowStart = firstDataOffset;

    const state = createCsvByteScanState(delimiter);

    let lastLogT = Date.now();
    let lastLogBytes = firstDataOffset;

    function maybeLogProgress() {
      const now = Date.now();
      if (now - lastLogT < 1500 && offset - lastLogBytes < (512 << 20)) return;
      lastLogT = now;
      lastLogBytes = offset;
      const pct = ((offset / fileSize) * 100).toFixed(2);
      process.stdout.write(
        `[index] ${fmtGiB(offset)} / ${fmtGiB(fileSize)} GiB (${pct}%) rows~${rowCount}\r`,
      );
    }

    rs.on("data", (buf) => {
      const chunkStartOffset = offset;
      offset += buf.length;

      for (let i = 0; i < buf.length; i++) {
        const b = buf[i];

        if (state.sawCR) {
          state.sawCR = false;
          if (b === 0x0a) {
            rowStart = chunkStartOffset + i + 1;
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

        if (b === delimByte) {
          state.fieldStart = true;
          continue;
        }

        if (b === 0x0a || b === 0x0d) {
          rowCount++;
          if ((rowCount - 1) % INDEX_STRIDE === 0) {
            const baseRow = rowCount - 1;
            index.push({ row: baseRow, offset: rowStart });
          }

          let nextStart = chunkStartOffset + i + 1;
          if (b === 0x0d) {
            const next = buf[i + 1];
            if (next === 0x0a) {
              nextStart += 1;
              i++;
            } else {
              state.sawCR = true;
            }
          }

          rowStart = nextStart;
          state.fieldStart = true;
          state.inQuotes = false;
          continue;
        }

        state.fieldStart = false;
      }

      maybeLogProgress();
    });

    rs.on("error", reject);

    rs.on("end", async () => {
      if (rowStart < offset) {
        rowCount++;
        if ((rowCount - 1) % INDEX_STRIDE === 0) {
          const baseRow = rowCount - 1;
          index.push({ row: baseRow, offset: rowStart });
        }
      }

      process.stdout.write("\n");

      const payload = {
        version: INDEX_VERSION,
        file: path.resolve(filePath),
        fileSize,
        indexStride: INDEX_STRIDE,
        delimiter,
        header,
        firstDataOffset,
        dataRowCountEstimate: rowCount,
        index,
        createdAt: new Date().toISOString(),
        node: process.version,
        platform: `${os.platform()} ${os.release()}`,
      };

      await fs.promises.writeFile(outPath, JSON.stringify(payload));
      console.log(`[index] wrote ${outPath}`);
      console.log(`[index] indexed points: ${index.length}`);
      console.log(`[index] data rows estimate: ${rowCount}`);
      resolve(payload);
    });
  });
}

async function loadOrBuildIndex(filePath, indexPath) {
  try {
    const raw = await fs.promises.readFile(indexPath, "utf8");
    const idx = JSON.parse(raw);

    const st = await statFile(filePath);
    if (idx.fileSize !== st.size) {
      console.log("[index] index fileSize mismatch, rebuilding");
      return await buildIndex(filePath, indexPath);
    }
    if ((idx.version | 0) < INDEX_VERSION) {
      console.log("[index] index version too old, rebuilding");
      return await buildIndex(filePath, indexPath);
    }
    if (
      !idx.delimiter ||
      !idx.header ||
      idx.firstDataOffset == null ||
      !Array.isArray(idx.index)
    ) {
      console.log("[index] index missing fields, rebuilding");
      return await buildIndex(filePath, indexPath);
    }

    console.log(`[index] loaded ${indexPath}`);
    console.log(
      `[index] delimiter: ${idx.delimiter === "\t" ? "\\t (tab)" : JSON.stringify(idx.delimiter)}`,
    );
    console.log(`[index] header columns: ${idx.header.length}`);
    return idx;
  } catch (e) {
    if (e && e.code === "ENOENT") {
      console.log("[index] no index found, building");
    } else {
      console.log("[index] could not load index, rebuilding");
      console.error(e);
    }
    return await buildIndex(filePath, indexPath);
  }
}

class DelimRowDecoder {
  constructor(delimChar) {
    this._delim = delimChar;
    this._inQuotes = false;
    this._fieldStart = true;
    this._field = "";
    this._row = [];
    this._sawCR = false;
  }

  push(chunk) {
    const rows = [];
    const s = Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk);

    for (let i = 0; i < s.length; i++) {
      const ch = s[i];

      if (this._sawCR) {
        this._sawCR = false;
        if (ch === "\n") continue;
      }

      if (this._inQuotes) {
        if (ch === '"') {
          const next = s[i + 1];
          if (next === '"') {
            this._field += '"';
            i++;
          } else {
            this._inQuotes = false;
            this._fieldStart = false;
          }
        } else {
          this._field += ch;
        }
        continue;
      }

      if (this._fieldStart && ch === '"') {
        this._inQuotes = true;
        this._fieldStart = false;
        continue;
      }

      if (ch === this._delim) {
        this._row.push(this._field);
        this._field = "";
        this._fieldStart = true;
        continue;
      }

      if (ch === "\n" || ch === "\r") {
        if (ch === "\r") this._sawCR = true;
        this._row.push(this._field);
        this._field = "";
        rows.push(this._row);
        this._row = [];
        this._fieldStart = true;
        continue;
      }

      this._field += ch;
      this._fieldStart = false;
    }

    return rows;
  }

  finish() {
    const rows = [];
    if (this._sawCR) this._sawCR = false;
    if (this._inQuotes) this._inQuotes = false;

    if (this._field.length > 0 || this._row.length > 0) {
      this._row.push(this._field);
      this._field = "";
      rows.push(this._row);
      this._row = [];
      this._fieldStart = true;
    }

    return rows;
  }
}

function getAllColumnIndexes(header) {
  const n = Math.max(1, Array.isArray(header) ? header.length | 0 : 0);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

async function readPage(idx, startRow, limit, asObject, headerOverride) {
  const safeLimit = Math.min(Math.max(1, limit), MAX_LIMIT);
  const { seekOffset, baseRow } = findNearestOffset(idx, startRow);

  const wantStart = startRow >>> 0;
  const wantEndExclusive = (wantStart + safeLimit) >>> 0;
  const header = headerOverride || idx.header || [];
  const wantCols = getAllColumnIndexes(header);

  const READ_BYTES =
    Math.max(
      256 << 10,
      Math.min(Number(process.env.CSV_PAGE_READ_BYTES || (4 << 20)) | 0, 64 << 20),
    ) | 0;

  const buf = Buffer.allocUnsafe(READ_BYTES);

  const ownedFd = !idx._sharedCsvFd;
  const fd = idx._sharedCsvFd || (await fs.promises.open(idx.file, "r"));

  try {
    const parser = new CsvByteParser(idx.delimiter, wantCols);
    parser.resetOffsets(seekOffset);
    parser.resetRowId(baseRow);

    const out = [];
    let pos = seekOffset;
    let abs = seekOffset;

    while (out.length < safeLimit) {
      const { bytesRead } = await fd.read(buf, 0, buf.length, pos);
      if (bytesRead <= 0) break;

      const view = buf.subarray(0, bytesRead);
      const got = parser.push(view, abs);

      abs += bytesRead;
      pos += bytesRead;

      for (let i = 0; i < got.length; i++) {
        const r = got[i];
        const rowId = r.rowId >>> 0;
        if (rowId < wantStart) continue;
        if (rowId >= wantEndExclusive) {
          return {
            start: wantStart,
            limit: safeLimit,
            returned: out.length,
            baseRow,
            delimiter: idx.delimiter,
            header,
            rows: out,
          };
        }

        if (asObject) {
          const obj = {};
          const vals = r.values;
          const n = Math.min(header.length, vals.length);
          for (let c = 0; c < n; c++) obj[header[c]] = vals[c];
          out.push(obj);
        } else {
          out.push(r.values);
        }

        if (out.length >= safeLimit) {
          return {
            start: wantStart,
            limit: safeLimit,
            returned: out.length,
            baseRow,
            delimiter: idx.delimiter,
            header,
            rows: out,
          };
        }
      }
    }

    const tail = parser.finish();
    for (let i = 0; i < tail.length && out.length < safeLimit; i++) {
      const r = tail[i];
      const rowId = r.rowId >>> 0;
      if (rowId < wantStart) continue;
      if (rowId >= wantEndExclusive) break;

      if (asObject) {
        const obj = {};
        const vals = r.values;
        const n = Math.min(header.length, vals.length);
        for (let c = 0; c < n; c++) obj[header[c]] = vals[c];
        out.push(obj);
      } else {
        out.push(r.values);
      }
    }

    return {
      start: wantStart,
      limit: safeLimit,
      returned: out.length,
      baseRow,
      delimiter: idx.delimiter,
      header,
      rows: out,
      eof: true,
    };
  } finally {
    if (ownedFd) {
      await fd.close().catch((e) => {
        console.error(e);
      });
    }
  }
}

async function readRowsByRowIdsSelectiveFd({
  fd,
  delimiter,
  startOffset,
  startRowId,
  stopRowIdExclusive,
  rowIdsSorted,
  wantCols,
  limit,
}) {
  if (!rowIdsSorted.length) return [];

  const wantSet = new Set(rowIdsSorted);
  const out = [];

  const parser = new CsvByteParser(delimiter, wantCols);
  parser.resetOffsets(startOffset);
  parser.resetRowId(startRowId);
  parser.setRowKeepFn((rid) => wantSet.has(rid));

  const READ_BYTES =
    Math.max(
      256 << 10,
      Math.min(Number(process.env.CSV_READ_BYTES || (4 << 20)) | 0, 64 << 20),
    ) | 0;

  const buf = Buffer.allocUnsafe(READ_BYTES);

  let pos = startOffset;
  let abs = startOffset;

  while (wantSet.size && out.length < limit) {
    if (
      stopRowIdExclusive != null &&
      ((parser.rowId >>> 0) >= (stopRowIdExclusive >>> 0))
    ) {
      break;
    }

    const { bytesRead } = await fd.read(buf, 0, buf.length, pos);
    if (bytesRead <= 0) break;

    const view = buf.subarray(0, bytesRead);
    const got = parser.push(view, abs);

    abs += bytesRead;
    pos += bytesRead;

    for (let i = 0; i < got.length; i++) {
      const r = got[i];
      out.push(r.values);
      wantSet.delete(r.rowId);
      if (out.length >= limit) break;
    }
  }

  if (wantSet.size && out.length < limit) {
    const tail = parser.finish();
    for (let i = 0; i < tail.length; i++) {
      const r = tail[i];
      if (!wantSet.has(r.rowId)) continue;
      out.push(r.values);
      wantSet.delete(r.rowId);
      if (out.length >= limit) break;
    }
  }

  return out;
}

async function readRowsByRowIdsSelective({
  filePath,
  delimiter,
  startOffset,
  startRowId,
  rowIdsSorted,
  wantCols,
  limit,
}) {
  if (!rowIdsSorted.length) return [];

  const wantSet = new Set(rowIdsSorted);
  const out = [];

  const parser = new CsvByteParser(delimiter, wantCols);
  parser.resetOffsets(startOffset);
  parser.resetRowId(startRowId);
  parser.setRowKeepFn((rid) => wantSet.has(rid));

  const rs = fs.createReadStream(filePath, {
    start: startOffset,
    highWaterMark: 2 << 20,
  });

  let abs = startOffset;
  let doneEarly = false;
  let settled = false;

  await new Promise((resolve, reject) => {
    function finish() {
      if (settled) return;
      settled = true;
      resolve();
    }

    rs.on("data", (buf) => {
      if (doneEarly) return;

      const got = parser.push(buf, abs);
      abs += buf.length;

      for (const r of got) {
        out.push(r.values);
        wantSet.delete(r.rowId);
        if (out.length >= limit) {
          doneEarly = true;
          rs.destroy();
          finish();
          return;
        }
      }

      if (!wantSet.size) {
        doneEarly = true;
        rs.destroy();
        finish();
        return;
      }
    });

    rs.on("error", (e) => {
      if (settled) return;
      settled = true;
      reject(e);
    });

    rs.on("close", finish);
    rs.on("end", finish);
  });

  if (!doneEarly && wantSet.size) {
    const tail = parser.finish();
    for (const r of tail) {
      if (!wantSet.has(r.rowId)) continue;
      out.push(r.values);
      wantSet.delete(r.rowId);
      if (out.length >= limit) break;
    }
  }

  return out;
}

async function readRowsByRowIdsSelectiveRecordsFd({
  fd,
  delimiter,
  startOffset,
  startRowId,
  stopRowIdExclusive,
  rowIdsSorted,
  wantCols,
  limit,
}) {
  if (!rowIdsSorted.length) return [];

  const wantSet = new Set(rowIdsSorted);
  const out = [];

  const parser = new CsvByteParser(delimiter, wantCols);
  parser.resetOffsets(startOffset);
  parser.resetRowId(startRowId);
  parser.setRowKeepFn((rid) => wantSet.has(rid));

  const READ_BYTES =
    Math.max(
      256 << 10,
      Math.min(Number(process.env.CSV_READ_BYTES || (4 << 20)) | 0, 64 << 20),
    ) | 0;

  const buf = Buffer.allocUnsafe(READ_BYTES);

  let pos = startOffset;
  let abs = startOffset;

  while (wantSet.size && out.length < limit) {
    if (
      stopRowIdExclusive != null &&
      ((parser.rowId >>> 0) >= (stopRowIdExclusive >>> 0))
    ) {
      break;
    }

    const { bytesRead } = await fd.read(buf, 0, buf.length, pos);
    if (bytesRead <= 0) break;

    const view = buf.subarray(0, bytesRead);
    const got = parser.push(view, abs);

    abs += bytesRead;
    pos += bytesRead;

    for (let i = 0; i < got.length; i++) {
      const r = got[i];
      out.push({
        rowId: r.rowId >>> 0,
        rowStartAbs: r.rowStartAbs,
        values: r.values,
      });
      wantSet.delete(r.rowId);
      if (out.length >= limit) break;
    }
  }

  if (wantSet.size && out.length < limit) {
    const tail = parser.finish();
    for (let i = 0; i < tail.length; i++) {
      const r = tail[i];
      if (!wantSet.has(r.rowId)) continue;
      out.push({
        rowId: r.rowId >>> 0,
        rowStartAbs: r.rowStartAbs,
        values: r.values,
      });
      wantSet.delete(r.rowId);
      if (out.length >= limit) break;
    }
  }

  return out;
}

module.exports = {
  splitHeaderLine,
  detectDelimiterFromHeaderBytes,
  readHeaderAndFirstDataOffset,
  chooseHeader,
  findColIndex,
  getAllColumnIndexes,
  CsvByteParser,
  getBestLookupStride,
  groupRowIdsByBaseRow,
  findNearestOffset,
  getLookupWindowForBaseRow,
  locateRawRangesFd,
  buildIndex,
  loadOrBuildIndex,
  DelimRowDecoder,
  readPage,
  readRowsByRowIdsSelectiveFd,
  readRowsByRowIdsSelective,
  readRowsByRowIdsSelectiveRecordsFd,
};