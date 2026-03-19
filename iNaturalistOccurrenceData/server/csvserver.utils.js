"use strict";

// ./server/csvserver.utils.js

const fs = require("fs");
const path = require("path");
const os = require("os");

const FILE = process.argv[2] || "occurrences.csv";
const PORT = Number(process.env.PORT || 8787);

const INDEX_STRIDE = Math.max(1, Number(process.env.INDEX_STRIDE || 2000));
const MAX_LIMIT = Math.max(1, Number(process.env.MAX_LIMIT || 1000));
const INDEX_PATH = process.env.INDEX_PATH || `${FILE}.idx.json`;

const TAXA_DIR = process.env.TAXA_DIR || `${FILE}.taxaidx`;
const ROW_STRIDE = Math.max(1, Number(process.env.ROW_STRIDE || 8192));
const PAIR_CHUNK_ROWS = Math.max(
  1,
  Number(process.env.PAIR_CHUNK_ROWS || 2_000_000),
);
const SKIP_EVERY = Math.max(256, Number(process.env.SKIP_EVERY || 4096));
const SEGMENT_MAX_ROWS = Math.max(
  50_000,
  Number(process.env.SEGMENT_MAX_ROWS || 2_000_000),
);
const MERGE_READ_BYTES = Math.max(
  1 << 20,
  Number(process.env.MERGE_READ_BYTES || 8 << 20),
);
const POSTINGS_READ_BYTES = Math.max(
  256 << 10,
  Number(process.env.POSTINGS_READ_BYTES || 4 << 20),
);

const ARGS = new Set(process.argv.slice(3));
const WANT_BUILD_TAXA = ARGS.has("--build-taxa");
const WANT_FINALIZE_TAXA = ARGS.has("--finalize-taxa");

const DEFAULT_EXPORT_ROOT = `${FILE}.exports`;

const TAXON_RANK_NAMES = [
  "root",
  "kingdom",
  "phylum",
  "class",
  "order",
  "family",
  "genus",
  "species",
];

const GUESSED_HEADERS_50 = [
  "gbifID",
  "datasetKey",
  "occurrenceID",
  "kingdom",
  "phylum",
  "class",
  "order",
  "family",
  "genus",
  "species",
  "infraspecificEpithet",
  "taxonRank",
  "scientificName",
  "verbatimScientificName",
  "taxonID",
  "countryCode",
  "country",
  "stateProvince",
  "occurrenceStatus",
  "individualCount",
  "publishingOrgKey",
  "decimalLatitude",
  "decimalLongitude",
  "elevation",
  "coordinateUncertaintyInMeters",
  "coordinatePrecision",
  "geodeticDatum",
  "coordinateDeterminedBy",
  "georeferenceSources",
  "eventDate",
  "day",
  "month",
  "year",
  "taxonKey",
  "acceptedTaxonKey",
  "basisOfRecord",
  "institutionCode",
  "collectionCode",
  "catalogNumber",
  "recordedByID",
  "recordedBy",
  "dateIdentified",
  "license",
  "rightsHolder",
  "identifiedBy",
  "occurrenceRemarks",
  "identificationRemarks",
  "modified",
  "type",
  "issues",
];

function taxonomyRankName(rankId) {
  return TAXON_RANK_NAMES[rankId >>> 0] || "unknown";
}

function captureMemoryStats() {
  const mu = process.memoryUsage();
  return {
    rssBytes: Number(mu.rss || 0),
    heapUsedBytes: Number(mu.heapUsed || 0),
    heapTotalBytes: Number(mu.heapTotal || 0),
    rssMB: Number(((mu.rss || 0) / 1024 ** 2).toFixed(1)),
    heapUsedMB: Number(((mu.heapUsed || 0) / 1024 ** 2).toFixed(1)),
    heapTotalMB: Number(((mu.heapTotal || 0) / 1024 ** 2).toFixed(1)),
  };
}

function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  if (n <= 0) return 0;
  if (n >= 1) return 1;
  return n;
}

function fmtJobCount(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "?";
  return v.toLocaleString();
}

function makeAsciiProgressBar(progress, width = 28) {
  const p = clamp01(progress);
  const fill = Math.round(p * width);
  return `[${"#".repeat(fill)}${"-".repeat(Math.max(0, width - fill))}]`;
}

const JOB_STATE = {
  id: 0,
  active: false,
  done: false,
  type: "",
  title: "",
  phase: "",
  message: "",
  startedAt: null,
  finishedAt: null,
  updatedAt: null,
  error: null,
  result: null,

  progress: 0,
  targetNodeCount: 0,
  plannedSpeciesCount: 0,
  speciesDone: 0,
  speciesTotal: 0,
  rowsDone: 0,
  rowsTotalEstimate: 0,
  fileCount: 0,
  currentSpecies: "",
  currentSpeciesRowsDone: 0,
  currentSpeciesRowsTotal: 0,

  rssBytes: 0,
  heapUsedBytes: 0,
  heapTotalBytes: 0,
  rssMB: 0,
  heapUsedMB: 0,
  heapTotalMB: 0,
};

function getJobState() {
  return { ...JOB_STATE };
}

function setJobState(patch) {
  Object.assign(JOB_STATE, patch || {}, {
    updatedAt: new Date().toISOString(),
    ...captureMemoryStats(),
  });
  return getJobState();
}

async function runExclusiveJob(type, title, fn) {
  if (JOB_STATE.active) {
    const err = new Error(
      `Job already running: ${JOB_STATE.title || JOB_STATE.type || "unknown"}`,
    );
    err.code = "BUSY";
    throw err;
  }

  const nextId = ((JOB_STATE.id || 0) + 1) >>> 0 || 1;

  setJobState({
    id: nextId,
    active: true,
    done: false,
    type,
    title,
    phase: "start",
    message: title,
    startedAt: new Date().toISOString(),
    finishedAt: null,
    error: null,
    result: null,

    progress: 0,
    targetNodeCount: 0,
    plannedSpeciesCount: 0,
    speciesDone: 0,
    speciesTotal: 0,
    rowsDone: 0,
    rowsTotalEstimate: 0,
    fileCount: 0,
    currentSpecies: "",
    currentSpeciesRowsDone: 0,
    currentSpeciesRowsTotal: 0,
  });

  let lastLogAt = 0;
  let lastLogPhase = "";
  let lastLogMessage = "";
  let lastLogProgress = -1;

  function maybeConsoleLog(force) {
    const s = getJobState();
    const now = Date.now();
    const prog = clamp01(s.progress);

    if (
      !force &&
      now - lastLogAt < 1000 &&
      Math.abs(prog - lastLogProgress) < 0.01 &&
      s.phase === lastLogPhase &&
      s.message === lastLogMessage
    ) {
      return;
    }

    lastLogAt = now;
    lastLogPhase = s.phase || "";
    lastLogMessage = s.message || "";
    lastLogProgress = prog;

    const bits = [];
    bits.push(`[job ${s.id}]`);
    bits.push(s.title || s.type || "job");
    bits.push(makeAsciiProgressBar(prog));
    bits.push(`${(prog * 100).toFixed(1)}%`);

    if (s.phase) bits.push(`phase=${s.phase}`);
    if (s.message) bits.push(s.message);

    if (s.speciesTotal > 0 || s.speciesDone > 0) {
      bits.push(
        `species=${fmtJobCount(s.speciesDone)}/${fmtJobCount(s.speciesTotal)}`,
      );
    }

    if (s.rowsTotalEstimate > 0 || s.rowsDone > 0) {
      bits.push(
        `rows=${fmtJobCount(s.rowsDone)}/${fmtJobCount(s.rowsTotalEstimate)}`,
      );
    }

    if (s.fileCount > 0) {
      bits.push(`files=${fmtJobCount(s.fileCount)}`);
    }

    if (s.currentSpecies) {
      bits.push(`current=${s.currentSpecies}`);
    }

    bits.push(`rss=${s.rssMB}MB`);
    bits.push(`heap=${s.heapUsedMB}/${s.heapTotalMB}MB`);

    console.log(bits.join(" | "));
  }

  maybeConsoleLog(true);

  const report = (patch) => {
    if ((JOB_STATE.id >>> 0) !== nextId) return;
    setJobState(patch || {});
    maybeConsoleLog(false);
  };

  try {
    const result = await fn(report);

    report({
      active: false,
      done: true,
      phase: "done",
      message: `${title} complete`,
      finishedAt: new Date().toISOString(),
      error: null,
      result,
      progress: 1,
    });

    maybeConsoleLog(true);
    return result;
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    report({
      active: false,
      done: false,
      phase: "error",
      message: msg,
      finishedAt: new Date().toISOString(),
      error: msg,
    });
    maybeConsoleLog(true);
    throw e;
  }
}

function statFile(p) {
  return fs.promises.stat(p);
}

function fmtGiB(bytes) {
  return (bytes / 1024 ** 3).toFixed(2);
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function maybeGC(tag) {
  if (typeof global.gc !== "function") return;

  const mu = process.memoryUsage();
  const heapUsed = mu.heapUsed || 0;
  const heapTotal = mu.heapTotal || 0;

  if (heapTotal > 0 && heapUsed / heapTotal > 0.85) {
    try {
      global.gc();
      const mu2 = process.memoryUsage();
      process.stdout.write(
        `[gc] ${tag} heap ${(mu.heapUsed / 1024 ** 2).toFixed(0)}MB -> ${(mu2.heapUsed / 1024 ** 2).toFixed(0)}MB\n`,
      );
    } catch (e) {
      console.error(e);
    }
  }
}

function sanitizePathSegment(value) {
  let s = String(value == null ? "" : value).trim();

  if (!s) s = "(missing)";

  s = s.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_");
  s = s.replace(/\s+/g, " ");
  s = s.replace(/[. ]+$/g, "");

  if (!s) s = "_";

  const upper = s.toUpperCase();
  if (/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/.test(upper)) {
    s = "_" + s;
  }

  if (s.length > 180) {
    s = s.slice(0, 180).replace(/[. ]+$/g, "");
  }

  return s || "_";
}

async function writeStreamChunk(ws, chunk) {
  if (!chunk || chunk.length === 0) return;
  if (ws.write(chunk)) return;

  await new Promise((resolve, reject) => {
    function onDrain() {
      cleanup();
      resolve();
    }

    function onError(err) {
      cleanup();
      reject(err);
    }

    function cleanup() {
      ws.off("drain", onDrain);
      ws.off("error", onError);
    }

    ws.on("drain", onDrain);
    ws.on("error", onError);
  });
}


async function readFileRangeFd(fd, start, endExclusive) {
  const startPos = Number(start);
  const endPos = Number(endExclusive);

  if (!Number.isFinite(startPos) || startPos < 0) {
    throw new Error(`Bad start offset: ${start}`);
  }

  if (!Number.isFinite(endPos) || endPos < startPos) {
    throw new Error(`Bad end offset: ${endExclusive}`);
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

async function copyRangeToStream(fd, start, endExclusive, ws, chunkBytes) {
  let pos = Math.max(0, Number(start));
  const end = Math.max(pos, Number(endExclusive));
  const bufSize = Math.max(64 << 10, Number(chunkBytes) | 0);
  const buf = Buffer.allocUnsafe(bufSize);

  while (pos < end) {
    const want = Math.min(buf.length, end - pos);
    const { bytesRead } = await fd.read(buf, 0, want, pos);
    if (bytesRead <= 0) break;
    await writeStreamChunk(ws, buf.subarray(0, bytesRead));
    pos += bytesRead;
  }
}

function csvEscapeValue(value) {
  const s = String(value == null ? "" : value);
  if (/[",\r\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

function send(res, code, contentType, body) {
  res.writeHead(code, {
    "content-type": contentType,
    "cache-control": "no-store",
  });
  res.end(body);
}

function sendJson(res, code, obj) {
  send(res, code, "application/json; charset=utf-8", JSON.stringify(obj));
}

module.exports = {
  fs,
  path,
  os,

  FILE,
  PORT,
  INDEX_STRIDE,
  MAX_LIMIT,
  INDEX_PATH,
  TAXA_DIR,
  ROW_STRIDE,
  PAIR_CHUNK_ROWS,
  SKIP_EVERY,
  SEGMENT_MAX_ROWS,
  MERGE_READ_BYTES,
  POSTINGS_READ_BYTES,
  WANT_BUILD_TAXA,
  WANT_FINALIZE_TAXA,
  DEFAULT_EXPORT_ROOT,
  TAXON_RANK_NAMES,
  GUESSED_HEADERS_50,

  taxonomyRankName,
  captureMemoryStats,
  clamp01,
  fmtJobCount,
  makeAsciiProgressBar,
  getJobState,
  setJobState,
  runExclusiveJob,
  statFile,
  fmtGiB,
  ensureDir,
  maybeGC,
  sanitizePathSegment,
  writeStreamChunk,
  readFileRangeFd,
  copyRangeToStream,
  csvEscapeValue,
  send,
  sendJson,
};