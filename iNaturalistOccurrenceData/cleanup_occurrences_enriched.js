#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const readline = require("readline");

function parseArgs(argv) {
  const out = {
    target: "",
    mode: "invalid",
    inplace: false,
    backup: true,
    outPath: "",
    tempSuffix: ".cleaning.tmp",
    backupSuffix: ".bak",
    progressEvery: 250000,
  };

  function needValue(i, flag) {
    if (i + 1 >= argv.length) throw new Error(`Missing value for ${flag}`);
    return argv[i + 1];
  }

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    } else if (a === "--mode") {
      out.mode = String(needValue(i, a)).toLowerCase();
      i++;
    } else if (a === "--out") {
      out.outPath = String(needValue(i, a));
      i++;
    } else if (a === "--inplace") {
      out.inplace = true;
    } else if (a === "--no-backup") {
      out.backup = false;
    } else if (a === "--backup-suffix") {
      out.backupSuffix = String(needValue(i, a));
      i++;
    } else if (a === "--temp-suffix") {
      out.tempSuffix = String(needValue(i, a));
      i++;
    } else if (a === "--progress-every") {
      out.progressEvery = Math.max(1, Number(needValue(i, a)) | 0);
      i++;
    } else if (!out.target) {
      out.target = a;
    } else {
      throw new Error(`Unknown option: ${a}`);
    }
  }

  if (!out.target) {
    printHelp();
    throw new Error("Missing target path");
  }

  if (!["none", "invalid", "soilgrids", "invalid+soilgrids"].includes(out.mode)) {
    throw new Error(`Bad --mode value: ${out.mode}`);
  }

  return out;
}

function printHelp() {
  console.log([
    "Usage:",
    "  node cleanup_occurrences_enriched.js <out-root-or-csv> [options]",
    "",
    "Target:",
    "  You can pass either:",
    "    - the out-root folder that contains occurrences_enriched.csv",
    "    - or the full path to occurrences_enriched.csv",
    "",
    "Modes:",
    "  --mode invalid             drop rows with missing / bad decimalLatitude or decimalLongitude",
    "  --mode soilgrids           drop rows with an empty SoilGrids payload",
    "  --mode invalid+soilgrids   do both",
    "  --mode none                copy through unchanged",
    "",
    "Output:",
    "  --inplace                  replace occurrences_enriched.csv in place",
    "  --out FILE                 write to a specific output path",
    "  --no-backup                with --inplace, do not keep a .bak copy",
    "",
    "Examples:",
    "  node cleanup_occurrences_enriched.js D:/envpull_daucus --mode invalid+soilgrids --inplace",
    "  node cleanup_occurrences_enriched.js D:/envpull_daucus/occurrences_enriched.csv --mode invalid",
    "  node cleanup_occurrences_enriched.js D:/envpull_daucus --mode soilgrids --out D:/envpull_daucus/occurrences_enriched.clean.csv",
  ].join("\n"));
}

function nowIso() {
  return new Date().toISOString();
}

function csvEscapeValue(v) {
  const s = String(v ?? "");
  if (s.includes('"') || s.includes(",") || s.includes("\n") || s.includes("\r")) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }

  out.push(cur);
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

function resolveInputCsv(target) {
  const p = path.resolve(target);
  const st = fs.statSync(p);
  if (st.isDirectory()) return path.join(p, "occurrences_enriched.csv");
  return p;
}

function defaultOutputPath(inCsvPath, outPath, inplace) {
  if (outPath) return path.resolve(outPath);
  if (inplace) return inCsvPath + ".cleaning.tmp";
  const dir = path.dirname(inCsvPath);
  const ext = path.extname(inCsvPath);
  const base = path.basename(inCsvPath, ext);
  return path.join(dir, `${base}.cleaned${ext || ".csv"}`);
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

function makePredicate(header, mode) {
  const latIdx = header.indexOf("decimalLatitude");
  const lonIdx = header.indexOf("decimalLongitude");
  const soilgridsIdxs = detectSoilgridsValueColumns(header);

  function isInvalidRow(vals) {
    if (latIdx < 0 || lonIdx < 0) return false;
    return !isValidLatLon(vals[latIdx], vals[lonIdx]);
  }

  function isMissingSoilgrids(vals) {
    if (!soilgridsIdxs.length) return false;
    for (let i = 0; i < soilgridsIdxs.length; i++) {
      if (!isBlankLike(vals[soilgridsIdxs[i]])) return false;
    }
    return true;
  }

  return {
    latIdx,
    lonIdx,
    soilgridsIdxs,
    shouldDrop(vals) {
      if (mode === "none") return false;
      if (mode === "invalid") return isInvalidRow(vals);
      if (mode === "soilgrids") return isMissingSoilgrids(vals);
      if (mode === "invalid+soilgrids") return isInvalidRow(vals) || isMissingSoilgrids(vals);
      return false;
    },
    invalidRow: isInvalidRow,
    missingSoilgrids: isMissingSoilgrids,
  };
}

async function cleanupCsv({ inCsvPath, outCsvPath, mode, progressEvery }) {
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
        predicate = makePredicate(header, mode);
        ws.write(header.map(csvEscapeValue).join(",") + "\n");
        console.log(`[cleanup] header columns=${header.length} soilgrids_value_columns=${predicate.soilgridsIdxs.length}`);
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

async function finalizeInplace({ inCsvPath, tempOutPath, backup, backupSuffix }) {
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

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const inCsvPath = resolveInputCsv(opts.target);
  const outCsvPath = defaultOutputPath(inCsvPath, opts.outPath, opts.inplace);

  await fs.promises.access(inCsvPath, fs.constants.R_OK);
  await fs.promises.mkdir(path.dirname(outCsvPath), { recursive: true });

  console.log(`[cleanup] started=${nowIso()}`);
  console.log(`[cleanup] in=${inCsvPath}`);
  console.log(`[cleanup] mode=${opts.mode}`);
  console.log(`[cleanup] out=${outCsvPath}`);

  const stats = await cleanupCsv({
    inCsvPath,
    outCsvPath,
    mode: opts.mode,
    progressEvery: opts.progressEvery,
  });

  let backupPath = "";
  if (opts.inplace) {
    backupPath = await finalizeInplace({
      inCsvPath,
      tempOutPath: outCsvPath,
      backup: opts.backup,
      backupSuffix: opts.backupSuffix,
    });
  }

  console.log(`[cleanup] finished=${nowIso()}`);
  console.log(`[cleanup] rows_read=${stats.rowsRead.toLocaleString()}`);
  console.log(`[cleanup] rows_written=${stats.rowsWritten.toLocaleString()}`);
  console.log(`[cleanup] dropped_invalid=${stats.droppedInvalid.toLocaleString()}`);
  console.log(`[cleanup] dropped_soilgrids=${stats.droppedSoilgrids.toLocaleString()}`);
  console.log(`[cleanup] dropped_both=${stats.droppedBoth.toLocaleString()}`);
  if (opts.inplace && backupPath) console.log(`[cleanup] backup=${backupPath}`);
  console.log(`[cleanup] final=${opts.inplace ? inCsvPath : outCsvPath}`);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
