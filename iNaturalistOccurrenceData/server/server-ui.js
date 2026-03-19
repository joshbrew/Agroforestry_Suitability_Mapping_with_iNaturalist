"use strict";

// ./server/server-ui.js

const fs = require("fs");
const http = require("http");
const path = require("path");

const RANK_NAMES = [
  "root",
  "kingdom",
  "phylum",
  "class",
  "order",
  "family",
  "genus",
  "species",
];

const UI_DIR = path.join(__dirname, "ui");
const UI_HTML_PATH = path.join(UI_DIR, "index.html");
const UI_CSS_PATH = path.join(UI_DIR, "style.css");
const UI_JS_PATH = path.join(UI_DIR, "app.js");

function rankName(rankId) {
  return RANK_NAMES[rankId] || "unknown";
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

function sendText(res, code, body) {
  send(res, code, "text/plain; charset=utf-8", body);
}

function sendHtml(res, code, body) {
  send(res, code, "text/html; charset=utf-8", body);
}

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

function helpText({ scriptName, fileDefault }) {
  return [
    "CSV/TSV Pager + Taxonomy Index Server",
    "",
    "Interactive UI:",
    "  GET /",
    "",
    "JSON Endpoints:",
    "  GET /meta",
    "  GET /job/status",
    "  GET /page?start=0&limit=200&as=array|object&header=file|guess",
    "  GET /tree/root?start=0&limit=200&sort=id|name|count|count_desc",
    "  GET /tree/children?nodeId=123&start=0&limit=200&sort=id|name|count|count_desc",
    "  GET /tree/node?nodeId=123",
    "  GET /species/page?speciesNodeId=456&start=0&limit=200&header=guess|file",
    "  POST /admin/build-taxa",
    "  POST /admin/finalize-taxa",
    "  POST /export/node",
    "",
    "Build taxonomy index (CLI):",
    `  node ${scriptName} ${fileDefault} --build-taxa`,
    `  node ${scriptName} ${fileDefault} --finalize-taxa`,
    "",
  ].join("\n");
}

function parseIntParam(url, key, def) {
  const raw = url.searchParams.get(key);
  if (raw == null || raw === "") return def;
  const n = Number(raw);
  if (!Number.isFinite(n)) return def;
  return (n | 0) >>> 0;
}

function clampLimit(n, max) {
  const x = Number(n);
  if (!Number.isFinite(x) || x <= 0) return 1;
  return Math.min(max, x | 0);
}

function normName(s) {
  const t = String(s || "").trim();
  return t.length ? t : "(missing)";
}

function childSummaries(taxa, ids) {
  const out = new Array(ids.length);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i] >>> 0;
    const n = taxa.readNode(id);
    out[i] = {
      nodeId: id,
      parentId: n.parentId,
      rankId: n.rankId,
      rank: rankName(n.rankId),
      name: normName(taxa.getString(n.nameId)),
      count: n.count,
    };
  }
  return out;
}

function sortChildrenSmallInPlace(taxa, ids, sort) {
  const mode = String(sort || "id");
  if (mode === "id" || mode === "") {
    ids.sort((a, b) => (a - b) | 0);
    return ids;
  }

  const decorated = new Array(ids.length);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i] >>> 0;
    const n = taxa.readNode(id);
    decorated[i] = {
      id,
      name: normName(taxa.getString(n.nameId)),
      count: n.count >>> 0,
    };
  }

  if (mode === "name") {
    decorated.sort((a, b) => {
      const c = a.name.localeCompare(b.name);
      if (c) return c;
      return (a.id - b.id) | 0;
    });
  } else if (mode === "count") {
    decorated.sort((a, b) => {
      const c = (a.count - b.count) | 0;
      if (c) return c;
      return (a.id - b.id) | 0;
    });
  } else if (mode === "count_desc") {
    decorated.sort((a, b) => {
      const c = (b.count - a.count) | 0;
      if (c) return c;
      return (a.id - b.id) | 0;
    });
  } else {
    decorated.sort((a, b) => (a.id - b.id) | 0);
  }

  for (let i = 0; i < ids.length; i++) ids[i] = decorated[i].id >>> 0;
  return ids;
}

function readChildrenPaged(
  taxa,
  nodeId,
  start,
  limit,
  sort,
  maxLimit,
  sortMaxChildren,
) {
  const total = taxa.childCount(nodeId) >>> 0;
  const safeStart = Math.min(total, Math.max(0, start | 0));
  const safeLimit = clampLimit(limit, maxLimit);
  const end = Math.min(total, safeStart + safeLimit);

  const mode = String(sort || "id");

  if (mode === "id" || mode === "") {
    const page = taxa.readChildrenPage(nodeId, safeStart, safeLimit);
    const children = childSummaries(taxa, page.ids);

    const returned = children.length >>> 0;
    const nextStart = safeStart + returned;
    const hasMore = nextStart < total;

    return {
      nodeId,
      start: safeStart,
      limit: safeLimit,
      returned,
      total,
      hasMore,
      nextStart: hasMore ? nextStart : null,
      sort: "id",
      children,
    };
  }

  if (total > sortMaxChildren >>> 0) {
    return {
      error: `Refusing sort="${mode}" for nodeId=${nodeId} with ${total} children (cap=${sortMaxChildren}). Use sort=id or drill down.`,
    };
  }

  const all = taxa.readChildrenAllBounded(nodeId, sortMaxChildren >>> 0);
  if (!all) {
    return {
      error: `Refusing sort="${mode}" for nodeId=${nodeId} (cap=${sortMaxChildren}).`,
    };
  }

  sortChildrenSmallInPlace(taxa, all.ids, mode);

  const pageIds = all.ids.slice(safeStart, end);
  const children = childSummaries(taxa, pageIds);

  const returned = children.length >>> 0;
  const nextStart = safeStart + returned;
  const hasMore = nextStart < total;

  return {
    nodeId,
    start: safeStart,
    limit: safeLimit,
    returned,
    total,
    hasMore,
    nextStart: hasMore ? nextStart : null,
    sort: mode,
    children,
  };
}

function readStaticFile(filePath) {
  return fs.promises.readFile(filePath);
}

function createServer({ idx, getTaxa, reloadTaxa, config, fns }) {
  const {
    fileDefault,
    maxLimit,
    treeMaxLimit,
    treeSortMaxChildren,
    taxaDir,
    guessedHeaderColumns,
    scriptPath,
    defaultExportRoot,
  } = config;

  const maxTree = Math.max(1, Number(treeMaxLimit || maxLimit || 1000));
  const sortMax = Math.max(10_000, Number(treeSortMaxChildren || 200_000));

  const {
    readPage,
    chooseHeader,
    readSpeciesRows,
    buildTaxaPhase1,
    finalizeTaxaPhase2,
    exportTaxaNode,
    getJobState,
  } = fns;

  return http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      const taxa = typeof getTaxa === "function" ? getTaxa() : null;

      if (req.method === "GET" && url.pathname === "/") {
        const html = await readStaticFile(UI_HTML_PATH);
        return sendHtml(res, 200, html);
      }

      if (req.method === "GET" && url.pathname === "/help") {
        return sendText(
          res,
          200,
          helpText({ scriptName: path.basename(scriptPath), fileDefault }),
        );
      }

      if (req.method === "GET" && url.pathname === "/ui/style.css") {
        const css = await readStaticFile(UI_CSS_PATH);
        return send(res, 200, "text/css; charset=utf-8", css);
      }

      if (req.method === "GET" && url.pathname === "/ui/app.js") {
        const js = await readStaticFile(UI_JS_PATH);
        return send(res, 200, "application/javascript; charset=utf-8", js);
      }

      if (req.method === "GET" && url.pathname === "/meta") {
        const header = idx.header || [];
        const metaPath = path.join(path.resolve(taxaDir), "meta.json");

        let taxaMetaExists = false;
        let taxaPhase2Ready = false;

        try {
          if (fs.existsSync(metaPath)) {
            taxaMetaExists = true;
            const raw = JSON.parse(fs.readFileSync(metaPath, "utf8"));
            taxaPhase2Ready = !!raw.phase2;
          }
        } catch {}

        return sendJson(res, 200, {
          file: idx.file,
          fileSize: idx.fileSize,
          fileSizeGiB: Number((idx.fileSize / 1024 ** 3).toFixed(3)),
          delimiter: idx.delimiter,
          indexStride: idx.indexStride,
          indexedPoints: idx.index.length,
          headerColumns: header.length,
          header,
          guessedHeaderColumns,
          createdAt: idx.createdAt,
          dataRowCountEstimate: idx.dataRowCountEstimate,
          taxaIndexLoaded: !!taxa,
          taxaDir: path.resolve(taxaDir),
          taxaMeta: taxa ? taxa.meta.phase2 : null,
          taxaMetaExists,
          taxaPhase2Ready,
          treeMaxLimit: maxTree,
          treeSortMaxChildren: sortMax,
          defaultExportRoot,
        });
      }

      if (req.method === "GET" && url.pathname === "/job/status") {
        return sendJson(res, 200, getJobState());
      }

      if (req.method === "GET" && url.pathname === "/page") {
        const start = Number(url.searchParams.get("start") || "0");
        const limit = Number(url.searchParams.get("limit") || "200");
        const as = String(url.searchParams.get("as") || "array");
        const headerMode = String(url.searchParams.get("header") || "file");
        const asObject = as === "object";

        if (!Number.isFinite(start) || start < 0) {
          return sendJson(res, 400, { error: "start must be >= 0" });
        }
        if (!Number.isFinite(limit) || limit <= 0) {
          return sendJson(res, 400, { error: "limit must be > 0" });
        }

        const headerOverride = chooseHeader(idx, headerMode);
        const page = await readPage(
          idx,
          start,
          limit,
          asObject,
          headerOverride,
        );
        return sendJson(res, 200, page);
      }

      if (req.method === "GET" && url.pathname === "/tree/node") {
        if (!taxa) return sendJson(res, 400, { error: "taxa index not built" });

        const nodeId = Number(url.searchParams.get("nodeId") || "0") | 0;
        if (nodeId < 0 || nodeId >= taxa.nodeCount) {
          return sendJson(res, 400, { error: "bad nodeId" });
        }

        const n = taxa.readNode(nodeId);
        return sendJson(res, 200, {
          nodeId,
          parentId: n.parentId,
          rankId: n.rankId,
          rank: rankName(n.rankId),
          name: normName(taxa.getString(n.nameId)),
          count: n.count,
          childCount: taxa.childCount(nodeId) >>> 0,
        });
      }

      if (req.method === "GET" && url.pathname === "/tree/root") {
        if (!taxa) return sendJson(res, 400, { error: "taxa index not built" });

        const start = parseIntParam(url, "start", 0);
        const limit = parseIntParam(url, "limit", 200);
        const sort = String(url.searchParams.get("sort") || "id");

        const payload = readChildrenPaged(
          taxa,
          0,
          start,
          limit,
          sort,
          maxTree,
          sortMax,
        );

        if (payload.error) return sendJson(res, 400, { error: payload.error });
        return sendJson(res, 200, payload);
      }

      if (req.method === "GET" && url.pathname === "/tree/children") {
        if (!taxa) return sendJson(res, 400, { error: "taxa index not built" });

        const nodeId = Number(url.searchParams.get("nodeId") || "0") | 0;
        if (nodeId < 0 || nodeId >= taxa.nodeCount) {
          return sendJson(res, 400, { error: "bad nodeId" });
        }

        const start = parseIntParam(url, "start", 0);
        const limit = parseIntParam(url, "limit", 200);
        const sort = String(url.searchParams.get("sort") || "id");

        const payload = readChildrenPaged(
          taxa,
          nodeId,
          start,
          limit,
          sort,
          maxTree,
          sortMax,
        );

        if (payload.error) return sendJson(res, 400, { error: payload.error });
        return sendJson(res, 200, payload);
      }

      if (req.method === "GET" && url.pathname === "/species/page") {
        if (!taxa) return sendJson(res, 400, { error: "taxa index not built" });

        const speciesNodeId =
          Number(url.searchParams.get("speciesNodeId") || "0") | 0;
        const start = Number(url.searchParams.get("start") || "0") | 0;
        const limit = Math.min(
          maxLimit,
          Number(url.searchParams.get("limit") || "200") | 0,
        );
        const headerMode = String(url.searchParams.get("header") || "guess");
        const colsRaw = String(url.searchParams.get("cols") || "").trim();
        const requestedColNames = colsRaw
          ? colsRaw
              .split(",")
              .map((s) => String(s || "").trim())
              .filter(Boolean)
          : null;
        const headerOverride = chooseHeader(idx, headerMode);

        const result = await readSpeciesRows(
          idx,
          taxa,
          speciesNodeId,
          start,
          limit,
          headerOverride,
          requestedColNames,
        );
        if (result.error) return sendJson(res, 400, result);

        return sendJson(res, 200, {
          ...result,
          delimiter: idx.delimiter,
          header: result.header || headerOverride,
        });
      }

      if (req.method === "POST" && url.pathname === "/admin/build-taxa") {
        await readJsonBody(req);
        try {
          const result = await buildTaxaPhase1();
          return sendJson(res, 200, { ok: true, result, job: getJobState() });
        } catch (e) {
          if (e && e.code === "BUSY") {
            return sendJson(res, 409, {
              error: String(e.message || e),
              job: getJobState(),
            });
          }
          throw e;
        }
      }

      if (req.method === "POST" && url.pathname === "/admin/finalize-taxa") {
        await readJsonBody(req);
        try {
          const result = await finalizeTaxaPhase2();
          if (typeof reloadTaxa === "function") await reloadTaxa();
          return sendJson(res, 200, { ok: true, result, job: getJobState() });
        } catch (e) {
          if (e && e.code === "BUSY") {
            return sendJson(res, 409, {
              error: String(e.message || e),
              job: getJobState(),
            });
          }
          throw e;
        }
      }

      if (req.method === "POST" && url.pathname === "/export/node") {
        const body = await readJsonBody(req);
        try {
          const result = await exportTaxaNode({
            scope: String(body.scope || "selected"),
            nodeId: Number(body.nodeId || 0) | 0,
            currentNodeId: Number(body.currentNodeId || 0) | 0,
            nodeIds: Array.isArray(body.nodeIds) ? body.nodeIds : [],
            mode: String(body.mode || "coords"),
            outRoot: String(body.outRoot || "").trim(),
          });
          return sendJson(res, 200, { ok: true, result, job: getJobState() });
        } catch (e) {
          if (e && e.code === "BUSY") {
            return sendJson(res, 409, {
              error: String(e.message || e),
              job: getJobState(),
            });
          }
          throw e;
        }
      }

      return sendText(res, 404, "Not found");
    } catch (e) {
      return sendJson(res, 500, {
        error: String(e && e.message ? e.message : e),
      });
    }
  });
}

module.exports = {
  createServer,
};
