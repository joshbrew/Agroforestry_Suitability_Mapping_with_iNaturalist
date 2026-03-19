"use strict";

//server/ui/app.js

const UI_OCCURRENCE_COLS = [
  "gbifID",
  "occurrenceID",
  "references",
  "identifier",
  "catalogNumber",
  "recordNumber",
  "scientificName",
  "species",
  "genus",
  "family",
  "country",
  "stateProvince",
  "eventDate",
  "year",
  "month",
  "day",
  "decimalLatitude",
  "decimalLongitude",
  "basisOfRecord",
];

function byId(id) {
  return document.getElementById(id);
}

async function fetchJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  const txt = await r.text();
  let j = null;
  try {
    j = JSON.parse(txt);
  } catch {}
  if (!r.ok) {
    const msg = j && j.error ? j.error : txt || "HTTP " + r.status;
    throw new Error(msg);
  }
  return j;
}

async function postJson(url, body) {
  const r = await fetch(url, {
    method: "POST",
    cache: "no-store",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const txt = await r.text();
  let j = null;
  try {
    j = JSON.parse(txt);
  } catch {}
  if (!r.ok) {
    const msg = j && j.error ? j.error : txt || "HTTP " + r.status;
    throw new Error(msg);
  }
  return j || {};
}

let _map = null;
let _pointsLayer = null;
let _jobPoll = 0;

function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  if (n <= 0) return 0;
  if (n >= 1) return 1;
  return n;
}

function initMapOnce() {
  if (_map) return;

  _map = L.map("map", {
    zoomControl: true,
    preferCanvas: true,
    worldCopyJump: true,
  }).setView([20, 0], 2);

  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(_map);

  _pointsLayer = new CanvasPointsLayer([]);
  _pointsLayer.addTo(_map);
}

class CanvasPointsLayer extends L.Layer {
  constructor(points) {
    super();
    this._points = Array.isArray(points) ? points : [];
    this._canvas = null;
    this._ctx = null;
    this._map = null;
    this._origin = L.point(0, 0);
    this._raf = 0;
  }

  setPoints(points) {
    this._points = Array.isArray(points) ? points : [];
    this._scheduleRedraw(true);
  }

  onAdd(map) {
    this._map = map;

    const canvas = (this._canvas = L.DomUtil.create("canvas", "leaflet-layer"));
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.pointerEvents = "none";
    canvas.style.background = "transparent";

    this._ctx = canvas.getContext("2d", { alpha: true });

    map.getPanes().overlayPane.appendChild(canvas);

    map.on("moveend", this._reset, this);
    map.on("zoomend", this._reset, this);
    map.on("resize", this._resize, this);

    this._resize();
    this._reset();
  }

  _redraw() {
    if (!this._map || !this._canvas || !this._ctx) return;

    const map = this._map;
    const ctx = this._ctx;
    const size = map.getSize();
    const dpr = this._dpr || 1;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, size.x, size.y);

    const pts = this._points;
    if (!pts.length) return;

    const bounds = map.getBounds();

    const z = map.getZoom();
    let px = 3 + Math.floor((z - 2) * 0.4);
    if (pts.length > 500) px = Math.max(2, px - 1);
    if (pts.length > 2000) px = 2;
    px = Math.max(2, Math.min(10, px));

    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "#ff0d00ff";

    const half = px * 0.5;
    const origin = this._origin;

    for (let i = 0; i < pts.length; i++) {
      const lat = pts[i][0];
      const lon = pts[i][1];
      if (!bounds.contains([lat, lon])) continue;

      const p = map.latLngToLayerPoint([lat, lon]).subtract(origin);
      ctx.fillRect(p.x - half, p.y - half, px, px);
    }
  }

  onRemove(map) {
    map.off("moveend", this._reset, this);
    map.off("zoomend", this._reset, this);
    map.off("resize", this._resize, this);

    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;

    if (this._canvas && this._canvas.parentNode) {
      this._canvas.parentNode.removeChild(this._canvas);
    }

    this._canvas = null;
    this._ctx = null;
    this._map = null;
  }

  _resize() {
    if (!this._map || !this._canvas) return;

    const size = this._map.getSize();
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));

    this._canvas.width = Math.floor(size.x * dpr);
    this._canvas.height = Math.floor(size.y * dpr);
    this._canvas.style.width = size.x + "px";
    this._canvas.style.height = size.y + "px";

    this._dpr = dpr;
    this._scheduleRedraw(true);
  }

  _reset() {
    if (!this._map || !this._canvas) return;

    const topLeft = this._map.containerPointToLayerPoint([0, 0]);
    this._origin = topLeft;
    L.DomUtil.setPosition(this._canvas, topLeft);

    this._scheduleRedraw(true);
  }

  _scheduleRedraw(force) {
    if (!this._map || !this._canvas || !this._ctx) return;
    if (this._raf && !force) return;

    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this._redraw();
    });
  }
}

function updateMapFromOccurrencePayload(payload) {
  const header = payload.header || [];
  const rows = payload.rows || [];

  const latI = header.indexOf("decimalLatitude");
  const lonI = header.indexOf("decimalLongitude");

  const elMap = byId("map");
  if (latI < 0 || lonI < 0) {
    if (elMap) elMap.style.display = "none";
    return 0;
  }

  const pts = [];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const lat = Number(r[latI]);
    const lon = Number(r[lonI]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) continue;
    pts.push([lat, lon]);
  }

  if (elMap) elMap.style.display = "block";

  initMapOnce();
  _map.invalidateSize(true);
  _pointsLayer.setPoints(pts);

  if (pts.length) {
    const bounds = L.latLngBounds(pts);
    _map.fitBounds(bounds.pad(0.1), { maxZoom: 10 });
  }

  return pts.length;
}

function esc(s) {
  return String(s == null ? "" : s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtInt(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "";
  return x.toLocaleString();
}

function pickOccurrenceColumns(header) {
  const prefer = [
    "gbifID",
    "scientificName",
    "species",
    "genus",
    "family",
    "country",
    "stateProvince",
    "eventDate",
    "year",
    "month",
    "day",
    "decimalLatitude",
    "decimalLongitude",
    "basisOfRecord",
  ];

  const idx = new Map();
  for (let i = 0; i < header.length; i++) idx.set(String(header[i]), i);

  const cols = [];
  for (const k of prefer) {
    if (idx.has(k)) cols.push(k);
    if (cols.length >= 10) break;
  }
  if (cols.length >= 4) return cols;

  const out = [];
  for (let i = 0; i < Math.min(10, header.length); i++) out.push(header[i]);
  return out;
}

function normalizeInatUrl(urlOrId) {
  const s = String(urlOrId || "").trim();
  if (!s) return null;

  const mUrl = s.match(
    /https?:\/\/(?:www\.)?inaturalist\.org\/observations\/(\d+)/i,
  );
  if (mUrl) return "https://www.inaturalist.org/observations/" + mUrl[1];

  if (/\binaturalist\b|\binat\b/i.test(s)) {
    const mId = s.match(/(\d{2,})/);
    if (mId) return "https://www.inaturalist.org/observations/" + mId[1];
  }

  return null;
}

function inferInatUrlFromRow(header, row) {
  if (!header || !row) return null;

  const lower = new Array(header.length);
  for (let i = 0; i < header.length; i++) {
    lower[i] = String(header[i] || "").toLowerCase();
  }

  function col(name) {
    const k = String(name || "").toLowerCase();
    for (let i = 0; i < lower.length; i++) {
      if (lower[i] === k) return i;
    }
    return -1;
  }

  const primaryCols = [
    "occurrenceID",
    "references",
    "identifier",
    "catalogNumber",
    "recordNumber",
  ];

  for (let i = 0; i < primaryCols.length; i++) {
    const ci = col(primaryCols[i]);
    if (ci >= 0) {
      const u = normalizeInatUrl(row[ci]);
      if (u) return u;
    }
  }

  for (let i = 0; i < lower.length; i++) {
    const name = lower[i];
    if (!name) continue;
    if (name.includes("inaturalist") || name.includes("inat")) {
      const u = normalizeInatUrl(row[i]);
      if (u) return u;

      const s = String(row[i] || "").trim();
      if (s && /^\d{2,}$/.test(s)) {
        return "https://www.inaturalist.org/observations/" + s;
      }
    }
  }

  for (let i = 0; i < row.length; i++) {
    const s = String(row[i] || "").trim();
    if (!s) continue;
    if (s.indexOf("inaturalist.org") !== -1) {
      const u = normalizeInatUrl(s);
      if (u) return u;
    }
  }

  return null;
}

function gbifUrlFromRow(header, row) {
  if (!header || !row) return null;
  const gbifI = header.indexOf("gbifID");
  if (gbifI >= 0) {
    const id = String(row[gbifI] || "").trim();
    if (id && /^\d+$/.test(id)) {
      return "https://www.gbif.org/occurrence/" + id;
    }
  }
  return null;
}

const state = {
  meta: null,
  taxaLoaded: false,

  nodeId: 0,
  parentId: 0,
  treeStart: 0,
  treeLimit: 200,
  treeSort: "id",
  loadedTreeNodeIds: [],
  loadedTreeRank: "root",
  loadedTreeParentName: "(root)",
  loadedTreeTotal: 0,
  loadedTreeReturned: 0,

  crumbs: [{ nodeId: 0, name: "(root)", rank: "root" }],

  speciesNodeId: -1,
  occStart: 0,
  occLimit: 200,

  selectedNodeId: 0,
  selectedRankId: 0,
  selectedRank: "root",
  selectedName: "(root)",
};

const el = {
  crumbs: byId("crumbs"),
  btnUp: byId("btnUp"),
  sort: byId("sort"),
  treeLimit: byId("treeLimit"),
  btnPrev: byId("btnPrev"),
  btnNext: byId("btnNext"),
  treeStatus: byId("treeStatus"),
  treeBody: byId("treeBody"),

  occTitle: byId("occTitle"),
  occLimit: byId("occLimit"),
  btnOccPrev: byId("btnOccPrev"),
  btnOccNext: byId("btnOccNext"),
  occJsonLink: byId("occJsonLink"),
  occStatus: byId("occStatus"),
  occHead: byId("occHead"),
  occBody: byId("occBody"),

  indexSection: byId("indexSection"),
  btnBuildTaxa: byId("btnBuildTaxa"),
  btnFinalizeTaxa: byId("btnFinalizeTaxa"),

  jobStatus: byId("jobStatus"),
  jobProgressFill: byId("jobProgressFill"),
  jobProgressMeta: byId("jobProgressMeta"),

  exportNodeInfo: byId("exportNodeInfo"),
  exportLevelInfo: byId("exportLevelInfo"),
  exportMode: byId("exportMode"),
  exportRoot: byId("exportRoot"),
  btnExportSelected: byId("btnExportSelected"),
  btnExportLevelAll: byId("btnExportLevelAll"),
  btnExportLoaded: byId("btnExportLoaded"),
  exportResult: byId("exportResult"),
  adminMeta: byId("adminMeta"),
};

function setTreeStatus(msg, isErr) {
  el.treeStatus.innerHTML = isErr
    ? '<span class="err">' + esc(msg) + "</span>"
    : esc(msg);
}

function setOccStatus(msg, isErr) {
  el.occStatus.innerHTML = isErr
    ? '<span class="err">' + esc(msg) + "</span>"
    : esc(msg);
}

function setExportResultHtml(html) {
  el.exportResult.innerHTML = html || "";
}

function setSelectedNode(info) {
  state.selectedNodeId = info.nodeId | 0;
  state.selectedRankId = info.rankId | 0;
  state.selectedRank = String(info.rank || "unknown");
  state.selectedName = String(info.name || "(missing)");
  renderSelectedNode();
}

function renderSelectedNode() {
  const loadedCount = state.loadedTreeNodeIds.length;
  const returned = Number(state.loadedTreeReturned || 0);
  const total = Number(state.loadedTreeTotal || 0);

  el.exportNodeInfo.innerHTML =
    "Selected node: " +
    esc(state.selectedName) +
    " · " +
    esc(state.selectedRank) +
    " · nodeId " +
    esc(state.selectedNodeId);

  el.exportLevelInfo.innerHTML =
    "Current level: children of " +
    esc(state.loadedTreeParentName) +
    " · loaded " +
    esc(fmtInt(loadedCount)) +
    " node(s)" +
    (returned > 0 || total > 0
      ? " · page " +
        esc(fmtInt(state.treeStart)) +
        "–" +
        esc(fmtInt(state.treeStart + returned)) +
        " of " +
        esc(fmtInt(total))
      : "");

  syncExportControls();
}

function renderAdminMeta() {
  const m = state.meta;
  if (!m) {
    el.adminMeta.innerHTML = "";
    return;
  }

  let buildState = "none";
  if (m.taxaPhase2Ready) buildState = "finalized";
  else if (m.taxaMetaExists) buildState = "phase 1";

  const bits = [];
  bits.push(
    "Taxa index: " +
      (m.taxaIndexLoaded
        ? '<span class="ok">loaded</span>'
        : '<span class="err">not loaded</span>'),
  );
  bits.push('Build state: <span class="mono">' + esc(buildState) + "</span>");
  bits.push(
    'CSV size: <span class="mono">' +
      esc(String(m.fileSizeGiB)) +
      " GiB</span>",
  );
  bits.push('Taxa dir: <span class="mono">' + esc(m.taxaDir || "") + "</span>");
  bits.push(
    'Default export root: <span class="mono">' +
      esc(m.defaultExportRoot || "") +
      "</span>",
  );
  bits.push(
    'Tree page cap: <span class="mono">' +
      esc(fmtInt(m.treeMaxLimit || 0)) +
      "</span>",
  );
  bits.push(
    'Tree sort cap: <span class="mono">' +
      esc(fmtInt(m.treeSortMaxChildren || 0)) +
      "</span>",
  );
  bits.push(
    'Header columns: <span class="mono">' +
      esc(fmtInt(m.headerColumns || 0)) +
      "</span>",
  );
  el.adminMeta.innerHTML = bits.join("<br/>");
}

function syncIndexSection() {
  const m = state.meta;
  if (!m) return;

  if (m.taxaIndexLoaded || m.taxaPhase2Ready) {
    el.indexSection.style.display = "none";
    return;
  }

  el.indexSection.style.display = "";

  if (m.taxaMetaExists) {
    el.btnBuildTaxa.style.display = "none";
    el.btnFinalizeTaxa.style.display = "";
  } else {
    el.btnBuildTaxa.style.display = "";
    el.btnFinalizeTaxa.style.display = "none";
  }
}

function syncExportControls() {
  const canExport = !!state.taxaLoaded;
  const hasLoadedSubset = state.loadedTreeNodeIds.length > 0;

  el.btnExportSelected.disabled = !canExport;
  el.btnExportLevelAll.disabled = !canExport;
  el.btnExportLoaded.disabled = !canExport || !hasLoadedSubset;
  el.btnExportLoaded.style.display = hasLoadedSubset ? "" : "none";
}

function renderCrumbs() {
  el.crumbs.innerHTML = state.crumbs
    .map((c, i) => {
      return (
        '<div class="crumb" data-i="' +
        i +
        '">' +
        '<span class="name">' +
        esc(c.name) +
        "</span>" +
        '<span class="meta">' +
        esc(c.rank) +
        " · " +
        c.nodeId +
        "</span>" +
        "</div>"
      );
    })
    .join("");

  for (const n of el.crumbs.querySelectorAll(".crumb")) {
    n.addEventListener("click", async () => {
      const i = Number(n.getAttribute("data-i") || "0") | 0;
      const c = state.crumbs[i];
      if (!c) return;
      state.crumbs = state.crumbs.slice(0, i + 1);
      await goNode(c.nodeId, true);
    });
  }
}

async function loadMeta() {
  const m = await fetchJson("/meta");
  state.meta = m;
  state.taxaLoaded = !!m.taxaIndexLoaded;

  if (!el.exportRoot.value) {
    el.exportRoot.value = m.defaultExportRoot || "";
  }

  if (Number.isFinite(Number(m.treeMaxLimit)) && Number(m.treeMaxLimit) > 0) {
    const cap = Number(m.treeMaxLimit) | 0;
    state.treeLimit = Math.max(1, Math.min(state.treeLimit, cap));
    if (el.treeLimit) {
      const current = Number(el.treeLimit.value || state.treeLimit) | 0;
      el.treeLimit.value = String(Math.max(1, Math.min(current, cap)));
      state.treeLimit = Math.max(
        1,
        Math.min(Number(el.treeLimit.value || state.treeLimit) | 0, cap),
      );
    }
  }

  renderAdminMeta();
  syncIndexSection();
  syncExportControls();

  if (!state.taxaLoaded) {
    setTreeStatus("Taxa index not loaded. Build phase 1, then finalize.", true);
  }
}

async function refreshJobStatusOnce() {
  const job = await fetchJson("/job/status");
  renderJobState(job);
  return job;
}

function startJobPolling() {
  if (_jobPoll) return;
  _jobPoll = window.setInterval(async () => {
    try {
      const job = await refreshJobStatusOnce();
      if (!job.active) stopJobPolling();
    } catch {}
  }, 1000);
}

function stopJobPolling() {
  if (!_jobPoll) return;
  clearInterval(_jobPoll);
  _jobPoll = 0;
}

function renderJobState(job) {
  if (!job) {
    el.jobStatus.innerHTML = "Idle";
    el.jobProgressFill.style.width = "0%";
    el.jobProgressMeta.innerHTML = "";
    syncExportControls();
    return;
  }

  const p = clamp01(job.progress);
  el.jobProgressFill.style.width = (p * 100).toFixed(1) + "%";

  const meta = [];

  if (Number(job.targetNodeCount || 0) > 0) {
    meta.push("targets " + fmtInt(job.targetNodeCount || 0));
  }

  if (Number(job.plannedSpeciesCount || 0) > 0) {
    meta.push("planned species " + fmtInt(job.plannedSpeciesCount || 0));
  }

  if (Number(job.speciesTotal || 0) > 0 || Number(job.speciesDone || 0) > 0) {
    meta.push(
      "species " +
        fmtInt(job.speciesDone || 0) +
        " / " +
        fmtInt(job.speciesTotal || 0),
    );
  }

  if (Number(job.rowsTotalEstimate || 0) > 0 || Number(job.rowsDone || 0) > 0) {
    meta.push(
      "rows " +
        fmtInt(job.rowsDone || 0) +
        " / " +
        fmtInt(job.rowsTotalEstimate || 0),
    );
  }

  if (
    Number(job.currentSpeciesRowsTotal || 0) > 0 ||
    Number(job.currentSpeciesRowsDone || 0) > 0
  ) {
    meta.push(
      "current rows " +
        fmtInt(job.currentSpeciesRowsDone || 0) +
        " / " +
        fmtInt(job.currentSpeciesRowsTotal || 0),
    );
  }

  if (Number(job.fileCount || 0) > 0) {
    meta.push("files " + fmtInt(job.fileCount || 0));
  }

  if (job.currentSpecies) {
    meta.push("current " + esc(job.currentSpecies));
  }

  if (Number.isFinite(Number(job.rssMB))) {
    meta.push("rss " + esc(job.rssMB) + " MB");
  }

  if (Number.isFinite(Number(job.heapUsedMB))) {
    meta.push("heap " + esc(job.heapUsedMB) + " MB");
  }

  el.jobProgressMeta.innerHTML = meta.join(" · ");

  if (job.active) {
    const parts = [];
    parts.push(String(job.title || job.type || "job"));
    if (job.phase) parts.push("phase=" + String(job.phase));
    parts.push((p * 100).toFixed(1) + "%");
    if (job.message) parts.push(String(job.message));
    el.jobStatus.innerHTML = esc(parts.join(" · "));

    el.btnBuildTaxa.disabled = true;
    el.btnFinalizeTaxa.disabled = true;
    el.btnExportSelected.disabled = true;
    el.btnExportLevelAll.disabled = true;
    el.btnExportLoaded.disabled = true;
    return;
  }

  if (job.error) {
    el.jobStatus.innerHTML =
      '<span class="err">' + esc(String(job.error)) + "</span>";
  } else if (job.done) {
    const parts = [];
    parts.push("Done");
    parts.push(String(job.title || job.type || "job"));
    if (job.message) parts.push(String(job.message));
    el.jobStatus.innerHTML =
      '<span class="ok">' + esc(parts.join(" · ")) + "</span>";
  } else {
    el.jobStatus.innerHTML = "Idle";
    el.jobProgressFill.style.width = "0%";
  }

  el.btnBuildTaxa.disabled = false;
  el.btnFinalizeTaxa.disabled = false;
  syncExportControls();
}

async function goNode(nodeId, keepPaging) {
  state.nodeId = nodeId | 0;
  if (!keepPaging) state.treeStart = 0;

  if (nodeId === 0) {
    state.parentId = 0;
    state.crumbs = [{ nodeId: 0, name: "(root)", rank: "root" }];
    setSelectedNode({ nodeId: 0, rankId: 0, rank: "root", name: "(root)" });
  } else {
    const info = await fetchJson("/tree/node?nodeId=" + nodeId);
    state.parentId = info.parentId | 0;

    const last = state.crumbs[state.crumbs.length - 1];
    if (!last || last.nodeId !== nodeId) {
      state.crumbs.push({
        nodeId,
        name: info.name || "(missing)",
        rank: info.rank || "unknown",
      });
    } else {
      last.name = info.name || last.name;
      last.rank = info.rank || last.rank;
    }

    setSelectedNode({
      nodeId,
      rankId: info.rankId,
      rank: info.rank,
      name: info.name || "(missing)",
    });
  }

  renderCrumbs();
  await loadTreePageWithFallback();
}

async function loadTreePageWithFallback() {
  try {
    await loadTreePage();
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    setTreeStatus(msg, true);

    if (state.treeSort !== "id") {
      state.treeSort = "id";
      el.sort.value = "id";
      try {
        await loadTreePage();
        setTreeStatus("Sort fallback -> id. Original: " + msg, false);
      } catch (e2) {
        setTreeStatus(String(e2 && e2.message ? e2.message : e2), true);
      }
    }
  }
}

async function loadTreePage() {
  if (!state.taxaLoaded) return;

  const cap =
    state.meta && Number.isFinite(Number(state.meta.treeMaxLimit))
      ? Math.max(1, Number(state.meta.treeMaxLimit) | 0)
      : 200;

  state.treeLimit = Math.max(1, Math.min(state.treeLimit | 0, cap));
  if (el.treeLimit) {
    el.treeLimit.value = String(state.treeLimit);
  }

  const qs = new URLSearchParams();
  qs.set("start", String(state.treeStart));
  qs.set("limit", String(state.treeLimit));
  qs.set("sort", state.treeSort);

  let payload;
  if (state.nodeId === 0) {
    payload = await fetchJson("/tree/root?" + qs.toString());
  } else {
    qs.set("nodeId", String(state.nodeId));
    payload = await fetchJson("/tree/children?" + qs.toString());
  }

  setTreeStatus(
    "nodeId=" +
      payload.nodeId +
      " · rows " +
      fmtInt(payload.start) +
      "–" +
      fmtInt(payload.start + payload.returned) +
      " of " +
      fmtInt(payload.total) +
      " · sort=" +
      payload.sort,
    false,
  );

  const rows = payload.children || [];
  state.loadedTreeNodeIds = rows.map((r) => r.nodeId | 0);
  state.loadedTreeTotal = Number(payload.total || 0);
  state.loadedTreeReturned = Number(payload.returned || rows.length || 0);
  state.loadedTreeParentName =
    state.nodeId === 0
      ? "(root)"
      : (state.crumbs[state.crumbs.length - 1] &&
          state.crumbs[state.crumbs.length - 1].name) ||
        "(missing)";

  renderSelectedNode();

  el.treeBody.innerHTML = rows
    .map((r) => {
      const isSpecies = (r.rankId | 0) === 7;
      const badge = isSpecies ? ' <span class="badge">species</span>' : "";
      return (
        '<tr data-node="' +
        r.nodeId +
        '" data-rank="' +
        r.rankId +
        '" data-name="' +
        esc(r.name) +
        '" data-rank-name="' +
        esc(r.rank) +
        '">' +
        "<td>" +
        r.nodeId +
        "</td>" +
        "<td>" +
        esc(r.rank) +
        badge +
        "</td>" +
        "<td>" +
        esc(r.name) +
        "</td>" +
        '<td style="text-align:right;">' +
        fmtInt(r.count) +
        "</td>" +
        "</tr>"
      );
    })
    .join("");

  for (const tr of el.treeBody.querySelectorAll("tr")) {
    tr.addEventListener("click", async () => {
      const nodeId = Number(tr.getAttribute("data-node") || "0") | 0;
      const rankId = Number(tr.getAttribute("data-rank") || "0") | 0;
      const name = tr.getAttribute("data-name") || "(missing)";
      const rank = tr.getAttribute("data-rank-name") || "unknown";

      setSelectedNode({ nodeId, rankId, rank, name });

      if (rankId === 7) {
        state.speciesNodeId = nodeId;
        state.occStart = 0;
        await loadOccurrencePage();
      } else {
        await goNode(nodeId, false);
      }
    });
  }

  el.btnPrev.disabled = payload.start <= 0;
  el.btnNext.disabled = !payload.hasMore;
  el.btnUp.disabled = state.nodeId === 0 || state.parentId === state.nodeId;
}

async function loadOccurrencePage() {
  if (!state.taxaLoaded) return;
  if (state.speciesNodeId < 0) return;

  const qs = new URLSearchParams();
  qs.set("speciesNodeId", String(state.speciesNodeId));
  qs.set("start", String(state.occStart));
  qs.set("limit", String(state.occLimit));
  qs.set("header", "guess");
  qs.set("cols", UI_OCCURRENCE_COLS.join(","));

  const url = "/species/page?" + qs.toString();
  el.occJsonLink.href = url;

  el.occTitle.textContent = "Loading… (nodeId " + state.speciesNodeId + ")";
  setOccStatus("Loading…", false);

  el.occHead.innerHTML = "";
  el.occBody.innerHTML = "";
  el.btnOccPrev.disabled = true;
  el.btnOccNext.disabled = true;

  let payload = null;
  try {
    payload = await fetchJson(url);
    const nPts = updateMapFromOccurrencePayload(payload);
    setOccStatus(
      "rows " +
        fmtInt(payload.start) +
        "–" +
        fmtInt(payload.start + payload.returned) +
        " of " +
        fmtInt(payload.totalCount) +
        " · points " +
        fmtInt(nPts),
      false,
    );
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    el.occTitle.textContent = "Error (nodeId " + state.speciesNodeId + ")";
    setOccStatus(msg, true);
    return;
  }

  el.occTitle.textContent = payload.name
    ? payload.name + " (nodeId " + state.speciesNodeId + ")"
    : "species nodeId " + state.speciesNodeId;

  setOccStatus(
    "rows " +
      fmtInt(payload.start) +
      "–" +
      fmtInt(payload.start + payload.returned) +
      " of " +
      fmtInt(payload.totalCount),
    false,
  );

  const header = payload.header || [];
  const rows = payload.rows || [];
  const cols = pickOccurrenceColumns(header);

  el.occHead.innerHTML =
    "<tr>" + cols.map((c) => "<th>" + esc(c) + "</th>").join("") + "</tr>";

  const colIdx = cols.map((c) => header.indexOf(c));
  el.occBody.innerHTML = rows
    .map((r, ri) => {
      return (
        '<tr data-ri="' +
        ri +
        '">' +
        colIdx
          .map((ci) => {
            const v = ci >= 0 && ci < r.length ? r[ci] : "";
            return "<td>" + esc(v) + "</td>";
          })
          .join("") +
        "</tr>"
      );
    })
    .join("");

  for (const tr of el.occBody.querySelectorAll("tr")) {
    tr.addEventListener("click", () => {
      const ri = Number(tr.getAttribute("data-ri") || "0") | 0;
      const row = rows[ri];
      if (!row) return;

      const inat = inferInatUrlFromRow(header, row);
      if (inat) {
        window.open(inat, "_blank", "noopener");
        return;
      }

      const gbif = gbifUrlFromRow(header, row);
      if (gbif) {
        window.open(gbif, "_blank", "noopener");
      }
    });

    const ri = Number(tr.getAttribute("data-ri") || "0") | 0;
    const row = rows[ri];
    const inat = row ? inferInatUrlFromRow(header, row) : null;
    if (inat) tr.title = "Open iNaturalist";
  }

  el.btnOccPrev.disabled = payload.start <= 0;
  el.btnOccNext.disabled =
    payload.start + payload.returned >= payload.totalCount;
}

function renderExportResult(result) {
  if (!result) {
    setExportResultHtml("");
    return;
  }

  const parts = [];
  parts.push(
    'target nodes <span class="ok">' +
      fmtInt(result.targetNodeCount || 0) +
      "</span>",
  );
  parts.push(
    'species files <span class="ok">' +
      fmtInt(result.speciesFileCount || result.fileCount || 0) +
      "</span>",
  );
  parts.push(
    'species total <span class="ok">' +
      fmtInt(result.speciesTotal || 0) +
      "</span>",
  );
  parts.push(
    'rows <span class="ok">' + fmtInt(result.rowCount || 0) + "</span>",
  );

  if (Number(result.rowsTotalEstimate || 0) > 0) {
    parts.push(
      'row estimate <span class="ok">' +
        fmtInt(result.rowsTotalEstimate || 0) +
        "</span>",
    );
  }

  if (Number(result.shardCount || 0) > 0) {
    parts.push(
      'shards <span class="ok">' + fmtInt(result.shardCount || 0) + "</span>",
    );
  }

  if (Number(result.baseRowRuns || 0) > 0) {
    parts.push(
      'base-row runs <span class="ok">' +
        fmtInt(result.baseRowRuns || 0) +
        "</span>",
    );
  }

  if (Number(result.sortWorkers || 0) > 0) {
    parts.push(
      'sort workers <span class="ok">' +
        fmtInt(result.sortWorkers || 0) +
        "</span>",
    );
  }

  parts.push(
    'root <span class="mono">' + esc(result.outRoot || "") + "</span>",
  );

  if (Array.isArray(result.files) && result.files.length) {
    parts.push(
      'first output <span class="mono">' + esc(result.files[0]) + "</span>",
    );
  }

  setExportResultHtml(parts.join("<br/>"));
}

async function runServerJob(url, body, onSuccess) {
  startJobPolling();
  try {
    const result = await postJson(url, body || {});
    await loadMeta();
    const job = result.job || (await refreshJobStatusOnce());
    renderJobState(job);
    if (typeof onSuccess === "function") await onSuccess(result);
  } catch (e) {
    el.jobStatus.innerHTML =
      '<span class="err">' +
      esc(String(e && e.message ? e.message : e)) +
      "</span>";
  } finally {
    stopJobPolling();
    try {
      const job = await refreshJobStatusOnce();
      renderJobState(job);
    } catch {}
    syncExportControls();
  }
}

async function exportWithScope(scope) {
  const outRoot = String(el.exportRoot.value || "").trim();
  if (!outRoot) {
    setExportResultHtml('<span class="err">Output root is required.</span>');
    return;
  }

  const body = {
    scope,
    mode: String(el.exportMode.value || "coords"),
    outRoot,
    nodeId: state.selectedNodeId,
    currentNodeId: state.nodeId,
    nodeIds: state.loadedTreeNodeIds.slice(),
  };

  setExportResultHtml("");
  await runServerJob("/export/node", body, async (result) => {
    renderExportResult(result.result || null);
  });
}

el.sort.addEventListener("change", async () => {
  state.treeSort = String(el.sort.value || "id");
  state.treeStart = 0;
  await loadTreePageWithFallback();
});

el.treeLimit.addEventListener("change", async () => {
  const cap =
    state.meta && Number.isFinite(Number(state.meta.treeMaxLimit))
      ? Math.max(1, Number(state.meta.treeMaxLimit) | 0)
      : 200;
  const n = Number(el.treeLimit.value || "200");
  state.treeLimit = Math.max(1, Math.min(n | 0, cap));
  el.treeLimit.value = String(state.treeLimit);
  state.treeStart = 0;
  await loadTreePageWithFallback();
});

el.btnPrev.addEventListener("click", async () => {
  state.treeStart = Math.max(0, state.treeStart - state.treeLimit);
  await loadTreePageWithFallback();
});

el.btnNext.addEventListener("click", async () => {
  state.treeStart = state.treeStart + state.treeLimit;
  await loadTreePageWithFallback();
});

el.btnUp.addEventListener("click", async () => {
  if (state.nodeId === 0) return;
  const pid = state.parentId | 0;
  const idx = state.crumbs.findIndex((c) => c.nodeId === pid);
  if (idx >= 0) state.crumbs = state.crumbs.slice(0, idx + 1);
  else state.crumbs = [{ nodeId: 0, name: "(root)", rank: "root" }];
  await goNode(pid, false);
});

el.occLimit.addEventListener("change", async () => {
  const n = Number(el.occLimit.value || "200");
  state.occLimit = Math.max(1, n | 0);
  state.occStart = 0;
  try {
    await loadOccurrencePage();
  } catch (e) {
    setOccStatus(String(e.message || e), true);
  }
});

el.btnOccPrev.addEventListener("click", async () => {
  state.occStart = Math.max(0, state.occStart - state.occLimit);
  try {
    await loadOccurrencePage();
  } catch (e) {
    setOccStatus(String(e.message || e), true);
  }
});

el.btnOccNext.addEventListener("click", async () => {
  state.occStart = state.occStart + state.occLimit;
  try {
    await loadOccurrencePage();
  } catch (e) {
    setOccStatus(String(e.message || e), true);
  }
});

el.btnBuildTaxa.addEventListener("click", async () => {
  setExportResultHtml("");
  await runServerJob("/admin/build-taxa", {}, async () => {});
});

el.btnFinalizeTaxa.addEventListener("click", async () => {
  setExportResultHtml("");
  await runServerJob("/admin/finalize-taxa", {}, async () => {
    if (state.taxaLoaded) {
      await goNode(state.nodeId || 0, true);
    }
  });
});

el.btnExportSelected.addEventListener("click", async () => {
  await exportWithScope("selected");
});

el.btnExportLevelAll.addEventListener("click", async () => {
  await exportWithScope("current_level_all");
});

el.btnExportLoaded.addEventListener("click", async () => {
  await exportWithScope("loaded_subset");
});

(async () => {
  try {
    await loadMeta();
    renderCrumbs();
    renderSelectedNode();
    await refreshJobStatusOnce().catch(() => {});
    if (state.taxaLoaded) await loadTreePageWithFallback();
  } catch (e) {
    setTreeStatus(String(e && e.message ? e.message : e), true);
  }
})();
