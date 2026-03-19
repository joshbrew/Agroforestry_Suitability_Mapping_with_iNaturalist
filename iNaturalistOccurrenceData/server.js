#!/usr/bin/env node
"use strict";

// ./server.js

if (!process.env.EXPORT_SPOOL_SHARDS) process.env.EXPORT_SPOOL_SHARDS = "512";
if (!process.env.EXPORT_OPEN_WRITERS) process.env.EXPORT_OPEN_WRITERS = "128";
if (!process.env.CSV_READ_BYTES) process.env.CSV_READ_BYTES = "8388608";
if (!process.env.CSV_PAGE_READ_BYTES) process.env.CSV_PAGE_READ_BYTES = "8388608";
if (!process.env.EXPORT_SPOOL_BATCH_TRIPLES) process.env.EXPORT_SPOOL_BATCH_TRIPLES = "262144";
if (!process.env.EXPORT_WRITER_FLUSH_BYTES) process.env.EXPORT_WRITER_FLUSH_BYTES = "4194304";
if (!process.env.EXPORT_SORT_WORKERS) process.env.EXPORT_SORT_WORKERS = "2";
if (!process.env.EXPORT_SHARD_REPORT_ROWS) process.env.EXPORT_SHARD_REPORT_ROWS = "200";

const {
  fs,
  FILE,
  PORT,
  MAX_LIMIT,
  INDEX_PATH,
  TAXA_DIR,
  DEFAULT_EXPORT_ROOT,
  GUESSED_HEADERS_50,
  WANT_BUILD_TAXA,
  WANT_FINALIZE_TAXA,
  statFile,
  runExclusiveJob,
  getJobState,
} = require("./server/csvserver.utils.js");

const {
  loadOrBuildIndex,
  readPage,
  chooseHeader,
} = require("./server/csvserver.core.js");

const {
  buildTaxaIndexPhase1,
  finalizeTaxaIndexPhase2,
  loadTaxaIndex,
  readSpeciesRows,
  listChildNodeIds,
} = require("./server/csvserver.taxa.js");

const { exportTaxaTargets } = require("./server/csvserver.export.js");

const { createServer } = require("./server/server-ui.js");

(async function main() {
  const st = await statFile(FILE).catch((e) => {
    console.error(e);
    return null;
  });

  if (!st || !st.isFile()) {
    console.error(`File not found: ${FILE}`);
    process.exit(1);
  }

  const idx = await loadOrBuildIndex(FILE, INDEX_PATH);

  if (WANT_BUILD_TAXA) {
    await buildTaxaIndexPhase1(idx);
    process.exit(0);
  }

  if (WANT_FINALIZE_TAXA) {
    await finalizeTaxaIndexPhase2();
    process.exit(0);
  }

  idx._sharedCsvFd = await fs.promises.open(idx.file, "r");

  let taxa = await loadTaxaIndex().catch((e) => {
    console.error(e);
    return null;
  });

  if (taxa && !taxa._sharedPostingsFd) {
    taxa._sharedPostingsFd = await fs.promises
      .open(taxa.postingsPath, "r")
      .catch((e) => {
        console.error(e);
        return null;
      });
  }

  async function reloadTaxa() {
    if (taxa && taxa._sharedPostingsFd) {
      await taxa._sharedPostingsFd.close().catch((e) => {
        console.error(e);
      });
    }

    taxa = await loadTaxaIndex().catch((e) => {
      console.error(e);
      return null;
    });

    if (taxa && !taxa._sharedPostingsFd) {
      taxa._sharedPostingsFd = await fs.promises
        .open(taxa.postingsPath, "r")
        .catch((e) => {
          console.error(e);
          return null;
        });
    }

    return taxa;
  }

  async function shutdown() {
    if (idx && idx._sharedCsvFd) {
      await idx._sharedCsvFd.close().catch((e) => {
        console.error(e);
      });
    }

    if (taxa && taxa._sharedPostingsFd) {
      await taxa._sharedPostingsFd.close().catch((e) => {
        console.error(e);
      });
    }
  }

  process.on("SIGINT", async () => {
    await shutdown();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    await shutdown();
    process.exit(0);
  });

  if (taxa) console.log(`[taxa] loaded ${TAXA_DIR}`);
  else {
    console.log(
      `[taxa] not loaded (run --build-taxa then --finalize-taxa to enable taxonomy browsing)`,
    );
  }

  console.log(
    `[perf] EXPORT_SPOOL_SHARDS=${process.env.EXPORT_SPOOL_SHARDS} EXPORT_OPEN_WRITERS=${process.env.EXPORT_OPEN_WRITERS} CSV_READ_BYTES=${process.env.CSV_READ_BYTES} CSV_PAGE_READ_BYTES=${process.env.CSV_PAGE_READ_BYTES} EXPORT_SPOOL_BATCH_TRIPLES=${process.env.EXPORT_SPOOL_BATCH_TRIPLES} EXPORT_WRITER_FLUSH_BYTES=${process.env.EXPORT_WRITER_FLUSH_BYTES}`,
  );

  const TREE_MAX_LIMIT = Math.max(
    1,
    Number(process.env.TREE_MAX_LIMIT || 5000),
  );

  const TREE_SORT_MAX_CHILDREN = Math.max(
    10_000,
    Number(process.env.TREE_SORT_MAX_CHILDREN || 200_000),
  );

  const server = createServer({
    idx,
    getTaxa: () => taxa,
    reloadTaxa,
    config: {
      fileDefault: FILE,
      maxLimit: MAX_LIMIT,
      treeMaxLimit: TREE_MAX_LIMIT,
      treeSortMaxChildren: TREE_SORT_MAX_CHILDREN,
      taxaDir: TAXA_DIR,
      guessedHeaderColumns: GUESSED_HEADERS_50.length,
      scriptPath: process.argv[1],
      defaultExportRoot: DEFAULT_EXPORT_ROOT,
    },
    fns: {
      readPage,
      chooseHeader,
      readSpeciesRows,

      buildTaxaPhase1: async () =>
        runExclusiveJob(
          "build-phase1",
          "Build taxonomy phase 1",
          async (report) => {
            report({
              phase: "phase1",
              message: "Building taxonomy phase 1",
            });
            const meta = await buildTaxaIndexPhase1(idx);
            report({
              phase: "phase1_done",
              message: "Taxonomy phase 1 complete",
            });
            return meta;
          },
        ),

      finalizeTaxaPhase2: async () =>
        runExclusiveJob(
          "finalize-phase2",
          "Finalize taxonomy index",
          async (report) => {
            report({
              phase: "phase2",
              message: "Finalizing taxonomy index",
            });
            await finalizeTaxaIndexPhase2();
            await reloadTaxa();
            report({
              phase: "phase2_done",
              message: "Taxonomy index finalized",
            });
            return {
              taxaDir: TAXA_DIR,
              taxaIndexLoaded: !!taxa,
            };
          },
        ),

      exportTaxaNode: async ({
        scope,
        nodeId,
        currentNodeId,
        nodeIds,
        mode,
        outRoot,
      }) =>
        runExclusiveJob(
          "export-node",
          "Export taxonomy selection",
          async (report) => {
            const liveTaxa = taxa || (await reloadTaxa());
            if (!liveTaxa) {
              throw new Error("taxa index not built");
            }

            const exportScope = String(scope || "selected");
            let resolvedNodeIds = [];

            if (exportScope === "selected") {
              resolvedNodeIds = [nodeId >>> 0];
            } else if (exportScope === "current_level_all") {
              resolvedNodeIds = await listChildNodeIds(
                liveTaxa,
                currentNodeId >>> 0,
              );
            } else if (exportScope === "loaded_subset") {
              resolvedNodeIds = Array.isArray(nodeIds) ? nodeIds.slice() : [];
            } else {
              throw new Error(`Unknown export scope "${exportScope}"`);
            }

            report({
              phase: "resolve",
              message: `Resolved ${resolvedNodeIds.length} node(s)`,
              scope: exportScope,
            });

            return await exportTaxaTargets({
              idx,
              taxa: liveTaxa,
              nodeIds: resolvedNodeIds,
              mode,
              outRoot,
              reporter: report,
            });
          },
        ),

      getJobState,
    },
  });

  server.listen(PORT, "127.0.0.1", () => {
    console.log(`server: http://127.0.0.1:${PORT}`);
  });
})().catch((e) => {
  console.error(e);
  process.exit(1);
});