# Design: Concurrent batch upserts for PDF ingestion

Date: 2026-06-15
Status: Approved (pending user spec review)

## Problem

`LegalAnalysisBot.ingest_pdfs` (`main.py`) is slow on a full ingest of ~995 Thai
legal PDFs. Embedding runs on a local Ollama GPU model (`qwen3-embedding:0.6b`),
but the GPU sits idle during two phases:

1. PDF extraction at the start (all PDFs parsed sequentially before any embedding).
2. DB writes between batches — the batch loop calls `vector_store.add_documents`
   one batch at a time, so the GPU waits while each batch's vectors are written
   to Postgres.

The dominant, easy-to-recover waste is (2): embedding and DB writes never overlap.

## Goal

Increase throughput of the one-time full ingest (995 files, GPU) by:
1. overlapping DB writes with embedding, and
2. skipping files already ingested so a re-run only embeds what is missing
   (resume after a crash / incremental adds to the corpus),

without adding structural complexity.

Out of scope: retrieval/`analyze_case`, parallelizing PDF extraction, a multi-stage
producer/consumer pipeline, new modules/abstractions. (`AGENTS.md`: keep `main.py`
simple and single-file.)

## Approach: Concurrent batch upserts (chosen over full pipeline / extract-only)

Replace the sequential batch loop in `ingest_pdfs` with a bounded
`ThreadPoolExecutor`. Each task calls `vector_store.add_documents(batch, ids=...)`
(which does embed + DB write together). While one worker is blocked writing a
batch to Postgres (IO, GIL released), another worker can be embedding via Ollama
(HTTP, GIL released). The GPU is fed continuously instead of stalling on each write.

This does **not** increase peak GPU load: Ollama still serializes embedding compute.
The overlap happens between *embed* and *DB write*, not between two embeds. Peak
Ollama load is therefore ~equal to the sequential run, just without the idle gaps —
so a machine that runs the model today cannot be overloaded by this change.

## Changes (all within `main.py`, `ingest_pdfs`)

1. **Concurrency.** Build the list of `(docs, ids)` batches as today, then submit
   each to a `ThreadPoolExecutor(max_workers=max_workers)`. Collect results with
   `concurrent.futures.as_completed`.

2. **Conservative defaults.** `max_workers=2`, `batch_size=64` (unchanged).
   `max_workers` stays `<=` PGVector's default SQLAlchemy pool size (5) so workers
   never contend for DB connections. Both remain function parameters so they can be
   raised later without code edits.

3. **Error handling.** Today an exception in any batch aborts the whole run. Change
   to catch per batch: record failed batch indices, let other batches commit, and
   print a summary of failures at the end. This is safe because ingestion is
   idempotent (deterministic `<case_id>:<n>` IDs) — re-running repairs failed
   batches. Progress (`upserted X/N`) is printed from the main thread inside the
   `as_completed` loop to avoid interleaved output.

4. **Skip already-ingested files (`skip_existing=True`, default).** Before
   building docs, query the existing collection for `case_id`s already present and
   drop those PDFs from `pdf_paths`. Granularity is per-file (matches the user's
   "which files" question and is the common resume case).

   Query via `psycopg` directly against the connection string (strip the
   `+psycopg` driver suffix `langchain` uses), avoiding langchain private APIs:

   ```sql
   SELECT DISTINCT e.cmetadata->>'case_id'
   FROM langchain_pg_embedding e
   JOIN langchain_pg_collection c ON e.collection_id = c.uuid
   WHERE c.name = %s;
   ```

   If the collection/tables don't exist yet (fresh DB), treat as "nothing ingested"
   and proceed with all files. Print how many files were skipped vs. will be embedded.

   `skip_existing` is a parameter; set it to `False` when iterating on the
   cleaner/splitter so existing files get re-embedded (deterministic IDs overwrite
   in place). **Caveat (per `AGENTS.md`):** per-file skip means a file re-chunked
   into *fewer* pieces leaves stale high-index chunks; that is the existing
   idempotency caveat and is unchanged by this feature — soft-wipe before such runs.

5. **Operational note.** Add a note (code comment + `AGENTS.md`) that
   `OLLAMA_NUM_PARALLEL>=2` (set on the `ollama serve` side) lets two embeds run
   concurrently, but is optional — the embed↔DB-write overlap works even at the
   default of 1.

## Data flow (after change)

```
if skip_existing: query existing case_ids -> drop already-ingested PDFs
extract remaining PDFs (sequential, unchanged)
  -> build all_docs / all_ids (unchanged)
  -> slice into batches of batch_size
  -> ThreadPoolExecutor(max_workers=2):
        worker: vector_store.add_documents(batch, ids)   # embed (GPU) + write (DB)
     main thread: as_completed -> print progress, collect failures
  -> print "Done. N chunks ... (F batches failed)" if any failed
```

## Risks

- **Ollama client thread-safety.** `langchain_ollama.OllamaEmbeddings` uses a shared
  `httpx.Client`, which is safe for concurrent independent requests across threads.
  Low risk; `max_workers=2` keeps concurrency minimal.
- **DB connection contention.** `max_workers (2) <= pool size (5)` avoids it.
- **Partial-failure semantics change.** Run no longer aborts on first error; it
  reports failures at the end instead. Acceptable and arguably better for a long run.

## Testing / verification

No test suite exists (`AGENTS.md`). Verify by:
- A dry sanity run on a small subset of `documents/` confirming all batches upsert
  and progress prints correctly.
- Re-running on the same subset (idempotency) — counts unchanged, no duplicate rows.
- Re-running with `skip_existing=True` after a partial run — confirms already-ingested
  files are skipped and only the missing ones are embedded; with `skip_existing=False`
  all files are re-embedded.
- Comparing wall-clock against the sequential version on the same subset to confirm
  the overlap actually reduces time.
