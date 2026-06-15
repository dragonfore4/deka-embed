# Concurrent + Resumable PDF Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up the full ingest of ~995 Thai legal PDFs by overlapping embedding with DB writes, and let re-runs skip files already embedded.

**Architecture:** All changes live in `main.py`'s `ingest_pdfs`. A `ThreadPoolExecutor` runs batch upserts concurrently so the GPU keeps embedding while Postgres writes the previous batch. A new module-level helper queries the existing collection for already-ingested `case_id`s so they can be skipped. Failures are collected per batch instead of aborting the run.

**Tech Stack:** Python 3.12, `uv`, `psycopg` v3 (already a dep), `langchain-postgres` PGVector, local Ollama embeddings, `concurrent.futures` (stdlib).

---

## Testing note (read first)

`AGENTS.md` states the project has **no test suite and forbids adding lint/format/type tooling without asking**. Per instruction priority (user/project instructions override the TDD default), this plan does **not** add a `pytest` suite. Verification instead uses small runnable checks against the real environment (Ollama running + DB up from `compose.yaml`). These verification steps run on the user's machine; this checkout has no `documents/` folder and may not have Ollama's HTTP endpoint live, so the engineer executes them where the corpus and DB exist.

Each task is still: make a focused change → verify with an exact command → commit.

---

## File Structure

- Modify: `main.py`
  - Add imports: `from concurrent.futures import ThreadPoolExecutor, as_completed` and `import psycopg`.
  - Add module-level helper `_existing_case_ids(connection, collection_name) -> set[str]`.
  - Rewrite the body of `LegalAnalysisBot.ingest_pdfs` (new params `max_workers`, `skip_existing`; concurrent upserts; per-batch failure handling).
- Modify: `AGENTS.md`
  - Add a note about the new `ingest_pdfs` params and optional `OLLAMA_NUM_PARALLEL`.

No new files, no new modules (per `AGENTS.md`: keep `main.py` simple and single-file).

---

## Task 1: Add the `_existing_case_ids` helper

**Files:**
- Modify: `main.py` (imports near top; new function after the `# --- Config ---` block / before `# --- PDF extraction`)

- [ ] **Step 1: Add imports**

At the top of `main.py`, alongside the existing `import os` / `import re`, add the stdlib and psycopg imports. Place near the other stdlib imports:

```python
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import psycopg
from dotenv import load_dotenv
```

(Keep the existing `from typing import List` and `from dotenv import load_dotenv` — just ensure `concurrent.futures` and `psycopg` are imported. `psycopg` is already a project dependency via `psycopg[binary]`.)

- [ ] **Step 2: Add the helper function**

Add this module-level function after the config block (after the `SPLITTER = ...` definition), before `def extract_pdf`:

```python
def _existing_case_ids(connection: str | None, collection_name: str) -> set[str]:
    """Return case_ids already embedded in `collection_name`.

    Used by ingest_pdfs(skip_existing=True) to skip files that are already in
    the vector store (resume after a crash / incremental adds). Returns an empty
    set if there is no connection string or the langchain tables don't exist yet.

    psycopg/libpq doesn't understand the SQLAlchemy `+psycopg` driver suffix that
    main.py uses for PGVector, so strip it back to a plain postgresql:// DSN here.
    """
    if not connection:
        return set()
    dsn = connection.replace("postgresql+psycopg://", "postgresql://", 1)
    query = """
        SELECT DISTINCT e.cmetadata->>'case_id'
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(query, (collection_name,))
            return {row[0] for row in cur.fetchall() if row[0] is not None}
    except psycopg.errors.UndefinedTable:
        # Fresh DB: collection tables not created yet → nothing ingested.
        return set()
```

- [ ] **Step 3: Verify it imports and runs against the seeded DB**

Prerequisite: DB is up from the dump (`podman compose up -d`, which loads `legal_db_2005.sql`).

Run:

```bash
uv run python -c "from main import _existing_case_ids, CONNECTION_STRING, COLLECTION_NAME; ids = _existing_case_ids(CONNECTION_STRING, COLLECTION_NAME); print(len(ids)); print(sorted(ids)[:3])"
```

Expected output (the dump has 995 case_ids):

```
995
['deka_666539', 'deka_670363', 'deka_679242']
```

If the DB is empty/freshly wiped instead, expected output is `0` and `[]` — also acceptable (proves the empty path works).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(ingest): add _existing_case_ids helper for skip-existing"
```

---

## Task 2: Rewrite `ingest_pdfs` — skip-existing + concurrent upserts + per-batch failures

**Files:**
- Modify: `main.py` — replace the body of `LegalAnalysisBot.ingest_pdfs` (currently `main.py:93-130`)

- [ ] **Step 1: Replace the method**

Replace the entire existing `ingest_pdfs` method with this version. The PDF-extraction loop is unchanged; the signature gains `max_workers` and `skip_existing`, a skip-existing pre-filter is added, and the sequential upsert loop becomes a concurrent one with per-batch failure collection.

```python
    def ingest_pdfs(
        self,
        pdf_paths: List[str],
        batch_size: int = 64,
        max_workers: int = 2,
        skip_existing: bool = True,
    ) -> None:
        """Embed and upsert chunks for the given PDFs.

        Idempotent: chunk IDs are deterministic ('<case_id>:<n>'), so re-running
        on the same files updates existing rows instead of duplicating them.

        skip_existing=True (default) drops PDFs whose case_id is already in the
        collection (per-file) — use it to resume a partial run or add new files
        without re-embedding the whole corpus. Set it False when iterating on the
        cleaner/splitter so existing files get re-embedded in place.

        Upserts run concurrently (max_workers): while one batch is being written
        to Postgres, another can be embedding via Ollama, so the GPU is not idle
        during DB writes. max_workers stays <= PGVector's default pool size (5).
        Set OLLAMA_NUM_PARALLEL>=2 on the Ollama server to also let two embeds run
        at once — optional; the embed/DB-write overlap helps even at the default.
        """
        if skip_existing:
            existing = _existing_case_ids(CONNECTION_STRING, COLLECTION_NAME)
            before = len(pdf_paths)
            pdf_paths = [
                p for p in pdf_paths
                if os.path.splitext(os.path.basename(p))[0] not in existing
            ]
            print(
                f"skip_existing: {before - len(pdf_paths)} already-ingested file(s) "
                f"skipped, {len(pdf_paths)} to embed."
            )
            if not pdf_paths:
                print("Nothing to ingest.")
                return

        all_docs: List[Document] = []
        all_ids: List[str] = []

        for path in pdf_paths:
            case_id = os.path.splitext(os.path.basename(path))[0]  # e.g. 'deka_666539'
            text, case_no = extract_pdf(path)
            chunks = SPLITTER.split_text(clean_thai_legal(text))
            for i, chunk in enumerate(chunks):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "case_id": case_id,
                        "case_no": case_no,
                        "chunk": i,
                        "source": path,
                    },
                ))
                all_ids.append(f"{case_id}:{i}")
            print(f"  prepared {len(chunks):>3} chunks  {case_id}  ({case_no})")

        n = len(all_docs)
        if n == 0:
            print("No chunks to embed.")
            return

        # Concurrent upserts: each task does embed + DB write. While one worker is
        # blocked on a Postgres write (GIL released), another embeds via Ollama
        # (GIL released), so the GPU stays fed. One failed batch is recorded and
        # skipped, not fatal — re-running repairs it (deterministic IDs).
        batches = [
            (all_docs[i : i + batch_size], all_ids[i : i + batch_size])
            for i in range(0, n, batch_size)
        ]
        print(
            f"\nEmbedding & upserting {n} chunks in {len(batches)} batches of "
            f"{batch_size} ({max_workers} workers)..."
        )

        done = 0
        failures: list[tuple[int, Exception]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.vector_store.add_documents, docs, ids=ids): bi
                for bi, (docs, ids) in enumerate(batches)
            }
            for future in as_completed(future_to_batch):
                bi = future_to_batch[future]
                try:
                    future.result()
                    done += len(batches[bi][0])
                    print(f"  upserted {done}/{n}")
                except Exception as exc:  # noqa: BLE001 - report & continue
                    failures.append((bi, exc))
                    print(f"  [WARN] batch {bi} failed: {exc}")

        if failures:
            print(
                f"Done with errors. {done}/{n} chunks upserted from "
                f"{len(pdf_paths)} files; {len(failures)} batch(es) failed. "
                f"Re-run to retry (ingestion is idempotent)."
            )
        else:
            print(f"Done. {n} chunks from {len(pdf_paths)} files.")
```

- [ ] **Step 2: Verify skip_existing against the seeded DB (no embedding happens)**

Prerequisite: DB up from the dump (995 case_ids present), `documents/` present.

Run:

```bash
uv run python -c "
from main import LegalAnalysisBot
import os
docs_dir = './documents'
pdfs = [f'{docs_dir}/{f}' for f in os.listdir(docs_dir) if f.endswith('.pdf')]
LegalAnalysisBot().ingest_pdfs(pdfs)
"
```

Expected: a line like `skip_existing: 995 already-ingested file(s) skipped, 0 to embed.` followed by `Nothing to ingest.` — and the process exits without calling Ollama.

- [ ] **Step 3: Verify concurrent upsert + idempotency on a tiny subset**

Pick 2 files and force re-embed (`skip_existing=False`). Prerequisite: Ollama running with `qwen3-embedding:0.6b`, DB up.

Run:

```bash
uv run python -c "
from main import LegalAnalysisBot, _existing_case_ids, CONNECTION_STRING, COLLECTION_NAME
import os
docs_dir = './documents'
pdfs = sorted(f'{docs_dir}/{f}' for f in os.listdir(docs_dir) if f.endswith('.pdf'))[:2]
print('files:', pdfs)
LegalAnalysisBot().ingest_pdfs(pdfs, skip_existing=False, max_workers=2)
print('case_ids now present (sample):', len(_existing_case_ids(CONNECTION_STRING, COLLECTION_NAME)))
"
```

Expected: prints `prepared N chunks` per file, then `upserted X/N` progress lines, then `Done. N chunks from 2 files.` with no `[WARN]` lines. Re-running the same command must print the same chunk counts (idempotent upsert — `Done.` again, row count unchanged, not doubled).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "perf(ingest): concurrent batch upserts + skip already-ingested files"
```

---

## Task 3: Document the new behavior in AGENTS.md

**Files:**
- Modify: `AGENTS.md` (the "Gotchas" section)

- [ ] **Step 1: Add a gotcha entry**

In `AGENTS.md`, under `## Gotchas`, add this bullet (after the existing idempotency bullet):

```markdown
- **`ingest_pdfs` is concurrent and resumable.** Defaults: `batch_size=64`,
  `max_workers=2`, `skip_existing=True`. `skip_existing` drops PDFs whose
  `case_id` is already in the collection (per-file) — so a re-run resumes a
  partial ingest / adds only new files. Set `skip_existing=False` when you tweak
  the cleaner/splitter and want existing files re-embedded in place. `max_workers`
  overlaps Ollama embedding with Postgres writes so the GPU isn't idle during
  writes; keep it `<=` PGVector's default pool size (5). Optionally set
  `OLLAMA_NUM_PARALLEL>=2` on the `ollama serve` side to let two embeds run at
  once (not required — the embed/DB-write overlap helps even at the default of 1).
  **Per-file skip keeps the existing re-chunk caveat:** a file re-chunked into
  *fewer* pieces leaves stale high-index chunks; soft-wipe before such runs.
```

- [ ] **Step 2: Verify**

Run:

```bash
grep -n "concurrent and resumable" AGENTS.md
```

Expected: one matching line.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: note concurrent + resumable ingest_pdfs in AGENTS.md"
```

---

## Self-Review

- **Spec coverage:**
  - Concurrent upserts (overlap embed/DB write) → Task 2. ✓
  - Conservative defaults `max_workers=2`, `batch_size=64` → Task 2 signature. ✓
  - Per-batch error handling (no whole-run abort, summary at end) → Task 2. ✓
  - `skip_existing=True` default, per-file, psycopg query, fresh-DB safe → Task 1 + Task 2. ✓
  - Operational `OLLAMA_NUM_PARALLEL` note → Task 2 docstring + Task 3 AGENTS.md. ✓
  - Validation against `legal_db_2005.sql` seeded data (995 case_ids) → Task 1 Step 3, Task 2 Step 2. ✓
- **Placeholder scan:** No TBD/TODO; all code shown in full. ✓
- **Type consistency:** `_existing_case_ids(connection, collection_name) -> set[str]` defined in Task 1 and called identically in Task 2. `ingest_pdfs(pdf_paths, batch_size=64, max_workers=2, skip_existing=True)` consistent across docstring, body, and verification calls. ✓
