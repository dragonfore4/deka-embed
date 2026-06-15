# AGENTS.md

Compact, repo-specific notes. Read this before editing.

## Project

Thai legal-case RAG prototype. Single-file (`main.py`) script that:
- ingests Thai court-decision PDFs from `./documents/` into Postgres + pgvector,
- answers Thai legal-case drafts via Ollama (embeddings + chat).

No tests, no lint/format/typecheck config, no CI. It is a research script — keep it simple.

## Run it

- Python 3.12, managed by `uv` (see `.python-version`, `uv.lock`).
- `uv sync` to install deps.
- DB: `podman compose up -d` (uses `compose.yaml`, image `pgvector/pgvector:pg16`, port 5432, db `legal_db`, creds `postgres/postgres`, persists in `./pgdata`). `init.sql` runs against the default `postgres` db (not `legal_db`), so the `vector` extension isn't pre-installed there — `langchain-postgres.PGVector` creates it on first connect. Don't be confused if `\dx` on a freshly-reset `legal_db` shows no `vector` until the script runs.
- Local **Ollama** must be running with models pulled:
  - `ollama pull qwen3-embedding:0.6b` (embeddings — 1024-dim; the collection is locked to this dimension once any vectors are written)
  - the chat model referenced in `main.py` is `gemma4:31b-cloud` — a **cloud-proxied** model routed via ollama.com (the local `ollama list` entry shows `remote_host: https://ollama.com:443`). Requires network and an Ollama account configured for cloud models. It is not a typo.
- `.env` requires `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_db`. Keep the plain `postgresql://` form — `main.py` rewrites it to `postgresql+psycopg://` at runtime so the env var stays portable.
- `COLLECTION_NAME` env var optional, defaults to `legal_cases`.
- Run: `uv run python main.py`.

## Gotchas

- **The `__main__` block toggles between ingest and analyze.** It currently runs a **full ingest** — `pdf_files = sorted(str(p) for p in Path(docs_dir).rglob("*.pdf"))` (recursive, so nested per-year subfolders under `./documents/` work) then `bot.ingest_pdfs(pdf_files)`. The `analyze_case(...)` example is commented out. To do analysis-only, comment the ingest call and uncomment a `test_case` + `bot.analyze_case(...)`. The user iterates on the embedding model, so this toggling is normal.
- **Ingestion is idempotent.** Chunk IDs are deterministic (`<case_id>:<n>`, e.g. `deka_666539:0`), so `add_documents(..., ids=...)` upserts existing rows on re-run. Tweaking the cleaner or splitter and re-running on the same files is safe — old chunks get overwritten. **Caveat:** if a doc is *re-chunked into fewer pieces*, the leftover high-index chunks are not deleted; they go stale. For that case, soft-wipe the collection first.
- **`ingest_pdfs` is concurrent and resumable.** Defaults: `batch_size=256`, `max_workers=3`, `skip_existing=True`. `skip_existing` drops PDFs whose `case_id` is already in the collection (per-file, via `_existing_case_ids` querying `langchain_pg_embedding.cmetadata->>'case_id'`) — so a re-run resumes a partial ingest / adds only new files without re-embedding the corpus. Set `skip_existing=False` when you tweak the cleaner/splitter and want existing files re-embedded in place. `max_workers` overlaps Ollama embedding with Postgres writes so the GPU isn't idle during writes; keep it `<=` PGVector's default pool size (5). A failed batch is reported and skipped, not fatal; re-run to retry. **Per-file skip keeps the re-chunk caveat above** — soft-wipe before re-chunking into fewer pieces. **Skip is per-file, so a run interrupted mid-file leaves that file's `case_id` present → it'd be skipped on resume while partial; re-embed it with `skip_existing=False` (or soft-wipe) if you killed a run.**
- **Embedding throughput is GPU-bound.** `OllamaEmbeddings(num_ctx=2048)` (`main.py`) caps the context so a slot is ~1.3 GB (vs ~5.8 GB at the model's default 32768) — chunks are ~1000 chars, so 2048 is ample and the vectors are unchanged (embedding depends on input, not num_ctx). For big corpora (tens of thousands of chunks), set `OLLAMA_NUM_PARALLEL` (Ollama **server** env, not code) to run multiple embeds at once; with `num_ctx=2048` on a 32 GB machine, `OLLAMA_NUM_PARALLEL=3` + `max_workers=3` (~4–5 GB) is a safe ~2–3× speedup. Restart Ollama after setting it (`launchctl setenv OLLAMA_NUM_PARALLEL 3` for the macOS app, or `OLLAMA_NUM_PARALLEL=3 ollama serve`).
- **Changing the embedding model invalidates the whole collection** — the vector dimension is locked at first write (1024 for `qwen3-embedding:0.6b`). You must reset before swapping embedders. See "DB reset".
- **PDF extraction uses PyMuPDF (`fitz`), not pypdf.** pypdf garbles this corpus's Thai fonts (tone marks come back as `\x00`). Don't switch back. PyMuPDF is AGPL-3.0; that is acceptable for this internal research script. PyMuPDF also exposes the case number via `doc.metadata["title"]` (e.g. `'283/2565'`); `extract_pdf` returns it for use in chunk metadata.
- **Vector store uses `langchain-postgres.PGVector`** (the supported package), **not** the deprecated `langchain_community.vectorstores.PGVector`. New API: `connection=`, `embeddings=`, `use_jsonb=True`. Driver is `psycopg` v3 (not `psycopg2`). Don't revert to the old import.
- **`pgvector` Python client is pinned `<0.4`** because `langchain-postgres` requires it. Don't bump past 0.4 unless `langchain-postgres` relaxes its constraint.
- Thai text is normalized in `clean_thai_legal` (`main.py`) using `pythainlp.normalize` + minimal whitespace cleanup. With PyMuPDF input this is mostly a no-op, but keep it: it absorbs any future loader change.
- Prompts and user-facing strings are Thai. Preserve locale.

## DB reset (when re-embedding)

- Hard wipe: `podman compose down && rm -rf pgdata && podman compose up -d`
- Soft wipe (keep db, drop collection tables):
  ```
  podman exec -it legal-pgvector psql -U postgres -d legal_db \
    -c "DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE;"
  ```

## Repo layout quirks

- `documents/` — ~995 PDFs, the real corpus, git-ignored. Collected **recursively** (`Path(docs_dir).rglob("*.pdf")`), so PDFs may be nested in subfolders (e.g. per-year). `case_id` derives from the filename only, so filenames must be unique across folders.
- `documents-tmp/`, `documentssdf/` — leftover sample folders, **not used by code**. Ignore.
- `ingest/` — only contains `__pycache__`. No real module.
- `.env.py` — unused legacy sketch referencing Supabase + OpenAI. Current code uses Ollama + local Postgres only. Don't follow it.
- **`.gitignore` reality (it lies elsewhere):** it currently ignores only `documents`, `documents-tmp`, `documentssdf`, `pgdata`, `*.pyc`. It does **not** ignore `*.sql` or `.env`.
  - `legal_db_2005.sql` (~174 MB) is **committed on purpose** — it is the seed `compose.yaml` mounts as the DB init script (full embedded corpus: 995 case_ids / 11,742 chunks). Don't delete it casually.
  - Other dumps (`full_dump.sql`, `legal_db.sql`, `*.dump`) should **not** be committed (large / may contain copyrighted court text) — but they are not auto-ignored, so don't `git add .` blindly.
  - `.env` is **git-ignored and untracked** — it holds only localhost `postgres/postgres` creds. Copy `.env.example` → `.env` to set up locally.
- `pgdata/` — local Postgres volume, git-ignored.
- `README.md` is a full bilingual (EN/TH) guide. Keep it in sync with `main.py` (ingest options, reset commands, layout).

## Don't

- Don't introduce lint/format/type tools without asking — none are configured.
- Don't commit `.env`, `pgdata/`, `documents/`, or any `*.sql`.
- Don't run a full ingest casually (≈995 PDFs through a local embedding model is slow and writes a lot of vectors).
- Don't add abstractions/modules to `main.py` unprompted. The user wants it simple.
- Don't switch back to `docker` in commands or `compose.yaml` — project standard is `podman`.
