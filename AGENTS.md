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

- **`bot.ingest_pdfs(...)` in the `__main__` block is commented out on purpose.** Default run = analysis only against existing vectors. The user uncomments it when (re)ingesting. The user is actively iterating on the embedding model, so this toggling is normal.
- **Ingestion is idempotent.** Chunk IDs are deterministic (`<case_id>:<n>`, e.g. `deka_666539:0`), so `add_documents(..., ids=...)` upserts existing rows on re-run. Tweaking the cleaner or splitter and re-running on the same files is safe — old chunks get overwritten. **Caveat:** if a doc is *re-chunked into fewer pieces*, the leftover high-index chunks are not deleted; they go stale. For that case, soft-wipe the collection first.
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

- `documents/` — ~995 PDFs, the real corpus, git-ignored.
- `documents-tmp/`, `documentssdf/` — leftover sample folders, **not used by code**. Ignore.
- `ingest/` — only contains `__pycache__`. No real module.
- `.env.py` — unused legacy sketch referencing Supabase + OpenAI. Current code uses Ollama + local Postgres only. Don't follow it.
- `full_dump.sql` (~190 MB), `pgdata/` — local-only, git-ignored (`.gitignore` excludes `*.sql` and `pgdata`).
- `README.md` is intentionally empty.

## Don't

- Don't introduce lint/format/type tools without asking — none are configured.
- Don't commit `.env`, `pgdata/`, `documents/`, or any `*.sql`.
- Don't run a full ingest casually (≈995 PDFs through a local embedding model is slow and writes a lot of vectors).
- Don't add abstractions/modules to `main.py` unprompted. The user wants it simple.
- Don't switch back to `docker` in commands or `compose.yaml` — project standard is `podman`.
