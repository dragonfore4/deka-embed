# Thai Legal-Case RAG Prototype
# ระบบวิเคราะห์คดีความภาษาไทยด้วย RAG (ต้นแบบ)

A single-file research script (`main.py`) that ingests Thai Supreme Court decision PDFs (คำพิพากษาฎีกา) into Postgres + pgvector, then drafts a Thai legal analysis for a user-supplied case using Ollama.

สคริปต์ไฟล์เดียว (`main.py`) สำหรับนำคำพิพากษาฎีกาในรูปแบบ PDF เข้าสู่ฐานข้อมูล Postgres + pgvector และใช้ Ollama ในการร่างบทวิเคราะห์คดีภาษาไทยจากร่างคดีที่ผู้ใช้ป้อน

> ⚠️ **Disclaimer / คำเตือน**
> Internal research prototype. AI-generated output is **not** legal advice and must not be relied on for real legal decisions.
> เป็นต้นแบบเพื่อการวิจัยภายในเท่านั้น ผลลัพธ์จาก AI **ไม่ใช่** คำแนะนำทางกฎหมาย ห้ามนำไปใช้ตัดสินใจทางกฎหมายจริง

---

## 📋 What it does / ระบบทำอะไร

**EN**
- Extracts text from Thai Supreme Court PDFs with PyMuPDF (`fitz`), collected **recursively** from `./documents/` (PDFs may sit directly in the folder or in nested subfolders, e.g. per-year).
- Normalizes Thai text and splits it into overlapping chunks (1000 / 200).
- Embeds chunks locally via Ollama and stores them in Postgres + pgvector. Ingestion is **concurrent** (embedding overlaps DB writes so the GPU isn't idle) and **resumable** (a re-run skips PDFs already in the store).
- For a given case draft, retrieves the top-5 most similar chunks and asks a chat model to produce a structured Thai analysis (win/lose odds, strengths, weaknesses, comparisons, recommendations).

**TH**
- ดึงข้อความจากไฟล์ PDF คำพิพากษาฎีกาด้วย PyMuPDF (`fitz`) โดยเก็บไฟล์จาก `./documents/` แบบ **recursive** (วาง PDF ไว้ในโฟลเดอร์ตรง ๆ หรือจัดเป็น subfolder ซ้อน เช่นแยกตามปี ก็ได้)
- ปรับมาตรฐานข้อความภาษาไทยและแบ่งเป็นชิ้น (chunk) ขนาด 1000 ตัวอักษร เหลื่อม 200
- สร้าง embedding ผ่าน Ollama ที่รันในเครื่องและเก็บลง Postgres + pgvector การ ingest ทำแบบ **concurrent** (embedding ทำคู่ขนานกับการเขียน DB เพื่อไม่ให้ GPU ว่าง) และ **resumable** (รันซ้ำจะข้ามไฟล์ที่ embed ไปแล้ว)
- เมื่อรับร่างคดีจากผู้ใช้ ระบบจะค้นหาชิ้นข้อความที่คล้ายที่สุด 5 ชิ้น แล้วส่งให้โมเดล chat สร้างบทวิเคราะห์ภาษาไทยตามหัวข้อ (โอกาสชนะ/แพ้ จุดแข็ง จุดอ่อน การเปรียบเทียบ และคำแนะนำ)

---

## 🧩 Stack / เทคโนโลยีที่ใช้

| Layer / ส่วนประกอบ | Tool / เครื่องมือ |
|---|---|
| PDF extraction | PyMuPDF (`fitz`) |
| Thai normalization | `pythainlp.normalize` |
| Chunking | `RecursiveCharacterTextSplitter` (1000 / 200) |
| Embeddings | Ollama `qwen3-embedding:0.6b` (1024-dim, local) |
| Vector store | Postgres 16 + pgvector (via `langchain-postgres`) |
| Chat model | Ollama `gemma4:31b-cloud` (cloud-proxied) |
| Runtime | Python 3.12 + [`uv`](https://docs.astral.sh/uv/) |
| Container | `podman` + `podman compose` |

---

## ✅ Requirements / สิ่งที่ต้องมี

**EN**
- Python 3.12 (managed by `uv`).
- `podman` and `podman compose`.
- A running local Ollama, with:
  - `qwen3-embedding:0.6b` pulled locally for embeddings.
  - An Ollama account configured for cloud models (for `gemma4:31b-cloud`) and a working internet connection.

**TH**
- Python 3.12 (จัดการผ่าน `uv`)
- `podman` และ `podman compose`
- Ollama ที่รันในเครื่อง พร้อมกับ:
  - ดึงโมเดล `qwen3-embedding:0.6b` มาไว้ในเครื่องเพื่อใช้ทำ embedding
  - บัญชี Ollama ที่ตั้งค่าใช้งานโมเดล cloud ได้ (สำหรับ `gemma4:31b-cloud`) และมีอินเทอร์เน็ต

---

## 📦 Setup / การติดตั้ง

**EN**

1. Install Python deps:
   ```bash
   uv sync
   ```
2. Pull the embedding model (cloud chat model is fetched on demand by Ollama):
   ```bash
   ollama pull qwen3-embedding:0.6b
   ```
3. Start the database (see the next section).
4. Create a `.env` file in the repo root:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_db
   COLLECTION_NAME=legal_cases   # optional, this is the default
   ```
   Keep the plain `postgresql://` form — `main.py` rewrites it to `postgresql+psycopg://` at runtime.
5. Drop your PDF corpus into `./documents/` (one file per case). Subfolders are fine — PDFs are collected recursively (`Path(docs_dir).rglob("*.pdf")`). `case_id` comes from the **filename** only (e.g. `deka_718365.pdf` → `deka_718365`), so keep filenames unique across folders.

**TH**

1. ติดตั้ง dependency ของ Python:
   ```bash
   uv sync
   ```
2. ดึงโมเดล embedding (ส่วนโมเดล chat แบบ cloud ระบบ Ollama จะจัดการให้เอง):
   ```bash
   ollama pull qwen3-embedding:0.6b
   ```
3. สตาร์ทฐานข้อมูล (ดูหัวข้อถัดไป)
4. สร้างไฟล์ `.env` ที่รากของโปรเจกต์:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_db
   COLLECTION_NAME=legal_cases   # ไม่ใส่ก็ได้ ค่าเริ่มต้นคือชื่อนี้
   ```
   ใส่เป็น `postgresql://` ตามปกติ — โค้ดใน `main.py` จะแปลงเป็น `postgresql+psycopg://` เองตอนรัน
5. นำไฟล์ PDF ของคดีมาวางไว้ใน `./documents/` (หนึ่งไฟล์ต่อหนึ่งคดี) จะวางใน subfolder ก็ได้ ระบบเก็บแบบ recursive (`Path(docs_dir).rglob("*.pdf")`) ทั้งนี้ `case_id` มาจาก**ชื่อไฟล์**อย่างเดียว (เช่น `deka_718365.pdf` → `deka_718365`) จึงควรตั้งชื่อไฟล์ไม่ให้ซ้ำกันข้ามโฟลเดอร์

---

## 🐳 Database (Podman) / ฐานข้อมูล

Container is defined in `compose.yaml`: image `pgvector/pgvector:pg16`, name `legal-pgvector`, port `5432`, db `legal_db`, user/password `postgres/postgres`. Data persists in `./pgdata/` (git-ignored).

คอนเทนเนอร์ถูกประกาศไว้ใน `compose.yaml` ใช้ image `pgvector/pgvector:pg16` ชื่อ `legal-pgvector` พอร์ต `5432` ฐานข้อมูล `legal_db` ผู้ใช้/รหัสผ่าน `postgres/postgres` ข้อมูลถูกเก็บไว้ที่ `./pgdata/` (ถูก ignore โดย git)

```bash
# Start in background / สตาร์ทแบบทำงานเบื้องหลัง
podman compose up -d

# Stop (keep container) / หยุด (เก็บคอนเทนเนอร์ไว้)
podman compose stop

# Stop + remove container (data in ./pgdata is kept) / หยุดและลบคอนเทนเนอร์ (ข้อมูลใน ./pgdata ยังอยู่)
podman compose down

# Tail logs / ดู log
podman compose logs -f db

# Status / ดูสถานะ
podman ps

# Open psql shell / เปิด psql
podman exec -it legal-pgvector psql -U postgres -d legal_db
```

---

## 🚀 Running / การรัน

```bash
uv run python main.py
```

**EN**
- The `__main__` block in `main.py` is currently set to **ingest** every PDF found recursively under `./documents/`.
- The `analyze_case(...)` example is commented out. To switch to "analysis only":
  1. Comment out `bot.ingest_pdfs(pdf_files)`.
  2. Uncomment one of the `test_case = "..."` blocks and the `bot.analyze_case(test_case)` call.
- Ingestion is **idempotent**: chunk IDs are deterministic (`<case_id>:<n>`, e.g. `deka_666539:0`), so re-running upserts existing rows instead of duplicating them.
- Ingestion is **concurrent and resumable** — see "Ingest options" below.

**TH**
- บล็อก `__main__` ใน `main.py` ตั้งค่าให้ **นำเข้า (ingest)** ทุกไฟล์ PDF ที่หาเจอแบบ recursive ใต้ `./documents/`
- ส่วน `analyze_case(...)` ถูกคอมเมนต์ไว้ หากต้องการให้รันเฉพาะการวิเคราะห์:
  1. คอมเมนต์ `bot.ingest_pdfs(pdf_files)` ออก
  2. ยกเลิกคอมเมนต์ของบล็อก `test_case = "..."` ตัวใดตัวหนึ่ง และบรรทัด `bot.analyze_case(test_case)`
- การ ingest เป็นแบบ **idempotent**: chunk ID ถูกกำหนดแบบ deterministic (`<case_id>:<n>` เช่น `deka_666539:0`) ดังนั้นการรันซ้ำจะ upsert ทับข้อมูลเดิม ไม่สร้างซ้ำ
- การ ingest ทำแบบ **concurrent และ resumable** — ดูหัวข้อ "Ingest options" ด้านล่าง

### ⚙️ Ingest options / ตัวเลือกการนำเข้า

`bot.ingest_pdfs(pdf_paths, batch_size=64, max_workers=2, skip_existing=True)`

| Param | Default | EN | TH |
|---|---|---|---|
| `batch_size` | `64` | Chunks embedded + upserted per batch. | จำนวน chunk ต่อ batch ที่ embed + เขียนลง DB |
| `max_workers` | `2` | Concurrent upsert workers. Embedding overlaps DB writes so the GPU stays fed. Keep `<=` PGVector's pool size (5). | จำนวน worker ที่ upsert พร้อมกัน ให้ embedding ทับซ้อนกับการเขียน DB เพื่อไม่ให้ GPU ว่าง ควร `<=` ขนาด pool ของ PGVector (5) |
| `skip_existing` | `True` | Skip PDFs whose `case_id` is already in the store (per-file). Resume a partial run / add new files without re-embedding. Set `False` to re-embed existing files (e.g. after tweaking the cleaner/splitter). | ข้ามไฟล์ที่ `case_id` มีใน DB แล้ว (ระดับไฟล์) ใช้ทำ resume / เพิ่มไฟล์ใหม่โดยไม่ต้อง embed ซ้ำ ตั้งเป็น `False` เมื่ออยาก re-embed ไฟล์เดิม (เช่นหลังแก้ cleaner/splitter) |

**EN** — A failed batch is reported (`[WARN]`) and skipped, not fatal; the run continues and you re-run to retry (idempotent). Optionally set `OLLAMA_NUM_PARALLEL>=2` on the Ollama server to let two embeds run at once — not required; the embed/DB-write overlap helps even at the default of 1.

**TH** — ถ้า batch ใด fail จะถูกรายงาน (`[WARN]`) แล้วข้าม ไม่ล้มทั้ง run รันซ้ำเพื่อ retry ได้ (idempotent) ถ้าต้องการให้ embed 2 ก้อนพร้อมกันจริง ตั้ง `OLLAMA_NUM_PARALLEL>=2` ฝั่ง Ollama server (ไม่จำเป็น — การ overlap embed/เขียน DB ช่วยอยู่แล้วแม้ค่าเริ่มต้นเป็น 1)

> ⚠️ Caveat: if a document is later re-chunked into **fewer** pieces, the leftover high-index chunks from a previous run are not deleted automatically. For that case, soft-wipe the collection first (see "Resetting" below).
>
> ⚠️ ข้อควรระวัง: ถ้าเอกสารถูกแบ่งใหม่เหลือ chunk **น้อยลง** ตัว chunk เก่าที่มี index สูงจะไม่ถูกลบให้ ต้องล้าง collection ก่อน (ดูหัวข้อ "Resetting" ด้านล่าง)

---

## 🗄️ DB dump & restore / สำรองและกู้คืนฐานข้อมูล

```bash
# --- Open a shell in the DB / เข้าไปใน DB ---
podman exec -it legal-pgvector psql -U postgres -d legal_db   # exit with \q

# --- Dump / สำรองข้อมูล ---
# NOTE: do NOT pass -t when redirecting to a file — `podman exec -t` allocates a
# TTY and rewrites \n -> \r\n, which corrupts the dump. Omit it (as below).
# หมายเหตุ: อย่าใส่ -t ตอน redirect ลงไฟล์ — `podman exec -t` จะจอง TTY แล้วแปลง \n เป็น \r\n ทำให้ไฟล์เพี้ยน ใช้แบบไม่มี -t (ตามด้านล่าง)

# Single DB, plain SQL — same format as legal_db_2005.sql / เฉพาะ legal_db เป็น SQL ธรรมดา รูปแบบเดียวกับ legal_db_2005.sql
podman exec legal-pgvector pg_dump -U postgres -d legal_db > legal_db_latest.sql

# ...or stamp the filename with today's date / หรือใส่วันที่ในชื่อไฟล์
podman exec legal-pgvector pg_dump -U postgres -d legal_db > "legal_db_$(date +%Y%m%d).sql"

# --- Restore / กู้คืน ---

# From plain SQL / กู้คืนจากไฟล์ SQL
podman exec -i legal-pgvector psql -U postgres -d legal_db < legal_db_latest.sql

# From custom format / กู้คืนจาก custom format
podman exec -i legal-pgvector pg_restore -U postgres -d legal_db --clean --if-exists < legal_db.dump
```

### Regenerate the compose seed / สร้าง seed ของ compose ใหม่

`compose.yaml` mounts `legal_db_2005.sql` as the init script, loaded **only when `pgdata/` is empty** (first boot). To refresh that seed from the current DB:

`compose.yaml` mount ไฟล์ `legal_db_2005.sql` เป็น init script ซึ่งจะถูกโหลด **เฉพาะตอน `pgdata/` ว่าง** (บูตครั้งแรก) ถ้าจะอัปเดต seed จากข้อมูลปัจจุบัน:

```bash
# Re-dumping without -t also rewrites the file with clean LF endings, fixing any
# CRLF left by an older `pg_dump -t` dump (harmless to Postgres COPY, but cleaner).
# การ re-dump แบบไม่มี -t จะเขียนไฟล์ใหม่เป็น LF สะอาด แก้ CRLF ที่อาจค้างจาก dump เก่าที่ใช้ -t
podman exec legal-pgvector pg_dump -U postgres -d legal_db > legal_db_2005.sql

# Verify clean LF (should print 0) / ตรวจว่าเป็น LF สะอาด (ควรได้ 0)
LC_ALL=C grep -c $'\r' legal_db_2005.sql

# To actually reload it / ถ้าจะให้โหลดใหม่จริง:
podman compose down && rm -rf pgdata && podman compose up -d
```

> ⚠️ **`*.sql` is NOT git-ignored** — only `documents/`, `documents-tmp/`, `documentssdf/`, `pgdata/`, `*.pyc`, and `.env` are. `legal_db_2005.sql` is committed on purpose (it is the seed). **Don't commit other dumps** (`legal_db_latest.sql`, `full_dump.sql`, `*.dump`) — they are large and may contain copyrighted court text; `git add` them explicitly only if you mean to.
> ⚠️ **`*.sql` ไม่ได้ถูก git-ignore** — ที่ ignore มีแค่ `documents/`, `documents-tmp/`, `documentssdf/`, `pgdata/`, `*.pyc`, `.env` ส่วน `legal_db_2005.sql` ถูก commit ตั้งใจ (เป็น seed) **อย่า commit dump อื่น** (`legal_db_latest.sql`, `full_dump.sql`, `*.dump`) เพราะไฟล์ใหญ่และอาจมีข้อความคำพิพากษาที่มีลิขสิทธิ์

---

## 🔄 Resetting / รีเซ็ตเมื่อเปลี่ยน embedding model

The vector dimension is **locked at first write** (currently 1024 for `qwen3-embedding:0.6b`). If you swap to a different embedding model, you must wipe existing vectors first.

มิติของ vector จะ **ถูกล็อกตั้งแต่ครั้งแรกที่เขียนข้อมูล** (ปัจจุบันคือ 1024 มิติ สำหรับโมเดล `qwen3-embedding:0.6b`) ถ้าจะเปลี่ยนโมเดล embedding ต้องล้างข้อมูลเก่าก่อน

```bash
# Hard wipe (destroy DB volume) / ล้างทั้งหมด (ลบ volume)
podman compose down && rm -rf pgdata && podman compose up -d

# Soft wipe (keep DB, drop only collection tables) / ล้างเฉพาะ collection tables
podman exec -it legal-pgvector psql -U postgres -d legal_db \
  -c "DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE;"
```

---

## 📁 Project layout / โครงสร้างไฟล์

```
.
├── main.py              # The whole script / สคริปต์หลักทั้งหมด
├── compose.yaml         # Postgres + pgvector container / ไฟล์ตั้งค่าคอนเทนเนอร์
├── legal_db_2005.sql    # Seed dump mounted by compose on first boot (full embedded corpus) / ดัมป์ seed ที่ compose โหลดตอน init (มี embedding ครบ)
├── init.sql             # Alt init script (currently not mounted; see compose.yaml) / สคริปต์ init สำรอง (ยังไม่ถูก mount)
├── pyproject.toml       # Python deps / dependency
├── uv.lock              # uv lockfile
├── .python-version      # Python 3.12
├── .env                 # Local config (git-ignored) / ตั้งค่าเฉพาะเครื่อง (ignored)
├── .env.example         # Template — copy to .env / เทมเพลต คัดลอกเป็น .env
├── documents/           # PDF corpus (~995 files, may be nested, git-ignored) / คลังไฟล์ PDF (ซ้อนโฟลเดอร์ได้, ignored)
├── pgdata/              # Postgres data volume (git-ignored) / ข้อมูล Postgres (ignored)
├── AGENTS.md            # Deeper notes & gotchas for contributors / บันทึกเชิงลึกสำหรับผู้พัฒนา
└── README.md            # This file / ไฟล์นี้
```

**Unused legacy / ไฟล์เก่าที่ไม่ใช้แล้ว — ignore:** `documents-tmp/`, `documentssdf/`, `ingest/`, `.env.py`.

---

## 🧠 Notes & gotchas / ข้อควรระวัง

**EN**
- **PyMuPDF, not pypdf.** pypdf garbles this corpus's Thai fonts (tone marks come back as `\x00`). Don't switch back.
- **`gemma4:31b-cloud` is cloud-routed via ollama.com.** It needs network and a configured Ollama cloud account. Not a typo.
- **Embedding dimension is locked at first write** — 1024 for the current model. Reset before changing models.
- **Don't commit** `.env`, `pgdata/`, `documents/`, or any `*.sql` dumps.
- **Don't run a full ingest casually** — ~995 PDFs through a local embedding model is slow.
- For deeper notes (idempotency edge cases, dependency pins, etc.) see [`AGENTS.md`](./AGENTS.md).

**TH**
- **ใช้ PyMuPDF ห้ามกลับไปใช้ pypdf** เพราะ pypdf อ่านฟอนต์ไทยในคลังนี้ผิด (วรรณยุกต์กลายเป็น `\x00`)
- **`gemma4:31b-cloud` เป็นโมเดล cloud ผ่าน ollama.com** ต้องมีอินเทอร์เน็ตและบัญชี Ollama ที่เปิด cloud ไว้ ไม่ใช่พิมพ์ผิด
- **มิติของ embedding ถูกล็อกตั้งแต่ครั้งแรกที่เขียน** — ปัจจุบัน 1024 ถ้าเปลี่ยนโมเดลต้องรีเซ็ตก่อน
- **ห้าม commit** `.env`, `pgdata/`, `documents/` หรือไฟล์ `*.sql`
- **อย่ารัน ingest เต็มชุดเล่น ๆ** — ~995 ไฟล์ผ่านโมเดล embedding ในเครื่องใช้เวลานานพอสมควร
- รายละเอียดเชิงลึกเพิ่มเติม (กรณี edge ของ idempotency, การ pin เวอร์ชัน ฯลฯ) ดูที่ [`AGENTS.md`](./AGENTS.md)

---

## ⚖️ License / ใบอนุญาต

This is an internal research prototype with no public license attached. Note that **PyMuPDF is AGPL-3.0** — fine for internal use, but relevant if you ever redistribute the code or expose it as a network service.

โปรเจกต์นี้เป็นต้นแบบเพื่อการวิจัยภายใน ไม่มี license สาธารณะแนบมา ข้อสังเกต: **PyMuPDF ใช้ license AGPL-3.0** ซึ่งใช้ภายในได้ปกติ แต่ต้องระวังถ้าจะเผยแพร่โค้ดหรือเปิดเป็นบริการบนเครือข่ายในอนาคต
