"""Thai legal-case RAG prototype.

Ingests Thai Supreme Court decision PDFs into Postgres + pgvector,
then drafts a Thai legal analysis for a user-supplied case.

See AGENTS.md for setup, gotchas, and DB-reset commands.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import psycopg

# PyMuPDF (fitz) is used because pypdf garbles Thai PDFs from this corpus:
# the embedded font's tone-mark glyphs come back as null bytes, e.g.
# 'เกี่ยว' -> 'เกี\x00ยว'. PyMuPDF returns clean Thai text with all
# tone marks intact, and is also ~5x faster on this corpus.
import fitz  # type: ignore[import-untyped]

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pythainlp.util import normalize

# --- Config ----------------------------------------------------------------

load_dotenv()
CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_cases")

# langchain-postgres expects a SQLAlchemy-style URL with the psycopg v3 driver.
# Keep .env portable by normalizing the scheme here.
if CONNECTION_STRING and CONNECTION_STRING.startswith("postgresql://"):
    CONNECTION_STRING = CONNECTION_STRING.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)


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


# --- PDF extraction & cleaning --------------------------------------------

def extract_pdf(path: str) -> tuple[str, str]:
    """Read a Thai Supreme Court PDF and return (text, case_no).

    case_no is taken from the PDF's title metadata (e.g. '283/2565'),
    falling back to the filename stem.
    """
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    case_no = (doc.metadata or {}).get("title") or os.path.splitext(
        os.path.basename(path)
    )[0]
    doc.close()
    return text, case_no


def clean_thai_legal(text: str) -> str:
    """Light normalization for Thai legal text.

    PyMuPDF output is already clean, so this only:
      - normalizes Thai variants (e.g. sara-am sequences) via pythainlp,
      - collapses runs of spaces/tabs and excessive blank lines.
    """
    text = normalize(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Bot --------------------------------------------------------------------

class LegalAnalysisBot:
    def __init__(self) -> None:
        # Embeddings: local Ollama, 1024-dim. Run: `ollama pull qwen3-embedding:0.6b`.
        self.embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
        # Chat: cloud-proxied via ollama.com. Requires network + Ollama cloud setup.
        self.llm = ChatOllama(model="gemma4:31b-cloud", temperature=0)
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )

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

    def analyze_case(self, case_draft: str) -> str:
        # 1. Retrieve similar cases
        docs = self.vector_store.similarity_search(case_draft, k=5)
        context = "\n\n".join(
            f"Case Reference {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

        print(context)
        print()

        # 2. Construct Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """คุณคือผู้เชี่ยวชาญด้านกฎหมายระดับอาวุโส หน้าที่ของคุณคือวิเคราะห์ร่างคดีความโดยใช้ข้อมูลอ้างอิงจากคำพิพากษาที่ใกล้เคียงที่สุด
            
            กรุณาวิเคราะห์ตามหัวข้อดังนี้:
            1. **โอกาสในการชนะ/แพ้**: ประเมินเป็นเปอร์เซ็นต์โดยประมาณ พร้อมเหตุผลประกอบ
            2. **จุดแข็งของคดี**: สิ่งที่สนับสนุนข้อเรียกร้องของผู้ใช้
            3. **จุดอ่อนและสิ่งที่น่ากังวล**: ความเสี่ยงหรือช่องโหว่ที่ฝ่ายตรงข้ามอาจยกขึ้นอ้าง
            4. **การเปรียบเทียบกับคดีในอดีต**: ระบุว่ามีความคล้ายคลึงหรือแตกต่างจากคดีอ้างอิงอย่างไร
            5. **คำแนะนำเพิ่มเติม**: สิ่งที่ควรปรับปรุงในร่างคดีเพื่อให้มีโอกาสชนะมากขึ้น

            คำเตือน: การวิเคราะห์นี้เป็นเพียงการประเมินเบื้องต้นโดย AI ไม่ใช่คำแนะนำทางกฎหมายจากทนายความที่มีใบอนุญาต
            
            ข้อมูลอ้างอิงจาก Vector DB:
            {context}
            """),
            ("user", "{case_draft}")
        ])

        # 3. Execute Chain
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "case_draft": case_draft})


if __name__ == "__main__":
    bot = LegalAnalysisBot()

    # Ingest documents from the documents folder.
    # Uncomment when (re)ingesting; ingestion is idempotent (deterministic IDs).
    # rglob recurses into subfolders, so docs_dir may hold PDFs directly or be
    # organized into nested folders (e.g. per-year ./downloads/2564-2565-991/*.pdf).
    # case_id/id come from the filename only (not the folder), so the layout is
    # irrelevant as long as filenames are unique across folders.
    docs_dir = "./documents"
    pdf_files = sorted(str(p) for p in Path(docs_dir).rglob("*.pdf"))
    bot.ingest_pdfs(pdf_files)

    # test_case = "ร่างคดี: นาย ก. ถูกเลิกจ้างโดยไม่เป็นธรรมเนื่องจากบริษัทอ้างว่าผลงานไม่ถึงเกณฑ์ แต่นาย ก. มีหลักฐานการประเมินย้อนหลัง 3 ปีที่อยู่ในระดับดีมาก และไม่เคยได้รับคำเตือนเป็นลายลักษณ์อักษร"
    # test_case = """
    # ร่างคดี: นายสมชาย (โจทก์) พบว่า นายบุญมี (จำเลย) ซึ่งเป็นเจ้าของที่ดินข้างเคียง 
    # ได้ทำการก่อสร้างกำแพงรั้วและต่อเติมหลังคาโรงรถรุกล้ำเข้ามาในเขตที่ดินของนายสมชายประมาณ 50 เซนติเมตร ตลอดแนวเขตด้านทิศตะวันออก 
    # นอกจากนี้ นายบุญมียังนำวัสดุก่อสร้างมาวางกองปิดทับเส้นทางภาระจำยอมที่นายสมชายใช้สัญจรออกสู่ทางสาธารณะมานานกว่า 15 ปี 
    # ทำให้นายสมชายไม่สามารถนำรถยนต์เข้า-ออกบ้านได้ 

    # ข้อเรียกร้อง: 
    # 1. ขอให้จำเลยรื้อถอนกำแพงและหลังคาที่รุกล้ำออกไป พร้อมปรับปรุงสภาพที่ดินให้เป็นดังเดิม
    # 2. ขอให้เปิดทางภาระจำยอมและห้ามจำเลยทำการปิดกั้นอีก
    # 3. เรียกค่าเสียหายจากการเสียประโยชน์ในการใช้สอยที่ดินและทางเดินรถเป็นเงิน 5,000 บาทต่อเดือน
    # """
    # print("\n--- Analyzing Case ---\n")
    # result = bot.analyze_case(test_case)
    # print(result)
