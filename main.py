import os
import re
from typing import List
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pythainlp.util import normalize

# Load environment variables
load_dotenv()
CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_cases")

def clean_legal_thai_text(text: str) -> str:
    text = normalize(text)
    text = re.sub(r'\s*\.\s*', '.', text)
    text = re.sub(r'([^\n])\n([^\n])', r'\1\2', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

class LegalAnalysisBot:
    def __init__(self):
        # Use Ollama for embeddings (local)
        # Note: Ensure you have run 'ollama pull nomic-embed-text'
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Use Ollama for LLM (local)
        # Note: Ensure you have run 'ollama pull llama3' or your preferred model
        self.llm = ChatOllama(model="gemma4:31b-cloud", temperature=0)
        
        self.vector_store = PGVector(
            connection_string=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    def ingest_pdfs(self, pdf_paths: List[str]):
        all_chunks = []
        for path in pdf_paths:
            print(f"Processing {path}...")
            loader = PyPDFLoader(path)
            raw_docs = loader.load()
            
            cleaned_docs = []
            for doc in raw_docs:
                clean_content = clean_legal_thai_text(doc.page_content)
                cleaned_docs.append(Document(page_content=clean_content, metadata=doc.metadata))
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            all_chunks.extend(text_splitter.split_documents(cleaned_docs))
        
        self.vector_store.add_documents(all_chunks)
        print(f"Successfully ingested {len(all_chunks)} chunks from {len(pdf_paths)} files.")

    def analyze_case(self, case_draft: str):
        # 1. Retrieve similar cases
        docs = self.vector_store.similarity_search(case_draft, k=5)
        context = "\n\n".join([f"Case Reference {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

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
    
    # Ingest documents from the documents folder
    pdf_files = [f"./documents/{f}" for f in os.listdir("./documents") if f.endswith(".pdf")]
    # bot.ingest_pdfs(pdf_files)

    # test_case = "ร่างคดี: นาย ก. ถูกเลิกจ้างโดยไม่เป็นธรรมเนื่องจากบริษัทอ้างว่าผลงานไม่ถึงเกณฑ์ แต่นาย ก. มีหลักฐานการประเมินย้อนหลัง 3 ปีที่อยู่ในระดับดีมาก และไม่เคยได้รับคำเตือนเป็นลายลักษณ์อักษร"
    test_case = """
    ร่างคดี: นายสมชาย (โจทก์) พบว่า นายบุญมี (จำเลย) ซึ่งเป็นเจ้าของที่ดินข้างเคียง 
    ได้ทำการก่อสร้างกำแพงรั้วและต่อเติมหลังคาโรงรถรุกล้ำเข้ามาในเขตที่ดินของนายสมชายประมาณ 50 เซนติเมตร ตลอดแนวเขตด้านทิศตะวันออก 
    นอกจากนี้ นายบุญมียังนำวัสดุก่อสร้างมาวางกองปิดทับเส้นทางภาระจำยอมที่นายสมชายใช้สัญจรออกสู่ทางสาธารณะมานานกว่า 15 ปี 
    ทำให้นายสมชายไม่สามารถนำรถยนต์เข้า-ออกบ้านได้ 

    ข้อเรียกร้อง: 
    1. ขอให้จำเลยรื้อถอนกำแพงและหลังคาที่รุกล้ำออกไป พร้อมปรับปรุงสภาพที่ดินให้เป็นดังเดิม
    2. ขอให้เปิดทางภาระจำยอมและห้ามจำเลยทำการปิดกั้นอีก
    3. เรียกค่าเสียหายจากการเสียประโยชน์ในการใช้สอยที่ดินและทางเดินรถเป็นเงิน 5,000 บาทต่อเดือน
    """
    print("\n--- Analyzing Case ---\n")
    result = bot.analyze_case(test_case)
    print(result)
