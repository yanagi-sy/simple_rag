from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document



class PDFRagSystem:
    def __init__(self):
        # ① 文章 → ベクトル（意味の数値）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ② 回答生成用のAI（ローカル）
        self.llm = Ollama(
            model="llama3:latest",
            temperature=0
        )

        self.vectorstore = None

    def load_pdf(self, pdf_path):
        # ③ PDF → Document
        loader = PyPDFLoader(pdf_path)
        documents =  loader.load()

        # ④ チャンク化
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        # ⑤ ベクトル化して保存
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

    def query(self, question, k=3):
        # ⑥ 検索 → 生成
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            return_source_documents=True
        )
        return qa.invoke({"query": question})

rag = PDFRagSystem()
rag.load_pdf("linuxtext_ver4.0.0.pdf")

result = rag.query(
    "Linuxとは何ですか？初心者向けに日本語で説明してください。",
    k=10
)

print("=== 回答 ===")
print(result["result"])

