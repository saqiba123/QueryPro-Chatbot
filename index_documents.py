from utils.rag_chain import create_vectorstore
from langchain_community.document_loaders import PyPDFLoader
import os

def index_pdf_document(pdf_path: str, session_id: str):
    loader = PyPDFLoader(pdf_path)
    documents =  loader.load()
    save_path = f"faiss_index/{session_id}"
    os.makedirs("faiss_index", exist_ok=True)
    create_vectorstore(documents, save_path)
