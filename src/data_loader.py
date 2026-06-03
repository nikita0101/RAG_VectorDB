from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import json_loader

def load_all_documents(data_dir: str):
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Datapath: {data_path}")

    documents = []

    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")

    for file_path in pdf_files:
        print(f"[DEBUG] Loading file: {file_path}")

        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            print(f"[DEBUG] Loaded {len(docs)} documents from {file_path}")

            documents.extend(docs)

        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")

    return documents
    
# Text files
# CSV files
# SQL files