import sys
import io
import os
import re
import shutil

# FIX 1: Force UTF-8 encoding at top for Windows
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace'
)
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding='utf-8',
    errors='replace'
)

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Default path (used only for ingest_folder)
DEFAULT_CHROMA_PATH = "chroma_db"

def clean_text(text: str) -> str:
    if not text:
        return text
    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    replacements = {
        '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--',
        '\u2022': '*', '\u00a0': ' ',
        '\u2012': '-', '\u2015': '--',
        '\u00b7': '*', '\u25cf': '*',
        '\u25cb': '*', '\u2764': '',
        '\u2665': '', '\u00e9': 'e',
        '\u00e0': 'a', '\u00e8': 'e',
        '\u00f1': 'n', '\u00fc': 'u',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_docx_safe(filepath: str):
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(filepath)
        return loader.load()
    except Exception as e:
        print(f"Docx2txt failed: {e}, falling back to python-docx...")
        try:
            import docx
            doc_obj = docx.Document(filepath)
            full_text = []
            for para in doc_obj.paragraphs:
                text = para.text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                if text.strip():
                    full_text.append(text)
            return [Document(
                page_content='\n'.join(full_text),
                metadata={"source": filepath}
            )]
        except Exception as e2:
            print(f"python-docx failed: {e2}")
            return []

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=[
            "\n\n", "\n", ".", "Skills",
            "Experience", "Projects", "Education",
            "Certifications", "Summary", " ", ""
        ]
    )

def get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Please check your config.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

# ✅ KEY FIX: chroma_path is now a parameter, not a global
def ingest_single_file(filepath: str, chroma_path: str):
    print(f"Ingesting: {filepath} → ChromaDB: {chroma_path}")

    ext = filepath.rsplit(".", 1)[-1].lower()
    docs = []

    if ext == "pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        docs = loader.load()
    elif ext == "txt":
        try:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
        except Exception:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            docs = [Document(page_content=content, metadata={"source": filepath})]
    elif ext == "docx":
        docs = load_docx_safe(filepath)
    elif ext == "pptx":
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(filepath)
        docs = loader.load()
    else:
        print(f"Unsupported extension: {ext}")
        return

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    docs = [d for d in docs if d.page_content.strip()]
    print(f"Loaded and cleaned {len(docs)} documents")

    splitter = get_splitter()
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"Created {len(chunks)} chunks")

    embeddings = get_embeddings()

    # ✅ Only clear THIS user's ChromaDB — not everyone's!
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            print(f"Cleared old ChromaDB for session: {chroma_path}")
        except Exception as e:
            print(f"Warning: Could not clear old ChromaDB: {e}")

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=chroma_path
    )
    print(f"Created new ChromaDB at: {chroma_path}")
    print(f"Done. Ingested: {os.path.basename(filepath)}")


def ingest_folder(folder_path="data/"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
        return

    print(f"Loading from folder: {folder_path}...")
    from langchain_community.document_loaders import DirectoryLoader
    loader = DirectoryLoader(folder_path, glob="**/*.*")
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    docs = [d for d in docs if d.page_content.strip()]
    print(f"Loaded {len(docs)} documents")

    splitter = get_splitter()
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()

    if os.path.exists(DEFAULT_CHROMA_PATH):
        try:
            shutil.rmtree(DEFAULT_CHROMA_PATH)
            print("Old ChromaDB cleared.")
        except:
            print("Warning: Could not clear old ChromaDB. Appending instead.")

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DEFAULT_CHROMA_PATH
    )
    print("Done. ChromaDB ready.")


if __name__ == "__main__":
    ingest_folder()
