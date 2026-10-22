import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")


def get_embeddings():
    if LLM_PROVIDER == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        return OpenAIEmbeddings(model=model, api_key=api_key)


def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    print(f"Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print(f"Initializing embeddings model (provider={LLM_PROVIDER})...")
    embeddings = get_embeddings()

    print(f"Storing chunks in PostgreSQL with pgVector...")
    print(f"Database: {DATABASE_URL}")
    print(f"Collection: {COLLECTION_NAME}")

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DATABASE_URL,
    )

    print(f"✅ Successfully ingested {len(chunks)} chunks into vector store!")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        exit(1)