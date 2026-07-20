from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents")
    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "./data/vectorstore")

    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP_ORACIONES: int = int(os.getenv("CHUNK_OVERLAP_ORACIONES", "2"))

settings = Settings()