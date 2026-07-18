from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "./data/vectores")
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documentos")
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

settings = Settings()