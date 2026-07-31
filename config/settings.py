from dotenv import load_dotenv
import os


load_dotenv()


class Settings:
    # Ollama
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Documentos y corpus RAG
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents")

    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "./data/vectorstore")

    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))

    RAG_MAX_DISTANCE: float = float(os.getenv("RAG_MAX_DISTANCE", "0.75"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))

    CHUNK_OVERLAP_ORACIONES: int = int(os.getenv("CHUNK_OVERLAP_ORACIONES", "2"))

    # Memoria Buffer en SQLite
    MEMORY_SQLITE_PATH: str = os.getenv("MEMORY_SQLITE_PATH","./data/memoria.db")

    MEMORY_BUFFER_TURNS: int = int(os.getenv("MEMORY_BUFFER_TURNS", "5"))

    # Memoria vectorial en ChromaDB
    MEMORY_VECTOR_COLLECTION: str = os.getenv("MEMORY_VECTOR_COLLECTION","memoria_conversacional")

    MEMORY_VECTOR_TOP_K: int = int(os.getenv("MEMORY_VECTOR_TOP_K", "3"))

    # Chroma usa distancia coseno:
    # 0 significa muy similar y valores mayores son menos similares.
    MEMORY_VECTOR_MAX_DISTANCE: float = float(os.getenv("MEMORY_VECTOR_MAX_DISTANCE", "0.75"))


settings = Settings()