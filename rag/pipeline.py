from memory.conversation_memory import ConversationMemory
from rag.retriever import Retriever
from rag.generator import generar_respuesta
from ingestion.embedder import Embedder
from config.settings import settings

_embedder = Embedder(settings.VECTORSTORE_PATH)
_retriever = Retriever(_embedder)
_memoria = ConversationMemory()


def procesar_mensaje(user_id: str, pregunta: str) -> str:
    historial = _memoria.obtener_historial(user_id)
    chunks = _retriever.buscar_contexto(pregunta, top_k=settings.RAG_TOP_K)

    respuesta = generar_respuesta(pregunta, chunks, historial)

    _memoria.agregar_mensaje(user_id, "user", pregunta)
    _memoria.agregar_mensaje(user_id, "assistant", respuesta)

    return respuesta