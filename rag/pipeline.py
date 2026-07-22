from memory.conversation_memory import ConversationMemory
from rag.retriever import Retriever, es_posible_sigla
from rag.generator import generar_respuesta
from ingestion.embedder import Embedder
from ingestion.loader import cargar_todos_los_documentos
from ingestion.chunker import limpiar_texto
from config.settings import settings

_embedder = Embedder(settings.VECTORSTORE_PATH)
_retriever = Retriever(_embedder)
_memoria = ConversationMemory()

_documentos_originales = {
    nombre: limpiar_texto(texto)
    for nombre, texto in cargar_todos_los_documentos(settings.DOCUMENTS_PATH).items()
}

def procesar_mensaje(user_id: str, pregunta: str, modelo: str = None) -> str:
    historial = _memoria.obtener_historial(user_id)
    chunks = _retriever.buscar_contexto(pregunta, top_k=settings.RAG_TOP_K)

    siglas = es_posible_sigla(pregunta)
    for sigla in siglas:
        chunks += _retriever.buscar_por_keyword(sigla, _documentos_originales, top_k=2)

    respuesta = generar_respuesta(pregunta, chunks, historial, modelo=modelo)

    _memoria.agregar_mensaje(user_id, "user", pregunta)
    _memoria.agregar_mensaje(user_id, "assistant", respuesta)

    return respuesta