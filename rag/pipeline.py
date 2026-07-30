import re
from memory.conversation_memory import ConversationMemory
from rag.retriever import Retriever, es_posible_sigla
from rag.generator import generar_respuesta
from ingestion.embedder import Embedder
from ingestion.loader import cargar_todos_los_documentos
from ingestion.chunker import limpiar_texto
from config.settings import settings

PATRONES_SALUDO = re.compile(
    r'^(hola|buenas|buen[oa]s? d[ií]as|hey|hi|hello)[\s!.,¿?]*$', re.IGNORECASE
)

def es_saludo_simple(mensaje: str) -> bool:
    return bool(PATRONES_SALUDO.match(mensaje.strip()))


_embedder = Embedder(settings.VECTORSTORE_PATH)
_retriever = Retriever(_embedder)
_memoria = ConversationMemory()
_documentos_originales = {
    nombre: limpiar_texto(texto)
    for nombre, texto in cargar_todos_los_documentos(settings.DOCUMENTS_PATH).items()
}


def procesar_mensaje(user_id: str, pregunta: str, modelo: str = None) -> str:
    historial = _memoria.obtener_historial(user_id)

    if es_saludo_simple(pregunta):
        chunks = []
    else:
        chunks = _retriever.buscar_contexto(pregunta, top_k=settings.RAG_TOP_K)
        for sigla in es_posible_sigla(pregunta):
            chunks += _retriever.buscar_por_keyword(sigla, _documentos_originales, top_k=2)

        chunks_relevantes = _retriever.filtrar_por_relevancia(chunks)

        if not chunks_relevantes:
            respuesta = "No tengo información suficiente en los documentos para responder esa pregunta."
        else:
            respuesta = generar_respuesta(pregunta, chunks_relevantes, historial, modelo=modelo)

    _memoria.agregar_mensaje(user_id, "user", pregunta)
    _memoria.agregar_mensaje(user_id, "assistant", respuesta)

    return respuesta