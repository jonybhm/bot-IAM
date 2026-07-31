import re

from config.settings import settings
from ingestion.chunker import limpiar_texto
from ingestion.embedder import Embedder
from ingestion.loader import cargar_todos_los_documentos
from memory.conversation_memory import ConversationMemory
from memory.hybrid_memory import HybridConversationMemory
from memory.vector_memory import VectorConversationMemory
from rag.generator import (FRASE_SIN_INFO, generar_respuesta)
from rag.retriever import (Retriever, es_posible_sigla)


PATRONES_SALUDO = re.compile(
    (
        r"^(hola|buenas|buen[oa]s? d[ií]as|"
        r"hey|hi|hello)[\s!.,¿?]*$"
    ),
    re.IGNORECASE
)


def es_saludo_simple(mensaje: str) -> bool:
    return bool(
        PATRONES_SALUDO.match(
            mensaje.strip()
        )
    )


def deduplicar_chunks(chunks: list[dict]) -> list[dict]:
    """
    Evita enviar dos veces un fragmento recuperado tanto
    por búsqueda semántica como por búsqueda léxica.
    """
    chunks_unicos = []
    vistos = set()

    for chunk in chunks:
        clave = (
            chunk["fuente"],
            chunk["texto"].strip()
        )

        if clave in vistos:
            continue

        vistos.add(clave)
        chunks_unicos.append(chunk)

    return chunks_unicos


_embedder = Embedder(
    settings.VECTORSTORE_PATH
)

_retriever = Retriever(
    _embedder
)

_memoria_buffer = ConversationMemory(
    ruta_db=settings.MEMORY_SQLITE_PATH,
    max_turnos=settings.MEMORY_BUFFER_TURNS
)


_memoria_vectorial = VectorConversationMemory(
    ruta_vectorstore=settings.VECTORSTORE_PATH,
    modelo_embeddings=_embedder.modelo,
    nombre_coleccion=(
        settings.MEMORY_VECTOR_COLLECTION
    ),
    top_k=settings.MEMORY_VECTOR_TOP_K,
    distancia_maxima=(
        settings.MEMORY_VECTOR_MAX_DISTANCE
    )
)


_memoria = HybridConversationMemory(
    buffer_memory=_memoria_buffer,
    vector_memory=_memoria_vectorial
)


_documentos_originales = {
    nombre: limpiar_texto(texto)
    for nombre, texto
    in cargar_todos_los_documentos(
        settings.DOCUMENTS_PATH
    ).items()
}


def procesar_mensaje(user_id: str, pregunta: str, modelo: str = None) -> str:
    """
    Flujo completo:

    1. Recupera memoria reciente desde SQLite.
    2. Recupera memoria semántica desde ChromaDB.
    3. Ejecuta RAG semántico y búsqueda léxica.
    4. Genera la respuesta.
    5. Guarda el intercambio en ambas memorias.
    """

    if es_saludo_simple(pregunta):
        respuesta = (
            "¡Hola! Soy el asistente de soporte IAM. "
            "Podés consultarme sobre autenticación, "
            "autorización, gestión de identidades "
            "y control de acceso."
        )

        _memoria.guardar_intercambio(user_id=user_id, pregunta=pregunta, respuesta=respuesta, indexar_vectorialmente=False)

        return respuesta

    historial_reciente, memoria_semantica = (_memoria.obtener_contexto(user_id=user_id, consulta=pregunta))

    chunks = _retriever.buscar_contexto(pregunta, top_k=settings.RAG_TOP_K)

    siglas_detectadas = es_posible_sigla(pregunta)

    for sigla in siglas_detectadas:
        chunks_lexicos = (_retriever.buscar_por_keyword(termino=sigla,documentos_originales=(_documentos_originales),top_k=2))

        chunks.extend(chunks_lexicos)

    chunks = deduplicar_chunks(chunks)

    chunks_relevantes = (_retriever.filtrar_por_relevancia(chunks))

    if not chunks_relevantes:
        respuesta = FRASE_SIN_INFO

    else:
        respuesta = generar_respuesta(
            pregunta=pregunta,
            chunks_contexto=chunks_relevantes,
            historial=historial_reciente,
            memoria_semantica=memoria_semantica,
            modelo=modelo
        )

    debe_indexar_vectorialmente = (
        respuesta != FRASE_SIN_INFO
    )

    _memoria.guardar_intercambio(user_id=user_id, pregunta=pregunta, respuesta=respuesta, indexar_vectorialmente=(debe_indexar_vectorialmente))

    return respuesta