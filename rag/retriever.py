from config.settings import settings
from ingestion.embedder import Embedder
import re


def es_posible_sigla(pregunta: str) -> list[str]:
    """
    Detecta tokens que parecen siglas: 2 a 5 letras mayúsculas seguidas.
    """
    siglas = re.findall(r"\b[A-ZÁÉÍÓÚÑ]{2,5}\b", pregunta)

    return list(dict.fromkeys(siglas))


class Retriever:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def buscar_contexto(self, pregunta: str, top_k: int = 5) -> list[dict]:
        """
        Busca los 'top_k' chunks más relevantes semánticamente a la pregunta.
        Devuelve una lista de dicts con el texto del chunk y su fuente.
        """
        vector_pregunta = self.embedder.modelo.encode([pregunta]).tolist()

        resultados = self.embedder.coleccion.query(
            query_embeddings=vector_pregunta,
            n_results=top_k
        )

        chunks_encontrados = []
        for texto, metadata, distancia in zip(
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0]
        ):
            chunks_encontrados.append({
                "texto": texto,
                "fuente": metadata["fuente"],
                "distancia": distancia
            })

        return chunks_encontrados

    
    def filtrar_por_relevancia(self, chunks: list[dict], umbral: float = settings.RAG_MAX_DISTANCE) -> list[dict]:
        return [c for c in chunks if c["distancia"] <= umbral]

    def buscar_por_keyword(self, termino: str, documentos_originales: dict[str, str], top_k: int = 2) -> list[dict]:
        """
        Fallback de búsqueda literal: busca un término corto (típicamente
        una sigla) directamente en el texto de los documentos originales,
        sin pasar por embeddings.
        """
        resultados = []

        # Ejemplo: encuentra "AAA", pero no "baaa".
        patron = re.compile(
            rf"(?<!\w){re.escape(termino)}(?!\w)",
            flags=re.IGNORECASE
        )

        for nombre_documento, texto in documentos_originales.items():
            for coincidencia in patron.finditer(texto):
                inicio = max(
                    0,
                    coincidencia.start() - 200
                )

                fin = min(
                    len(texto),
                    coincidencia.end() + 500
                )

                fragmento = texto[inicio:fin].strip()

                resultados.append({
                    "texto": fragmento,
                    "fuente": nombre_documento,
                    "distancia": 0.0,
                    "tipo_busqueda": "lexica"
                })

                if len(resultados) >= top_k:
                    return resultados

        return resultados
    
