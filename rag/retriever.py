from ingestion.embedder import Embedder
import re

UMBRAL_RELEVANCIA = 0.75


def es_posible_sigla(pregunta: str) -> list[str]:
    """
    Detecta tokens que parecen siglas: 2 a 5 letras mayúsculas seguidas.
    """
    return re.findall(r'\b[A-Z]{2,5}\b', pregunta)

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

    
    def filtrar_por_relevancia(self, chunks: list[dict], umbral: float = UMBRAL_RELEVANCIA) -> list[dict]:
        return [c for c in chunks if c["distancia"] <= umbral]

    def buscar_por_keyword(self, termino: str, documentos_originales: dict[str, str], top_k: int = 2) -> list[dict]:
        """
        Fallback de búsqueda literal: busca un término corto (típicamente
        una sigla) directamente en el texto de los documentos originales,
        sin pasar por embeddings.
        """
        resultados = []
        for nombre, texto in documentos_originales.items():
            idx = texto.lower().find(termino.lower())
            if idx == -1:
                continue
            fragmento = texto[max(0, idx - 150): idx + 400]
            resultados.append({"texto": fragmento, "fuente": nombre, "distancia": 0.0})
            if len(resultados) >= top_k:
                break
        return resultados
    
