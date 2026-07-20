from ingestion.embedder import Embedder

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