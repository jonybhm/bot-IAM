from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime

class VectorialMemory:
    def __init__(self, ruta_vectorstore: str, modelo_embeddings: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.modelo = SentenceTransformer(modelo_embeddings)
        self.cliente = chromadb.PersistentClient(path=ruta_vectorstore)
        self.coleccion = self.cliente.get_or_create_collection(
            "memoria_conversacional", metadata={"hnsw:space": "cosine"}
        )
        self._contador = 0

    def agregar_mensaje(self, user_id: str, rol: str, contenido: str):
        embedding = self.modelo.encode([contenido]).tolist()
        self._contador += 1
        self.coleccion.add(
            ids=[f"{user_id}_{self._contador}_{datetime.utcnow().timestamp()}"],
            embeddings=embedding,
            documents=[contenido],
            metadatas=[{"user_id": user_id, "rol": rol}]
        )

    def obtener_historial_relevante(self, user_id: str, pregunta_actual: str, top_k: int = 4) -> list[dict]:
        """
        A diferencia del buffer (últimos N turnos), recupera los turnos
        pasados MÁS RELEVANTES semánticamente a la pregunta actual,
        sin importar cuándo ocurrieron en la conversación.
        """
        embedding_pregunta = self.modelo.encode([pregunta_actual]).tolist()
        resultados = self.coleccion.query(
            query_embeddings=embedding_pregunta,
            n_results=top_k,
            where={"user_id": user_id}
        )
        if not resultados["documents"] or not resultados["documents"][0]:
            return []

        mensajes = []
        for texto, metadata in zip(resultados["documents"][0], resultados["metadatas"][0]):
            mensajes.append({"role": metadata["rol"], "content": texto})
        return mensajes