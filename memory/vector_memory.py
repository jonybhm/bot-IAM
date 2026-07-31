from datetime import datetime, timezone
from uuid import uuid4

import chromadb


class VectorConversationMemory:
    """
    Memoria conversacional semántica persistida en ChromaDB.

    Cada registro representa un intercambio completo:

        Usuario: pregunta
        Asistente: respuesta

    La recuperación siempre se filtra por user_id para impedir
    que un usuario recupere conversaciones de otro.
    """

    def __init__(self, ruta_vectorstore: str, modelo_embeddings, nombre_coleccion: str = "memoria_conversacional", top_k: int = 3, distancia_maxima: float = 0.75):

        self.modelo = modelo_embeddings
        self.top_k = top_k
        self.distancia_maxima = distancia_maxima

        self.cliente = chromadb.PersistentClient(path=ruta_vectorstore)

        # Se utiliza una colección distinta de "documentos_iam".
        self.coleccion = self.cliente.get_or_create_collection(name=nombre_coleccion,
            metadata={
                "hnsw:space": "cosine",
                "descripcion": (
                    "Memoria conversacional semántica "
                    "separada del corpus RAG"
                )
            }
        )

    @staticmethod
    def formatear_intercambio(pregunta: str, respuesta: str) -> str:
        """
        Formato único utilizado tanto al guardar como al deduplicar.
        """
        return (
            f"Usuario: {pregunta.strip()}\n"
            f"Asistente: {respuesta.strip()}"
        )

    @staticmethod
    def _convertir_embeddings_a_lista(embeddings):
        """
        SentenceTransformer devuelve normalmente un numpy.ndarray,
        mientras que los modelos falsos de los tests pueden devolver
        directamente una lista de Python.
        """
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()

        return embeddings

    def agregar_intercambio(self, user_id: str, pregunta: str, respuesta: str) -> str:
        """
        Genera el embedding del intercambio y lo guarda en ChromaDB.

        Devuelve el ID generado para el registro.
        """
        contenido = self.formatear_intercambio(pregunta=pregunta, respuesta=respuesta)

        embedding_generado = self.modelo.encode([contenido])

        embedding = self._convertir_embeddings_a_lista(embedding_generado)

        registro_id = (f"memory_{user_id}_{uuid4().hex}")

        timestamp = datetime.now(timezone.utc).isoformat()

        self.coleccion.add(ids=[registro_id], embeddings=embedding, documents=[contenido],
            metadatas=[
                {
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "tipo": "intercambio_conversacional"
                }
            ]
        )

        return registro_id

    def buscar_memoria(self, user_id: str, consulta: str, top_k: int | None = None, distancia_maxima: float | None = None) -> list[dict]:
        """
        Recupera intercambios relacionados semánticamente con la consulta.

        La búsqueda se limita exclusivamente al user_id recibido.
        """
        if not user_id.strip() or not consulta.strip():
            return []

        cantidad = (
            top_k
            if top_k is not None
            else self.top_k
        )

        umbral = (
            distancia_maxima
            if distancia_maxima is not None
            else self.distancia_maxima
        )

        vector_generado = self.modelo.encode([consulta])

        vector_consulta = self._convertir_embeddings_a_lista(vector_generado)
        
        resultados = self.coleccion.query(
            query_embeddings=vector_consulta,
            n_results=cantidad,

            where={
                "user_id": user_id
            },

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documentos = resultados.get("documents")
        metadatas = resultados.get("metadatas")
        distancias = resultados.get("distances")

        if not documentos or not documentos[0]:
            return []

        recuerdos = []

        for contenido, metadata, distancia in zip(
            documentos[0],
            metadatas[0],
            distancias[0]
        ):
            if distancia > umbral:
                continue

            recuerdos.append({
                "contenido": contenido,
                "user_id": metadata["user_id"],
                "timestamp": metadata.get("timestamp"),
                "distancia": distancia,
                "tipo_memoria": "vectorial"
            })

        return recuerdos