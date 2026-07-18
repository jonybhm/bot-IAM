from sentence_transformers import SentenceTransformer
import chromadb

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

class Embedder:
    def __init__(self, ruta_vectorstore: str):
        self.modelo = SentenceTransformer(MODELO_EMBEDDINGS)
        self.cliente = chromadb.PersistentClient(path=ruta_vectorstore)
        self.coleccion = self.cliente.get_or_create_collection("documentos_iam")

    def indexar_chunks(self, chunks: list[str], nombre_documento: str):
        """
        Genera el embedding de cada chunk y lo guarda en ChromaDB,
        junto con metadata (de qué documento vino) para poder citarlo después.
        """
        embeddings = self.modelo.encode(chunks).tolist()
        ids = [f"{nombre_documento}_{i}" for i in range(len(chunks))]
        metadatas = [{"fuente": nombre_documento} for _ in chunks]

        self.coleccion.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )