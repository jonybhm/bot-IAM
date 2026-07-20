from ingestion.embedder import Embedder
from rag.retriever import Retriever
from rag.generator import generar_respuesta
from config.settings import settings

embedder = Embedder(settings.VECTORSTORE_PATH)
retriever = Retriever(embedder)

pregunta = "¿qué es el principio de menor privilegio?"
chunks = retriever.buscar_contexto(pregunta, top_k=5)
respuesta = generar_respuesta(pregunta, chunks, historial=[])

print(respuesta)