from ingestion.loader import cargar_todos_los_documentos
from ingestion.chunker import chunkear_por_oraciones, limpiar_texto
from ingestion.embedder import Embedder
from config.settings import settings

def ejecutar_ingestion():
    documentos = cargar_todos_los_documentos(settings.DOCUMENTS_PATH)
    embedder = Embedder(settings.VECTORSTORE_PATH)

    for nombre, texto in documentos.items():
        texto_limpio = limpiar_texto(texto)
        chunks = chunkear_por_oraciones(texto_limpio)
        embedder.indexar_chunks(chunks, nombre)
        print(f"Indexado {nombre}: {len(chunks)} chunks")

if __name__ == "__main__":
    ejecutar_ingestion()