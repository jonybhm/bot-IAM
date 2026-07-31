# from ingestion.loader import cargar_documento
# from ingestion.chunker import limpiar_texto

# texto = limpiar_texto(cargar_documento("data/documents/AccessControl.pdf"))
# idx = texto.upper().find("AAA")
# print(texto[max(0, idx-100):idx+500])

from rag.pipeline import procesar_mensaje
print(procesar_mensaje("test_grounding", "¿que es OAuth?"))
