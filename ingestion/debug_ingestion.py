from ingestion.loader import cargar_documento
from ingestion.chunker import limpiar_texto

texto = limpiar_texto(cargar_documento("data/documents/TiposDeAutenticacion.pdf"))
print(len(texto.split()), "palabras totales")