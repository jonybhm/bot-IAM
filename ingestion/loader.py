import pypdf
from pathlib import Path

def cargar_documento(ruta: str) -> str:
    """
    Extrae todo el texto de un PDF y lo devuelve como un único string.
    """
    lector = pypdf.PdfReader(ruta)
    texto_completo = []

    for numero_pagina, pagina in enumerate(lector.pages):
        texto_pagina = pagina.extract_text()
        texto_completo.append(texto_pagina)

    return "\n".join(texto_completo)


def cargar_todos_los_documentos(carpeta: str) -> dict[str, str]:
    """
    Recorre una carpeta y carga todos los PDFs que encuentre.
    Devuelve un diccionario {nombre_archivo: texto_extraido}
    """
    documentos = {}
    for archivo in Path(carpeta).glob("*.pdf"):
        documentos[archivo.name] = cargar_documento(str(archivo))
    return documentos