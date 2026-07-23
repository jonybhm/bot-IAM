import re
from config.settings import settings

def limpiar_texto(texto: str) -> str:
    """
    Elimina ruido típico de la extracción de PDFs:
    - Líneas de tabla de contenidos (títulos seguidos de puntos y número de página)
    - Espacios múltiples raros que a veces deja pypdf entre letras
    """
    lineas = texto.split("\n")
    lineas_limpias = []

    for linea in lineas:
        if re.search(r"\.{4,}", linea):
            continue
        lineas_limpias.append(linea)

    texto_limpio = "\n".join(lineas_limpias)

    texto_limpio = re.sub(r" {2,}", " ", texto_limpio)

    return texto_limpio

def dividir_en_oraciones(texto: str) -> list[str]:
    """
    Divide el texto en oraciones usando un patrón simple:
    corta después de '.', '?' o '!' seguido de espacio y mayúscula.
    """
    oraciones = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', texto)
    return [o.strip() for o in oraciones if o.strip()]


def chunkear_por_oraciones(texto: str, max_palabras: int = settings.CHUNK_SIZE, overlap_oraciones: int = settings.CHUNK_OVERLAP_ORACIONES) -> list[str]:
    """
    Agrupa oraciones completas hasta acercarse a max_palabras.
    Nunca corta una oración a la mitad.
    """
    oraciones = dividir_en_oraciones(texto)
    chunks = []
    buffer = []
    palabras_buffer = 0

    for oracion in oraciones:
        palabras_oracion = len(oracion.split())

        if palabras_buffer + palabras_oracion > max_palabras and buffer:
            chunks.append(" ".join(buffer))
            buffer = buffer[-overlap_oraciones:]
            palabras_buffer = sum(len(o.split()) for o in buffer)

        buffer.append(oracion)
        palabras_buffer += palabras_oracion

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks