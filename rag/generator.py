import re
import ollama
from config.settings import settings

cliente_ollama = ollama.Client(
    host=settings.OLLAMA_HOST
)


FRASE_SIN_INFO = (
    "No tengo información suficiente en los documentos "
    "para responder esa pregunta."
)


FRASES_ADMISION_SIN_INFO = [
    r"\bno tengo información\b",
    r"\bno tengo suficiente información\b",
    r"\bno cuento con información\b",
    r"\bno cuento con suficiente información\b",
    r"\bno dispongo de información\b",
    r"\bno dispongo de suficiente información\b",
    r"\bno encontré información\b",
    r"\bno encuentro información\b",
]


PROMPT_SISTEMA = f"""
Sos un asistente para el equipo de IAM
(Identity and Access Management).

Tu trabajo es responder preguntas sobre seguridad,
autenticación, autorización, gestión de identidades
y control de acceso.

Reglas obligatorias:

- Las respuestas técnicas deben basarse ÚNICAMENTE
  en el contexto documental recuperado.

- La memoria conversacional sirve para comprender
  referencias, preferencias o temas mencionados anteriormente,
  pero NO es una fuente de verdad técnica.

- Si una afirmación de la memoria conversacional no está
  respaldada por el contexto documental, no la presentes
  como información correcta.

- Si el contexto documental no contiene la información necesaria,
  respondé EXACTAMENTE con el siguiente texto:

  "{FRASE_SIN_INFO}"

- No agregues nada antes ni después de esa frase.

- No utilices conocimiento general externo al contexto.

- Respondé siempre en español.

- Sé preciso y técnico, pero claro.
"""


def armar_prompt(pregunta: str, chunks_contexto: list[dict], historial: list[dict], memoria_semantica: list[dict] | None = None) -> list[dict]:
    """
    Combina:

    - instrucciones del sistema;
    - contexto documental RAG;
    - memoria vectorial recuperada;
    - historial reciente de SQLite;
    - pregunta actual.
    """
    memoria_semantica = memoria_semantica or []

    contexto_documental = "\n\n".join(
        (
            f"[Fuente documental: {chunk['fuente']}]\n"
            f"{chunk['texto']}"
        )
        for chunk in chunks_contexto
    )

    if memoria_semantica:
        contexto_memoria = "\n\n".join(
            (
                f"[Recuerdo conversacional "
                f"- distancia: {recuerdo['distancia']:.4f}]\n"
                f"{recuerdo['contenido']}"
            )
            for recuerdo in memoria_semantica
        )
    else:
        contexto_memoria = (
            "No se recuperaron recuerdos conversacionales "
            "adicionales."
        )

    contenido_sistema = (
        f"{PROMPT_SISTEMA}\n\n"
        "=== CONTEXTO DOCUMENTAL AUTORIZADO ===\n"
        f"{contexto_documental}\n\n"
        "=== MEMORIA CONVERSACIONAL RECUPERADA ===\n"
        "La siguiente información solo ayuda a interpretar "
        "la conversación. No reemplaza al contexto documental.\n\n"
        f"{contexto_memoria}"
    )

    mensaje_sistema = {
        "role": "system",
        "content": contenido_sistema
    }

    return [
        mensaje_sistema,
        *historial,
        {
            "role": "user",
            "content": pregunta
        }
    ]


def _admite_falta_de_info(texto: str) -> bool:
    """
    Detecta respuestas donde el LLM admite no tener información,
    incluso si luego intenta responder con conocimiento general.
    """
    texto_normalizado = texto.strip().lower()

    return any(
        re.search(
            patron,
            texto_normalizado,
            flags=re.IGNORECASE
        )
        for patron in FRASES_ADMISION_SIN_INFO
    )


def generar_respuesta(pregunta: str, chunks_contexto: list[dict], historial: list[dict], memoria_semantica: list[dict] | None = None, modelo: str = None) -> str:
    """
    Genera una respuesta con Ollama utilizando documentos,
    memoria reciente y memoria vectorial.
    """
    mensajes = armar_prompt(pregunta=pregunta, chunks_contexto=chunks_contexto, historial=historial, memoria_semantica=memoria_semantica)

    modelo_a_usar = (modelo or settings.OLLAMA_MODEL)

    respuesta = cliente_ollama.chat(
        model=modelo_a_usar,
        messages=mensajes,
        options={
            "temperature": 0.0,
            "num_predict": 400
        },
        keep_alive="30m"
    )

    texto_respuesta = (respuesta["message"]["content"].strip())

    if _admite_falta_de_info(texto_respuesta):
        return FRASE_SIN_INFO

    return texto_respuesta