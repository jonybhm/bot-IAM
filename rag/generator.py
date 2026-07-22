import ollama
from config.settings import settings

cliente_ollama = ollama.Client(host=settings.OLLAMA_HOST)


FRASE_SIN_INFO = "No tengo información suficiente en los documentos para responder esa pregunta."

PROMPT_SISTEMA = f"""Sos un asistente para el equipo de IAM (Identity and Access Management).
Tu trabajo es responder preguntas sobre conceptos de seguridad, autenticación y control de acceso
basándote ÚNICAMENTE en el contexto que se te proporciona a continuación.

Reglas importantes:
- Si el contexto no contiene la información necesaria para responder, tu respuesta debe ser
  EXACTAMENTE el siguiente texto, sin nada antes ni después:
  "{FRASE_SIN_INFO}"
- Respondé siempre en español, sin importar el idioma del contexto.
- Sé preciso y técnico, pero claro.
"""

def armar_prompt(pregunta: str, chunks_contexto: list[dict], historial: list[dict]) -> list[dict]:
    """
    Arma la lista de mensajes en el formato que espera el LLM,
    combinando: instrucciones del sistema + contexto recuperado + historial + pregunta actual.
    """
    contexto_texto = "\n\n".join(
        f"[Fuente: {c['fuente']}]\n{c['texto']}" for c in chunks_contexto
    )

    mensaje_sistema = {
        "role": "system",
        "content": f"{PROMPT_SISTEMA}\n\nContexto recuperado de los documentos:\n{contexto_texto}"
    }

    mensajes = [mensaje_sistema] + historial + [{"role": "user", "content": pregunta}]
    return mensajes




def generar_respuesta(pregunta: str, chunks_contexto: list[dict], historial: list[dict], modelo: str = None) -> str:
    mensajes = armar_prompt(pregunta, chunks_contexto, historial)
    """
    Llama al LLM local (vía Ollama) con el prompt armado y devuelve la respuesta.
    """
    modelo_a_usar = modelo or settings.OLLAMA_MODEL

    respuesta = cliente_ollama.chat(
        model=modelo_a_usar,
        messages=mensajes,
        keep_alive="30m"
    )
    texto_respuesta = respuesta["message"]["content"]
 
    if FRASE_SIN_INFO.lower()[:30] in texto_respuesta.lower():
        return FRASE_SIN_INFO

    return texto_respuesta
