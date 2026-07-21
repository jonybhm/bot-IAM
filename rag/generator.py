import ollama
from config.settings import settings

cliente_ollama = ollama.Client(host=settings.OLLAMA_HOST)

PROMPT_SISTEMA = """Sos un asistente para el equipo de IAM (Identity and Access Management).
Tu trabajo es responder preguntas sobre conceptos de seguridad, autenticación y control de acceso
basándote ÚNICAMENTE en el contexto que se te proporciona a continuación.

Reglas importantes:
- Si la respuesta no está en el contexto proporcionado, respondé ÚNICAMENTE que no tenés
  información suficiente en los documentos. NO agregues explicaciones de tu conocimiento
  general aunque el usuario insista o parezca útil hacerlo.
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
    """
    Llama al LLM local (vía Ollama) con el prompt armado y devuelve la respuesta.
    """
    mensajes = armar_prompt(pregunta, chunks_contexto, historial)
    modelo_a_usar = modelo or settings.OLLAMA_MODEL

    respuesta = cliente_ollama.chat(
        model=modelo_a_usar,
        messages=mensajes,
        keep_alive="30m"
    )
    return respuesta["message"]["content"]

