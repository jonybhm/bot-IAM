# tests/test_conversacional.py (o donde prefieras dejarlo, fuera de la carpeta de tests unitarios si tenés)
import requests

PREGUNTAS = [
    "¿Qué tipos de métodos de autenticación existen?",
    "¿Por qué no es recomendable tener token de sesión con fecha de expiración grande?",
    "¿Qué significa AAA?",
    "¿Qué es PKCE y para qué sirve?",
    "¿Qué diferencia hay entre autenticación y autorización?",
    "¿Qué es OpenID Connect y en qué se diferencia de OAuth?",
    "¿Qué fases tiene el ciclo de vida de una identidad?",
    "¿Qué es el user provisioning?",
    "¿Qué es un ataque de replay de tokens?",
    "¿Cuál es la capital de Francia?",
]

for i, pregunta in enumerate(PREGUNTAS):
    resp = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"user_id": f"test_batch_{i}", "message": pregunta}
    )
    print(f"\n{'='*60}\nP{i+1}: {pregunta}\n{'-'*60}")
    print(resp.json()["response"])