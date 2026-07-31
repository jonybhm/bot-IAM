import time
import json
from rag.pipeline import procesar_mensaje

MODELOS = ["llama3.2:3b", "mistral:7b", "llama3.1:8b"]

PREGUNTAS = [
    "¿Qué tipos de métodos de autenticación existen?",
    "¿Por qué no es recomendable tener token de sesión con fecha de expiración grande?",
    "¿Qué significa AAA?",
    "¿Qué es PKCE y para qué sirve?",
    "¿Qué diferencia hay entre autenticación y autorización?",
]


def correr_benchmark():
    resultados = []

    for modelo in MODELOS:
        print(f"\n{'='*60}\nModelo: {modelo}\n{'='*60}")

        for i, pregunta in enumerate(PREGUNTAS):
            # user_id único por modelo+pregunta para que no se mezcle memoria
            # entre corridas de distintos modelos
            user_id = f"bench_{modelo.replace(':', '_')}_{i}"

            inicio = time.time()
            respuesta = procesar_mensaje(user_id, pregunta, modelo=modelo)
            duracion = time.time() - inicio

            resultados.append({
                "modelo": modelo,
                "pregunta": pregunta,
                "duracion_segundos": round(duracion, 2),
                "longitud_respuesta_caracteres": len(respuesta),
                "respuesta": respuesta
            })

            print(f"  [{duracion:.2f}s, {len(respuesta)} caracteres] {pregunta[:50]}...")

    with open("benchmark_resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\nResultados guardados en benchmark_resultados.json")


if __name__ == "__main__":
    correr_benchmark()