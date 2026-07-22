# IAM Support Bot

Bot conversacional para el equipo de IAM (Identity and Access Management), que responde preguntas sobre conceptos y tendencias de protección de cuentas de usuario basándose en un corpus de documentos técnicos. Utiliza un LLM Open Source ejecutado localmente mediante Ollama, combinado con una arquitectura de RAG (Retrieval-Augmented Generation) y memoria conversacional propia.

## Índice

- [Screenshots](#screenshots)
- [Arquitectura](#arquitectura)
- [Cómo ejecutar](#cómo-ejecutar)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Pruebas conversacionales y resultados](#pruebas-conversacionales-y-resultados)
- [Próximos pasos](#próximos-pasos)

## Screenshots

![Screenshot conversación](assets/images/Screenshot_IAM_Support_Bot_01.png)
![Screenshot conversación](assets/images/Screenshot_IAM_Support_Bot_02.png)


## Arquitectura

El sistema sigue un flujo de RAG clásico, dividido en etapas independientes:

![Arquitectura del sistema](assets/images/Arquitectura.png)

**Módulos del proyecto:**

| Carpeta | Responsabilidad |
|---|---|
| `ingestion/` | Extracción de texto de PDFs (`loader.py`), limpieza y chunking (`chunker.py`), generación de embeddings y persistencia en ChromaDB (`embedder.py`) |
| `rag/` | Búsqueda semántica de contexto relevante (`retriever.py`), armado de prompt y llamada al LLM (`generator.py`), orquestación del flujo completo (`pipeline.py`) |
| `memory/` | Memoria conversacional propia, implementada con SQLite (`conversation_memory.py`) — **no se usa memoria nativa de ninguna librería ni del LLM**, tal como exige la consigna |
| `api/` | Capa HTTP: definición de endpoints (`routes.py`) y schemas de request/response (`schemas.py`) |
| `config/` | Configuración centralizada, leída desde variables de entorno (`settings.py`) |
| `data/` | Documentos originales, persistencia de la base vectorial y de la memoria conversacional |

**Stack:**

- **API:** FastAPI + Uvicorn
- **LLM:** Ollama (local), modelo por defecto `llama3.1:8b`
- **Embeddings:** `sentence-transformers`, modelo `paraphrase-multilingual-MiniLM-L12-v2` (multilingüe, ya que los documentos fuente están en inglés y las preguntas de los usuarios se esperan en español)
- **Base de datos vectorial:** ChromaDB (persistente en disco)
- **Memoria conversacional:** SQLite, manejada íntegramente por la propia API

## Cómo ejecutar

### Requisitos previos

- Python 3.10+
- [Ollama](https://ollama.com/download) instalado y corriendo localmente
- Un modelo descargado en Ollama:
  ```bash
  ollama pull llama3.1:8b
  ```

### Instalación

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt --break-system-packages
```

### Configuración

Copiar `.env.example` a `.env` (o crear `.env` con las siguientes variables):

```
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
DOCUMENTS_PATH=./data/documents
VECTORSTORE_PATH=./data/vectorstore
RAG_TOP_K=4
```

### Indexar los documentos

Colocar los PDFs del corpus en `data/documents/` y correr, desde la raíz del proyecto:

```bash
python -m ingestion.run_ingestion
```

Esto genera los embeddings de todos los documentos y los persiste en `data/vectorstore/`. Solo es necesario correrlo una vez, o cada vez que se agreguen o modifiquen documentos.

### Levantar la API

```bash
uvicorn main:app --reload
```

La API queda disponible en `http://localhost:8000`, con documentación interactiva (Swagger) en `http://localhost:8000/docs`.

### Probar el bot

**Vía Swagger:** entrar a `/docs`, desplegar `POST /chat`, usar "Try it out".

**Vía HTTP directo:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "usuario1", "message": "¿qué es PKCE?"}'
```

El campo `user_id` identifica la conversación — mismo `user_id` mantiene memoria entre preguntas; distintos `user_id` no comparten historial entre sí.

### Ejecución con Docker Compose (alternativa)

Requiere Docker Desktop instalado y corriendo.

```bash
docker compose up -d --build
```

La primera vez, bajar el modelo dentro del contenedor de Ollama:
```bash
docker compose exec ollama ollama pull llama3.1:8b
```

E indexar los documentos dentro del contenedor:
```bash
docker compose exec api python -m ingestion.run_ingestion
```

La API queda disponible igual que en la ejecución manual, en `http://localhost:8000`, y el frontend en `http://localhost:8000/app`.

Para pausar sin perder el modelo descargado ni los contenedores construidos:
```bash
docker compose stop     # pausa
docker compose start    # retoma
```

## Decisiones de diseño

### RAG en lugar de fine-tuning

La consigna pide "entrenar un LLM con la información de los documentos". Se optó por **RAG (Retrieval-Augmented Generation)** en lugar de fine-tuning de los pesos del modelo, por varias razones:

- Fine-tuning requiere recursos de cómputo (GPU) significativamente mayores, no disponibles en el entorno de desarrollo.
- RAG permite actualizar el conocimiento del bot simplemente re-indexando documentos, sin reentrenar nada — más apropiado para un caso de uso donde los documentos de referencia pueden cambiar.
- Reduce el riesgo de alucinación: el modelo responde en base a texto recuperado explícitamente, no a lo que "memorizó" durante un entrenamiento.

### Chunking: de párrafos a oraciones completas

Se probaron tres estrategias de chunking, iterando sobre los resultados:

1. **Chunking por tamaño fijo de palabras:** simple, pero corta ideas a la mitad sin ningún criterio.
2. **Chunking por párrafos (`\n\n`):** falló en la práctica porque los PDFs del corpus no preservan saltos de línea dobles de forma confiable al extraer el texto (`pypdf` concatena líneas de forma continua) — en varios documentos, todo el texto quedaba interpretado como un único "párrafo" gigante.
3. **Chunking por oraciones completas (solución final):** se acumulan oraciones completas hasta acercarse a un límite de ~400 palabras, con overlap de 2 oraciones entre chunks consecutivos. Esto garantiza que ningún chunk corte una idea a la mitad, independientemente de cómo esté formateado el PDF original.

### Limpieza de texto antes de chunkear

Se detectó que las tablas de contenido de los PDFs (líneas tipo `TERMINOLOGY....4`) se colaban como texto dentro de los primeros chunks, diluyendo el embedding con ruido no semántico. Se agregó un paso de limpieza (`limpiar_texto`) que descarta líneas con secuencias de 4 o más puntos seguidos, antes de chunkear.

### Métrica de similitud: coseno en lugar de distancia euclidiana (L2)

ChromaDB usa distancia euclidiana (L2) por defecto, que no tiene un rango acotado y es difícil de interpretar o de usar como umbral de corte. Se configuró explícitamente la colección para usar **similitud coseno** (`metadata={"hnsw:space": "cosine"}`), que produce valores interpretables (cercano a 0 = muy similar) y es el estándar para comparar embeddings de texto.

### Grounding del LLM (evitar alucinaciones)

El prompt de sistema instruye explícitamente al modelo a responder únicamente en base al contexto recuperado, y a indicar cuando no tiene información suficiente en lugar de inventar una respuesta. En las pruebas se observó un comportamiento **mayormente estricto, con una excepción parcial** (ver [Limitaciones conocidas](#limitaciones-conocidas)) — queda documentado como un punto de mejora concreto: endurecer aún más la instrucción para eliminar por completo la mezcla de conocimiento general del modelo con el contexto de los documentos.

Se reforzó el grounding con una segunda capa de seguridad a nivel código, no solo de prompt: si la respuesta generada comienza indicando que no hay información suficiente, se descarta cualquier texto adicional que el modelo agregue después (evitando que complete igual con conocimiento general), devolviendo únicamente el mensaje canónico de "no tengo información suficiente".

### Memoria conversacional: buffer simple en SQLite

Se implementó memoria de tipo "buffer" (los últimos N turnos de conversación, sin resumir), persistida en SQLite y separada por `user_id`. Se eligió SQLite por ser parte de la biblioteca estándar de Python (sin dependencias ni infraestructura adicional) y por ofrecer persistencia real entre reinicios de la API, a diferencia de una estructura en memoria (diccionario).

La memoria es manejada íntegramente por la API (tabla propia, lógica de inserción y recuperación propia) — no se utiliza memoria nativa de LangChain ni del LLM, conforme lo exige la consigna.

### Manejo de dependencias async/sync en FastAPI

El endpoint `/chat` está declarado como `async`, pero la lógica interna (retriever, LLM, SQLite) es sincrónica y potencialmente bloqueante. Se utiliza `run_in_threadpool` para ejecutar esa lógica en un thread separado, evitando que bloquee el event loop de FastAPI ante múltiples requests concurrentes.

### Búsqueda híbrida para siglas y acrónimos cortos

Se detectó que la búsqueda semántica pura fallaba en preguntas sobre siglas cortas (ej. "¿Qué significa AAA?"), porque un embedding de 2-5 letras aporta poca señal semántica comparado con el texto que lo rodea. Se implementó una **búsqueda híbrida**: cuando la pregunta contiene un token que matchea el patrón de una sigla (2 a 5 letras mayúsculas), se complementa la búsqueda semántica habitual con una búsqueda léxica directa sobre el texto original de los documentos, agregando los fragmentos encontrados al contexto pasado al LLM. Validado con la pregunta de ejemplo del challenge ("¿Qué significa AAA?"), que ahora responde correctamente citando la sección exacta del documento.

## Limitaciones conocidas

### Ejecución en CPU

El entorno de desarrollo no cuenta con GPU dedicada, por lo que el LLM corre sobre CPU vía Ollama. Esto es funcional pero introduce latencia (varios segundos por respuesta) que sería menor en un entorno con GPU o usando un modelo más liviano.

### Límite de memoria de Ollama dentro de Docker

Al correr Ollama dentro de un contenedor gestionado por WSL2, el proceso solo tiene visibilidad de una porción de la RAM total del host (11.7 GiB observados, sobre 32GB físicos disponibles) — es el límite de memoria que Docker Desktop asigna a WSL2 por defecto, no un límite de la aplicación en sí. No representó un problema para `llama3.1:8b`, pero es un punto a tener en cuenta si se quisiera correr un modelo más grande dentro de Docker: el límite es ajustable en Docker Desktop → Settings → Resources → Memory.

## Pruebas conversacionales y resultados

Se ejecutó un set de 10 preguntas contra el endpoint `/chat`, cubriendo la mayoría de los documentos del corpus y un caso negativo (pregunta fuera de dominio), con `user_id` distintos para evitar interferencia de memoria entre pruebas.

| # | Pregunta | Documento esperado | Resultado |
|---|---|---|---|
| 1 | ¿Qué tipos de métodos de autenticación existen? | TiposDeAutenticacion.pdf | ✅ Correcto y completo |
| 2 | ¿Por qué no es recomendable tener token de sesión con fecha de expiración grande? | BuenasPracticasTiempoDeSesion2025.pdf | ✅ Correcto y completo |
| 3 | ¿Qué significa AAA? | AccessControl.pdf | ✅ Correcto |
| 4 | ¿Qué es PKCE y para qué sirve? | PKCE.pdf | ✅ Correcto |
| 5 | ¿Qué diferencia hay entre autenticación y autorización? | Autenticacion+Autorizacion.pdf | ✅ Correcto, combinó bien dos fuentes |
| 6 | ¿Qué es OpenID Connect y en qué se diferencia de OAuth? | OpenIDC.pdf | ✅ Correcto y completo |
| 7 | ¿Qué fases tiene el ciclo de vida de una identidad? | LifeCycle.pdf | ✅ Correcto, distinguió ciclo de cliente vs. trabajador |
| 8 | ¿Qué es el user provisioning? | UserProvisioning.pdf | ✅ Correcto |
| 9 | ¿Qué es un ataque de replay de tokens? | BuenasPracticasTiempoDeSesion2025.pdf | ✅ Correcto |
| 10 | ¿Cuál es la capital de Francia? (caso negativo, fuera de dominio) | — | ✅ Rechazó correctamente responder |

**Prueba de memoria conversacional:** se validó adicionalmente una conversación de dos turnos, donde la segunda pregunta ("¿y ese tipo de token es más o menos seguro que uno de corta duración?") solo tiene sentido en conjunto con la primera ("¿qué es un refresh token?"). El bot respondió correctamente utilizando el contexto del turno anterior, confirmando que la memoria conversacional funciona de punta a punta. Se validó también que conversaciones con distinto `user_id` no comparten historial entre sí.

## Benchmark de modelos

Se comparó el mismo set de 5 preguntas (subconjunto del set de pruebas anterior) ejecutado contra tres modelos Open Source distintos, corriendo en CPU vía Ollama, midiendo tiempo de respuesta:

| Modelo | Tamaño | Tiempo promedio | Rango (min-max) |
|---|---|---|---|
| `llama3.2:3b` | 3B parámetros | **36.5s** | 24.5s - 44.8s |
| `llama3.1:8b` | 8B parámetros | 79.7s | 44.0s - 125.7s |
| `mistral:7b` | 7B parámetros | 99.3s | 62.5s - 158.0s |

**Observaciones:**

- `llama3.2:3b` resultó más del doble de rápido que los otros dos modelos, consistente con ser el más chico de los tres — es el candidato natural si se prioriza latencia sobre profundidad de respuesta (por ejemplo, para una demo interactiva).
- `mistral:7b`, a pesar de tener menos parámetros que `llama3.1:8b`, resultó el más lento del grupo. Esto contradice la intuición de que un modelo más chico es siempre más rápido — probablemente se deba a diferencias de arquitectura interna entre ambas familias de modelos, no solo al conteo de parámetros. Es un hallazgo relevante: el tamaño nominal del modelo no es un predictor confiable de latencia real sin medirlo.
- Cualitativamente, las respuestas de `llama3.2:3b` fueron notablemente más cortas (promedio de ~730 caracteres) que las de `mistral:7b` (~850 caracteres) y `llama3.1:8b` (~690 caracteres, aunque afectado por la respuesta corta de la pregunta de AAA que el modelo no pudo responder). No se observó una degradación evidente en la corrección de las respuestas del modelo más chico para este set de preguntas, aunque el análisis fue cualitativo, no una evaluación automatizada de exactitud.
- **Recomendación:** para el caso de uso de este bot (soporte interno, no tiempo real crítico), `llama3.1:8b` sigue siendo el default razonable por su balance entre calidad y velocidad. Para escenarios donde la latencia es más crítica que el detalle de la respuesta, `llama3.2:3b` es una alternativa a considerar.

Resultados completos (con el texto de cada respuesta) disponibles en `benchmark_resultados.json`, generado por `tests/benchmark_modelos.py`.

### Frontend propio en lugar de chat-ui

La consigna sugiere `chat-ui` (proyecto open source de HuggingFace) como opción para el frontend. Se evaluó y se decidió no adoptarlo tal cual, por una diferencia de alcance significativa respecto al resto del proyecto: `chat-ui` está construido en SvelteKit (stack de JavaScript distinto al resto del proyecto, en Python), y espera una base de datos MongoDB para persistir conversaciones — dos dependencias nuevas que no aportan valor proporcional al tiempo que tomaría integrarlas, dado que la memoria conversacional del bot ya está resuelta del lado del backend.

En su lugar, se optó por un **frontend propio, minimalista** (HTML + CSS + JavaScript vanilla, sin frameworks), servido directamente desde la misma API de FastAPI mediante archivos estáticos (`app.mount(...)`). Con estética de burbujas de chat similar a ChatGPT. Ventajas de este enfoque:

- Cero dependencias nuevas (no requiere Node.js, npm, ni una base de datos adicional).
- Reutiliza el endpoint `/chat` existente sin modificaciones ni adaptación de protocolo.
- Cada sesión de navegador genera un `user_id` aleatorio propio al cargar la página, aprovechando la separación de memoria por usuario ya implementada en el backend — sin necesidad de un sistema de login.

El frontend queda disponible en `frontend/index.html`, accesible en `http://localhost:8000/app` una vez levantada la API.

## Próximos pasos

- [x] Dockerizar la aplicación (API + Ollama) con Docker Compose — validado y funcionando de punta a punta
- [x] Benchmark comparativo entre múltiples modelos (`llama3.1:8b`, `mistral:7b`, `llama3.2:3b`)
- [x] Frontend conversacional
- [ ] Despliegue en un servicio web (fly.io / Render)
- [ ] Búsqueda híbrida (semántica + keyword) para mejorar el caso de siglas/acrónimos
