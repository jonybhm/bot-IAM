from rag.pipeline import procesar_mensaje

# Grounding
print(procesar_mensaje("test1", "¿qué es un ataque man in the middle?"))

# Búsqueda híbrida
print(procesar_mensaje("test2", "¿Qué significa AAA?"))