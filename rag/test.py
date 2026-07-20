from rag.pipeline import procesar_mensaje

user_id = "test_user_1"

print("--- Turno 1 ---")
r1 = procesar_mensaje(user_id, "¿qué es un refresh token?")
print(r1)

print("\n--- Turno 2 (pregunta que depende del contexto anterior) ---")
r2 = procesar_mensaje(user_id, "¿y ese tipo de token es más o menos seguro que uno de corta duración?")
print(r2)