import sqlite3
from pathlib import Path
from datetime import datetime

class ConversationMemory:
    def __init__(self, ruta_db: str = "./data/memoria.db", max_turnos: int = 5):
        """
        max_turnos: cuántos intercambios (pregunta+respuesta) pasados
        se recuperan como contexto para la próxima pregunta.
        """
        self.max_turnos = max_turnos
        Path(ruta_db).parent.mkdir(parents=True, exist_ok=True)
        self.conexion = sqlite3.connect(ruta_db, check_same_thread=False)
        self._crear_tabla()

    def _crear_tabla(self):
        self.conexion.execute("""
            CREATE TABLE IF NOT EXISTS mensajes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                rol TEXT NOT NULL,
                contenido TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        self.conexion.commit()

    def agregar_mensaje(self, user_id: str, rol: str, contenido: str):
        """
        rol: "user" o "assistant"
        """
        self.conexion.execute(
            "INSERT INTO mensajes (user_id, rol, contenido, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, rol, contenido, datetime.utcnow().isoformat())
        )
        self.conexion.commit()

    def obtener_historial(self, user_id: str) -> list[dict]:
        """
        Devuelve los últimos N turnos de este usuario, en el formato
        {"role": ..., "content": ...} que espera el generator.
        """
        cursor = self.conexion.execute(
            """
            SELECT rol, contenido FROM mensajes
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, self.max_turnos * 2)  # *2 porque cada turno = 1 user + 1 assistant
        )
        filas = cursor.fetchall()
        filas.reverse()  # las traemos en orden DESC, hay que volver a orden cronológico

        return [{"role": rol, "content": contenido} for rol, contenido in filas]