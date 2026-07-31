import logging

from memory.conversation_memory import ConversationMemory
from memory.vector_memory import VectorConversationMemory


logger = logging.getLogger(__name__)


class HybridConversationMemory:
    """
    Combina dos estrategias:

    1. Buffer SQLite:
       recupera los últimos turnos en orden cronológico.

    2. Memoria vectorial ChromaDB:
       recupera intercambios antiguos semánticamente relevantes.
    """

    def __init__(self, buffer_memory: ConversationMemory, vector_memory: VectorConversationMemory):

        self.buffer_memory = buffer_memory
        self.vector_memory = vector_memory

    @staticmethod
    def _serializar_intercambios_recientes(historial: list[dict]) -> set[str]:
        """
        Convierte el historial reciente al mismo formato utilizado por la memoria vectorial.
        """
        intercambios = set()
        pregunta_pendiente = None

        for mensaje in historial:
            rol = mensaje.get("role")
            contenido = mensaje.get("content", "")

            if rol == "user":
                pregunta_pendiente = contenido

            elif (rol == "assistant" and pregunta_pendiente is not None):
                intercambio = (
                    VectorConversationMemory
                    .formatear_intercambio(
                        pregunta=pregunta_pendiente,
                        respuesta=contenido
                    )
                )

                intercambios.add(intercambio)
                pregunta_pendiente = None

        return intercambios

    def obtener_contexto(self, user_id: str, consulta: str) -> tuple[list[dict], list[dict]]:
        """
        Devuelve:

        - historial_reciente: últimos turnos desde SQLite.
        - memoria_semantica: recuerdos relevantes desde ChromaDB.
        """
        historial_reciente = (self.buffer_memory.obtener_historial(user_id))

        try:
            memoria_semantica = (
                self.vector_memory.buscar_memoria(
                    user_id=user_id,
                    consulta=consulta
                )
            )

        except Exception:

            logger.exception(
                "No se pudo recuperar la memoria vectorial "
                "para el usuario %s",
                user_id
            )

            memoria_semantica = []

        intercambios_recientes = (self._serializar_intercambios_recientes(historial_reciente))

        memoria_sin_duplicados = [
            recuerdo
            for recuerdo in memoria_semantica
            if recuerdo["contenido"]
            not in intercambios_recientes
        ]

        return (historial_reciente, memoria_sin_duplicados)

    def guardar_intercambio(self, user_id: str, pregunta: str, respuesta: str, indexar_vectorialmente: bool = True) -> None:
        """
        Siempre guarda el intercambio en SQLite.

        Solo lo indexa en ChromaDB cuando indexar_vectorialmente=True.
        """
        self.buffer_memory.agregar_mensaje( user_id=user_id,rol="user",contenido=pregunta)

        self.buffer_memory.agregar_mensaje(user_id=user_id, rol="assistant", contenido=respuesta)

        if not indexar_vectorialmente:
            return

        try:
            self.vector_memory.agregar_intercambio(user_id=user_id, pregunta=pregunta, respuesta=respuesta)

        except Exception:
            logger.exception(
                "No se pudo guardar la memoria vectorial "
                "para el usuario %s",
                user_id
            )