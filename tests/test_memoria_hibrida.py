import tempfile
import unittest

from memory.conversation_memory import ConversationMemory
from memory.hybrid_memory import HybridConversationMemory
from memory.vector_memory import VectorConversationMemory
from rag.generator import armar_prompt


class FakeEmbeddingModel:
    """
    Modelo de embeddings determinista para los tests.

    No descarga modelos reales ni necesita Ollama.
    Los textos sobre el mismo concepto reciben el mismo vector.
    """

    def encode(self, textos):
        return [
            self._vectorizar(texto)
            for texto in textos
        ]

    @staticmethod
    def _vectorizar(texto):
        texto = texto.lower()

        if any(
            termino in texto
            for termino in [
                "mfa",
                "multifactor",
                "múltiples factores",
                "doble factor"
            ]
        ):
            return [1.0, 0.0, 0.0, 0.0]

        if "oauth" in texto:
            return [0.0, 1.0, 0.0, 0.0]

        if "pkce" in texto:
            return [0.0, 0.0, 1.0, 0.0]

        return [0.0, 0.0, 0.0, 1.0]


class TestMemoriaHibrida(unittest.TestCase):

    def setUp(self):
        # ignore_cleanup_errors evita problemas de archivos abiertos
        # por ChromaDB en Windows.
        self.directorio_temporal = (
            tempfile.TemporaryDirectory(
                ignore_cleanup_errors=True
            )
        )

        ruta_sqlite = (
            f"{self.directorio_temporal.name}/memoria.db"
        )

        ruta_chroma = (
            f"{self.directorio_temporal.name}/vectorstore"
        )

        self.modelo = FakeEmbeddingModel()

        self.buffer = ConversationMemory(
            ruta_db=ruta_sqlite,
            max_turnos=1
        )

        self.vector = VectorConversationMemory(
            ruta_vectorstore=ruta_chroma,
            modelo_embeddings=self.modelo,
            nombre_coleccion=(
                "memoria_conversacional_test"
            ),
            top_k=3,
            distancia_maxima=0.20
        )

        self.hibrida = HybridConversationMemory(
            buffer_memory=self.buffer,
            vector_memory=self.vector
        )

    def tearDown(self):
        self.directorio_temporal.cleanup()

    def test_guarda_y_recupera_memoria_vectorial(self):
        self.vector.agregar_intercambio(
            user_id="usuario_a",
            pregunta="¿Qué es MFA?",
            respuesta=(
                "MFA utiliza múltiples factores "
                "de autenticación."
            )
        )

        recuerdos = self.vector.buscar_memoria(
            user_id="usuario_a",
            consulta=(
                "¿Qué habíamos hablado sobre "
                "autenticación multifactor?"
            )
        )

        self.assertEqual(
            len(recuerdos),
            1
        )

        self.assertIn(
            "¿Qué es MFA?",
            recuerdos[0]["contenido"]
        )

        self.assertEqual(
            recuerdos[0]["user_id"],
            "usuario_a"
        )

        self.assertEqual(
            recuerdos[0]["tipo_memoria"],
            "vectorial"
        )

    def test_memoria_vectorial_aisla_usuarios(self):
        self.vector.agregar_intercambio(
            user_id="usuario_a",
            pregunta="¿Qué es MFA?",
            respuesta=(
                "MFA utiliza múltiples factores."
            )
        )

        recuerdos_usuario_b = (
            self.vector.buscar_memoria(
                user_id="usuario_b",
                consulta=(
                    "¿Qué habíamos hablado sobre MFA?"
                )
            )
        )

        self.assertEqual(
            recuerdos_usuario_b,
            []
        )

    def test_recupera_recuerdo_fuera_del_buffer(self):
        # El Buffer guarda solamente un turno.
        self.hibrida.guardar_intercambio(
            user_id="usuario_a",
            pregunta="¿Qué es MFA?",
            respuesta=(
                "MFA utiliza múltiples factores "
                "de autenticación."
            )
        )

        # Este segundo turno desplaza al primero del Buffer.
        self.hibrida.guardar_intercambio(
            user_id="usuario_a",
            pregunta="¿Qué es OAuth?",
            respuesta=(
                "OAuth es un protocolo de autorización."
            )
        )

        historial, recuerdos = (
            self.hibrida.obtener_contexto(
                user_id="usuario_a",
                consulta=(
                    "Recordame qué hablamos sobre "
                    "autenticación multifactor"
                )
            )
        )

        # SQLite solo conserva el último turno: OAuth.
        historial_completo = " ".join(
            mensaje["content"]
            for mensaje in historial
        )

        self.assertIn(
            "OAuth",
            historial_completo
        )

        self.assertNotIn(
            "¿Qué es MFA?",
            historial_completo
        )

        # Chroma recupera el intercambio antiguo sobre MFA.
        self.assertEqual(
            len(recuerdos),
            1
        )

        self.assertIn(
            "¿Qué es MFA?",
            recuerdos[0]["contenido"]
        )

    def test_no_duplica_recuerdo_presente_en_buffer(self):
        self.hibrida.guardar_intercambio(
            user_id="usuario_a",
            pregunta="¿Qué es MFA?",
            respuesta=(
                "MFA utiliza múltiples factores."
            )
        )

        _, recuerdos = self.hibrida.obtener_contexto(
            user_id="usuario_a",
            consulta="Explicame nuevamente MFA"
        )

        # El turno todavía está en SQLite.
        # No debe volver a enviarse también desde Chroma.
        self.assertEqual(
            recuerdos,
            []
        )

    def test_puede_guardar_solo_en_sqlite(self):
        self.hibrida.guardar_intercambio(
            user_id="usuario_a",
            pregunta="Hola",
            respuesta="¡Hola!",
            indexar_vectorialmente=False
        )

        historial = self.buffer.obtener_historial(
            "usuario_a"
        )

        recuerdos = self.vector.buscar_memoria(
            user_id="usuario_a",
            consulta="Hola"
        )

        self.assertEqual(
            len(historial),
            2
        )

        self.assertEqual(
            recuerdos,
            []
        )

    def test_umbral_descarta_memoria_irrelevante(self):
        self.vector.agregar_intercambio(
            user_id="usuario_a",
            pregunta="¿Qué es OAuth?",
            respuesta=(
                "OAuth es un protocolo de autorización."
            )
        )

        recuerdos = self.vector.buscar_memoria(
            user_id="usuario_a",
            consulta="¿Qué es MFA?"
        )

        # Los vectores de OAuth y MFA son ortogonales.
        # La distancia supera el límite configurado.
        self.assertEqual(
            recuerdos,
            []
        )

    def test_prompt_incluye_memoria_separada(self):
        chunks = [
            {
                "texto": (
                    "MFA requiere más de un factor "
                    "de autenticación."
                ),
                "fuente": "Autenticacion.pdf",
                "distancia": 0.10
            }
        ]

        historial = [
            {
                "role": "user",
                "content": "¿Qué es OAuth?"
            },
            {
                "role": "assistant",
                "content": (
                    "OAuth es un protocolo "
                    "de autorización."
                )
            }
        ]

        memoria_semantica = [
            {
                "contenido": (
                    "Usuario: ¿Qué es MFA?\n"
                    "Asistente: MFA utiliza "
                    "múltiples factores."
                ),
                "user_id": "usuario_a",
                "timestamp": "2026-07-30T12:00:00+00:00",
                "distancia": 0.05,
                "tipo_memoria": "vectorial"
            }
        ]

        mensajes = armar_prompt(
            pregunta=(
                "¿Qué habíamos hablado sobre MFA?"
            ),
            chunks_contexto=chunks,
            historial=historial,
            memoria_semantica=memoria_semantica
        )

        mensaje_sistema = mensajes[0]["content"]

        self.assertIn(
            "CONTEXTO DOCUMENTAL AUTORIZADO",
            mensaje_sistema
        )

        self.assertIn(
            "MEMORIA CONVERSACIONAL RECUPERADA",
            mensaje_sistema
        )

        self.assertIn(
            "Usuario: ¿Qué es MFA?",
            mensaje_sistema
        )

        self.assertIn(
            "NO es una fuente de verdad técnica",
            mensaje_sistema
        )

        # El historial reciente conserva su formato normal.
        self.assertEqual(
            mensajes[1],
            historial[0]
        )

        self.assertEqual(
            mensajes[-1]["role"],
            "user"
        )

    def test_coleccion_separada_del_corpus(self):
        self.assertEqual(
            self.vector.coleccion.name,
            "memoria_conversacional_test"
        )

        self.assertNotEqual(
            self.vector.coleccion.name,
            "documentos_iam"
        )


if __name__ == "__main__":
    unittest.main()