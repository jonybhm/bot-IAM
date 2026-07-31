import unittest

from rag.generator import (
    FRASE_SIN_INFO,
    _admite_falta_de_info
)

from rag.retriever import (
    Retriever,
    es_posible_sigla
)


class TestFeedbackChallenge(unittest.TestCase):

    def test_detecta_respuesta_contradictoria_muhammad_ali(self):
        respuesta = (
            "No tengo información sobre la muerte de Muhammad Ali. "
            "Sin embargo, murió el 3 de junio de 2016."
        )

        self.assertTrue(
            _admite_falta_de_info(respuesta)
        )

    def test_no_detecta_respuesta_valida_como_falta_de_info(self):
        respuesta = (
            "AAA significa Authentication, Authorization "
            "and Accounting."
        )

        self.assertFalse(
            _admite_falta_de_info(respuesta)
        )

    def test_frase_canonica(self):
        self.assertEqual(
            FRASE_SIN_INFO,
            (
                "No tengo información suficiente en los documentos "
                "para responder esa pregunta."
            )
        )

    def test_detecta_sigla_aaa(self):
        resultado = es_posible_sigla(
            "¿Qué significa AAA?"
        )

        self.assertEqual(
            resultado,
            ["AAA"]
        )

    def test_detecta_varias_siglas(self):
        resultado = es_posible_sigla(
            "¿Qué relación existe entre IAM, MFA y SSO?"
        )

        self.assertEqual(
            resultado,
            ["IAM", "MFA", "SSO"]
        )

    def test_no_detecta_palabras_normales_como_siglas(self):
        resultado = es_posible_sigla(
            "¿Qué significa autenticación?"
        )

        self.assertEqual(
            resultado,
            []
        )

    def test_busqueda_lexica_encuentra_aaa(self):
        documentos = {
            "AccessControl.pdf": (
                "El modelo AAA representa Authentication, "
                "Authorization and Accounting."
            ),
            "OtroDocumento.pdf": (
                "Este documento no contiene el término buscado."
            )
        }

        retriever = Retriever(embedder=None)

        resultados = retriever.buscar_por_keyword(
            termino="AAA",
            documentos_originales=documentos,
            top_k=2
        )

        self.assertEqual(
            len(resultados),
            1
        )

        self.assertEqual(
            resultados[0]["fuente"],
            "AccessControl.pdf"
        )

        self.assertEqual(
            resultados[0]["tipo_busqueda"],
            "lexica"
        )

        self.assertIn(
            "AAA",
            resultados[0]["texto"]
        )

    def test_busqueda_lexica_no_encuentra_subcadena(self):
        documentos = {
            "Documento.pdf": (
                "La palabra baaa no representa la sigla buscada."
            )
        }

        retriever = Retriever(embedder=None)

        resultados = retriever.buscar_por_keyword(
            termino="AAA",
            documentos_originales=documentos
        )

        self.assertEqual(
            resultados,
            []
        )


if __name__ == "__main__":
    unittest.main()