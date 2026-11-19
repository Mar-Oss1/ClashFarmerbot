"""
tests/test_damage_reader.py
Pruebas unitarias para el lector de porcentaje de destrucción (DamageReader).
"""
import unittest
import os
import cv2
import numpy as np
from vision.damage_reader import DamageReader
from core.config import PROJECT_ROOT


class TestDamageReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DamageReader()

    def test_templates_loaded(self):
        """Verifica que todos los dígitos 0-9 y el símbolo % se cargan correctamente."""
        self.assertEqual(len(self.reader.digit_templates), 10, "Deben cargarse 10 dígitos (0-9)")
        for d in range(10):
            self.assertIn(d, self.reader.digit_templates, f"Falta la plantilla del dígito {d}")
            tpl = self.reader.digit_templates[d]
            self.assertGreater(tpl.shape[0], 20, f"Altura de plantilla {d} inválida")
            self.assertGreater(tpl.shape[1], 8, f"Ancho de plantilla {d} inválida")

        self.assertIsNotNone(self.reader.percent_template, "Debe cargarse la plantilla de %")
        self.assertGreater(self.reader.percent_template.shape[0], 20)

    def test_read_sample_12s(self):
        """Verifica la lectura de porcentaje en la captura a 12s."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_edrag_atk1_12s.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertIn(pct, (43, 44, 55), "Debe leer el porcentaje válido (43%, 44% o 55%)")

    def test_read_sample_107s(self):
        """Verifica la lectura de 77% en la captura a 107s."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_edrag_atk1_107s.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertEqual(pct, 77, "Debe leer exactamente 77%")

    def test_read_sample_progress_atk2(self):
        """Verifica la lectura de porcentaje en battle_progress_atk2."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_atk2.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertIn(pct, (44, 50, 54, 55, 57, 61, 62), f"Debe leer porcentaje válido ({pct})")

    def test_read_sample_progress_atk4(self):
        """Verifica la lectura de porcentaje en battle_progress_atk4."""
        path = os.path.join(PROJECT_ROOT, "logs", "runs", "battle_progress_atk4.png")
        if not os.path.exists(path):
            self.skipTest(f"Captura {path} no encontrada")
        img = cv2.imread(path)
        pct = self.reader.read_destruction_percentage(img)
        self.assertIn(pct, (39, 52, 54, 55, 58, 68), f"Debe leer porcentaje válido ({pct})")

    def test_invalid_and_blank_screens(self):
        """Verifica que pantallas nulas, vacías o negras retornan None."""
        self.assertIsNone(self.reader.read_destruction_percentage(None))
        self.assertIsNone(self.reader.read_destruction_percentage(np.array([], dtype=np.uint8)))

        black_screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.assertIsNone(self.reader.read_destruction_percentage(black_screen))

    def test_roi_scaling(self):
        """Verifica que el cálculo de ROI escala correctamente en distintas resoluciones."""
        rx, ry, rw, rh = self.reader._get_roi(1080, 2400)
        self.assertEqual((rx, ry, rw, rh), (2130, 780, 200, 60))

        # En 720p (1280x720)
        rx720, ry720, rw720, rh720 = self.reader._get_roi(720, 1280)
        self.assertAlmostEqual(rx720, int(2130 * 1280 / 2400), delta=1)
        self.assertAlmostEqual(ry720, int(780 * 720 / 1080), delta=1)


if __name__ == "__main__":
    unittest.main()
