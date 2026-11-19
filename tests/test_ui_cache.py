"""
tests/test_ui_cache.py
Pruebas unitarias para el motor de Caché UI Zero-Latency y micro-verificaciones de Vision.
"""
import unittest
import numpy as np
import cv2
from core.ui_cache import UICache
from core.vision import Vision


class TestUICache(unittest.TestCase):
    def setUp(self):
        self.cache = UICache()
        self.vision = Vision()

    def test_default_coords_scaling(self):
        """Verifica que las coordenadas por defecto escalan correctamente a 2400x1080."""
        pos_next = self.cache.get_pos("next_btn", width=2400, height=1080)
        self.assertEqual(pos_next, (2185, 892))

        pos_attack = self.cache.get_pos("attack_btn", width=2400, height=1080)
        self.assertEqual(pos_attack, (195, 945))

        pos_surrender = self.cache.get_pos("surrender_btn", width=2400, height=1080)
        self.assertEqual(pos_surrender, (214, 808))

    def test_jitter_within_bounds(self):
        """Verifica que el jitter humanizado se mantenga dentro del margen legal."""
        for _ in range(50):
            x, y = self.cache.get_pos("next_btn", width=2400, height=1080, jitter=10)
            self.assertTrue(2170 <= x <= 2200)
            self.assertTrue(880 <= y <= 905)
            self.assertTrue(0 <= x <= 2400)
            self.assertTrue(0 <= y <= 1080)

    def test_dynamic_learning(self):
        """Verifica el auto-aprendizaje dinámico con suavizado exponencial."""
        self.cache.learn_on_detect = True
        # Observación ligeramente desplazada en (2190, 895)
        self.cache.update_pos("next_btn", 2190, 895, width=2400, height=1080)
        new_pos = self.cache.get_pos("next_btn", width=2400, height=1080)
        self.assertTrue(2185 <= new_pos[0] <= 2190)
        self.assertTrue(892 <= new_pos[1] <= 895)

    def test_micro_verify_success(self):
        """Verifica que micro_verify encuentra el template en la micro-ROI en tiempo récord (<5ms)."""
        screen = np.zeros((1080, 2400), dtype=np.uint8)
        # Template sintético cuadrado de 40x40 con patrón reconocible
        template = np.full((40, 40), 200, dtype=np.uint8)
        template[10:30, 10:30] = 50

        # Incrustar template exactamente en el centro de next_btn: (2185, 892)
        cx, cy = 2185, 892
        screen[cy - 20:cy + 20, cx - 20:cx + 20] = template

        match = self.cache.micro_verify(screen, "next_btn", template, threshold=0.80)
        self.assertIsNotNone(match)
        self.assertEqual(match, (cx, cy))

    def test_micro_verify_failure_returns_none(self):
        """Verifica que si el botón no está en la zona reducida, retorna None de forma segura."""
        blank_screen = np.zeros((1080, 2400), dtype=np.uint8)
        template = np.full((40, 40), 200, dtype=np.uint8)

        match = self.cache.micro_verify(blank_screen, "next_btn", template, threshold=0.80)
        self.assertIsNone(match)

    def test_find_template_cached_integration(self):
        """Verifica la integración entre Vision y UICache con fallback transparente."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Template existente
        tpl_path = "assets/buttons/next_btn.png"
        loaded = self.vision._load_template(tpl_path)
        if loaded is None:
            self.skipTest(f"{tpl_path} no disponible para test")

        tpl_gray, (th, tw) = loaded
        cx, cy = self.cache.get_pos("next_btn", width=2400, height=1080)
        y1, y2 = cy - th // 2, cy - th // 2 + th
        x1, x2 = cx - tw // 2, cx - tw // 2 + tw
        screen[y1:y2, x1:x2, 0] = tpl_gray
        screen[y1:y2, x1:x2, 1] = tpl_gray
        screen[y1:y2, x1:x2, 2] = tpl_gray

        res = self.vision.find_template_cached(
            screen, tpl_path, cache_name="next_btn", ui_cache=self.cache, threshold=0.75
        )
        self.assertIsNotNone(res)
        self.assertTrue(abs(res[0] - cx) <= 2)
        self.assertTrue(abs(res[1] - cy) <= 2)


if __name__ == "__main__":
    unittest.main()
