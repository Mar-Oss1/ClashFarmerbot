"""
tests/test_drop_validation.py
Tests unitarios sintéticos para validar puntos de drop y regresiones de máscara.
"""
import unittest
import numpy as np

from vision.deploy_zone_detector import DeployZoneDetector


class TestDropValidation(unittest.TestCase):

    def setUp(self):
        # Detector ligero sin sesión ONNX para tests puramente sintéticos
        self.detector = DeployZoneDetector.__new__(DeployZoneDetector)
        self.detector.session = None
        self.detector.input_name = "input"
        self.detector.output_name = "output"
        self.detector.input_shape = (1, 3, 288, 512)
        self.detector.last_mask = None
        self.detector.last_pts_bld = None
        self.detector.last_pts_grass = None

    def test_01_isolated_blob_forbidden(self):
        """Verifica que un componente conexo aislado en la máscara sea considerado prohibido."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        # Aldea principal
        mask[300:700, 900:1500] = 1
        # Edificio aislado exterior (segundo blob)
        mask[200:260, 400:460] = 1

        # Punto dentro del edificio aislado debe ser NO desplegable
        self.assertFalse(
            self.detector.is_point_deployable(430, 230, mask=mask, safety_margin_px=5),
            "El edificio aislado debe considerarse zona prohibida"
        )
        # Punto en césped exterior debe ser desplegable
        self.assertTrue(
            self.detector.is_point_deployable(200, 200, mask=mask, safety_margin_px=5),
            "Césped libre debe ser desplegable"
        )

    def test_02_none_mask_behavior(self):
        """Verifica el comportamiento seguro cuando la máscara es None."""
        self.detector.last_mask = None
        # Por compatibilidad con tests legados retorna True, pero no lanza excepción
        res = self.detector.is_point_deployable(500, 500, mask=None, safety_margin_px=10)
        self.assertIsInstance(res, bool)

    def test_03_find_closest_deployable_point_outside_mask(self):
        """Verifica que un punto dentro de la máscara sea proyectado fuera de ella."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        mask[400:600, 1000:1400] = 1

        px, py = self.detector.find_closest_deployable_point(
            1200, 500, margin_px=20, mask=mask
        )
        # El punto resultante debe estar en césped (mask == 0)
        self.assertEqual(mask[py, px], 0, "El punto proyectado debe estar en césped")
        self.assertTrue(self.detector.is_point_deployable(px, py, mask=mask, safety_margin_px=15))


if __name__ == "__main__":
    unittest.main()
