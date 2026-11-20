"""
tests/test_deploy_zone_detector.py
Pruebas unitarias para el detector de zona de despliegue IA (DeployZoneDetector)
y su integración con detect_village_boundary.
"""
import os
import unittest
import numpy as np
import cv2

from core.config import PROJECT_ROOT
from vision.deploy_zone_detector import DeployZoneDetector, get_deploy_zone_detector
from attack.deployment_strategy import detect_village_boundary


class TestDeployZoneDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = get_deploy_zone_detector()
        cls.test_img_path = os.path.join(PROJECT_ROOT, "tests", "fixtures", "current_screen.png")
        if os.path.exists(cls.test_img_path):
            cls.test_img = cv2.imread(cls.test_img_path)
        else:
            # Fallback a imagen sintética 2400x1080 con círculo central simulando aldea
            cls.test_img = np.zeros((1080, 2400, 3), dtype=np.uint8)
            cv2.circle(cls.test_img, (1200, 540), 300, (60, 140, 60), -1)

    def test_01_detector_initialization(self):
        """Verifica que el detector cargue la sesión ONNX y reporte estado listo."""
        self.assertIsNotNone(self.detector)
        self.assertTrue(self.detector.is_ready(), "DeployZoneDetector debería estar listo con models/deploy_zone_finder.onnx")

    def test_02_predict_mask_shape_and_values(self):
        """Verifica que la predicción retorne una máscara binaria 2D del tamaño original."""
        mask = self.detector.predict_mask(self.test_img)
        self.assertIsNotNone(mask, "La máscara retornada no debe ser None")
        self.assertEqual(mask.shape, self.test_img.shape[:2], "La máscara debe coincidir con la resolución de entrada")
        self.assertTrue(np.all((mask == 0) | (mask == 1)), "La máscara debe ser estrictamente binaria (0 y 1)")

    def test_03_is_point_deployable(self):
        """Verifica que el validador distinga coordenadas válidas de inválidas."""
        mask = self.detector.predict_mask(self.test_img)
        self.assertIsNotNone(mask)

        # Césped exterior despejado (debe ser desplegable)
        self.assertTrue(self.detector.is_point_deployable(200, 500, mask=mask, safety_margin_px=5))

        # HUD superior izquierdo (debe ser no desplegable)
        self.assertFalse(self.detector.is_point_deployable(100, 100, mask=mask, safety_margin_px=0))

        # Puntos fuera de los límites de pantalla (deben ser rechazados)
        self.assertFalse(self.detector.is_point_deployable(-5, 500, mask=mask))
        self.assertFalse(self.detector.is_point_deployable(2405, 500, mask=mask))

    def test_04_find_closest_deployable_point(self):
        """Verifica que proyecte un punto prohibido hacia una coordenada legal segura."""
        mask = self.detector.predict_mask(self.test_img)
        self.assertIsNotNone(mask)

        # Si el centro es no desplegable, proyectar hacia el borde exterior
        center_x, center_y = 1200, 540
        if not self.detector.is_point_deployable(center_x, center_y, mask=mask, safety_margin_px=0):
            proj_x, proj_y = self.detector.find_closest_deployable_point(center_x, center_y, margin_px=20, mask=mask)
            self.assertTrue(self.detector.is_point_deployable(proj_x, proj_y, mask=mask, safety_margin_px=15),
                            f"El punto proyectado ({proj_x}, {proj_y}) debe ser legal para despliegue")

    def test_05_boundary_polygons(self):
        """Verifica la generación de contornos pts_bld y pts_grass."""
        pts_bld, pts_grass = self.detector.get_boundary_polygons(self.test_img, margin_px=25)
        self.assertIsNotNone(pts_bld, "pts_bld debe ser generado")
        self.assertIsNotNone(pts_grass, "pts_grass debe ser generado")
        self.assertGreater(len(pts_bld), 10, "El contorno de la aldea debe tener suficientes vértices")
        self.assertGreater(len(pts_grass), 10, "El contorno de césped debe tener suficientes vértices")

    def test_06_integration_detect_village_boundary(self):
        """Verifica la integración transparente dentro de detect_village_boundary con modo IA."""
        pts_bld, pts_grass = detect_village_boundary(self.test_img, mode="deploy_zone_finder")
        self.assertIsNotNone(pts_bld)
        self.assertIsNotNone(pts_grass)
        self.assertGreater(len(pts_bld), 10)
        self.assertGreater(len(pts_grass), 10)


if __name__ == "__main__":
    unittest.main()
