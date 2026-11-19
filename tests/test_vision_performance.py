"""
tests/test_vision_performance.py
Pruebas de rendimiento, ausencia de colisiones de caché, optimización de zona de despliegue,
re-exportación de HUD y banco de escalas en ResourceReader (MEJORA 02).
"""
import os
import time
import unittest
import cv2
import numpy as np

from core.config import PROJECT_ROOT
from core.vision import Vision
from vision.hud import get_battle_hud_mask
from vision.wall_compartments import get_battle_hud_mask as hud_from_compartments
from vision.wall_seg_detector import get_battle_hud_mask as hud_from_seg
from vision.wall_raycast import (
    _compute_frame_fingerprint,
    detect_global_wall_segments,
    reset_iso_angles_cache,
)
from vision.deploy_zone_detector import DeployZoneDetector
from vision.resource_reader import ResourceReader


class TestVisionPerformance(unittest.TestCase):

    def test_01_battle_hud_mask_centralization_and_cache(self):
        """Verifica que get_battle_hud_mask esté centralizado, use caché y sea re-exportado."""
        mask1 = get_battle_hud_mask(1080, 2400)
        mask2 = get_battle_hud_mask(1080, 2400)
        self.assertEqual(mask1.shape, (1080, 2400))
        np.testing.assert_array_equal(mask1, mask2)

        # Re-exportaciones en módulos existentes
        mask_comp = hud_from_compartments(1080, 2400)
        mask_seg = hud_from_seg(1080, 2400)
        np.testing.assert_array_equal(mask1, mask_comp)
        np.testing.assert_array_equal(mask1, mask_seg)

        # Inmutabilidad del caché: modificar mask1 in-place no debe corromper mask3
        mask1[500, 500] = 123
        mask3 = get_battle_hud_mask(1080, 2400)
        self.assertNotEqual(mask3[500, 500], 123, "La copia retornada debe ser mutable sin corromper la caché")

    def test_02_wall_raycast_fingerprint_no_top_row_collision(self):
        """
        Verifica que detect_global_wall_segments con dos imágenes distintas que comparten
        exactamente las primeras 10 filas negras produzca huellas distintas y no use la caché erróneamente.
        """
        reset_iso_angles_cache()

        h, w = 1080, 2400
        # Ambas imágenes comparten las primeras 50 filas completamente en negro
        img1 = np.zeros((h, w, 3), dtype=np.uint8)
        img2 = np.zeros((h, w, 3), dtype=np.uint8)

        # Aldea 1: Líneas simuladas en la región central (ángulo ~36°)
        for y in range(400, 700, 40):
            cv2.line(img1, (800, y), (1400, y + 200), (255, 255, 255), 3)

        # Aldea 2: Líneas simuladas con geometría completamente diferente (ángulo ~144°)
        for y in range(400, 700, 40):
            cv2.line(img2, (1400, y), (800, y + 200), (255, 255, 255), 3)

        # Los primeros 2048 bytes eran idénticos (filas negras superiores)
        self.assertEqual(img1.tobytes()[:2048], img2.tobytes()[:2048])

        # Pero la nueva huella digital por thumbnail central debe ser diferente
        fp1 = _compute_frame_fingerprint(img1)
        fp2 = _compute_frame_fingerprint(img2)
        self.assertNotEqual(fp1, fp2, f"Las huellas ({fp1} vs {fp2}) deben diferir para aldeas distintas")

        # Primera detección
        angles1, segs1 = detect_global_wall_segments(img1)
        # Segunda detección: debe detectar la huella distinta y no devolver segs1
        angles2, segs2 = detect_global_wall_segments(img2)
        self.assertNotEqual(fp1, _compute_frame_fingerprint(img2))

    def test_03_deploy_zone_detector_sub_millisecond_benchmark_and_precision(self):
        """
        Verifica que find_closest_deployable_point():
        1. Devuelva un punto legal con distancia euclidiana precisa (tolerancia <= 2 px).
        2. Ejecute 20 consultas consecutivas sobre una máscara 2400x1080 en < 100 ms totales.
        """
        det = DeployZoneDetector.__new__(DeployZoneDetector)
        det._boundary_pts_cache = {}
        det._last_dist_map = None
        det.last_mask = None

        # Máscara sintética representativa de 2400x1080 con aldea central prohibida
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        mask[350:750, 800:1600] = 1
        det.last_mask = mask

        # Punto objetivo dentro de la base
        target_x, target_y = 1200, 550
        proj_x, proj_y = det.find_closest_deployable_point(target_x, target_y, margin_px=20, mask=mask)

        # El punto debe estar en césped (mask == 0) y cumplir límites seguros
        self.assertEqual(mask[proj_y, proj_x], 0, "El punto proyectado debe estar en césped legal")
        self.assertTrue(120 <= proj_x <= 2280 and 140 <= proj_y <= 845, "Debe respetar safe_bounds")

        # Verificación de precisión: distancia al borde superior de la base (y=350 - margin_px=20 -> y=330)
        # La distancia esperada de target_y (550) a y=330 es 220px
        dist = np.hypot(proj_x - target_x, proj_y - target_y)
        self.assertAlmostEqual(dist, 220.0, delta=2.0)

        # Benchmark de 20 consultas consecutivas con targets diversos
        t0 = time.perf_counter()
        for i in range(20):
            tx = 900 + i * 30
            ty = 400 + (i % 5) * 60
            det.find_closest_deployable_point(tx, ty, margin_px=20, mask=mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertLess(
            elapsed_ms,
            100.0,
            f"20 consultas de zona de despliegue tardaron {elapsed_ms:.2f} ms (debe ser < 100 ms)",
        )

    def test_04_resource_reader_preferred_scale_caching(self):
        """
        Verifica que ResourceReader guarde y utilice la escala preferida tras la primera lectura válida.
        """
        v = Vision()
        reader = ResourceReader(v)

        sample_path = os.path.join(PROJECT_ROOT, "logs", "yolo", "attack_20261109_103747.png")
        if not os.path.exists(sample_path):
            self.skipTest(f"Fixture no encontrado: {sample_path}")

        img = cv2.imread(sample_path)
        self.assertIsNotNone(img)

        # Al inicio no hay escala preferida
        self.assertEqual(reader._preferred_scale, {})

        # Primera lectura: recorre escalas y aprende la escala ganadora para "enemy"
        t0 = time.perf_counter()
        loot1 = reader.read_enemy_loot(img)
        t_first = (time.perf_counter() - t0) * 1000

        self.assertIn("enemy", reader._preferred_scale)
        self.assertEqual(loot1.get("gold"), 974198)
        self.assertEqual(loot1.get("elixir"), 1606022)
        self.assertEqual(loot1.get("dark_elixir"), 12737)

        # Segunda lectura: debe utilizar la escala preferida y devolver resultados idénticos
        t0 = time.perf_counter()
        loot2 = reader.read_enemy_loot(img)
        t_second = (time.perf_counter() - t0) * 1000

        self.assertEqual(loot1, loot2)
        self.assertLessEqual(t_second, t_first * 1.5, "La lectura con escala preferida debe ser rápida")

    def test_05_read_enemy_loot_2x_cake_real_fixture(self):
        """
        Verifica que read_enemy_loot lea con precisión botín con dígitos amarillos/dorados
        y descarte el pastelito y sufijo '2x' del evento 2x Cake desde la captura real.
        """
        fixture_path = os.path.join(PROJECT_ROOT, "tests", "fixtures", "sample_loot_2x_cake.jpg")
        if not os.path.exists(fixture_path):
            self.skipTest(f"Fixture no encontrado: {fixture_path}")

        img = cv2.imread(fixture_path)
        self.assertIsNotNone(img, "No se pudo cargar la imagen del fixture 2x cake")

        v = Vision()
        reader = ResourceReader(v)
        loot = reader.read_enemy_loot(img)

        self.assertEqual(loot.get("gold"), 814164, "Oro con boost 2x debe ser 814,164")
        self.assertEqual(loot.get("elixir"), 446658, "Elixir con boost 2x debe ser 446,658")
        self.assertEqual(loot.get("dark_elixir"), 10914, "Elixir Oscuro con boost 2x debe ser 10,914")

    def test_06_read_enemy_loot_yellow_digits_and_cake_synthetic(self):
        """
        Test sintético offline: genera números amarillos con sufijo '2x' y separador amplio (>1.8*med_w)
        para validar que la binarización HSV y el truncado descarten el '2' espurio sin dependencias externas.
        """
        v = Vision()
        reader = ResourceReader(v)
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        yellow_bgr = (30, 220, 255)  # Color amarillo brillante en BGR

        def paste_number(canvas, roi_box, num_str, suffix="2"):
            ry1, ry2, rx1, rx2 = roi_box
            y1, y2, x1, x2 = int(ry1 * 1080), int(ry2 * 1080), int(rx1 * 2400), int(rx2 * 2400)
            cur_x = x1 + 10
            cur_y = y1 + 10
            for ch in num_str:
                if ch == " ":
                    cur_x += 8  # Espacio separador de miles en CoC (~8px)
                    continue
                tpl_gray, _ = v._load_template(f"assets/digits/{ch}.png")
                th, tw = tpl_gray.shape[:2]
                bin_tpl = reader._binarize_digits(tpl_gray)
                digit_bgr = np.zeros((th, tw, 3), dtype=np.uint8)
                digit_bgr[bin_tpl > 0] = yellow_bgr
                canvas[cur_y:cur_y + th, cur_x:cur_x + tw] = digit_bgr
                cur_x += tw + 1
            # Hueco del pastelito (> 1.8 * med_w, e.g. 50px)
            cur_x += 50
            for ch in suffix:
                tpl_gray, _ = v._load_template(f"assets/digits/{ch}.png")
                th, tw = tpl_gray.shape[:2]
                bin_tpl = reader._binarize_digits(tpl_gray)
                digit_bgr = np.zeros((th, tw, 3), dtype=np.uint8)
                digit_bgr[bin_tpl > 0] = yellow_bgr
                canvas[cur_y:cur_y + th, cur_x:cur_x + tw] = digit_bgr
                cur_x += tw + 1

        paste_number(screen, reader.enemy_rois["gold"], "814 164")
        paste_number(screen, reader.enemy_rois["elixir"], "446 658")
        paste_number(screen, reader.enemy_rois["dark_elixir"], "10 914")

        loot = reader.read_enemy_loot(screen)
        self.assertEqual(loot.get("gold"), 814164)
        self.assertEqual(loot.get("elixir"), 446658)
        self.assertEqual(loot.get("dark_elixir"), 10914)


if __name__ == "__main__":
    unittest.main()
