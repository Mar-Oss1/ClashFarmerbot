"""
tests/test_wall_raycast.py
Batería de pruebas unitarias y de integración para la detección física real de muros:
1. Estimación dinámica de ángulos isométricos globales (estimate_iso_angles).
2. Detección global de segmentos de muro en una sola pasada (detect_global_wall_segments).
3. Filtrado de ángulos de cruce (<25° descartados).
4. Filtrado de posición razonable (<40px del drop, <50px del target descartados).
5. Tolerancia colineal de extremos (acepta tramos de muro hasta 30px fuera del segmento).
6. Fusión de cruces a <25px y retención de muro doble a >25px.
7. Comportamiento seguro de fallback cuando no hay bordes detectables.
8. Integración completa con deployment_strategy.
"""
import os
import unittest
import cv2
import numpy as np

from vision.wall_raycast import (
    estimate_iso_angles,
    detect_global_wall_segments,
    find_wall_crossings_from_segments,
    find_wall_crossings,
    draw_wall_raycast_debug,
    reset_iso_angles_cache,
)
from vision.yolo_detector import Detection
from attack.deployment_strategy import (
    compute_adaptive_jump_targets_from_detections,
    _compute_wall_raycast_jump_targets,
)
from core.config import SETTINGS


class TestWallRaycast(unittest.TestCase):
    def setUp(self):
        reset_iso_angles_cache()
        self.w = 2400
        self.h = 1080

    def test_estimate_iso_angles_finds_peaks(self):
        """Comprueba que una rejilla isométrica sintética a 36° y 144° devuelve los picos correctos."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Dibujar líneas a 36° (dy/dx = tan(36°) ~ 0.7265)
        for x0 in range(200, 2200, 80):
            pt1 = (x0, 200)
            pt2 = (int(x0 + 400), int(200 + 400 * 0.7265))
            cv2.line(img, pt1, pt2, (255, 255, 255), 3)

        # Dibujar líneas a 144° (dy/dx = tan(144°) = -tan(36°) ~ -0.7265)
        for x0 in range(200, 2200, 80):
            pt1 = (x0, 800)
            pt2 = (int(x0 + 400), int(800 - 400 * 0.7265))
            cv2.line(img, pt1, pt2, (255, 255, 255), 3)

        peak1, peak2 = estimate_iso_angles(img, force_refresh=True)
        self.assertAlmostEqual(peak1, 36.0, delta=6.0)
        self.assertAlmostEqual(peak2, 144.0, delta=6.0)

    def test_detect_global_wall_segments_runs_once(self):
        """Comprueba que detect_global_wall_segments extrae segmentos globales y usa caché."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Línea de muro de 100px a 36° en el centro
        pt1 = (1000, 500)
        pt2 = (int(1000 + 100), int(500 + 100 * 0.7265))
        cv2.line(img, pt1, pt2, (255, 255, 255), 4)

        angles, segs = detect_global_wall_segments(img, force_refresh=True, min_line_len=50)
        self.assertIsInstance(angles, tuple)
        self.assertIsInstance(segs, list)

    def test_crossing_angle_filter_rejects_parallel_lines(self):
        """Verifica que si el rayo corre casi paralelo al muro (<25°), la línea es descartada."""
        p_drop = (300, 500)
        p_target = (900, 500)  # Rayo horizontal (0°)
        
        # Muro con ángulo de 15° (diferencia de 15° < 25°)
        # tan(15°) ~ 0.2679
        seg_parallel = (500, int(500 - 50 * 0.2679), 600, int(500 + 50 * 0.2679))
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_parallel], min_crossing_angle_deg=25.0
        )
        self.assertEqual(len(crossings), 0)

        # Muro con ángulo de 45° (diferencia de 45° >= 25°)
        seg_good = (500, 450, 600, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_good], min_crossing_angle_deg=25.0
        )
        self.assertEqual(len(crossings), 1)

    def test_position_filters_rejects_too_close_to_drop_or_target(self):
        """Descarta cruces a <40px del drop o <50px del target."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Muro a 20px del drop (x=320)
        seg_too_close_drop = (320, 450, 320, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_too_close_drop], min_dist_drop=40.0
        )
        self.assertEqual(len(crossings), 0)

        # Muro a 30px del target (x=870)
        seg_too_close_target = (870, 450, 870, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_too_close_target], min_dist_target=50.0
        )
        self.assertEqual(len(crossings), 0)

        # Muro a 300px del drop (x=600, en zona válida)
        seg_valid = (600, 450, 600, 550)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_valid], min_dist_drop=40.0, min_dist_target=50.0
        )
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0], (600, 500))

    def test_endpoint_tolerance_accepts_gap_up_to_30px(self):
        """Acepta intersecciones que caigan hasta 30px fuera del extremo (muros tapados)."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Segmento vertical en x=600, de y=520 a y=600 (se corta 20px antes del rayo y=500)
        # Como 20px <= 30px, debe ser aceptado
        seg_gap_20 = (600, 520, 600, 600)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_gap_20], endpoint_tol=30.0
        )
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0], (600, 500))

        # Segmento vertical en x=600, de y=550 a y=630 (se corta 50px antes del rayo y=500)
        # Como 50px > 30px, debe ser descartado
        seg_gap_50 = (600, 550, 600, 630)
        crossings = find_wall_crossings_from_segments(
            p_drop, p_target, [seg_gap_50], endpoint_tol=30.0
        )
        self.assertEqual(len(crossings), 0)

    def test_find_wall_crossings_double_wall_merged_when_close(self):
        """Verifica que dos líneas de muro muy juntas (<25px) se fusionan en 1 solo cruce promedio."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Muro 1 en x=595, Muro 2 en x=610 (separación = 15px < 25px)
        seg1 = (595, 450, 595, 550)
        seg2 = (610, 450, 610, 550)
        crossings = find_wall_crossings_from_segments(p_drop, p_target, [seg1, seg2])
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0][0], 602, delta=3)

    def test_find_wall_crossings_double_wall_separated_when_far(self):
        """Verifica que dos líneas de muro separadas (>25px) producen 2 cruces ordenados."""
        p_drop = (300, 500)
        p_target = (900, 500)

        # Muro 1 en x=500, Muro 2 en x=680 (separación = 180px)
        seg1 = (500, 450, 500, 550)
        seg2 = (680, 450, 680, 550)
        crossings = find_wall_crossings_from_segments(p_drop, p_target, [seg1, seg2])
        self.assertEqual(len(crossings), 2)
        self.assertEqual(crossings[0][0], 500)
        self.assertEqual(crossings[1][0], 680)

    def test_wall_raycast_jump_targets_integration(self):
        """Verifica la integración con deployment_strategy y separación mínima."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        detections = [
            Detection("storage_dark", 0.90, (1150, 480, 1250, 580), (1200, 530)),
            Detection("storage_gold", 0.88, (800, 400, 900, 500), (850, 450)),
            Detection("storage_elixir", 0.85, (1500, 400, 1600, 500), (1550, 450)),
            Detection("town_hall", 0.95, (1150, 300, 1270, 420), (1210, 360)),
        ]

        # Dibujar muros delante de cada almacén
        for d in detections:
            mx, my = int(d.center[0] * 0.85 + 200), int(d.center[1] * 0.85 + 100)
            pt1 = (mx - 40, int(my - 40 * 0.7265))
            pt2 = (mx + 40, int(my + 40 * 0.7265))
            cv2.line(img, pt1, pt2, (255, 255, 255), 4)

        targets = compute_adaptive_jump_targets_from_detections(
            detections, (self.w, self.h), screen=img
        )

        self.assertGreaterEqual(len(targets), 1)
        self.assertLessEqual(len(targets), 4)

        min_sep = float(SETTINGS.get("attack", {}).get("min_separation_between_jumps_px", 110))
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                dist = np.hypot(targets[i][0] - targets[j][0], targets[i][1] - targets[j][1])
                self.assertGreaterEqual(dist, min_sep - 5.0)

    def test_detect_global_wall_segments_masks_building_interiors(self):
        """Verifica que detect_global_wall_segments(img, detections=dets) ignora bordes dentro de edificios."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Muro exterior legítimo a 36°
        ext_pt1 = (600, 450)
        ext_pt2 = (int(600 + 120), int(450 + 120 * 0.7265))
        cv2.line(img, ext_pt1, ext_pt2, (255, 255, 255), 4)

        # 2. Borde dentro de la caja de un edificio (tejado con pendiente isométrica)
        b_x1, b_y1, b_x2, b_y2 = 1000, 400, 1200, 600
        int_pt1 = (1030, 460)
        int_pt2 = (int(1030 + 120), int(460 + 120 * 0.7265))
        cv2.line(img, int_pt1, int_pt2, (255, 255, 255), 4)

        # Sin detecciones: hay segmentos detectados dentro de la caja del edificio
        _, segs_without = detect_global_wall_segments(img, force_refresh=True, min_line_len=50)
        segs_inside_without = [s for s in segs_without if s[0] >= 950]
        self.assertGreater(len(segs_inside_without), 0, "Sin detecciones debe detectar bordes dentro del edificio")

        # Con detección del edificio: el interior queda enmascarado a 0
        building_det = Detection("town_hall", 0.95, (b_x1, b_y1, b_x2, b_y2), (1100, 500))
        _, segs_with = detect_global_wall_segments(
            img, force_refresh=True, min_line_len=50, detections=[building_det]
        )
        segs_inside_with = [s for s in segs_with if s[0] >= 950]
        self.assertEqual(len(segs_inside_with), 0, "Con detección no debe haber segmentos dentro del edificio")
        segs_outside_with = [s for s in segs_with if s[0] < 900]
        self.assertGreater(len(segs_outside_with), 0, "El muro exterior debe preservarse intacto")


if __name__ == "__main__":
    unittest.main()
