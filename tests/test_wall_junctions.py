"""
tests/test_wall_junctions.py
Batería de pruebas unitarias automatizadas para:
1. Detección precisa de uniones de muros isométricos (cruces en T y +).
2. Asignación adaptativa de Hechizos de Salto y Súper Rompemuros por demanda real.
3. No-solapamiento y límites estrictos de pantalla.
"""
import unittest
import numpy as np

from vision.yolo_detector import Detection
from vision.wall_raycast import (
    find_wall_junctions,
    calculate_adaptive_jump_and_breach_placements,
    _angle_diff_180
)


class TestWallJunctions(unittest.TestCase):
    def test_orthogonal_cross_junction(self):
        """Verifica que dos segmentos perpendiculares isométricos generen una unión tipo 'cross' (+)."""
        # Segmento 1 a 36° (x1=100, y1=100 -> x2=300, y2=245)
        # Segmento 2 a 144° (x3=100, y3=245 -> x4=300, y4=100)
        # Intersección teórica exacta en (200, 172)
        s1 = (100, 100, 300, 245)
        s2 = (100, 245, 300, 100)
        junctions = find_wall_junctions([s1, s2], iso_angles=(36.0, 144.0))

        self.assertEqual(len(junctions), 1)
        junc = junctions[0]
        self.assertEqual(junc["type"], "cross")
        self.assertEqual(junc["degree"], 4)
        self.assertEqual(junc["score"], 4.0)
        self.assertAlmostEqual(junc["pt"][0], 200, delta=5)
        self.assertAlmostEqual(junc["pt"][1], 172, delta=5)

    def test_tee_junction(self):
        """Verifica que un segmento que termina sobre otro forme una unión tipo 'tee' (T)."""
        # Segmento 1 continuo (100, 100 -> 300, 245)
        # Segmento 2 que topa con el centro (200, 172 -> 300, 100)
        s1 = (100, 100, 300, 245)
        s2 = (200, 172, 300, 100)
        junctions = find_wall_junctions([s1, s2], iso_angles=(36.0, 144.0))

        self.assertEqual(len(junctions), 1)
        junc = junctions[0]
        self.assertEqual(junc["type"], "tee")
        self.assertEqual(junc["degree"], 3)
        self.assertEqual(junc["score"], 3.0)

    def test_parallel_segments_no_junction(self):
        """Verifica que segmentos paralelos en el mismo ángulo no generen uniones falsas."""
        s1 = (100, 100, 200, 172)
        s2 = (150, 100, 250, 172)
        junctions = find_wall_junctions([s1, s2], iso_angles=(36.0, 144.0))
        self.assertEqual(len(junctions), 0)


class TestAdaptiveJumpAndBreachPlacements(unittest.TestCase):
    def setUp(self):
        # Muro exterior (Capa 1): cruza en x=400
        # Muro interior (Capa 2): cruza en x=600
        # Muro ortogonal formando un cruce en (600, 500)
        self.wall_segments = [
            (400, 300, 400, 700),  # Muro 1 exterior
            (600, 300, 600, 700),  # Muro 2 interior
            (500, 500, 700, 500),  # Muro transversal (forma cruz en 600, 500)
        ]

    def test_open_exterior_target_no_spells_needed(self):
        """Objetivo sin muros bloqueando no requiere ni saltos ni SWB."""
        det = Detection(cls_name="collector_gold", confidence=0.9, bbox_xyxy=(100, 100, 150, 150), center=(125, 125))
        drop_pt = (50, 125)
        targets = [(det, drop_pt)]

        jumps, swb, _ = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets,
            wall_segments=self.wall_segments,
            max_jumps=4,
        )
        self.assertEqual(len(jumps), 0)
        self.assertEqual(len(swb), 0)

    def test_single_outer_wall_uses_swb_no_jump(self):
        """Objetivo con solo 1 muro exterior usa SWB para romperlo y no gasta Salto."""
        det = Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(480, 480, 520, 520), center=(500, 500))
        # Drop exterior en césped a 40px del muro exterior (x=360)
        drop_pt = (360, 500)
        targets = [(det, drop_pt)]

        jumps, swb, _ = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets,
            wall_segments=self.wall_segments,
            max_jumps=4,
        )
        # Capa 1: se asigna a SWB
        self.assertEqual(len(swb), 1)
        # No hay capas 2 o 3, 0 saltos gastados (ahorro de hechizos por demanda real)
        self.assertEqual(len(jumps), 0)

    def test_multi_wall_target_places_jump_on_junction(self):
        """Objetivo con 2 capas de muro asigna Salto en la unión geométrica (600, 500)."""
        det = Detection(cls_name="storage_dark", confidence=0.95, bbox_xyxy=(780, 480, 820, 520), center=(800, 500))
        drop_pt = (350, 500)
        targets = [(det, drop_pt)]

        jumps, swb, _ = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets,
            wall_segments=self.wall_segments,
            max_jumps=4,
        )
        self.assertEqual(len(swb), 1)
        self.assertEqual(len(jumps), 1)
        # El salto debe quedar ubicado en la unión de muros interior (600, 500)
        self.assertAlmostEqual(jumps[0][0], 600, delta=20)
        self.assertAlmostEqual(jumps[0][1], 500, delta=20)


if __name__ == "__main__":
    unittest.main()
