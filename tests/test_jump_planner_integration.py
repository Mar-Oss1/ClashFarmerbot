"""
Tests unitarios y de integración para attack/jump_planner.py,
vision/village_analyzer.py y la modularización de attack/deployment_strategy.py
(Fase 4 - Mejora 05).
"""
import glob
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from vision.yolo_detector import Detection
from vision.wall_compartments import CompartmentAnalysisResult
from attack.jump_planner import (
    compute_adaptive_jump_targets_from_detections,
    _compute_compartment_jump_targets,
    _compute_wall_raycast_jump_targets,
    _compute_cluster_intersection_jump_targets,
    _compute_legacy_quadrant_jump_targets,
    compute_canonical_diamond_drop,
)
from vision.village_analyzer import VillageScene, VillageAnalyzer
from attack.deployment_strategy import DeploymentStrategy
import attack.deployment_strategy as ds_mod
from core.config import SETTINGS


class TestJumpPlannerIntegration(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 2400, 1080
        self.screen = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.detections = [
            Detection(cls_name="storage_gold", confidence=0.95, bbox_xyxy=(1000, 450, 1080, 530), center=(1040, 490)),
            Detection(cls_name="storage_elixir", confidence=0.92, bbox_xyxy=(1300, 450, 1380, 530), center=(1340, 490)),
            Detection(cls_name="storage_dark", confidence=0.98, bbox_xyxy=(1160, 460, 1240, 540), center=(1200, 500)),
            Detection(cls_name="town_hall", confidence=0.99, bbox_xyxy=(1150, 350, 1250, 450), center=(1200, 400)),
        ]

    @patch("vision.wall_compartments.optimize_jumps")
    def test_coverage_optimizer_invokes_optimize_jumps(self, mock_opt):
        """Verifica que jump_placement='coverage_optimizer' invoca optimize_jumps()."""
        mock_opt.return_value = MagicMock(
            jumps=[(1100, 480, 4.0), (1280, 480, 3.5)],
            debug_overlay=None
        )

        test_settings = dict(SETTINGS)
        test_settings["attack"] = dict(SETTINGS.get("attack", {}))
        test_settings["attack"]["jump_placement"] = "coverage_optimizer"

        with patch.dict(ds_mod.SETTINGS, test_settings):
            targets = compute_adaptive_jump_targets_from_detections(
                self.detections, (self.w, self.h), screen=self.screen
            )

        mock_opt.assert_called_once()
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0], (1100, 480))
        self.assertEqual(targets[1], (1280, 480))

    @patch("vision.wall_compartments.optimize_jumps")
    def test_compartments_mode_invokes_optimize_jumps(self, mock_opt):
        """Verifica que jump_placement='compartments' es un alias válido que invoca optimize_jumps()."""
        mock_opt.return_value = MagicMock(
            jumps=[(1150, 470, 5.0)],
            debug_overlay=None
        )

        test_settings = dict(SETTINGS)
        test_settings["attack"] = dict(SETTINGS.get("attack", {}))
        test_settings["attack"]["jump_placement"] = "compartments"

        with patch.dict(ds_mod.SETTINGS, test_settings):
            targets = compute_adaptive_jump_targets_from_detections(
                self.detections, (self.w, self.h), screen=self.screen
            )

        mock_opt.assert_called_once()
        self.assertEqual(targets, [(1150, 470)])

    def test_coverage_optimizer_fallback_when_screen_is_none(self):
        """Si screen es None bajo coverage_optimizer, debe recurrir con gracia a wall_raycast/cluster."""
        test_settings = dict(SETTINGS)
        test_settings["attack"] = dict(SETTINGS.get("attack", {}))
        test_settings["attack"]["jump_placement"] = "coverage_optimizer"

        with patch.dict(ds_mod.SETTINGS, test_settings):
            targets = compute_adaptive_jump_targets_from_detections(
                self.detections, (self.w, self.h), screen=None
            )

        # Debe devolver saltos calculados por fallback sin lanzar excepción
        self.assertIsInstance(targets, list)
        self.assertGreaterEqual(len(targets), 1)

    def test_reset_battle_inventory_loads_from_settings(self):
        """Verifica que _reset_battle_inventory() toma valores de SETTINGS['attack']['army_inventory']."""
        mock_dev = MagicMock()
        mock_dev.width = 2400
        mock_dev.height = 1080
        mock_vis = MagicMock()
        mock_army = MagicMock()

        strat = DeploymentStrategy(mock_dev, mock_vis, mock_army)

        # 1. Valores por defecto
        inv = strat._reset_battle_inventory()
        self.assertEqual(inv["goblins"], 96)
        self.assertEqual(inv["swb"], 8)
        self.assertEqual(inv["jump"], 4)
        self.assertEqual(inv["haste"], 3)
        self.assertEqual(strat._troop_inventory, inv)

        # 2. Valores personalizados configurados
        custom_settings = dict(SETTINGS)
        custom_settings["attack"] = dict(SETTINGS.get("attack", {}))
        custom_settings["attack"]["army_inventory"] = {
            "goblins": 120,
            "swb": 12,
            "jump": 5,
            "haste": 2,
        }

        with patch.dict(ds_mod.SETTINGS, custom_settings):
            custom_inv = strat._reset_battle_inventory()
            self.assertEqual(custom_inv["goblins"], 120)
            self.assertEqual(custom_inv["swb"], 12)
            self.assertEqual(custom_inv["jump"], 5)
            self.assertEqual(custom_inv["haste"], 2)
            self.assertEqual(strat._troop_inventory, custom_inv)

    @patch("cv2.imwrite")
    @patch("attack.deployment_strategy.save_debug_image_async")
    def test_combat_paths_do_not_call_sync_cv2_imwrite(self, mock_async_save, mock_cv2_imwrite):
        """Verifica que las rutas de combate no hacen llamadas síncronas a cv2.imwrite."""
        mock_dev = MagicMock()
        mock_dev.width = 2400
        mock_dev.height = 1080
        mock_vis = MagicMock()
        mock_army = MagicMock()

        strat = DeploymentStrategy(mock_dev, mock_vis, mock_army)

        # Probar _save_yolo_attack_debug
        strat._save_yolo_attack_debug(self.screen, self.detections, [(1100, 480)])
        mock_cv2_imwrite.assert_not_called()
        mock_async_save.assert_called()

        mock_async_save.reset_mock()

        # Probar _detect_red_exclusion_boundary
        strat._detect_red_exclusion_boundary(self.screen)
        mock_cv2_imwrite.assert_not_called()

    def test_no_device_class_name_hacks_in_attack_directory(self):
        """Verifica explícitamente que no queda ningún hack de .__class__.__name__ == 'Device' en attack/."""
        attack_py_files = glob.glob(os.path.join("attack", "**", "*.py"), recursive=True)
        self.assertGreater(len(attack_py_files), 0, "No se encontraron archivos en attack/")

        violations = []
        for file_path in attack_py_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if '__class__.__name__' in content:
                    violations.append(file_path)

        self.assertEqual(
            violations, [],
            f"Se encontraron hacks de __class__.__name__ en: {violations}"
        )

    def test_no_sync_cv2_imwrite_in_attack_directory(self):
        """Verifica explícitamente que no queda ninguna llamada síncrona a cv2.imwrite en attack/."""
        attack_py_files = glob.glob(os.path.join("attack", "**", "*.py"), recursive=True)
        self.assertGreater(len(attack_py_files), 0)

        violations = []
        for file_path in attack_py_files:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line_no, line in enumerate(lines, start=1):
                    # Ignorar comentarios
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if "cv2.imwrite" in line:
                        violations.append(f"{file_path}:{line_no} -> {stripped}")

        self.assertEqual(
            violations, [],
            f"Se encontraron llamadas directas a cv2.imwrite en: {violations}"
        )

    @patch("attack.deployment_strategy.detect_village_boundary")
    @patch("vision.village_analyzer.get_detector")
    def test_village_analyzer_analyze_e2e(self, mock_get_detector, mock_detect_boundary):
        """Verifica que VillageAnalyzer.analyze devuelve un VillageScene coherente."""
        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = self.detections
        mock_get_detector.return_value = mock_yolo

        fake_red = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)
        fake_green = np.array([[[120, 120]], [[180, 120]], [[180, 180]], [[120, 180]]], dtype=np.int32)
        mock_detect_boundary.return_value = (fake_red, fake_green)

        scene = VillageAnalyzer.analyze(self.screen, zone_name="CENTRO")

        self.assertIsInstance(scene, VillageScene)
        self.assertEqual(scene.zone_name, "CENTRO")
        self.assertEqual(scene.detections, self.detections)
        self.assertIs(scene.screen, self.screen)
        np.testing.assert_array_equal(scene.red_contour, fake_red)
        np.testing.assert_array_equal(scene.green_contour, fake_green)


if __name__ == "__main__":
    unittest.main()
