"""
tests/test_deep_corridor_and_swb.py
Pruebas para el corredor del núcleo profundo y despliegue de Súper Rompemuros.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np
from core.config import SETTINGS
from vision.yolo_detector import Detection
from attack.deployment_strategy import (
    _compute_cluster_intersection_jump_targets,
    compute_canonical_diamond_drop,
    CANONICAL_CENTER,
    DeploymentStrategy
)


class TestDeepCorridorAndSWB(unittest.TestCase):
    def test_deep_core_corridor_jump_generation(self):
        """Verifica que un almacén oscuro en el núcleo profundo genere saltos encadenados."""
        cx, cy = CANONICAL_CENTER
        # Almacén de elixir oscuro situado en el centro profundo de la base (núcleo)
        core_de = Detection(
            cls_name="storage_dark",
            confidence=0.92,
            bbox_xyxy=(int(cx - 25), int(cy - 25), int(cx + 25), int(cy + 25)),
            center=(int(cx), int(cy))
        )
        # Almacenes periféricos
        gold1 = Detection(
            cls_name="storage_gold",
            confidence=0.88,
            bbox_xyxy=(1500, 480, 1550, 530),
            center=(1525, 505)
        )
        elix1 = Detection(
            cls_name="storage_elixir",
            confidence=0.85,
            bbox_xyxy=(900, 480, 950, 530),
            center=(925, 505)
        )

        detections = [core_de, gold1, elix1]
        screen_size = (2400, 1080)

        targets = _compute_cluster_intersection_jump_targets(detections, screen_size)
        self.assertGreater(len(targets), 0, "Debe generar al menos 1 objetivo de salto")
        max_jumps = int(SETTINGS.get("attack", {}).get("max_jumps", 4))
        self.assertLessEqual(len(targets), max_jumps, f"No debe exceder max_jumps = {max_jumps}")

    def test_swb_graceful_fallback(self):
        """Verifica que si no hay SWB en la barra, _ensure_super_wall_breaker_selected retorne False."""
        mock_dev = MagicMock()
        mock_dev.width = 2400
        mock_dev.height = 1080
        mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        mock_vis = MagicMock()
        mock_army = MagicMock()
        mock_army.find_super_wall_breaker_in_battle_bar.return_value = None

        strat = DeploymentStrategy(mock_dev, mock_vis, mock_army)
        selected = strat._ensure_super_wall_breaker_selected(force_rescan=True)
        self.assertFalse(selected, "Debe retornar False cuando no hay SWB")

    def test_swb_selected_success(self):
        """Verifica que si hay SWB en la barra, lo seleccione y retorne True."""
        mock_dev = MagicMock()
        mock_dev.width = 2400
        mock_dev.height = 1080
        mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        mock_vis = MagicMock()
        mock_army = MagicMock()
        mock_army.find_super_wall_breaker_in_battle_bar.return_value = (520, 930)

        strat = DeploymentStrategy(mock_dev, mock_vis, mock_army)
        selected = strat._ensure_super_wall_breaker_selected(force_rescan=True)
        self.assertTrue(selected, "Debe retornar True cuando detecta SWB")
        mock_dev.tap.assert_called_with(520, 930)


if __name__ == "__main__":
    unittest.main()
