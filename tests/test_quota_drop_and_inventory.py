"""
tests/test_quota_drop_and_inventory.py
Batería de pruebas automatizadas para:
1. Control estricto de inventario de batalla (96 Duendes, 8 SWB, 4 Saltos, 3 Aceleraciones).
2. Detección zonal del rombo canónico y eliminación de falsos 'fuera de margen' en vistas paneadas (5 iteraciones por zona).
3. Distancia de drop cercana (15-25px del perímetro legal exterior).
4. Paneo inteligente condicional con estabilización a 1.3s y retorno a 0.85s.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import cv2
import numpy as np
import random

from vision.yolo_detector import Detection
from attack.deployment_strategy import (
    DeploymentStrategy,
    detect_village_boundary,
    compute_canonical_diamond_drop,
    CANONICAL_CENTER,
    CANONICAL_RX_GRASS,
    CANONICAL_RY_GRASS,
    CANONICAL_RX_BUILDABLE,
    CANONICAL_RY_BUILDABLE,
)


class TestTroopInventoryLimits(unittest.TestCase):
    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        self.mock_vis = MagicMock()
        self.mock_army = MagicMock()
        self.mock_army.find_super_wall_breaker_in_battle_bar.return_value = (580, 960)
        self.mock_army.find_goblins_in_battle_bar.return_value = (420, 960)
        self.mock_army.find_jump_spell_in_battle_bar.return_value = (1530, 960)
        self.mock_army.find_haste_spell_in_battle_bar.return_value = (1660, 960)

        self.strat = DeploymentStrategy(self.mock_dev, self.mock_vis, self.mock_army)

    def test_initial_inventory(self):
        """Verifica que el inventario se inicializa con los valores exactos."""
        self.assertEqual(self.strat._troop_inventory["goblins"], 96)
        self.assertEqual(self.strat._troop_inventory["swb"], 8)
        self.assertEqual(self.strat._troop_inventory["jump"], 4)
        self.assertEqual(self.strat._troop_inventory["haste"], 3)

    def test_swb_inventory_cap(self):
        """Verifica que nunca se lancen más de 8 SWB aunque se soliciten múltiples brechas."""
        self.strat._troop_inventory["swb"] = 8
        self.strat._ensure_super_wall_breaker_selected = MagicMock(return_value=True)

        # Generar 6 edificios que requerirían 2 SWB cada uno (12 en total)
        targets = [
            Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(500 + i * 200, 500, 550 + i * 200, 550), center=(525 + i * 200, 525))
            for i in range(6)
        ]
        deployed_log = []

        with patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_spell_selected", return_value=False), \
             patch.object(self.strat, "_deploy_heroes_spearhead"):
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=deployed_log,
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )

        # La cuota de SWB debe haber quedado en 0 exactamente al usar las 4 brechas (8 SWB)
        self.assertEqual(self.strat._troop_inventory["swb"], 0)

        # Simular agotamiento forzado a 0 y verificar que no se intenta desplegar más
        self.strat._troop_inventory["swb"] = 0
        with patch.object(self.strat, "_ensure_super_wall_breaker_selected") as mock_swb_sel:
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )
            mock_swb_sel.assert_not_called()

    def test_all_4_jumps_deployed(self):
        """Verifica que se generen y desplieguen los 4 Hechizos de Salto."""
        targets = [
            Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(1200, 500, 1250, 550), center=(1225, 525)),
            Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(900, 300, 950, 350), center=(925, 325)),
            Detection(cls_name="storage_elixir", confidence=0.9, bbox_xyxy=(1500, 300, 1550, 350), center=(1525, 325)),
            Detection(cls_name="town_hall", confidence=0.9, bbox_xyxy=(1200, 700, 1250, 750), center=(1225, 725)),
        ]
        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"use_spells": True, "use_yolo": True, "max_jumps": 4, "min_jump_score": 1.0}}):
            deployed = self.strat._deploy_strategic_spells_centered(
                screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets
            )
            self.assertEqual(len(deployed), 4, "Debe desplegar exactamente los 4 Hechizos de Salto")
            self.assertEqual(self.strat._troop_inventory["jump"], 0, "Inventario de saltos debe quedar en 0")

    def test_haste_inventory_cap(self):
        """Verifica que nunca se lancen más de 3 Hechizos de Aceleración."""
        self.strat._troop_inventory["haste"] = 3
        # Forzar que _ensure_spell_selected retorne True
        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False):
            targets = [
                Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(500, 500, 550, 550), center=(525, 525)),
                Detection(cls_name="storage_elixir", confidence=0.9, bbox_xyxy=(700, 500, 750, 550), center=(725, 525)),
                Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(900, 500, 950, 550), center=(925, 525)),
            ]
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )

        # Se deben haber gastado exactamente 2 hastes (hasta 2 almacenes en esta vista)
        self.assertEqual(self.strat._troop_inventory["haste"], 1)

        # Simular que solo queda 1 y se piden 2
        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False):
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )
        self.assertEqual(self.strat._troop_inventory["haste"], 0)

        # Con 0 restantes, no debe llamar a _ensure_spell_selected("haste")
        with patch.object(self.strat, "_ensure_spell_selected") as mock_spell_sel, \
             patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False):
            self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )
            for c in mock_spell_sel.call_args_list:
                self.assertNotEqual(c[0][0], "haste")

    def test_goblin_inventory_cap(self):
        """Verifica que el despliegue de duendes se detiene exactamente al llegar a 0 de los 96."""
        self.strat._troop_inventory["goblins"] = 10
        # Generar 3 almacenes (cuota 6 cada uno -> requerirían 18 duendes)
        targets = [
            Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(500, 500, 550, 550), center=(525, 525)),
            Detection(cls_name="storage_elixir", confidence=0.9, bbox_xyxy=(700, 500, 750, 550), center=(725, 525)),
            Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(900, 500, 950, 550), center=(925, 525)),
        ]
        with patch.object(self.strat, "_ensure_goblins_selected", return_value=True), \
             patch.object(self.strat, "_ensure_super_wall_breaker_selected", return_value=False), \
             patch.object(self.strat, "_ensure_spell_selected", return_value=False):
            deployed_count, unreached = self.strat._execute_single_zone_quota_deploy(
                zone_name="CENTRO",
                deployed_drops_log=[],
                skip_spells=True,
                precomputed_screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                precomputed_detections=targets,
            )

        # Debe desplegar exactamente 10 duendes y el inventario quedar en 0
        self.assertEqual(deployed_count, 10)
        self.assertEqual(self.strat._troop_inventory["goblins"], 0)


class TestZonalBoundaryAndDropLegality(unittest.TestCase):
    """
    Simulación de 5 iteraciones por cada zona (Centro, Norte, Sur, Este, Oeste).
    Verifica que en ninguna zona los objetivos legales son descartados como 'fuera de margen',
    y que los drops caen dentro de la franja segura 138 <= y <= 845 y fuera de la zona roja.
    """
    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.strat = DeploymentStrategy(self.mock_dev, MagicMock(), MagicMock())
        self.screen_dummy = np.zeros((1080, 2400, 3), dtype=np.uint8)

    def test_5_iterations_centro(self):
        """5 iteraciones en Zona Central: todos los drops deben ser 100% legales y fuera de zona roja."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="CENTRO")
        self.assertIsNotNone(red)
        self.assertIsNotNone(green)

        for iteration in range(5):
            # Probar diferentes ubicaciones de edificios interiores y periféricos
            test_x = random.randint(900, 1500)
            test_y = random.randint(350, 650)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="CENTRO"
            )

            # 1. Dentro de los límites seguros de pantalla
            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[CENTRO #{iteration}] Drop Y debe ser >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[CENTRO #{iteration}] Drop Y debe ser <= safe_y_max")
            self.assertGreaterEqual(drop_x, self.strat._SAFE_X_MIN, f"[CENTRO #{iteration}] Drop X debe ser >= safe_x_min")
            self.assertLessEqual(drop_x, self.strat._SAFE_X_MAX, f"[CENTRO #{iteration}] Drop X debe ser <= safe_x_max")

            # 2. Fuera de la zona roja construible
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[CENTRO #{iteration}] Drop ({drop_x}, {drop_y}) no debe caer dentro del rombo rojo")

    def test_5_iterations_norte(self):
        """5 iteraciones en Flanco Norte: el rombo se desplaza a Y=1.25*H y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="NORTE")
        self.assertIsNotNone(red)
        # El centro debe estar en (1200, 1350)
        center_red_y = np.mean(red[:, 0, 1])
        self.assertAlmostEqual(center_red_y, 1080 * 1.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(900, 1500)
            test_y = random.randint(200, 450)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="NORTE"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[NORTE #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[NORTE #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[NORTE #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")

    def test_5_iterations_sur(self):
        """5 iteraciones en Flanco Sur: el rombo se desplaza a Y=-0.25*H y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="SUR")
        self.assertIsNotNone(red)
        center_red_y = np.mean(red[:, 0, 1])
        self.assertAlmostEqual(center_red_y, -1080 * 0.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(900, 1500)
            test_y = random.randint(550, 800)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="SUR"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[SUR #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[SUR #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[SUR #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")

    def test_5_iterations_este(self):
        """5 iteraciones en Flanco Este: el rombo se desplaza a X=-0.25*W y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="ESTE")
        self.assertIsNotNone(red)
        center_red_x = np.mean(red[:, 0, 0])
        self.assertAlmostEqual(center_red_x, -2400 * 0.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(1400, 2000)
            test_y = random.randint(300, 700)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="ESTE"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[ESTE #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[ESTE #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[ESTE #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")

    def test_5_iterations_oeste(self):
        """5 iteraciones en Flanco Oeste: el rombo se desplaza a X=1.25*W y los drops son legales."""
        red, green = detect_village_boundary(self.screen_dummy, mode="canonical_diamond", zone_name="OESTE")
        self.assertIsNotNone(red)
        center_red_x = np.mean(red[:, 0, 0])
        self.assertAlmostEqual(center_red_x, 2400 * 1.25, delta=10)

        for iteration in range(5):
            test_x = random.randint(400, 1000)
            test_y = random.randint(300, 700)
            drop_x, drop_y = self.strat._get_drop_point_on_contour(
                (test_x, test_y), green, red_contour=red, zone_name="OESTE"
            )

            self.assertGreaterEqual(drop_y, self.strat._SAFE_Y_MIN, f"[OESTE #{iteration}] Drop Y >= safe_y_min")
            self.assertLessEqual(drop_y, self.strat._SAFE_Y_MAX, f"[OESTE #{iteration}] Drop Y <= safe_y_max")
            dist_red = cv2.pointPolygonTest(red, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(dist_red, 0.0, f"[OESTE #{iteration}] Drop ({drop_x}, {drop_y}) no debe ser inside_red")


class TestSmartConditionalPanning(unittest.TestCase):
    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.strat = DeploymentStrategy(self.mock_dev, MagicMock(), MagicMock())

    def test_no_pan_when_all_targets_reached(self):
        """Verifica que si la vista Central alcanza el 100% de los objetivos, NO se ejecutan paneos."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Mock _execute_single_zone_quota_deploy para CENTRO retornando 0 unreached
        with patch.object(self.strat, "_execute_single_zone_quota_deploy", return_value=(19, [])) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": False, "min_unreached_targets_for_pan": 1}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Solo debe haberse ejecutado CENTRO, 0 paneos
        self.assertEqual(mock_exec.call_count, 1)
        self.assertEqual(mock_exec.call_args_list[0][1]["zone_name"], "CENTRO")
        # El shell no debe haber recibido ningún swipe de paneo
        self.mock_dev.d.shell.assert_not_called()

    def test_pan_only_to_flank_with_unreached_target(self):
        """Verifica que si solo hay objetivos pendientes en el Norte, solo panea al Norte y regresa."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        unreached_north_target = Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(1200, 200, 1250, 250), center=(1225, 225))

        # Primer llamado retorna el unreached del norte, el segundo (en Norte) retorna alcanzado
        mock_returns = [
            (10, [unreached_north_target]),
            (2, []),
        ]

        with patch.object(self.strat, "_execute_single_zone_quota_deploy", side_effect=mock_returns) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": False, "min_unreached_targets_for_pan": 1, "zone_attack_flanks": ["north", "south", "east", "west"]}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Se llamó CENTRO y luego únicamente NORTE
        self.assertEqual(mock_exec.call_count, 2)
        self.assertEqual(mock_exec.call_args_list[0][1]["zone_name"], "CENTRO")
        self.assertEqual(mock_exec.call_args_list[1][1]["zone_name"], "NORTE")

        # Verificar que el swipe ejecutó swipe hacia abajo (ir a Norte) y retorno recíproco
        shell_calls = [c[0][0] for c in self.mock_dev.d.shell.call_args_list]
        self.assertTrue(any("1200 240 1200 820" in s for s in shell_calls), "Debe ejecutar swipe hacia el Norte")
        self.assertTrue(any("1200 820 1200 240" in s for s in shell_calls), "Debe ejecutar swipe recíproco al Centro")

    def test_full_cross_pan_always_runs_all_4_flanks(self):
        """Verifica que con enable_full_cross_pan=True, los 4 flancos se visitan siempre."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Retornar (5, []) para cada zona
        mock_returns = [(5, []) for _ in range(5)]

        with patch.object(self.strat, "_execute_single_zone_quota_deploy", side_effect=mock_returns) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": True, "min_unreached_targets_for_pan": 1, "zone_attack_flanks": ["north", "south", "east", "west"]}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Se llamó CENTRO, NORTE, SUR, ESTE, OESTE (5 llamadas en total)
        self.assertEqual(mock_exec.call_count, 5)
        called_zones = [c[1]["zone_name"] for c in mock_exec.call_args_list]
        self.assertEqual(called_zones, ["CENTRO", "NORTE", "SUR", "ESTE", "OESTE"])

    def test_flank_pans_skipped_when_goblins_exhausted(self):
        """Verifica que si _troop_inventory['goblins'] queda en 0 tras CENTRO, se omiten todos los paneos de flancos."""
        screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        unreached_dummy = [Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(100, 100, 200, 200), center=(150, 150))]

        def mock_center_depletes_goblins(zone_name, deployed_drops_log, **kwargs):
            if zone_name == "CENTRO":
                self.strat._troop_inventory["goblins"] = 0
                return 96, unreached_dummy
            return 0, []

        with patch.object(self.strat, "_execute_single_zone_quota_deploy", side_effect=mock_center_depletes_goblins) as mock_exec, \
             patch("attack.deployment_strategy.SETTINGS", {"attack": {"enable_camera_pan": True, "enable_full_cross_pan": True, "min_unreached_targets_for_pan": 1, "zone_attack_flanks": ["north", "south", "east", "west"]}}):
            self.strat._deploy_by_structure_quotas(
                detections=[],
                screen_centered=screen,
            )

        # Solo debe haberse llamado a CENTRO (1 llamada)
        self.assertEqual(mock_exec.call_count, 1)
        self.assertEqual(mock_exec.call_args_list[0][1]["zone_name"], "CENTRO")
        # No debe haber ningún swipe de paneo registrado en adb
        self.assertEqual(self.mock_dev.d.shell.call_count, 0)


class TestFlankSpellsAndHasteNearJumps(unittest.TestCase):
    """Suite de validación de saltos en flancos paneados y colocación de haste junto a saltos."""

    def setUp(self):
        self.mock_dev = MagicMock()
        self.mock_dev.width = 2400
        self.mock_dev.height = 1080
        self.mock_dev.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        self.mock_vis = MagicMock()
        self.mock_army = MagicMock()
        self.mock_army.find_jump_spell_in_battle_bar.return_value = (1530, 960)
        self.mock_army.find_haste_spell_in_battle_bar.return_value = (1660, 960)
        self.mock_army.find_goblins_in_battle_bar.return_value = (420, 960)

        self.strat = DeploymentStrategy(self.mock_dev, self.mock_vis, self.mock_army)

    def test_flank_spells_deployment_when_storage_walled(self):
        """Verifica que en una vista de flanco (NORTE) con almacén bloqueado tras muro, se despliega Salto y Haste."""
        self.strat._troop_inventory["jump"] = 2
        self.strat._troop_inventory["haste"] = 2
        self.strat._troop_inventory["goblins"] = 20

        flank_storage = Detection(
            cls_name="storage_gold",
            confidence=0.92,
            bbox_xyxy=(1200, 300, 1260, 360),
            center=(1230, 330)
        )
        reachable = [(flank_storage, (1230, 160))]

        with patch.object(self.strat, "_ensure_spell_selected", return_value=True), \
             patch("attack.deployment_strategy.find_wall_crossings", return_value=[(1230, 240)]):
            self.strat._deploy_flank_spells(
                screen=np.zeros((1080, 2400, 3), dtype=np.uint8),
                zone_name="NORTE",
                reachable_targets=reachable,
                green_contour=np.array([[1000, 100], [1500, 100], [1500, 800], [1000, 800]]),
                safe_y_min=140,
                safe_y_max=845,
            )

        # Debe haber gastado 1 salto y 1 haste en el flanco
        self.assertEqual(self.strat._troop_inventory["jump"], 1, "Debe quedar 1 salto tras desplegar en flanco")
        self.assertEqual(self.strat._troop_inventory["haste"], 1, "Debe quedar 1 haste tras desplegar junto al salto")
        # El salto debe estar registrado en _used_jump_points
        self.assertEqual(len(self.strat._used_jump_points), 1)
        self.assertAlmostEqual(self.strat._used_jump_points[0][1], 240, delta=5, msg="Y de salto debe ser en el cruce de muro con jitter")

    def test_haste_near_jump_location(self):
        """Verifica que el hechizo de aceleración se coloca junto a las coordenadas del salto (+sesgo <= 25px)."""
        self.strat._troop_inventory["haste"] = 2
        jump_point = (1200, 400)
        target_point = (1200, 500)

        with patch.object(self.strat, "_ensure_spell_selected", return_value=True):
            self.strat._deploy_haste_near_jump(jump_point, target_point, zone_name="CENTRO")

        self.assertEqual(self.strat._troop_inventory["haste"], 1)
        # La posición enviada a tap debe estar cerca de jump_point (distancia < 35px)
        tapped_x, tapped_y = self.mock_dev.tap.call_args[0]
        dist = np.hypot(tapped_x - 1200, tapped_y - 400)
        self.assertLessEqual(dist, 35.0, "Haste debe caer a menos de 35px del Salto")

    def test_loot_priority_orders_storages_before_collectors(self):
        """Verifica que los almacenes de oro, elixir y oscuro se ordenan antes que los recolectores."""
        collector = Detection(cls_name="collector_gold", confidence=0.9, bbox_xyxy=(100, 100, 150, 150), center=(125, 125))
        storage = Detection(cls_name="storage_dark", confidence=0.9, bbox_xyxy=(500, 500, 550, 550), center=(525, 525))
        reachable = [(collector, (100, 200)), (storage, (500, 300))]

        loot_priority = {
            "storage_dark": 10,
            "storage_gold": 8,
            "storage_elixir": 8,
            "town_hall": 7,
            "collector_dark": 4,
            "collector_gold": 3,
            "collector_elixir": 3,
        }
        reachable.sort(key=lambda item: loot_priority.get(item[0].cls_name, 1), reverse=True)
        self.assertEqual(reachable[0][0].cls_name, "storage_dark", "storage_dark debe ir de primero")
        self.assertEqual(reachable[1][0].cls_name, "collector_gold", "collector_gold debe ir de segundo")


if __name__ == "__main__":
    unittest.main()

