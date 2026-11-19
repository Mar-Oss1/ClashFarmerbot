"""
tests/test_heroes_haste_and_cross_pan.py
Pruebas unitarias para:
1. Detección de los 4 héroes en la barra de batalla.
2. Detección del Hechizo de Aceleración (Haste) en la barra de batalla.
3. Despliegue de héroes de vanguardia y activación instantánea de habilidades.
4. Despliegue táctico de Hechizos de Aceleración en brechas/corredores.
5. Paneo en cruz de 4 direcciones (Norte, Sur, Este, Oeste) y swipes recíprocos.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from core.vision import Vision
from vision.army_checker import ArmyChecker
from attack.deployment_strategy import DeploymentStrategy
from vision.yolo_detector import Detection


class TestHeroesAndHasteDetection(unittest.TestCase):
    def setUp(self):
        self.vision = Vision()
        self.checker = ArmyChecker(self.vision)
        self.frame_path = os.path.join("logs", "frame_25s.png")

    def test_heroes_templates_exist(self):
        """Verifica que existen los archivos de plantilla de todos los héroes."""
        self.assertTrue(os.path.exists("assets/buttons/hero_king_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_queen_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_warden_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_champion_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_minion_prince_slot.png"))
        self.assertTrue(os.path.exists("assets/buttons/hero_duke_dragon_slot.png"))

    def test_haste_template_exists(self):
        """Verifica que existe la plantilla del hechizo de aceleración."""
        self.assertTrue(os.path.exists("assets/buttons/spell_haste.png"))

    def test_detect_all_heroes_on_battle_frame(self):
        """Verifica que los héroes se detectan con precisión en los frames de batalla."""
        if not os.path.exists(self.frame_path):
            self.skipTest(f"Frame no encontrado: {self.frame_path}")

        img = cv2.imread(self.frame_path)
        heroes = self.checker.find_all_heroes_in_battle_bar(img)
        self.assertIn("king", heroes)
        self.assertIn("queen", heroes)
        self.assertIn("warden", heroes)

        # Si existe el frame real de EDrag con los 4 héroes activos (RC, Prince, Warden, Duke Dragon)
        edrag_frame = os.path.join("logs", "yolo", "edrag_attack_start_20261111_184514.png")
        if os.path.exists(edrag_frame):
            e_img = cv2.imread(edrag_frame)
            e_heroes = self.checker.find_all_heroes_in_battle_bar(e_img)
            self.assertIn("champion", e_heroes)
            self.assertIn("minion_prince", e_heroes)
            self.assertIn("warden", e_heroes)
            self.assertTrue("duke_dragon" in e_heroes or "king" in e_heroes)

        # Verificar coordenadas dentro de la barra de cartas (y > 800)
        for h_name, pos in heroes.items():
            self.assertGreater(pos[1], 800, f"Posición Y de {h_name} debe estar en la barra")
            self.assertGreater(pos[0], 500, f"Posición X de {h_name} debe estar en la barra")

    def test_detect_haste_spell_on_battle_frame(self):
        """Verifica que el hechizo de aceleración se detecta en el frame de batalla."""
        if not os.path.exists(self.frame_path):
            self.skipTest(f"Frame no encontrado: {self.frame_path}")

        img = cv2.imread(self.frame_path)
        haste_pos = self.checker.find_haste_spell_in_battle_bar(img)
        self.assertIsNotNone(haste_pos)
        self.assertGreater(haste_pos[1], 800)
        self.assertGreater(haste_pos[0], 1400)


class TestHeroSpearheadDeployment(unittest.TestCase):
    def setUp(self):
        self.mock_device = MagicMock()
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        self.mock_device.d = MagicMock()
        self.mock_vision = MagicMock()
        self.mock_army = MagicMock()
        self.strategy = DeploymentStrategy(self.mock_device, self.mock_vision, self.mock_army)

    @patch("attack.deployment_strategy.SETTINGS")
    def test_deploy_heroes_and_trigger_abilities(self, mock_settings):
        """Verifica que despliega los 4 héroes y toca inmediatamente sus slots para activar habilidades."""
        mock_settings.get.side_effect = lambda k, d=None: {
            "attack": {
                "use_heroes": True,
                "instant_hero_abilities": True,
            }
        }.get(k, d if d is not None else {})

        # Simular que el screenshot detecta los 4 héroes
        dummy_screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.mock_device.screenshot.return_value = dummy_screen
        self.strategy.army.find_all_heroes_in_battle_bar = MagicMock(return_value={
            "king": (920, 960),
            "queen": (1070, 970),
            "warden": (1230, 950),
            "champion": (1370, 950),
        })

        # Desplegar héroes en punto de penetración (1400, 500)
        deployed_count = self.strategy._deploy_heroes_spearhead(1400, 500, 138, 845)

        self.assertEqual(deployed_count, 4)
        self.assertTrue(self.strategy._heroes_deployed_in_battle)

        # Verificar que se seleccionaron los 4 slots y se tocaron sus habilidades
        # Total de taps en slots: 4 despliegues + 4 activaciones = 8 taps en slots
        slot_taps = [call for call in self.mock_device.tap.call_args_list]
        self.assertGreaterEqual(len(slot_taps), 8)

        # Verificar que se enviaron 4 clicks de despliegue al área de juego
        drop_clicks = [call for call in self.mock_device.d.click.call_args_list]
        self.assertEqual(len(drop_clicks), 4)

    @patch("attack.deployment_strategy.SETTINGS")
    def test_heroes_deploy_only_once_per_battle(self, mock_settings):
        """Verifica que los héroes no se vuelven a desplegar en fases subsecuentes."""
        mock_settings.get.side_effect = lambda k, d=None: {
            "attack": {"use_heroes": True}
        }.get(k, d if d is not None else {})

        self.strategy._heroes_deployed_in_battle = True
        count = self.strategy._deploy_heroes_spearhead(1200, 500, 138, 845)
        self.assertEqual(count, 0)


class TestCrossPan4Directions(unittest.TestCase):
    def setUp(self):
        self.mock_device = MagicMock()
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        self.mock_device.d = MagicMock()
        self.mock_vision = MagicMock()
        self.mock_army = MagicMock()
        self.strategy = DeploymentStrategy(self.mock_device, self.mock_vision, self.mock_army)

    @patch("attack.deployment_strategy.SETTINGS")
    def test_full_cross_pan_executes_all_4_flanks_with_reciprocals(self, mock_settings):
        """Verifica que con enable_full_cross_pan=True se recorren Norte, Sur, Este, Oeste con retornos recíprocos."""
        mock_settings.get.side_effect = lambda k, d=None: {
            "attack": {
                "enable_camera_pan": True,
                "enable_full_cross_pan": True,
                "min_unreached_targets_for_pan": 1,
                "zone_attack_flanks": ["north", "south", "east", "west"],
                "safe_drop_y_min": 138,
                "safe_drop_y_max": 845,
            }
        }.get(k, d if d is not None else {})

        # Mockear _execute_single_zone_quota_deploy para registrar las zonas visitadas
        visited_zones = []

        def fake_zone_deploy(zone_name, deployed_drops_log, skip_spells=False, **kwargs):
            visited_zones.append(zone_name)
            # Retorna 5 tropas y 1 objetivo inalcanzable para activar paneo
            unreached_dummy = [Detection(cls_name="storage_gold", confidence=0.9, bbox_xyxy=(100, 100, 200, 200), center=(150, 150))]
            return 5, unreached_dummy

        self.strategy._execute_single_zone_quota_deploy = fake_zone_deploy
        self.strategy._save_quota_attack_debug_image = MagicMock()

        self.strategy._deploy_by_structure_quotas(screen_centered=np.zeros((1080, 2400, 3), dtype=np.uint8))

        # Verificar que se visitaron las 5 zonas en orden
        self.assertEqual(visited_zones, ["CENTRO", "NORTE", "SUR", "ESTE", "OESTE"])

        # Verificar los swipes en shell de adb: deben ser 8 swipes (4 de ida y 4 de retorno)
        shell_calls = [call.args[0] for call in self.mock_device.d.shell.call_args_list]

        # Flanco Norte
        self.assertIn("input swipe 1200 240 1200 820 480", shell_calls)
        self.assertIn("input swipe 1200 820 1200 240 480", shell_calls)

        # Flanco Sur
        self.assertIn("input swipe 1200 780 1200 240 480", shell_calls)
        self.assertIn("input swipe 1200 240 1200 780 480", shell_calls)

        # Flanco Este (swipe calibrado 700px: 1550 -> 850 para no empujar almacenes fuera de pantalla)
        self.assertIn("input swipe 1550 540 850 540 480", shell_calls)
        self.assertIn("input swipe 850 540 1550 540 480", shell_calls)

        # Flanco Oeste (swipe calibrado 700px: 850 -> 1550)
        self.assertIn("input swipe 850 540 1550 540 480", shell_calls)
        self.assertIn("input swipe 1550 540 850 540 480", shell_calls)


if __name__ == "__main__":
    unittest.main()
