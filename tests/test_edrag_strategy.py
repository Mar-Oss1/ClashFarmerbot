"""
tests/test_edrag_strategy.py
Pruebas unitarias para la estrategia de Dragones Eléctricos + Furias (EDrag Rush 50%).
"""
import os
import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

from core.vision import Vision
from vision.army_checker import ArmyChecker
from vision.resource_reader import ResourceReader
from attack.edrag_strategy import EDragStrategy
from core.config import SETTINGS


class TestEDragStrategy(unittest.TestCase):
    """Verifica todos los componentes de la estrategia EDrag Rush 50%."""

    def setUp(self):
        self.vision = Vision()
        self.mock_device = MagicMock()
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        # Dummy screenshot de 2400x1080
        self.dummy_screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.mock_device.screenshot.return_value = self.dummy_screen
        self.strategy = EDragStrategy(self.mock_device, self.vision)

    def test_new_templates_exist_and_load(self):
        """Verifica que todos los archivos de plantilla nuevos existen y son válidos."""
        templates = [
            "assets/buttons/edrag_slot.png",
            "assets/buttons/spell_rage.png",
            "assets/buttons/hero_minion_prince_slot.png",
            "assets/buttons/hero_champion_slot.png",
            "assets/buttons/hero_warden_slot.png",
            "assets/buttons/hero_duke_dragon_slot.png",
            "assets/buttons/hero_king_slot.png",
        ]
        for tpl_path in templates:
            self.assertTrue(os.path.exists(tpl_path), f"Plantilla no encontrada: {tpl_path}")
            img = cv2.imread(tpl_path)
            self.assertIsNotNone(img, f"No se pudo cargar la imagen: {tpl_path}")
            self.assertGreater(img.shape[0], 20, f"Altura insuficiente: {tpl_path}")
            self.assertGreater(img.shape[1], 20, f"Anchura insuficiente: {tpl_path}")

    def test_north_triangle_drops_geometry(self):
        """Verifica que los drops calculados caen en cuadrantes seguros, altos y alejados en el norte."""
        nw_drops, ne_drops = self.strategy.compute_north_triangle_drops()
        self.assertGreaterEqual(len(nw_drops), 4)
        self.assertGreaterEqual(len(ne_drops), 4)

        for x, y in nw_drops:
            # Flanco Noroeste alto y seguro
            self.assertGreaterEqual(x, 600)
            self.assertLessEqual(x, 1100)
            self.assertGreaterEqual(y, 100)
            self.assertLessEqual(y, 300)

        for x, y in ne_drops:
            # Flanco Noreste alto y seguro
            self.assertGreaterEqual(x, 1300)
            self.assertLessEqual(x, 1800)
            self.assertGreaterEqual(y, 100)
            self.assertLessEqual(y, 300)

    def test_opening_and_core_rage_points(self):
        """Verifica que los puntos de Furia de apertura y núcleo están en zonas estratégicas."""
        opening = self.strategy.get_opening_rage_points()
        self.assertGreaterEqual(len(opening), 2)
        for x, y in opening:
            self.assertGreater(y, 180)
            self.assertLess(y, 450)

        core = self.strategy.get_core_rage_points()
        self.assertGreaterEqual(len(core), 2)
        for x, y in core:
            self.assertGreater(y, 300)
            self.assertLess(y, 650)

    def test_dynamic_loot_criteria_edrag_vs_sneaky(self):
        """Verifica que meets_attack_criteria aplica los umbrales reducidos para EDrag."""
        edrag_criteria = {
            "min_gold": 500000,
            "min_elixir": 500000,
            "min_dark_elixir": 6000,
            "conditions_required": 2,
        }
        loot_test = {"gold": 550000, "elixir": 520000, "dark_elixir": 4000}

        meets_edrag = ResourceReader.meets_attack_criteria(loot_test, criteria_override=edrag_criteria)
        self.assertTrue(meets_edrag)

        meets_classic = ResourceReader.meets_attack_criteria(loot_test)
        self.assertFalse(meets_classic)

    def test_army_detection_with_live_screenshot(self):
        """Verifica la detección automática del ejército EDrag sobre la captura real del usuario."""
        live_img_path = "logs/yolo/edrag_attack_start_20261111_184514.png"
        if not os.path.exists(live_img_path):
            self.skipTest(f"{live_img_path} no existe en este entorno")

        img = cv2.imread(live_img_path)
        army = ArmyChecker(self.vision)

        with unittest.mock.patch.dict(SETTINGS["attack"], {"attack_strategy": "auto"}):
            detected_type = army.detect_active_army_type(img)
            self.assertEqual(detected_type, "edrag_rush")

        heroes = army.find_all_heroes_in_battle_bar(img)
        self.assertIn("champion", heroes)
        self.assertIn("minion_prince", heroes)
        self.assertIn("warden", heroes)
        self.assertTrue("duke_dragon" in heroes or "king" in heroes)

    def test_execute_attack_mock_run(self):
        """Verifica que execute_attack corre sin excepciones y envía las secuencias de taps correctas."""
        army = self.strategy.army
        army.find_edrag_in_battle_bar = MagicMock(return_value=(407, 955))
        army.find_rage_spell_in_battle_bar = MagicMock(return_value=(1345, 960))
        army.find_all_heroes_in_battle_bar = MagicMock(return_value={
            "champion": (725, 960),
            "minion_prince": (880, 960),
            "warden": (1035, 960),
            "duke_dragon": (1190, 960),
        })

        self.strategy.rage_core_delay_sec = 0.05
        self.strategy.hero_ability_delay_sec = 0.05

        success = self.strategy.execute_attack()
        self.assertTrue(success)
        # Debe llamar zoom_half (modo half por defecto) y pan_camera hacia el norte
        self.mock_device.zoom_half.assert_called_once()
        self.mock_device.pan_camera.assert_called_once()
        # Debe ejecutar exactamente 2 taps sostenidos separados para los dragones
        self.assertEqual(self.mock_device.hold_tap.call_count, 2)
        # Debe desplegar los 4 héroes uno a uno en el mismo punto ancla de los primeros dragones
        nw_drops, _ = self.strategy.compute_north_triangle_drops(self.dummy_screen)
        anchor_pt = nw_drops[1]
        anchor_calls = [call for call in self.mock_device.tap.call_args_list if call[0] == anchor_pt or (len(call[0]) >= 2 and (call[0][0], call[0][1]) == anchor_pt)]
        self.assertEqual(len(anchor_calls), 4)

    def test_execute_attack_custom_heroes_only_champion_and_prince(self):
        """Verifica que si el usuario solo lleva Royal Champion y Minion Prince, se despliegan exactamente ellos dos sin toques fantasma."""
        self.mock_device.tap.reset_mock()
        army = self.strategy.army
        army.find_edrag_in_battle_bar = MagicMock(return_value=(407, 955))
        army.find_rage_spell_in_battle_bar = MagicMock(return_value=(1345, 960))
        army.find_all_heroes_in_battle_bar = MagicMock(return_value={
            "champion": (725, 960),
            "minion_prince": (880, 960),
        })

        self.strategy.rage_core_delay_sec = 0.01
        self.strategy.hero_ability_delay_sec = 0.01

        success = self.strategy.execute_attack()
        self.assertTrue(success)

        nw_drops, _ = self.strategy.compute_north_triangle_drops(self.dummy_screen)
        anchor_pt = nw_drops[1]
        anchor_calls = [call for call in self.mock_device.tap.call_args_list if call[0] == anchor_pt or (len(call[0]) >= 2 and (call[0][0], call[0][1]) == anchor_pt)]
        # Exactamente 2 héroes soltados en el ancla
        self.assertEqual(len(anchor_calls), 2)

        # Verificar que se seleccionaron las 2 ranuras y se activaron sus habilidades (2 taps en slot + 2 taps en slot para habilidad = 4 taps totales en slots)
        slot_taps = [call[0] for call in self.mock_device.tap.call_args_list if call[0] in [(725, 960), (880, 960)]]
        self.assertEqual(len(slot_taps), 4)


if __name__ == "__main__":
    unittest.main()
