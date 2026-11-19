"""
tests/test_army_checker.py
Pruebas unitarias para la detección de tropas y Súper Rompemuros en la barra inferior.
"""
import unittest
import numpy as np
import cv2
from core.vision import Vision
from vision.army_checker import ArmyChecker


class TestArmyChecker(unittest.TestCase):
    def setUp(self):
        self.vision = Vision()
        self.checker = ArmyChecker(self.vision)

    def test_find_super_wall_breaker_success(self):
        """Verifica que un frame con el icono de SWB en la mitad inferior se detecte correctamente."""
        tpl = cv2.imread(self.checker.tpl_swb_slot)
        if tpl is None:
            self.skipTest("No se encontró el template super_wall_breaker_slot.png")

        th, tw = tpl.shape[:2]
        frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Pegar el template en una posición realista de la barra de batalla
        target_x, target_y = 520, 930
        frame[target_y:target_y+th, target_x:target_x+tw] = tpl

        pos = self.checker.find_super_wall_breaker_in_battle_bar(frame)
        self.assertIsNotNone(pos, "Debe detectar el icono de Súper Rompemuros")
        # El punto retornado es el centro del match
        self.assertAlmostEqual(pos[0], target_x + tw // 2, delta=5)
        self.assertAlmostEqual(pos[1], target_y + th // 2, delta=5)

    def test_find_super_wall_breaker_absent(self):
        """Verifica que si no hay SWB en el frame retorne None sin lanzar excepción."""
        empty_frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        pos = self.checker.find_super_wall_breaker_in_battle_bar(empty_frame)
        self.assertIsNone(pos, "No debe detectar SWB en una imagen vacía")

    def test_find_super_wall_breaker_ignores_upper_half(self):
        """Verifica que si el icono está en la mitad superior (ej. en menús o aldea) se ignore."""
        tpl = cv2.imread(self.checker.tpl_swb_slot)
        if tpl is None:
            self.skipTest("No se encontró el template super_wall_breaker_slot.png")

        th, tw = tpl.shape[:2]
        frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Pegar en la mitad superior (y = 200)
        frame[200:200+th, 1000:1000+tw] = tpl

        pos = self.checker.find_super_wall_breaker_in_battle_bar(frame)
        self.assertIsNone(pos, "No debe detectar en la mitad superior por zona='MITAD_INFERIOR'")

    def test_find_royal_champion_and_minion_prince(self):
        """Verifica la detección específica de Royal Champion y Minion Prince."""
        rc_tpl = cv2.imread(self.checker.tpl_hero_champion)
        mp_tpl = cv2.imread(self.checker.tpl_hero_minion_prince)
        if rc_tpl is None or mp_tpl is None:
            self.skipTest("Plantillas de Champion o Minion Prince no encontradas.")

        frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Pegar Champion en x=725 y Minion Prince en x=880 (zona inferior y=940)
        frame[920:920+rc_tpl.shape[0], 690:690+rc_tpl.shape[1]] = rc_tpl
        frame[920:920+mp_tpl.shape[0], 845:845+mp_tpl.shape[1]] = mp_tpl

        rc_pos = self.checker.find_hero_champion_in_battle_bar(frame)
        mp_pos = self.checker.find_minion_prince_in_battle_bar(frame)

        self.assertIsNotNone(rc_pos, "Debe detectar a Royal Champion")
        self.assertIsNotNone(mp_pos, "Debe detectar a Minion Prince")
        self.assertAlmostEqual(rc_pos[0], 725, delta=10)
        self.assertAlmostEqual(mp_pos[0], 880, delta=10)

    def test_find_all_heroes_sorted_and_deduplicated(self):
        """Verifica que find_all_heroes desduplica plantillas en la misma ranura y las ordena de izq a der."""
        rc_tpl = cv2.imread(self.checker.tpl_hero_champion)
        mp_tpl = cv2.imread(self.checker.tpl_hero_minion_prince)
        w_tpl = cv2.imread(self.checker.tpl_hero_warden)
        if any(t is None for t in [rc_tpl, mp_tpl, w_tpl]):
            self.skipTest("Plantillas no encontradas")

        frame = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Pegar en orden inverso en X: Warden en x=1100, Prince en x=880, Champion en x=700
        frame[920:920+w_tpl.shape[0], 1100:1100+w_tpl.shape[1]] = w_tpl
        frame[920:920+mp_tpl.shape[0], 880:880+mp_tpl.shape[1]] = mp_tpl
        frame[920:920+rc_tpl.shape[0], 700:700+rc_tpl.shape[1]] = rc_tpl

        heroes = self.checker.find_all_heroes_in_battle_bar(frame)
        self.assertIn("champion", heroes)
        self.assertIn("minion_prince", heroes)
        self.assertIn("warden", heroes)

        # Verificar ordenamiento estricto de izquierda a derecha
        hero_x_coords = list(heroes.values())
        x_positions = [pt[0] for pt in hero_x_coords]
        self.assertEqual(x_positions, sorted(x_positions), "Las ranuras deben estar ordenadas de izq a der")
        hero_keys = list(heroes.keys())
        self.assertEqual(hero_keys, ["champion", "minion_prince", "warden"])


if __name__ == "__main__":
    unittest.main()
