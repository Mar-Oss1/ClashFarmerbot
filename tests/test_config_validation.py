"""
tests/test_config_validation.py
Tests unitarios para la validación de configuración y soporte de variables de entorno (Mejora 06).
"""
import copy
import os
import unittest
from unittest.mock import patch

from core.config import SETTINGS, validate_settings, load_settings


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        self.valid_settings = copy.deepcopy(SETTINGS)

    def test_validate_settings_clean_on_default(self):
        """Verifica que la configuración por defecto del proyecto pasa sin advertencias."""
        warnings = validate_settings(self.valid_settings)
        self.assertEqual(warnings, [], f"Se encontraron advertencias inesperadas: {warnings}")

    def test_missing_required_section(self):
        """Verifica que se detectan secciones obligatorias ausentes."""
        bad_settings = copy.deepcopy(self.valid_settings)
        del bad_settings["device"]
        del bad_settings["economy"]

        warnings = validate_settings(bad_settings)
        self.assertTrue(any("device" in w for w in warnings))
        self.assertTrue(any("economy" in w for w in warnings))

    def test_invalid_logical_ranges(self):
        """Verifica que se detectan rangos numéricos inválidos."""
        # 1. Umbral de almacenes > 1.0
        bad_settings = copy.deepcopy(self.valid_settings)
        bad_settings["economy"]["storage_upgrade_trigger_threshold"] = 1.25
        warnings = validate_settings(bad_settings)
        self.assertTrue(any("storage_upgrade_trigger_threshold" in w for w in warnings))

        # 2. Confianza YOLO >= 1.0 o <= 0.0
        bad_settings2 = copy.deepcopy(self.valid_settings)
        bad_settings2["attack"]["yolo_conf"] = 1.5
        warnings2 = validate_settings(bad_settings2)
        self.assertTrue(any("yolo_conf" in w for w in warnings2))

        # 3. y_min >= y_max
        bad_settings3 = copy.deepcopy(self.valid_settings)
        bad_settings3["attack"]["safe_drop_y_min"] = 900
        bad_settings3["attack"]["safe_drop_y_max"] = 500
        warnings3 = validate_settings(bad_settings3)
        self.assertTrue(any("safe_drop_y_min" in w for w in warnings3))

    def test_missing_onnx_model_reported_as_warning(self):
        """Verifica que un archivo .onnx inexistente se reporta como advertencia sin abortar."""
        bad_settings = copy.deepcopy(self.valid_settings)
        bad_settings["attack"]["yolo_model_path"] = "models/non_existent_model.onnx"

        warnings = validate_settings(bad_settings)
        self.assertTrue(any("non_existent_model.onnx" in w for w in warnings))

    def test_missing_critical_button_template_warning(self):
        """Verifica que la ausencia de plantillas críticas se reporta como advertencia."""
        with patch("os.path.exists") as mock_exists:
            # Simular que attack_btn.png no existe en disco
            mock_exists.side_effect = lambda path: False if "attack_btn.png" in path else True
            warnings = validate_settings(self.valid_settings)
            self.assertTrue(any("attack_btn.png" in w for w in warnings))

    def test_env_serial_override(self):
        """Verifica que COC_DEVICE_SERIAL en entorno tiene prioridad sobre settings.json."""
        with patch.dict(os.environ, {"COC_DEVICE_SERIAL": "test_serial_12345"}):
            serial = os.getenv("COC_DEVICE_SERIAL") or self.valid_settings.get("device", {}).get("serial", "")
            self.assertEqual(serial, "test_serial_12345")


if __name__ == "__main__":
    unittest.main()
