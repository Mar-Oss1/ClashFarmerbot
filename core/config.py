"""
core/config.py
Gestión centralizada de rutas y configuración JSON.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.json")


def load_settings() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


SETTINGS = load_settings()

# Constantes expuestas para compatibilidad directa con device.py y vision.py
DEVICE_SERIAL: str = SETTINGS["device"]["serial"]
APP_PACKAGE: str = SETTINGS["device"]["app_package"]
TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =====================================================
# BLOQUE DE ESTABILIDAD DE LOOT Y CÁMARA
# =====================================================
_LOOT = SETTINGS.get("loot", {})
_CAMERA = SETTINGS.get("camera", {})
_SWIPE_PROFILES = SETTINGS.get("swipe_profiles", {})
_BATTLE_END = SETTINGS.get("battle_end", {})
_DEBUG = SETTINGS.get("debug", {})
_ATTACK = SETTINGS.get("attack", {})
_ECONOMY = SETTINGS.get("economy", {})

# ---- Economía y Almacenes ----
STORAGE_UPGRADE_TRIGGER_THRESHOLD = float(_ECONOMY.get("storage_upgrade_trigger_threshold", 0.90))

# ---- Estabilidad de lectura de botín ----
LOOT_WAIT_AFTER_NEXT_MIN = float(_LOOT.get("wait_after_next_min", 0.55))
LOOT_WAIT_AFTER_NEXT_MAX = float(_LOOT.get("wait_after_next_max", 0.80))
LOOT_STABILITY_FRAMES = int(_LOOT.get("stability_frames", 2))
LOOT_STABILITY_DELAY = float(_LOOT.get("stability_delay", 0.12))
LOOT_MAX_READ_RETRIES = int(_LOOT.get("max_read_retries", 3))
LOOT_ALLOW_ATTACK_ON_UNSTABLE_READ = bool(_LOOT.get("allow_attack_on_unstable_read", False))

# ---- Cámara ----
CAMERA_SIGN_X = int(_CAMERA.get("sign_x", 1))
CAMERA_SIGN_Y = int(_CAMERA.get("sign_y", -1))
CAMERA_SAFE_ORIGIN_X = int(_CAMERA.get("safe_origin_x", 1200))
CAMERA_SAFE_ORIGIN_Y = int(_CAMERA.get("safe_origin_y", 540))
CAMERA_SETTLE_WAIT = float(_CAMERA.get("settle_wait", 0.25))

# ---- Perfiles de swipe y cierre de batalla ----
SWIPE_PROFILES = _SWIPE_PROFILES
BATTLE_END = _BATTLE_END

# ---- Debug ----
SEARCH_ONLY = bool(_DEBUG.get("search_only", True))

# ---- YOLO Detection ----
USE_YOLO = bool(_ATTACK.get("use_yolo", False))
YOLO_MODEL_PATH = str(_ATTACK.get("yolo_model_path", "models/yolov8s.onnx"))
YOLO_CONF = float(_ATTACK.get("yolo_conf", 0.40))
YOLO_IOU = float(_ATTACK.get("yolo_iou", 0.5))
YOLO_IMGSZ = int(_ATTACK.get("yolo_imgsz", 1280))
YOLO_MIN_AVG_CONF_FOR_SKIP_TAP = float(_ATTACK.get("yolo_min_avg_conf_for_skip_tap", 0.60))
