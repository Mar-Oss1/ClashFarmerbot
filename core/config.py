"""
core/config.py
Gestión centralizada de rutas, variables de entorno y configuración JSON validada.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.json")


def _load_env_file():
    """Carga variables desde .env si existe, usando python-dotenv o lector nativo."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip())
            except Exception:
                pass


_load_env_file()


def load_settings() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


SETTINGS = load_settings()

# Constantes expuestas para compatibilidad directa con device.py y vision.py
# Permite override seguro por variable de entorno sin exponer el serial en settings.json
DEVICE_SERIAL: str = os.getenv("COC_DEVICE_SERIAL") or SETTINGS.get("device", {}).get("serial", "")
if os.getenv("COC_DEVICE_SERIAL"):
    SETTINGS.setdefault("device", {})["serial"] = DEVICE_SERIAL

APP_PACKAGE: str = SETTINGS.get("device", {}).get("app_package", "com.supercell.clashofclans")
TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")


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

# ---- Perfiles de swipe y cierre de batalla (compatibilidad) ----
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


def validate_settings(settings: dict | None = None) -> list[str]:
    """
    Valida la integridad y coherencia lógica de settings.json al arranque.
    Verifica secciones requeridas, rangos lógicos de umbrales y presencia de modelos/templates.
    Retorna la lista de advertencias detectadas (vacía si todo es válido).
    """
    if settings is None:
        settings = SETTINGS

    warnings: list[str] = []

    # 1. Secciones obligatorias
    required_sections = ["device", "attack", "rois", "humanizer", "economy"]
    for sec in required_sections:
        if sec not in settings:
            warnings.append(f"Sección requerida faltante: '{sec}'")

    # 2. Rangos lógicos
    econ = settings.get("economy", {})
    storage_thresh = float(econ.get("storage_upgrade_trigger_threshold", 0.90))
    if not (0.0 < storage_thresh <= 1.0):
        warnings.append(f"storage_upgrade_trigger_threshold fuera de rango (0.0, 1.0]: {storage_thresh}")

    atk = settings.get("attack", {})
    yolo_conf = float(atk.get("yolo_conf", 0.40))
    if not (0.0 < yolo_conf < 1.0):
        warnings.append(f"attack.yolo_conf fuera de rango (0.0, 1.0): {yolo_conf}")

    y_min = int(atk.get("safe_drop_y_min", 140))
    y_max = int(atk.get("safe_drop_y_max", 845))
    if y_min >= y_max:
        warnings.append(f"safe_drop_y_min ({y_min}) debe ser menor que safe_drop_y_max ({y_max})")

    # 3. Modelos ONNX configurados
    onnx_keys = [
        ("attack.yolo_model_path", atk.get("yolo_model_path")),
        ("attack.deploy_zone_ai.model_path", atk.get("deploy_zone_ai", {}).get("model_path")),
        ("attack.jump_optimizer.wall_seg_model_path", atk.get("jump_optimizer", {}).get("wall_seg_model_path")),
    ]
    for key_name, m_path in onnx_keys:
        if m_path:
            full_path = os.path.join(PROJECT_ROOT, m_path) if not os.path.isabs(m_path) else m_path
            if not os.path.exists(full_path):
                warnings.append(f"Modelo ONNX configurado en '{key_name}' no existe en disco: '{m_path}'")

    # 4. Templates críticos de botones
    critical_templates = ["attack_btn.png", "next_btn.png", "return_home.png"]
    buttons_dir = os.path.join(PROJECT_ROOT, "assets", "buttons")
    for tmpl in critical_templates:
        tmpl_path = os.path.join(buttons_dir, tmpl)
        if not os.path.exists(tmpl_path):
            warnings.append(f"Plantilla crítica no encontrada en assets/buttons: '{tmpl}'")

    # Emitir advertencias vía logger si existen
    if warnings:
        try:
            from core.logger import get_logger
            log = get_logger("config")
            for w in warnings:
                log.warning("⚠️ [CONFIG VALIDATION] %s", w)
        except Exception:
            pass

    return warnings
