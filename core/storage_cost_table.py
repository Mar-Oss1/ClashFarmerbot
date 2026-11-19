"""
core/storage_cost_table.py
Fuente canónica y oficial de capacidades de almacenes (Gold Storage, Elixir Storage y Dark Elixir Storage)
en Clash of Clans (Aldea Principal).
Proporciona capacidades por nivel, cantidad de almacenes por Town Hall (TH), estimaciones teóricas
y valores de fallback predeterminados (TH18 con almacenes de TH17: 24M Oro/Elixir y 420k Oscuro).
"""
from typing import Dict, Any, Optional

# Tabla oficial de Gold Storage y Elixir Storage (Niveles 1 a 19)
# Misma capacidad, hitpoints y tiempos en ambos tipos.
GOLD_ELIXIR_STORAGE_LEVELS: Dict[int, Dict[str, Any]] = {
    1:  {"capacity": 1_500,     "hitpoints": 150,   "cost": 300,        "time": "10s",     "th_required": 1},
    2:  {"capacity": 3_000,     "hitpoints": 300,   "cost": 750,        "time": "2m",      "th_required": 2},
    3:  {"capacity": 6_000,     "hitpoints": 450,   "cost": 1_500,      "time": "5m",      "th_required": 2},
    4:  {"capacity": 12_000,    "hitpoints": 600,   "cost": 3_000,      "time": "15m",     "th_required": 3},
    5:  {"capacity": 25_000,    "hitpoints": 800,   "cost": 6_000,      "time": "30m",     "th_required": 3},
    6:  {"capacity": 45_000,    "hitpoints": 1_000, "cost": 12_000,     "time": "1h",      "th_required": 3},
    7:  {"capacity": 100_000,   "hitpoints": 1_250, "cost": 25_000,     "time": "2h",      "th_required": 4},
    8:  {"capacity": 225_000,   "hitpoints": 1_500, "cost": 50_000,     "time": "3h",      "th_required": 4},
    9:  {"capacity": 450_000,   "hitpoints": 1_700, "cost": 100_000,    "time": "4h",      "th_required": 5},
    10: {"capacity": 850_000,   "hitpoints": 1_900, "cost": 250_000,    "time": "5h",      "th_required": 6},
    11: {"capacity": 1_750_000, "hitpoints": 2_100, "cost": 500_000,    "time": "6h",      "th_required": 7},
    12: {"capacity": 2_000_000, "hitpoints": 2_500, "cost": 1_000_000,  "time": "12h",     "th_required": 11},
    13: {"capacity": 3_000_000, "hitpoints": 2_900, "cost": 1_800_000,  "time": "18h",     "th_required": 12},
    14: {"capacity": 4_000_000, "hitpoints": 3_300, "cost": 2_800_000,  "time": "1d",      "th_required": 13},
    15: {"capacity": 4_500_000, "hitpoints": 3_700, "cost": 3_000_000,  "time": "2d",      "th_required": 14},
    16: {"capacity": 5_000_000, "hitpoints": 3_900, "cost": 4_000_000,  "time": "3d",      "th_required": 15},
    17: {"capacity": 5_500_000, "hitpoints": 4_050, "cost": 5_500_000,  "time": "4d",      "th_required": 16},
    18: {"capacity": 6_000_000, "hitpoints": 4_200, "cost": 10_000_000, "time": "5d",      "th_required": 17},
    19: {"capacity": 6_500_000, "hitpoints": 4_300, "cost": 18_000_000, "time": "12d 12h", "th_required": 18},
}

# Tabla oficial de Dark Elixir Storage (Niveles 1 a 13)
DARK_ELIXIR_STORAGE_LEVELS: Dict[int, Dict[str, Any]] = {
    1:  {"capacity": 10_000,  "hitpoints": 2_000, "cost": 250_000,    "time": "8h",     "th_required": 7},
    2:  {"capacity": 17_500,  "hitpoints": 2_200, "cost": 500_000,    "time": "16h",    "th_required": 7},
    3:  {"capacity": 40_000,  "hitpoints": 2_400, "cost": 1_000_000,  "time": "1d",     "th_required": 8},
    4:  {"capacity": 75_000,  "hitpoints": 2_600, "cost": 1_500_000,  "time": "1d 12h", "th_required": 8},
    5:  {"capacity": 140_000, "hitpoints": 2_900, "cost": 2_000_000,  "time": "1d 16h", "th_required": 9},
    6:  {"capacity": 180_000, "hitpoints": 3_200, "cost": 2_400_000,  "time": "2d",     "th_required": 9},
    7:  {"capacity": 220_000, "hitpoints": 3_500, "cost": 3_800_000,  "time": "3d",     "th_required": 12},
    8:  {"capacity": 280_000, "hitpoints": 3_800, "cost": 5_400_000,  "time": "3d 12h", "th_required": 13},
    9:  {"capacity": 330_000, "hitpoints": 4_100, "cost": 6_600_000,  "time": "4d",     "th_required": 14},
    10: {"capacity": 360_000, "hitpoints": 4_300, "cost": 8_000_000,  "time": "4d 12h", "th_required": 15},
    11: {"capacity": 390_000, "hitpoints": 4_500, "cost": 10_000_000, "time": "5d",     "th_required": 16},
    12: {"capacity": 420_000, "hitpoints": 4_700, "cost": 16_000_000, "time": "7d",     "th_required": 17},
    13: {"capacity": 450_000, "hitpoints": 4_800, "cost": 25_000_000, "time": "13d",    "th_required": 18},
}

# Cantidad de Gold / Elixir Storage disponibles según Town Hall (TH 1 a 19)
GOLD_ELIXIR_STORAGE_COUNTS_BY_TH: Dict[int, int] = {
    1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3,
    9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4,
}

# Cantidad de Dark Elixir Storage disponibles según Town Hall (TH 1 a 19)
DARK_ELIXIR_STORAGE_COUNTS_BY_TH: Dict[int, int] = {
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
    7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
}

# Valores de fallback canónicos acordados (TH18 con almacenes de TH17 = 24M Oro/Elixir y 420k Oscuro)
# - Gold / Elixir: 4 almacenes x 6,000,000 (lvl 18, TH17) = 24,000,000
# - Dark Elixir: 1 almacén x 420,000 (lvl 12, TH17) = 420,000
FALLBACK_TH18_CAPACITIES: Dict[str, int] = {
    "gold": 24_000_000,
    "elixir": 24_000_000,
    "dark_elixir": 420_000,
}


def get_storage_capacity(resource_type: str, level: int) -> int:
    """
    Obtiene la capacidad unitaria de un almacén dado su nivel.
    resource_type: 'gold', 'elixir' o 'dark_elixir'.
    """
    res = resource_type.lower()
    if res in ("gold", "elixir"):
        if level in GOLD_ELIXIR_STORAGE_LEVELS:
            return int(GOLD_ELIXIR_STORAGE_LEVELS[level]["capacity"])
        raise KeyError(f"Nivel de Gold/Elixir Storage no válido: {level}")
    elif res in ("dark", "dark_elixir"):
        if level in DARK_ELIXIR_STORAGE_LEVELS:
            return int(DARK_ELIXIR_STORAGE_LEVELS[level]["capacity"])
        raise KeyError(f"Nivel de Dark Elixir Storage no válido: {level}")
    raise ValueError(f"Tipo de recurso no reconocido: {resource_type}")


def get_storage_count_for_th(resource_type: str, th: int) -> int:
    """Devuelve la cantidad de almacenes disponibles para un TH dado."""
    res = resource_type.lower()
    if res in ("gold", "elixir"):
        return GOLD_ELIXIR_STORAGE_COUNTS_BY_TH.get(th, 4 if th >= 9 else 1)
    elif res in ("dark", "dark_elixir"):
        return DARK_ELIXIR_STORAGE_COUNTS_BY_TH.get(th, 1 if th >= 7 else 0)
    raise ValueError(f"Tipo de recurso no reconocido: {resource_type}")


def calculate_estimated_storage_capacity(
    resource_type: str,
    th: int = 18,
    storage_level: Optional[int] = None,
) -> int:
    """
    Calcula la capacidad teórica multiplicando el nivel del almacén por la cantidad disponible en el TH.
    Si no se proporciona storage_level, utiliza el nivel máximo alcanzable para el TH especificado.
    """
    res = resource_type.lower()
    count = get_storage_count_for_th(res, th)
    if count == 0:
        return 0

    if storage_level is None:
        # Buscar el nivel máximo de almacén soportado por ese TH
        table = GOLD_ELIXIR_STORAGE_LEVELS if res in ("gold", "elixir") else DARK_ELIXIR_STORAGE_LEVELS
        valid_lvls = [lvl for lvl, data in table.items() if data["th_required"] <= th]
        storage_level = max(valid_lvls) if valid_lvls else 1

    unit_cap = get_storage_capacity(res, storage_level)
    return unit_cap * count


def get_fallback_storage_capacities(
    th_level: int = 18,
    use_th17_storages: bool = True,
) -> Dict[str, int]:
    """
    Retorna las capacidades de fallback estándar para oro, elixir y oscuro.
    Por defecto implementa la decisión de negocio: TH18 con almacenes de TH17 (24M Oro/Elixir y 420k Oscuro).
    """
    if th_level == 18 and use_th17_storages:
        return dict(FALLBACK_TH18_CAPACITIES)

    return {
        "gold": calculate_estimated_storage_capacity("gold", th_level),
        "elixir": calculate_estimated_storage_capacity("elixir", th_level),
        "dark_elixir": calculate_estimated_storage_capacity("dark_elixir", th_level),
    }
