"""
core/wall_cost_table.py
Fuente canónica y única de verdad para los costos oficiales de muros en Clash of Clans (Aldea Principal).
Define los requisitos exactos de Oro y Elixir para los niveles 1 a 19 según la tabla oficial de Supercell,
así como funciones de validación de recursos y soporte para extrapolación dinámica en niveles >= 20.
"""
from typing import Dict, Any, Optional

# Tabla canónica oficial de costos de muros (Niveles 1 a 19)
# Nota técnica:
# - Niveles 1 a 4: Elixir no se requiere (elixir_cost = None). Solo se puede mejorar con Oro.
# - Niveles 5 a 19: Se puede mejorar pagando con Oro O con Elixir (no se requieren ambos simultáneamente).
WALL_LEVEL_COSTS: Dict[int, Dict[str, Any]] = {
    # nivel: {hitpoints, gold_cost, elixir_cost (None si N/A), th_required}
    1:  {"hitpoints": 100,    "gold_cost": 0,          "elixir_cost": None,       "th_required": 2},
    2:  {"hitpoints": 200,    "gold_cost": 1_000,      "elixir_cost": None,       "th_required": 2},
    3:  {"hitpoints": 400,    "gold_cost": 5_000,      "elixir_cost": None,       "th_required": 3},
    4:  {"hitpoints": 800,    "gold_cost": 10_000,     "elixir_cost": None,       "th_required": 4},
    5:  {"hitpoints": 1_200,  "gold_cost": 20_000,     "elixir_cost": 20_000,     "th_required": 5},
    6:  {"hitpoints": 1_800,  "gold_cost": 30_000,     "elixir_cost": 30_000,     "th_required": 6},
    7:  {"hitpoints": 2_400,  "gold_cost": 50_000,     "elixir_cost": 50_000,     "th_required": 7},
    8:  {"hitpoints": 3_000,  "gold_cost": 75_000,     "elixir_cost": 75_000,     "th_required": 8},
    9:  {"hitpoints": 3_500,  "gold_cost": 100_000,    "elixir_cost": 100_000,    "th_required": 9},
    10: {"hitpoints": 4_000,  "gold_cost": 200_000,    "elixir_cost": 200_000,    "th_required": 9},
    11: {"hitpoints": 5_000,  "gold_cost": 500_000,    "elixir_cost": 500_000,    "th_required": 10},
    12: {"hitpoints": 7_000,  "gold_cost": 1_000_000,  "elixir_cost": 1_000_000,  "th_required": 11},
    13: {"hitpoints": 8_000,  "gold_cost": 1_500_000,  "elixir_cost": 1_500_000,  "th_required": 12},
    14: {"hitpoints": 9_000,  "gold_cost": 2_000_000,  "elixir_cost": 2_000_000,  "th_required": 13},
    15: {"hitpoints": 10_000, "gold_cost": 3_000_000,  "elixir_cost": 3_000_000,  "th_required": 14},
    16: {"hitpoints": 11_000, "gold_cost": 4_000_000,  "elixir_cost": 4_000_000,  "th_required": 15},
    17: {"hitpoints": 12_000, "gold_cost": 5_000_000,  "elixir_cost": 5_000_000,  "th_required": 16},
    18: {"hitpoints": 13_000, "gold_cost": 7_000_000,  "elixir_cost": 7_000_000,  "th_required": 17},
    19: {"hitpoints": 14_000, "gold_cost": 10_000_000, "elixir_cost": 10_000_000, "th_required": 18},
}


def get_required_resources(target_level: int) -> dict:
    """
    Obtiene los recursos requeridos (Oro y Elixir) para la mejora al nivel objetivo (target_level).
    - Para niveles 1 a 19: Lee directamente los valores canónicos de WALL_LEVEL_COSTS.
    - Para niveles >= 20: Calcula una extrapolación dinámica.
    
    ADVERTENCIA: Para niveles >= 20, los valores calculados son una aproximación extrapolada,
    ya que los valores futuros de Supercell pueden no seguir un patrón estrictamente lineal o geométrico.
    """
    if target_level in WALL_LEVEL_COSTS:
        row = WALL_LEVEL_COSTS[target_level]
        return {
            "gold": row["gold_cost"],
            "elixir": row["elixir_cost"],  # None indica que no requiere elixir en este nivel
        }
    elif target_level >= 20:
        # ADVERTENCIA DE APROXIMACIÓN (NIVEL >= 20):
        # Los valores por encima del nivel 19 son extrapolados dinámicamente.
        extra_cost = 10_000_000 + (target_level - 19) * 1_000_000
        return {
            "gold": extra_cost,
            "elixir": extra_cost,
        }
    else:
        raise KeyError(f"Nivel objetivo inválido: {target_level}")


def can_afford(
    target_level: int,
    available_gold: int,
    available_elixir: int,
    require_both: bool = False,
) -> bool:
    """
    Valida si el jugador puede costear la mejora al target_level según los saldos disponibles.
    
    Reglas oficiales de Clash of Clans:
    - Niveles 1 a 4: Solo se permite mejorar con Oro (req["elixir"] is None).
      Retorna True si available_gold >= req["gold"].
    - Niveles 5+: Los muros se pueden pagar con Oro O con Elixir de forma intercambiable.
      Retorna True si available_gold >= req["gold"] O available_elixir >= req["elixir"].
      Rechaza únicamente si no alcanza ninguno de los dos recursos (o ambos si require_both=True).
    """
    req = get_required_resources(target_level)
    gold_ok = available_gold >= req["gold"]
    if req["elixir"] is None:
        return gold_ok

    elixir_ok = available_elixir >= req["elixir"]
    if require_both:
        return gold_ok and elixir_ok
    return gold_ok or elixir_ok
