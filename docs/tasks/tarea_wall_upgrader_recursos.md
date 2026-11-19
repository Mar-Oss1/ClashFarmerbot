# Tarea: Corrección de reglas de recursos en `WallUpgradeStrategy`

## Contexto

El swarm anterior implementó `core/wall_upgrader.py` con `WallUpgradeStrategy` y
`WallTransition`, generalizando la lógica para no depender de valores cableados
de la transición 18→19. Sin embargo, la regla de validación de recursos que usó
es **incorrecta**: asumió que el elixir solo se requiere a partir del nivel 9
(transición 8→9). Según la tabla oficial de costos de muro de Clash of Clans
(adjunta más abajo), eso no es así.

## Corrección obligatoria

- El **oro** se requiere desde el nivel 1 en adelante (nunca `N/A`).
- El **elixir** se vuelve requerido a partir del **nivel 5** (es decir, la
  transición que termina en nivel 5 ya pide oro + elixir simultáneamente), y
  sigue siendo requerido junto con el oro en los niveles siguientes donde la
  tabla indique un valor numérico de elixir (no `N/A`).
- No existe un único "umbral mágico" fijo (ni 9, ni ningún otro número) a
  partir del cual el elixir se activa de forma permanente — la validación debe
  leer el costo de elixir **por nivel específico** desde una tabla de datos, no
  inferirlo de una regla de "a partir de X nivel siempre aplica Y".
- Eliminar cualquier condicional tipo `if level >= 9: requiere_elixir = True`
  en `core/wall_upgrader.py` y reemplazarlo por una lectura directa del campo
  `elixir_cost` de la tabla de transición (ver estructura de datos abajo).

## Tabla oficial de costos (fuente: captura de referencia del usuario)

| Nivel | Hitpoints | Costo Oro | Costo Oro Acum. | Costo Elixir | Costo Elixir Acum. | Costo Constructor | TH Requerido |
|---|---|---|---|---|---|---|---|
| 1  | 100    | 0          | 0          | N/A        | N/A        | N/A | 2  |
| 2  | 200    | 1,000      | 1,000      | N/A        | N/A        | 1   | 2  |
| 3  | 400    | 5,000      | 6,000      | N/A        | N/A        | 1   | 3  |
| 4  | 800    | 10,000     | 16,000     | N/A        | N/A        | 1   | 4  |
| 5  | 1,200  | 20,000     | 36,000     | 20,000     | 20,000     | 1   | 5  |
| 6  | 1,800  | 30,000     | 66,000     | 30,000     | 50,000     | 1   | 6  |
| 7  | 2,400  | 50,000     | 116,000    | 50,000     | 100,000    | 1   | 7  |
| 8  | 3,000  | 75,000     | 191,000    | 75,000     | 175,000    | 1   | 8  |
| 9  | 3,500  | 100,000    | 291,000    | 100,000    | 275,000    | 1   | 9  |
| 10 | 4,000  | 200,000    | 491,000    | 200,000    | 475,000    | 1   | 9  |
| 11 | 5,000  | 500,000    | 991,000    | 500,000    | 975,000    | 1   | 10 |
| 12 | 7,000  | 1,000,000  | 1,991,000  | 1,000,000  | 1,975,000  | 1   | 11 |
| 13 | 8,000  | 1,500,000  | 3,491,000  | 1,500,000  | 3,475,000  | 2   | 12 |
| 14 | 9,000  | 2,000,000  | 5,491,000  | 2,000,000  | 5,475,000  | 2   | 13 |
| 15 | 10,000 | 3,000,000  | 8,491,000  | 3,000,000  | 8,475,000  | 3   | 14 |
| 16 | 11,000 | 4,000,000  | 12,491,000 | 4,000,000  | 12,475,000 | 4   | 15 |
| 17 | 12,000 | 5,000,000  | 17,491,000 | 5,000,000  | 17,475,000 | 5   | 16 |
| 18 | 13,000 | 7,000,000  | 24,491,000 | 7,000,000  | 24,475,000 | 7   | 17 |
| 19 | 14,000 | 10,000,000 | 34,491,000 | 10,000,000 | 34,475,000 | 10  | 18 |

**Nota:** "Costo Constructor" no es un tercer recurso independiente del bot —
corresponde al tiempo/slot de constructor usado (Builder), que el proyecto ya
trata como instantáneo (`0s`) según lo implementado. No confundir con un
recurso tipo "Elixir Oscuro" (`gema` en la tabla original no aplica a muros).

## Estructura de datos recomendada para `core/wall_upgrader.py`

En vez de lógica condicional dispersa, usar una tabla de datos única que el
código consulte, por ejemplo como diccionario o archivo de configuración:

```python
# core/wall_cost_table.py (o config/wall_costs.json)
WALL_LEVEL_COSTS = {
    # nivel: {hitpoints, gold_cost, elixir_cost (None si N/A), th_required}
    1:  {"hitpoints": 100,   "gold_cost": 0,         "elixir_cost": None,     "th_required": 2},
    2:  {"hitpoints": 200,   "gold_cost": 1_000,     "elixir_cost": None,     "th_required": 2},
    3:  {"hitpoints": 400,   "gold_cost": 5_000,     "elixir_cost": None,     "th_required": 3},
    4:  {"hitpoints": 800,   "gold_cost": 10_000,    "elixir_cost": None,     "th_required": 4},
    5:  {"hitpoints": 1_200, "gold_cost": 20_000,    "elixir_cost": 20_000,   "th_required": 5},
    6:  {"hitpoints": 1_800, "gold_cost": 30_000,    "elixir_cost": 30_000,   "th_required": 6},
    7:  {"hitpoints": 2_400, "gold_cost": 50_000,    "elixir_cost": 50_000,   "th_required": 7},
    8:  {"hitpoints": 3_000, "gold_cost": 75_000,    "elixir_cost": 75_000,   "th_required": 8},
    9:  {"hitpoints": 3_500, "gold_cost": 100_000,   "elixir_cost": 100_000,  "th_required": 9},
    10: {"hitpoints": 4_000, "gold_cost": 200_000,   "elixir_cost": 200_000,  "th_required": 9},
    11: {"hitpoints": 5_000, "gold_cost": 500_000,   "elixir_cost": 500_000,  "th_required": 10},
    12: {"hitpoints": 7_000, "gold_cost": 1_000_000, "elixir_cost": 1_000_000,"th_required": 11},
    13: {"hitpoints": 8_000, "gold_cost": 1_500_000, "elixir_cost": 1_500_000,"th_required": 12},
    14: {"hitpoints": 9_000, "gold_cost": 2_000_000, "elixir_cost": 2_000_000,"th_required": 13},
    15: {"hitpoints": 10_000,"gold_cost": 3_000_000, "elixir_cost": 3_000_000,"th_required": 14},
    16: {"hitpoints": 11_000,"gold_cost": 4_000_000, "elixir_cost": 4_000_000,"th_required": 15},
    17: {"hitpoints": 12_000,"gold_cost": 5_000_000, "elixir_cost": 5_000_000,"th_required": 16},
    18: {"hitpoints": 13_000,"gold_cost": 7_000_000, "elixir_cost": 7_000_000,"th_required": 17},
    19: {"hitpoints": 14_000,"gold_cost": 10_000_000,"elixir_cost": 10_000_000,"th_required": 18},
}
```

La validación de recursos para cualquier transición `N → N+1` debe ser:

```python
def get_required_resources(target_level: int) -> dict:
    row = WALL_LEVEL_COSTS[target_level]
    return {
        "gold": row["gold_cost"],
        "elixir": row["elixir_cost"],  # None significa "no requiere elixir en este nivel"
    }
```

Y al validar si el jugador puede pagar la mejora:

```python
def can_afford(target_level: int, available_gold: int, available_elixir: int) -> bool:
    req = get_required_resources(target_level)
    if available_gold < req["gold"]:
        return False
    if req["elixir"] is not None and available_elixir < req["elixir"]:
        return False
    return True
```

Esto reemplaza cualquier `if level >= 9` por una consulta directa a la tabla,
que es la fuente de verdad real y evita suposiciones incorrectas sobre en qué
nivel empieza a requerirse cada recurso.

## Extrapolación para niveles ≥ 19 (N ≥ 19)

Mantener la extrapolación dinámica ya implementada (`WallTransition` genérico
para `N → N+1`), pero advertir en el código con un comentario que los valores
reales de Supercell para niveles por encima del 19 pueden no seguir un patrón
puramente lineal/geométrico, así que la extrapolación es una aproximación y
debe marcarse como tal (por ejemplo un flag `is_extrapolated: bool` en el
resultado), para que el bot no tome decisiones de gasto masivo basándose en un
número inventado sin saberlo.

## Qué corregir, puntualmente

1. En `core/wall_upgrader.py`, eliminar la condición `elixir solo desde nivel
   9` y reemplazarla por lectura directa de `WALL_LEVEL_COSTS[nivel]["elixir_cost"]`.
2. Actualizar `calculate_batch_cost` para sumar oro y elixir por separado nivel
   a nivel según la tabla, no asumiendo un único recurso por rango de niveles.
3. Agregar un test unitario que verifique explícitamente:
   - Niveles 1-4: `elixir_cost is None`.
   - Nivel 5 en adelante: `elixir_cost` tiene un valor numérico igual al de la
     tabla oficial.
   - Que `can_afford()` rechace correctamente una mejora de nivel 5+ si falta
     elixir, aunque sobre oro.
4. No tocar `assets/buttons/wall_row_generic.png` ni el fallback a
   `wall_row_item.png` — esa parte de la tarea anterior quedó bien y no se
   relaciona con este bug de recursos.

## Entregable esperado

- `core/wall_upgrader.py` corregido con la tabla de costos como fuente única
  de verdad (ya sea inline como diccionario, o cargada desde
  `config/wall_costs.json`).
- Tests actualizados que cubran específicamente la regla real de activación
  de elixir en nivel 5, no nivel 9.
- Breve resumen de qué se cambió respecto a la implementación anterior.
