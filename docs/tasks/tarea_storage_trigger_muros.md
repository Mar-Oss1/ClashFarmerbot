# Tarea: Trigger de mejora de muros basado en % de capacidad de almacenes

## Objetivo

Agregar una nueva señal de decisión al bot: cuando el oro o el elixir
disponibles se acerquen al límite de almacenamiento del jugador, priorizar
gastar ese recurso en mejoras de muro (`core/wall_upgrader.py`) antes de que
se desperdicie loot por almacén lleno.

## Datos de referencia (fuente: tablas oficiales de storages)

### Gold Storage y Elixir Storage (misma capacidad/HP/costo en ambos tipos,
solo cambia el ícono/recurso)

| Nivel | Capacidad | Hitpoints | Costo construir | Tiempo | TH requerido |
|---|---|---|---|---|---|
| 1  | 1,500      | 150   | 300        | 10s     | 1  |
| 2  | 3,000      | 300   | 750        | 2m      | 2  |
| 3  | 6,000      | 450   | 1,500      | 5m      | 2  |
| 4  | 12,000     | 600   | 3,000      | 15m     | 3  |
| 5  | 25,000     | 800   | 6,000      | 30m     | 3  |
| 6  | 45,000     | 1,000 | 12,000     | 1h      | 3  |
| 7  | 100,000    | 1,250 | 25,000     | 2h      | 4  |
| 8  | 225,000    | 1,500 | 50,000     | 3h      | 4  |
| 9  | 450,000    | 1,700 | 100,000    | 4h      | 5  |
| 10 | 850,000    | 1,900 | 250,000    | 5h      | 6  |
| 11 | 1,750,000  | 2,100 | 500,000    | 6h      | 7  |
| 12 | 2,000,000  | 2,500 | 1,000,000  | 12h     | 11 |
| 13 | 3,000,000  | 2,900 | 1,800,000  | 18h     | 12 |
| 14 | 4,000,000  | 3,300 | 2,800,000  | 1d      | 13 |
| 15 | 4,500,000  | 3,700 | 3,000,000  | 2d      | 14 |
| 16 | 5,000,000  | 3,900 | 4,000,000  | 3d      | 15 |
| 17 | 5,500,000  | 4,050 | 5,500,000  | 4d      | 16 |
| 18 | 6,000,000  | 4,200 | 10,000,000 | 5d      | 17 |
| 19 | 6,500,000  | 4,300 | 18,000,000 | 12d 12h | 18 |

### Dark Elixir Storage (capacidad distinta, recurso independiente)

| Nivel | Capacidad | Hitpoints | Costo construir (elixir) | Tiempo | TH requerido |
|---|---|---|---|---|---|
| 1  | 10,000  | 2,000 | 250,000    | 8h      | 7  |
| 2  | 17,500  | 2,200 | 500,000    | 16h     | 7  |
| 3  | 40,000  | 2,400 | 1,000,000  | 1d      | 8  |
| 4  | 75,000  | 2,600 | 1,500,000  | 1d 12h  | 8  |
| 5  | 140,000 | 2,900 | 2,000,000  | 1d 16h  | 9  |
| 6  | 180,000 | 3,200 | 2,400,000  | 2d      | 9  |
| 7  | 220,000 | 3,500 | 3,800,000  | 3d      | 12 |
| 8  | 280,000 | 3,800 | 5,400,000  | 3d 12h  | 13 |
| 9  | 330,000 | 4,100 | 6,600,000  | 4d      | 14 |
| 10 | 360,000 | 4,300 | 8,000,000  | 4d 12h  | 15 |
| 11 | 390,000 | 4,500 | 10,000,000 | 5d      | 16 |
| 12 | 420,000 | 4,700 | 16,000,000 | 7d      | 17 |
| 13 | 450,000 | 4,800 | 25,000,000 | 13d     | 18 |

### Cantidad de Gold Storage / Elixir Storage disponibles por nivel de TH

| TH | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9  | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cantidad | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 3 | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4  | 4  |

### Cantidad de Dark Elixir Storage disponibles por nivel de TH

| TH | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cantidad | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  | 1  |

## Reglas de negocio a implementar

1. **Capacidad total por recurso (gold o elixir) = suma de la capacidad
   individual de cada storage que el jugador tiene construido**, leída del
   nivel real de cada storage (NO asumir que todos están al mismo nivel; el
   bot debe leer/trackear el nivel de cada storage individual si es posible;
   si no es posible distinguirlos visualmente, usar como aproximación
   `capacidad_del_nivel_mas_bajo_detectado × cantidad_disponible_en_este_TH`
   como piso conservador, y documentar esa limitación en el código).

2. **El trigger de 90% se evalúa de forma INDEPENDIENTE por tipo de recurso**:
   - `gold_pct = gold_actual / capacidad_total_gold`
   - `elixir_pct = elixir_actual / capacidad_total_elixir`
   - `dark_elixir_pct = dark_elixir_actual / capacidad_total_dark_elixir`
   No existe un "90% global" combinado — cada recurso dispara su propia señal.

3. **El Elixir Oscuro NUNCA debe disparar una mejora de muro.** Los muros solo
   consumen oro y elixir normal (confirmado en la tabla de costos de muro ya
   implementada en `core/wall_upgrader.py`). Si `dark_elixir_pct >= 0.90`, esa
   señal debe dirigirse a otra lógica futura (tropas/hechizos oscuros), nunca
   a `wall_upgrader.py`. Dejar esto explícito en el código con un comentario,
   para que no se conecte por error.

4. **Trigger de mejora de muro:**
   - Si `gold_pct >= 0.90` → priorizar gastar oro en la siguiente mejora de
     muro más barata/disponible que use oro (consultar `WALL_LEVEL_COSTS`).
   - Si `elixir_pct >= 0.90` → priorizar gastar elixir en la siguiente mejora
     de muro disponible que use elixir (niveles 5+ en la tabla de muros).
   - Si ambos superan el umbral al mismo tiempo, priorizar la mejora que
     consuma AMBOS recursos si existe un muro pendiente en nivel ≥5 (así se
     gastan los dos excedentes en una sola operación).
   - El umbral `0.90` debe ser una constante configurable
     (`STORAGE_UPGRADE_TRIGGER_THRESHOLD = 0.90` en `config/settings.json`),
     no un valor cableado en el código, para poder ajustarlo después sin tocar
     lógica.

5. **No disparar la mejora si no hay un muro pendiente de subir** (todos al
   nivel máximo alcanzable para el TH actual) — en ese caso la señal de 90%
   debe loguearse pero no intentar ninguna acción, para no romper el bot
   intentando una mejora inexistente.

## Qué NO hacer

- No mezclar la lógica de Elixir Oscuro con `wall_upgrader.py`.
- No asumir que todos los storages de un mismo tipo están siempre al mismo
  nivel — dejar esa limitación documentada si no se puede resolver ahora.
- No cablear el umbral 90% directamente en el código; debe ser configurable.

## Entregable esperado

1. Tabla de datos de storages (gold/elixir y dark elixir) y cantidades por TH,
   como estructura reutilizable (similar a `WALL_LEVEL_COSTS`), por ejemplo en
   `core/storage_cost_table.py`.
2. Función que calcule `gold_pct`, `elixir_pct` y `dark_elixir_pct` a partir
   del estado actual leído por el bot (ledger.py) y la capacidad total.
3. Integración del trigger de 90% con `core/wall_upgrader.py`, respetando que
   el Elixir Oscuro nunca dispara mejoras de muro.
4. Constante configurable del umbral en `config/settings.json`.
5. Tests unitarios que cubran: umbral no alcanzado (no acción), umbral de oro
   alcanzado solo, umbral de elixir alcanzado solo, ambos alcanzados a la vez,
   y confirmación de que un umbral alto de elixir oscuro NO dispara ninguna
   mejora de muro.
