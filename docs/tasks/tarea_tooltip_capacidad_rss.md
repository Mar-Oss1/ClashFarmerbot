# Tarea: Leer capacidad máxima real por tooltip en vez de estimarla

## Contexto y motivo del cambio

En la tarea anterior (`tarea_storage_trigger_muros.md`) se dejó como limitación
conocida que el bot no puede saber con certeza el nivel real de cada storage
individual, por lo que la capacidad total se estimaba combinando tablas de
nivel × cantidad disponible por TH. **Esa limitación ya no aplica**: el juego
expone directamente la capacidad máxima real al tocar/dar clic sobre el icono
de cada tipo de recurso en la barra superior. Al tocarlo se despliega un
tooltip/label con 3 datos:

1. **Capacidad máxima** (del recurso correspondiente, considerando todos sus
   storages combinados — el número ya viene calculado por el juego).
2. **Producción por hora** (de collectors/mines de ese recurso).
3. **In Treasury** (cantidad guardada en el Castillo del Clan para ese
   recurso, dato totalmente aparte del storage normal).

Esto reemplaza la necesidad de usar `WALL_LEVEL_COSTS`/tablas de storage para
estimar capacidad — en vez de eso, se lee el valor real directamente del
juego, que es mucho más preciso.

## Objetivo

Agregar al bot la capacidad de tocar cada icono de recurso en la barra
superior, capturar el tooltip desplegado, hacer OCR sobre sus 3 valores, y
usar la **Capacidad máxima** real (no estimada) para calcular el porcentaje de
llenado de cada recurso y disparar el trigger de mejora de muros ya definido
(umbral configurable, 90% por defecto).

## Flujo esperado

Para cada tipo de recurso (oro, elixir, elixir oscuro):

1. Tocar el icono/zona del recurso en la barra superior (usar las ROIs ya
   definidas en `SETTINGS["rois"]["own_resources"]`, o el sub-ROI específico
   de cada icono si existe distinción por tipo).
2. Esperar un breve delay (el tooltip tarda en renderizarse) y capturar un
   screenshot de la zona donde aparece el tooltip desplegado.
3. Aplicar OCR (reutilizar la lógica de `resource_reader.py`, extendiéndola si
   hace falta un nuevo set de templates/dígitos o un motor OCR genérico, ya
   que el tooltip puede incluir texto además de números) sobre las 3 líneas:
   - `Capacidad máxima: <número>`
   - `Producción por hora: <número>`
   - `In Treasury: <número>`
4. Cerrar el tooltip (toque fuera de la zona, o el mismo icono de nuevo según
   corresponda) para no dejar la UI bloqueada antes de continuar el flujo
   normal del bot.
5. Repetir para los 3 recursos (oro, elixir, elixir oscuro).
6. Calcular:
   ```
   gold_pct   = gold_actual   / capacidad_maxima_oro_leida
   elixir_pct = elixir_actual / capacidad_maxima_elixir_leida
   dark_pct   = dark_actual   / capacidad_maxima_dark_leida
   ```
   donde `gold_actual`, `elixir_actual`, `dark_actual` siguen viniendo de la
   lectura normal de recursos ya implementada (no del tooltip).

## Reglas de negocio (se mantienen de la tarea anterior, sin cambios)

- El trigger de `STORAGE_UPGRADE_TRIGGER_THRESHOLD` (90% por defecto,
  configurable en `config/settings.json`) se evalúa de forma **independiente**
  por tipo de recurso.
- El **Elixir Oscuro nunca dispara mejoras de muro** — solo gold/elixir normal
  alimentan `core/wall_upgrader.py`. El dato de `dark_pct` queda disponible
  para una lógica futura (tropas/hechizos oscuros), no para muros.
- El dato de **"In Treasury"** (tesorería del Clan Castle) es informativo por
  ahora — no debe sumarse a la capacidad del storage normal ni afectar el
  cálculo de `*_pct`, ya que es un recurso que vive en un contenedor distinto
  con reglas de donación propias. Guardarlo en el ledger para uso futuro, pero
  no mezclarlo en esta fórmula.
- Si no hay una mejora de muro pendiente (todos al nivel máximo para el TH
  actual), loguear la señal del 90% pero no ejecutar ninguna acción.

## Consideraciones técnicas a resolver

1. **Posición y layout del tooltip**: puede variar según si se abre desde el
   lado izquierdo o derecho de la pantalla (para no salirse del borde) — el
   script de captura debe determinar dinámicamente el área a leer, no asumir
   coordenadas fijas absolutas si el popup se reposiciona.
2. **OCR mixto texto+número**: a diferencia del lector de recursos actual
   (solo dígitos con templates), aquí el tooltip trae etiquetas de texto fijas
   (siempre dicen "Capacidad máxima", "Producción por hora", "In Treasury" en
   el idioma configurado del juego) — se puede usar las etiquetas de texto
   como ancla posicional fija y aplicar el mismo motor de OCR de dígitos solo
   sobre la región numérica a la derecha de cada etiqueta, evitando necesitar
   un OCR de texto libre completo.
3. **Costo de tiempo por ciclo**: tocar y cerrar 3 tooltips añade latencia al
   loop principal del bot. Definir con qué frecuencia se debe revisar esto
   (ej. cada N minutos o cada vez que el bot vuelve a la vista de aldea
   principal tras un ataque) en vez de hacerlo en cada frame, para no
   ralentizar el resto del state machine.
4. **Fallback si el tooltip no se detecta** (popup no abrió, OCR falló): no
   romper el bot — simplemente omitir la actualización de capacidad máxima en
   ese ciclo y reintentar en el siguiente, sin bloquear el resto del flujo.

## Qué NO hacer

- No seguir usando la estimación de capacidad vía tabla de storages × cantidad
  como fuente principal — queda solo como fallback opcional si el tooltip no
  se puede leer, documentado como tal en el código.
- No mezclar el valor de "In Treasury" con la capacidad de storage normal.
- No disparar mejoras de muro desde el porcentaje de Elixir Oscuro.

## Entregable esperado

1. Nueva función/método en `core/device.py` o `core/vision.py` (según dónde
   encaje mejor con el patrón existente) para tocar el icono de recurso, capturar
   el tooltip, y hacer OCR sobre sus 3 valores.
2. Actualización de `core/ledger.py` para almacenar `capacidad_maxima_<rss>`,
   `produccion_hora_<rss>` e `in_treasury_<rss>` por tipo de recurso.
3. Actualización del trigger en `core/wall_upgrader.py` para calcular
   `*_pct` usando la capacidad máxima real leída del tooltip (con fallback a
   la tabla estimada de la tarea anterior solo si la lectura falla).
4. Tests que cubran: lectura exitosa de los 3 valores, fallback cuando el
   tooltip no se detecta, y que el Elixir Oscuro nunca dispare
   `wall_upgrader.py` aunque su `dark_pct` supere el umbral.
5. Breve nota de qué coordenadas/ROIs se usaron para tocar cada icono, para
   que puedas ajustarlas si no coinciden exactamente con tu dispositivo.
