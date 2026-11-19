---
name: mission
description: >-
  Acción maestra unificada que integra /teamwork, /goal, /plan, /boost y /grill-me en un solo flujo
  de desarrollo autónomo de máxima potencia. Úsalo cuando el usuario escriba /mission o requiera
  alineación interactiva previa, planificación rigurosa, orquestación de agentes especializados
  y ejecución autónoma hasta completar y verificar el 100% del objetivo.
---

# /mission - Modo Unificado de Desarrollo Autónomo

Esta habilidad combina las 5 herramientas avanzadas de Antigravity en una sola acción maestra:

1. **Alineación Interactiva (`/grill-me`)**:
   - Si existen decisiones de diseño, ambigüedades o alternativas técnicas, entrevista al usuario paso a paso (resolviendo una decisión a la vez o usando `ask_question`).
   - Si la tarea ya está completamente especificada o el usuario indica `[skip-interview]`, avanza directamente a la fase técnica.

2. **Diseño y Planificación Técnica (`/plan`)**:
   - Genera un artefacto de plan de implementación detallado (`implementation_plan.md`) con:
     - Objetivos y análisis de causas raíz.
     - Diagramas de arquitectura o flujo (Mermaid).
     - Diffs explícitos y propuestas de código por archivo/módulo.
     - Plan riguroso de verificación (pruebas automatizadas y manuales).

3. **Orquestación Multi-Agente y Razonamiento Profundo (`/boost` & `/teamwork`)**:
   - Para tareas complejas de codificación o investigación, delega a subagentes especializados (`DeepCoder`, `DeepInvestigator` o el equipo de `teamwork`).
   - Aplica revisión adversarial y contraste de perspectivas para garantizar que no existan cabos sueltos ni degradaciones en el código.

4. **Ejecución Imparable hasta Completar el Objetivo (`/goal`)**:
   - Opera de forma autónoma y continua sin detenerse prematuramente.
   - Audita y verifica activamente cada cambio en el entorno real (compilación, pruebas unitarias, diagnóstico visual o ejecución en dispositivo).
   - Solo concluye cuando todas las metas estén cumplidas y validadas, emitiendo el resumen y el walkthrough final.
