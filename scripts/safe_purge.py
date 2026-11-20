#!/usr/bin/env python3
"""
scripts/safe_purge.py
=====================
Script de saneamiento seguro para el repositorio ClashFarmerbot.

Comportamiento:
- Por defecto ejecuta en modo simulación (--dry-run), mostrando todos los
  archivos y carpetas identificados como PURGABLE y el espacio total a liberar.
- Con el flag --execute, mueve los elementos identificados a la carpeta
  '_quarantine/' en la raíz del proyecto SIN usar comandos destructivos ('rm' permanente).
- Asegura que los directorios runtime esenciales (ej. logs/ y runs/) sigan existiendo vacíos.
- Actualiza automáticamente el archivo .gitignore con las reglas de exclusión
  para evitar acumular basura, grabaciones pesadas de vídeo o contextos autogenerados.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Patrones / elementos catalogados como PURGABLE
# (logs viejos, debug images, backups duplicados, scratch, videos pesados, dumps de contexto)
PURGABLE_DIRECTORIES = [
    # Subdirectorios pesados en logs/ (> 8.5 GB)
    "logs/yolo",
    "logs/runs",
    "logs/diagnostics",
    "logs/debug_captures",
    "logs/tooltips",
    "logs/fails",
    "logs/findings",
    "logs/live_tests",

    # Subdirectorios en runs/ (salvo archivos de modelo)
    "runs/detect",
    "runs/segment",

    # Carpeta completa de pruebas y experimentos sueltos (> 790 MB)
    "scratch",

    # Backups redundantes desactualizados (> 1.7 MB)
    "backups/backup_jump_optimizer_20261114_170528",

    # Assets obsoletos / archivados
    "assets/_archive_buttons",
    "assets/custom_templates",
]

PURGABLE_ROOT_FILES = [
    # Dumps masivos de contexto autogenerados
    "proyecto_contexto.txt",
    "core_vision_contexto.txt",
    "clashfarmerbot_context.md",
    "muros_context.md",
    "muros_only.md",
]

PURGABLE_MODELS = [
    # Modelo genérico base sin referencias en runtime ni entrenamiento
    "models/yolov8n.pt",
]

GITIGNORE_ENTRIES = [
    "# Cuarentena y archivos temporales pesados",
    "_quarantine/",
    "_trash*/",
    "_backups*/",
    "*.h264",
    "*.mkv",
    "*.mp4",
    "*_context.md",
    "muros_context.md",
    "muros_only.md",
]


def get_dir_size_and_count(path: Path) -> Tuple[int, int]:
    """Retorna (cantidad_archivos, tamaño_en_bytes) de un directorio o archivo."""
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size

    count = 0
    total_size = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            if fp.is_file():
                count += 1
                try:
                    total_size += fp.stat().st_size
                except OSError:
                    pass
    return count, total_size


def collect_purgable_items(root: Path) -> List[Tuple[str, Path, int, int]]:
    """
    Identifica todos los elementos existentes marcados como PURGABLE.
    Retorna lista de tuplas: (categoría, path_origen, count_archivos, bytes_totales).
    """
    items = []

    # 1. Directorios purgables completos
    for rel_dir in PURGABLE_DIRECTORIES:
        p = root / rel_dir
        if p.exists():
            c, s = get_dir_size_and_count(p)
            items.append(("Directorio Purgable", p, c, s))

    # 2. Archivos sueltos en logs/ (imágenes debug .png/.jpg, json temporales, logs rotados)
    logs_dir = root / "logs"
    if logs_dir.exists():
        for f in logs_dir.iterdir():
            if f.is_file():
                # Conservar logs/clash_bot.log (log principal activo)
                if f.name == "clash_bot.log":
                    continue
                c, s = 1, f.stat().st_size
                items.append(("Log/Debug en logs/", f, c, s))

    # 3. Grabaciones y videos en runs/
    runs_dir = root / "runs"
    if runs_dir.exists():
        for f in runs_dir.iterdir():
            if f.is_file():
                # Conservar modelos como wall_unet_best.pt para revisión
                if f.suffix in (".h264", ".mp4", ".mkv"):
                    c, s = 1, f.stat().st_size
                    items.append(("Grabación de vídeo en runs/", f, c, s))

    # 4. Modelos sin uso
    for rel_m in PURGABLE_MODELS:
        p = root / rel_m
        if p.exists():
            c, s = get_dir_size_and_count(p)
            items.append(("Modelo sin uso", p, c, s))

    # 5. Archivos de contexto en raíz
    for rel_f in PURGABLE_ROOT_FILES:
        p = root / rel_f
        if p.exists():
            c, s = get_dir_size_and_count(p)
            items.append(("Contexto autogenerado en raíz", p, c, s))

    return items


def update_gitignore(root: Path, dry_run: bool = True) -> bool:
    """Verifica y añade las reglas de exclusión requeridas a .gitignore si faltan."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        current_content = ""
    else:
        current_content = gitignore_path.read_text(encoding="utf-8")

    missing = []
    for entry in GITIGNORE_ENTRIES:
        if entry.startswith("#"):
            continue
        if entry not in current_content:
            missing.append(entry)

    if not missing:
        print("  ✓ .gitignore ya contiene todas las reglas recomendadas.")
        return False

    print(f"  + Reglas a incorporar en .gitignore: {', '.join(missing)}")
    if not dry_run:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write("\n\n# Reglas agregadas automáticamente por safe_purge.py\n")
            for entry in GITIGNORE_ENTRIES:
                if entry not in current_content:
                    f.write(entry + "\n")
        print("  ✓ .gitignore actualizado exitosamente.")
    return True


def move_to_quarantine(src: Path, root: Path, quarantine_root: Path):
    """Mueve un archivo o carpeta preservando la estructura relativa dentro de _quarantine/."""
    rel_path = src.relative_to(root)
    dest = quarantine_root / rel_path

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()

    shutil.move(str(src), str(dest))


def main():
    parser = argparse.ArgumentParser(
        description="Mueve archivos y directorios PURGABLE a _quarantine/ de forma segura."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Simulación por defecto: inspecciona y calcula el tamaño sin mover nada.",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Ejecución real: mueve lo PURGABLE a _quarantine/ y actualiza .gitignore.",
    )

    args = parser.parse_args()
    dry_run = not args.execute

    print("=" * 80)
    print("🧹 ClashFarmerbot - Safe Repository Purge & Quarantine")
    print(f"Modo: {'SIMULACIÓN (--dry-run, ningún cambio será aplicado)' if dry_run else 'EJECUCIÓN REAL (--execute, moviendo a _quarantine/)'}")
    print(f"Directorio raíz: {PROJECT_ROOT}")
    print("=" * 80)

    items = collect_purgable_items(PROJECT_ROOT)

    total_files = sum(item[2] for item in items)
    total_bytes = sum(item[3] for item in items)
    total_mb = total_bytes / (1024 * 1024)

    print(f"\nSe identificaron {len(items)} entradas purgabiles ({total_files} archivos en total, ~{total_mb:.2f} MB):\n")

    for cat, p, count, b in items:
        rel = p.relative_to(PROJECT_ROOT)
        mb = b / (1024 * 1024)
        print(f"  [{cat:30}] {str(rel):45} -> {count:5d} arch, {mb:8.2f} MB")

    print("-" * 80)
    print(f"TOTAL ESTIMADO A CUARENTENAR: {total_files} archivos | {total_mb:.2f} MB (~{total_mb/1024:.2f} GB)")
    print("-" * 80)

    quarantine_dir = PROJECT_ROOT / "_quarantine"

    if dry_run:
        print("\n[INFO] Modo --dry-run activo. No se movió ningún archivo.")
        print("Revisión de .gitignore (simulada):")
        update_gitignore(PROJECT_ROOT, dry_run=True)
        print("\nPara proceder con la cuarentena segura y actualizar .gitignore, ejecuta:")
        print("    python scripts/safe_purge.py --execute\n")
    else:
        print(f"\n[ACCIÓN] Creando directorio de cuarentena: {quarantine_dir}")
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        for cat, p, count, b in items:
            if p.exists():
                print(f"  -> Moviendo a cuarentena: {p.relative_to(PROJECT_ROOT)}")
                move_to_quarantine(p, PROJECT_ROOT, quarantine_dir)

        # Recrear carpetas runtime esenciales vacías si fueron afectadas
        (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        (PROJECT_ROOT / "runs").mkdir(exist_ok=True)

        print("\nActualizando .gitignore:")
        update_gitignore(PROJECT_ROOT, dry_run=False)

        print("\n" + "=" * 80)
        print("✅ Cuarentena completada con éxito.")
        print(f"   Espacio aislado en _quarantine/: ~{total_mb:.2f} MB.")
        print("   Ningún archivo fue borrado de forma permanente.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
