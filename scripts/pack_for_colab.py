"""
scripts/pack_for_colab.py
Empaquetador automatizado para entrenamiento en Google Colab con GPU T4.

Acciones:
1. Comprime datasets/coc_walls_unet/ (train/ y val/) en colab_bundle/coc_walls_unet.zip.
2. Copia tools/train_wall_unet.py y tools/colab_train_walls.ipynb en colab_bundle/.
3. Agrega colab_bundle/ a .gitignore si aún no está presente.
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pack_colab_bundle() -> int:
    bundle_dir = PROJECT_ROOT / "colab_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = PROJECT_ROOT / "datasets" / "coc_walls_unet"
    if not dataset_dir.is_dir():
        print(f"❌ Error: No se encontró el dataset en {dataset_dir}")
        return 1

    zip_path = bundle_dir / "coc_walls_unet.zip"
    print(f"📦 Comprimiendo {dataset_dir} -> {zip_path}...")

    total_files = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                file_path = Path(root) / file
                rel_to_dataset = file_path.relative_to(dataset_dir)
                arcname = f"coc_walls_unet/{rel_to_dataset.as_posix()}"
                zf.write(file_path, arcname=arcname)
                total_files += 1

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✅ Dataset comprimido ({total_files} archivos, {zip_size_mb:.2f} MB)")

    # 2. Copiar train_wall_unet.py y colab_train_walls.ipynb
    script_src = PROJECT_ROOT / "tools" / "train_wall_unet.py"
    nb_src = PROJECT_ROOT / "tools" / "colab_train_walls.ipynb"

    if not script_src.exists():
        print(f"❌ Error: No se encontró {script_src}")
        return 1
    if not nb_src.exists():
        print(f"❌ Error: No se encontró {nb_src}")
        return 1

    script_dst = bundle_dir / "train_wall_unet.py"
    nb_dst = bundle_dir / "colab_train_walls.ipynb"

    shutil.copy2(script_src, script_dst)
    shutil.copy2(nb_src, nb_dst)
    print(f"✅ Archivo copiado: {script_dst.name} ({script_dst.stat().st_size / 1024:.1f} KB)")
    print(f"✅ Archivo copiado: {nb_dst.name} ({nb_dst.stat().st_size / 1024:.1f} KB)")

    # 3. Actualizar .gitignore
    gitignore_path = PROJECT_ROOT / ".gitignore"
    rule = "colab_bundle/"
    rule_added = False

    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        if rule not in content:
            if not content.endswith("\n"):
                content += "\n"
            content += f"\n# Bundle temporal para entrenamiento en Google Colab\n{rule}\n"
            gitignore_path.write_text(content, encoding="utf-8")
            rule_added = True
    else:
        gitignore_path.write_text(f"{rule}\n", encoding="utf-8")
        rule_added = True

    if rule_added:
        print(f"🔒 Regla '{rule}' agregada a .gitignore")
    else:
        print(f"ℹ️ '{rule}' ya existía en .gitignore")

    print("\n" + "=" * 70)
    print("🎉 BUNDLE PARA GOOGLE COLAB PREPARADO EXITOSAMENTE")
    print("=" * 70)
    print(f"Carpeta: {bundle_dir}")
    print("\nPasos para ejecutar en Google Colab con GPU T4:")
    print("1. En tu Google Drive, crea la carpeta: ClashFarmerbot_train")
    print("2. Sube a esa carpeta los 2 archivos de colab_bundle/:")
    print("   - coc_walls_unet.zip")
    print("   - train_wall_unet.py")
    print("3. Abre colab_train_walls.ipynb en Google Colab (https://colab.research.google.com).")
    print("4. Selecciona: Entorno de ejecución -> Cambiar tipo de entorno de ejecución -> T4 GPU.")
    print("5. Ejecuta las 4 celdas en orden.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(pack_colab_bundle())
