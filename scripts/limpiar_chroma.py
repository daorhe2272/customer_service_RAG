"""
Utility to safely delete the local ChromaDB storage folder.
Run from the project root: python scripts/limpiar_chroma.py
"""
import os
import stat
import shutil

def eliminar_carpeta_segura(path):
    def onerror(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    shutil.rmtree(path, onerror=onerror)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(_BASE_DIR, "storage", "chroma_db")

if os.path.exists(db_path):
    eliminar_carpeta_segura(db_path)
    print(f"ChromaDB eliminada en {db_path}")
else:
    print(f"No existe ChromaDB en {db_path}")