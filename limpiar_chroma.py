import os
import stat
import shutil

def eliminar_carpeta_segura(path):
    def onerror(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    shutil.rmtree(path, onerror=onerror)

# Ruta
db_path = "./chroma_db"

if os.path.exists(db_path):
    eliminar_carpeta_segura(db_path)
    print(f"🗑️ ChromaDB eliminada en {db_path}")
else:
    print(f"⚠️ No existe ChromaDB en {db_path}")
