"""
Standalone utility to index the policies document into ChromaDB.
Run from the project root: python scripts/indexar.py
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import dividir_y_indexar_texto
import asyncio

async def main():
    policies_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "policies", "politicas.txt"
    )
    with open(policies_path, "r", encoding="utf-8") as f:
        texto = f.read()

    total = await dividir_y_indexar_texto(texto, "politicas.txt")
    print(f"{total} fragmentos indexados.")

if __name__ == "__main__":
    asyncio.run(main())