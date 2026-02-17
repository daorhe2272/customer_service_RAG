"""
Basic test for the RAG retrieval pipeline.
Run from the project root: python tests/test_rag.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import buscar_contexto

pregunta = "¿Cuál es la política de devoluciones?"
print("Buscando contexto...")
contexto = buscar_contexto(pregunta)
print("Contexto encontrado:")
print(contexto)