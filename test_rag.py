from rag_service import buscar_contexto

pregunta = "¿Cuál es la política de devoluciones?"
print("🔍 Buscando contexto...")
contexto = buscar_contexto(pregunta)
print("✅ Contexto encontrado:")
print(contexto)
