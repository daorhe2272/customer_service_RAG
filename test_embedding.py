from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

print("🔐 Cargando modelo de embeddings...")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

texto = "Ejemplo de política de devoluciones en un ecommerce de moda."

print("🧠 Generando embedding...")
vector = embedding_model.embed_query(texto)

print("✅ Vector generado (primeros valores):")
print(vector[:5])
