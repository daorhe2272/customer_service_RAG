import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Inicializa el cliente de Chroma
client = chromadb.Client()
collection = client.get_or_create_collection(name="politicas")

def cargar_texto_y_indexar(nombre_archivo: str):
    with open(nombre_archivo, "r", encoding="utf-8") as f:
        texto = f.read()

    # Dividir texto en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(texto)

    # Embeddings con Gemini
    emb_fn = gen_embeddings.embed_content

    for i, chunk in enumerate(chunks):
        vector = emb_fn(chunk, model="models/embedding-001", task_type="retrieval_document")
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[f"chunk_{i}"]
        )
    print(f"{len(chunks)} fragmentos indexados.")

def buscar_contexto(pregunta: str, k=3):
    vector = gen_embeddings.embed_content(pregunta, model="models/embedding-001", task_type="retrieval_query")
    resultados = collection.query(query_embeddings=[vector], n_results=k)
    documentos = resultados["documents"][0]
    return "\n".join(documentos)
