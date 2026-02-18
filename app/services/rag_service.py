from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from chromadb import HttpClient
import asyncio
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

client = HttpClient(host="localhost", port=8001)
collection = client.get_or_create_collection(name="documentos")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

async def dividir_y_indexar_texto(texto: str, file_id: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(texto)

    print(f"Texto dividido en {len(chunks)} fragmentos.")

    for i, chunk in enumerate(chunks):
        try:
            print(f"Generando embedding para chunk {i}...")
            vector = await asyncio.to_thread(embedding_model.embed_query, chunk)
            print(f"Vector (chunk {i}): {vector[:5]}")
            chunk_id = f"{file_id}_chunk_{i}"

            try:
                collection.delete(ids=[chunk_id])
                print(f"Chunk duplicado eliminado: {chunk_id}")
            except Exception as delete_error:
                print(f"No se pudo eliminar chunk previo (puede no existir): {delete_error}")

            try:
                vector = list(map(float, vector))
                chunk_clean = chunk.replace("\x00", " ").strip()
                print(f"Longitud del vector: {len(vector)}")
                collection.add(
                    documents=[chunk_clean],
                    embeddings=[vector],
                    ids=[chunk_id]
                )
                print(f"Chunk {i} indexado.")
            except Exception as add_error:
                print(f"Error al agregar a Chroma en chunk {i}: {add_error}")
                raise

        except Exception as e:
            print(f"Error en chunk {i}: {e}")
            raise

    return len(chunks)

def buscar_contexto(pregunta: str, k: int = 5):
    query_vector = embedding_model.embed_query(pregunta)
    resultados = collection.query(query_embeddings=[query_vector], n_results=k)
    documentos = resultados.get("documents", [[]])[0]
    print(f"Fragmentos recuperados: {len(documentos)}")
    print(f"Fragmentos: {documentos}")
    if not documentos:
        return "No se encontraron fragmentos relevantes."
    return "\n".join(documentos)