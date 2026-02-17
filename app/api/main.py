from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.services.rag_service import dividir_y_indexar_texto, buscar_contexto
from app.services.gemini_service import generar_respuesta_con_contexto
from app.models.models import SessionLocal, Conversacion
from sqlalchemy import func
import fitz  # PyMuPDF
import os
import json
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Consulta(BaseModel):
    mensaje: str

class ConversacionRequest(BaseModel):
    mensaje: str
    session_id: str

# Logs stored under storage/logs/
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOG_PATH = os.path.join(_BASE_DIR, "storage", "logs", "interacciones.jsonl")

def guardar_log(entry):
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def limpiar_sesiones_expiradas(db, horas_inactividad=2):
    limite = datetime.utcnow() - timedelta(hours=horas_inactividad)
    subq = db.query(
        Conversacion.session_id,
        func.max(Conversacion.timestamp).label("ultimo")
    ).group_by(Conversacion.session_id).subquery()

    sesiones_expiradas = db.query(subq).filter(subq.c.ultimo < limite).all()
    expiradas_ids = [row.session_id for row in sesiones_expiradas]

    if expiradas_ids:
        db.query(Conversacion).filter(Conversacion.session_id.in_(expiradas_ids)).delete(synchronize_session=False)
        db.commit()
        print(f"Sesiones eliminadas por inactividad: {expiradas_ids}")

@app.post("/subir-y-indexar")
async def subir_y_indexar(file: UploadFile = File(...)):
    print("Recibiendo archivo:", file.filename)
    contenido = await file.read()
    texto = contenido.decode("utf-8")

    total = await dividir_y_indexar_texto(texto, file.filename)
    print("Archivo procesado con éxito.")

    return {"mensaje": f"{total} fragmentos indexados de '{file.filename}' exitosamente."}

@app.post("/subir-multiples")
async def subir_multiples(files: List[UploadFile] = File(...)):
    resultados = []

    for file in files:
        try:
            print("Procesando:", file.filename)
            contenido = await file.read()
            nombre = file.filename.lower()

            if nombre.endswith(".pdf"):
                with fitz.open(stream=contenido, filetype="pdf") as doc:
                    texto = "\n".join([page.get_text() for page in doc])
            else:
                texto = contenido.decode("utf-8")

            total = await dividir_y_indexar_texto(texto, file.filename)
            resultados.append({"archivo": file.filename, "fragmentos": total})

        except Exception as e:
            resultados.append({"archivo": file.filename, "error": str(e)})

    return {"resultados": resultados}

@app.post("/preguntar")
async def preguntar(data: Consulta):
    pregunta = data.mensaje
    print("Buscando contexto para la pregunta:", pregunta)

    contexto = buscar_contexto(pregunta)
    print("Contexto recuperado:\n", contexto)

    respuesta = generar_respuesta_con_contexto(pregunta, contexto)
    return {"respuesta": respuesta}

@app.post("/preguntar-conversacional")
async def preguntar_conversacional(data: ConversacionRequest):
    session_id = data.session_id
    pregunta = data.mensaje

    db = SessionLocal()

    limpiar_sesiones_expiradas(db)

    db.add(Conversacion(session_id=session_id, role="user", contenido=pregunta))
    db.commit()

    guardar_log({
        "session_id": session_id,
        "role": "user",
        "contenido": pregunta,
        "timestamp": datetime.utcnow().isoformat()
    })

    historial = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
    historial_conversacion = ""
    for mensaje in historial:
        historial_conversacion += f"{mensaje.role.capitalize()}: {mensaje.contenido}\n"

    contexto = buscar_contexto(pregunta)

    respuesta = generar_respuesta_con_contexto(pregunta, contexto, historial_conversacion)

    db.add(Conversacion(session_id=session_id, role="assistant", contenido=respuesta))
    db.commit()
    db.close()

    guardar_log({
        "session_id": session_id,
        "role": "assistant",
        "contenido": respuesta,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {"respuesta": respuesta}


@app.get("/historial/{session_id}")
async def obtener_historial(session_id: str):
    db = SessionLocal()
    historial = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
    historial_format = [{
        "role": h.role,
        "contenido": h.contenido,
        "timestamp": h.timestamp.isoformat()
    } for h in historial]
    db.close()
    return {"session_id": session_id, "historial": historial_format}
