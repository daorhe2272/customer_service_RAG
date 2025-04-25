import streamlit as st
import requests
import uuid

st.title("Agente Postventa IA - Conversacional")

# Crear un session_id único por usuario
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Mostrar el session_id actual
st.info(f"Session ID actual: {st.session_state.session_id}")

# Inicializar historial en frontend
if "historial" not in st.session_state:
    st.session_state.historial = []

# Botón para resetear historial local
if st.button("Resetear conversación local"):
    st.session_state.historial = []
    st.success("Conversación local reseteada.")

# Área para escribir la pregunta
pregunta = st.text_area("Escribe tu pregunta:")

if st.button("Preguntar"):
    if pregunta:
        with st.spinner("Consultando..."):
            payload = {
                "mensaje": pregunta,
                "session_id": st.session_state.session_id
            }
            response = requests.post("http://localhost:8000/preguntar-conversacional", json=payload)
            data = response.json()
            respuesta = data.get("respuesta", "Sin respuesta")

            # Guardar historial localmente (solo para mostrar)
            st.session_state.historial.append({"role": "user", "content": pregunta})
            st.session_state.historial.append({"role": "assistant", "content": respuesta})

# Mostrar historial de conversación local
if st.session_state.historial:
    st.subheader("Historial de conversación (local):")
    for mensaje in st.session_state.historial:
        if mensaje["role"] == "user":
            st.markdown(f"**Tú:** {mensaje['content']}")
        else:
            st.markdown(f"**IA:** {mensaje['content']}")

# Línea divisoria
st.markdown("---")

# Consultar historial desde el backend
st.subheader("Consultar historial desde el backend")

session_id_input = st.text_input("Introduce un session_id para consultar el historial:", value=st.session_state.session_id)

if st.button("Consultar historial desde el backend"):
    with st.spinner("Consultando historial desde el backend..."):
        response = requests.get(f"http://localhost:8000/historial/{session_id_input}")
        data = response.json()
        st.subheader(f"Historial completo de session_id: {session_id_input}")
        for mensaje in data.get("historial", []):
            timestamp = mensaje.get("timestamp", "")
            if mensaje["role"] == "user":
                st.markdown(f"**Tú ({timestamp}):** {mensaje['contenido']}")
            else:
                st.markdown(f"**IA ({timestamp}):** {mensaje['contenido']}")

# Línea divisoria
st.markdown("---")

# Subida de archivos (txt o pdf)
st.subheader("Subir archivos (.txt o .pdf):")
archivos = st.file_uploader("Selecciona archivos", type=["txt", "pdf"], accept_multiple_files=True)

if st.button("Subir archivos"):
    if archivos:
        with st.spinner("Subiendo archivos..."):
            files = [("files", (archivo.name, archivo.read())) for archivo in archivos]
            response = requests.post("http://localhost:8000/subir-multiples", files=files, timeout=60)
            data = response.json()
            st.success(f"Resultado: {data}")
