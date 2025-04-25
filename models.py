from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Conversacion(Base):
    __tablename__ = 'conversaciones'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True)
    role = Column(String(20))  # 'user' o 'assistant'
    contenido = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Configuración de SQLite
DATABASE_URL = 'sqlite:///historial.db'
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Crear las tablas
Base.metadata.create_all(engine)