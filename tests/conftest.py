"""
Pytest configuration and shared fixtures for RAG evaluation tests.
"""

import pytest
import json
import os
import sys
import uuid
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import buscar_contexto, collection
from app.services.gemini_service import generar_respuesta_con_contexto
from app.models.models import SessionLocal, Conversacion
from evaluators.llm_judge import judge_model


@pytest.fixture(scope="session")
def eval_dataset() -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset from JSON file.
    Shared across all tests in the session.
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        'fixtures',
        'eval_dataset.json'
    )

    with open(fixture_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"\n✓ Loaded {len(dataset)} evaluation cases")
    return dataset


@pytest.fixture(scope="session")
def judge_llm():
    """
    Provide the LLM judge model for evaluation.
    Uses JUDGE_LLM environment variable (defaults to gemini-2.5-flash).
    """
    model_name = os.getenv("JUDGE_LLM", "gemini-2.5-flash")
    print(f"\n✓ Using judge model: {model_name}")
    return judge_model


@pytest.fixture
def retrieval_function():
    """
    Wrapper around buscar_contexto that returns chunk IDs for metric calculation.
    """
    def retrieve(question: str, k: int = 5) -> List[str]:
        """
        Retrieve chunk IDs for a given question.
        """
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        query_vector = embedding_model.embed_query(question)
        resultados = collection.query(query_embeddings=[query_vector], n_results=k)

        chunk_ids = resultados.get("ids", [[]])[0]
        return chunk_ids

    return retrieve


@pytest.fixture
def rag_pipeline():
    """
    Full RAG pipeline: retrieval + generation.
    """
    def get_response(question: str, history: str = "") -> Dict[str, Any]:
        """
        Get full RAG response with context and metadata.

        Returns:
            Dict with keys:
                - question: Original question
                - context: Retrieved context
                - response: Generated response
                - timestamp: When response was generated
        """
        context = buscar_contexto(question, k=5)
        response = generar_respuesta_con_contexto(question, context, history)

        return {
            'question': question,
            'context': context,
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        }

    return get_response


@pytest.fixture
def conversational_rag():
    """
    Conversational RAG pipeline with session management.
    """
    def get_conversational_response(session_id: str, question: str) -> Dict[str, Any]:
        """
        Get response in a conversational context with history.

        Returns:
            Dict with keys:
                - session_id
                - question
                - context
                - history
                - response
                - timestamp
        """
        db = SessionLocal()

        # Add user message to history
        db.add(Conversacion(session_id=session_id, role="user", contenido=question))
        db.commit()

        # Retrieve conversation history
        historial = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
        historial_conversacion = ""
        for mensaje in historial:
            historial_conversacion += f"{mensaje.role.capitalize()}: {mensaje.contenido}\n"

        # Get context and generate response
        context = buscar_contexto(question)
        response = generar_respuesta_con_contexto(question, context, historial_conversacion)

        # Add assistant response to history
        db.add(Conversacion(session_id=session_id, role="assistant", contenido=response))
        db.commit()
        db.close()

        return {
            'session_id': session_id,
            'question': question,
            'context': context,
            'history': historial_conversacion,
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        }

    return get_conversational_response


@pytest.fixture
def test_session_id():
    """
    Generate a unique test session ID for conversational tests.
    """
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_test_sessions():
    """
    Cleanup fixture to remove test sessions from database after tests.
    Yields control to test, then cleans up afterward.
    """
    session_ids = []

    def register_session(session_id: str):
        """Register a session ID for cleanup."""
        session_ids.append(session_id)

    yield register_session

    # Cleanup after test
    if session_ids:
        db = SessionLocal()
        for session_id in session_ids:
            db.query(Conversacion).filter_by(session_id=session_id).delete()
        db.commit()
        db.close()
        print(f"\n✓ Cleaned up {len(session_ids)} test sessions")


@pytest.fixture(scope="session")
def eval_results_dir():
    """
    Create directory for storing evaluation results.
    """
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'eval_results'
    )
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n✓ Eval results directory: {results_dir}")
    return results_dir


def pytest_configure(config):
    """
    Pytest configuration hook.
    """
    # Add custom markers
    config.addinivalue_line(
        "markers", "retrieval: tests for retrieval quality (deterministic)"
    )
    config.addinivalue_line(
        "markers", "faithfulness: tests for answer grounding (LLM judge)"
    )
    config.addinivalue_line(
        "markers", "correctness: tests for factual correctness (LLM judge)"
    )
    config.addinivalue_line(
        "markers", "scope: tests for scope enforcement (LLM judge)"
    )
    config.addinivalue_line(
        "markers", "quality: tests for response quality (LLM judge)"
    )
    config.addinivalue_line(
        "markers", "conversational: tests for multi-turn conversations (LLM judge)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their file names.
    """
    for item in items:
        # Mark based on test file name
        if "retrieval" in item.nodeid:
            item.add_marker(pytest.mark.retrieval)
        if "faithfulness" in item.nodeid:
            item.add_marker(pytest.mark.faithfulness)
        if "correctness" in item.nodeid:
            item.add_marker(pytest.mark.correctness)
        if "scope" in item.nodeid:
            item.add_marker(pytest.mark.scope)
        if "quality" in item.nodeid:
            item.add_marker(pytest.mark.quality)
        if "conversational" in item.nodeid:
            item.add_marker(pytest.mark.conversational)