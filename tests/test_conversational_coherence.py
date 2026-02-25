"""
Tests for conversational coherence (multi-turn context awareness).
Uses LLM-as-judge to evaluate conversation quality.

Run: pytest tests/test_conversational_coherence.py -v
"""

import pytest
import uuid
from evaluators.llm_judge import evaluate_context_awareness


@pytest.mark.conversational
def test_multi_turn_context_maintenance(conversational_rag, cleanup_test_sessions, judge_llm):
    """
    Test that system maintains context across multiple conversation turns.
    """
    session_id = f"test_multi_turn_{uuid.uuid4().hex[:8]}"
    cleanup_test_sessions(session_id)

    print(f"\n{'='*60}")
    print(f"MULTI-TURN CONTEXT TEST")
    print(f"Session: {session_id}")
    print(f"{'='*60}")

    # Conversation scenario: returns inquiry with follow-ups
    conversation = [
        {
            'question': "¿Cómo puedo devolver un producto?",
            'should_maintain_context': True  # First question
        },
        {
            'question': "¿Y si lo compré hace 20 días?",
            'should_maintain_context': True  # Follows up on returns
        },
        {
            'question': "¿Necesito el recibo?",
            'should_maintain_context': True  # Still about returns
        },
        {
            'question': "¿Puedo hacerlo en cualquier tienda?",
            'should_maintain_context': True  # Still the same topic
        },
    ]

    failures = []

    for i, turn in enumerate(conversation, 1):
        question = turn['question']

        # Get conversational response
        result = conversational_rag(session_id, question)
        response = result['response']

        print(f"\nTurn {i}:")
        print(f"  User: {question}")
        print(f"  Agent: {response[:100]}...")

        # For turns 2+, evaluate context awareness
        if i > 1:
            # Get full conversation history
            from app.models.models import SessionLocal, Conversacion
            db = SessionLocal()
            history = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
            history_list = [{
                'role': h.role,
                'contenido': h.contenido
            } for h in history]
            db.close()

            # Judge context awareness
            judgment = evaluate_context_awareness(
                conversation_history=history_list,
                latest_response=response,
                model=judge_llm
            )

            print(f"  Context Maintained: {judgment['maintains_context']} (score: {judgment['context_score']:.2f})")

            if not judgment['maintains_context']:
                failures.append({
                    'turn': i,
                    'question': question,
                    'response': response,
                    'explanation': judgment['explanation']
                })

    if failures:
        print(f"\n{'='*60}")
        print(f"CONTEXT FAILURES ({len(failures)}):")
        print(f"{'='*60}")
        for fail in failures:
            print(f"\nTurn {fail['turn']}: {fail['question']}")
            print(f"  Issue: {fail['explanation']}")

    print(f"{'='*60}\n")

    # Should maintain context across all turns
    assert len(failures) == 0, (
        f"Lost context in {len(failures)} turns. "
        f"Conversational system should remember previous context."
    )


@pytest.mark.conversational
def test_pronoun_reference_resolution(conversational_rag, cleanup_test_sessions, judge_llm):
    """
    Test that system correctly handles pronouns and references.
    """
    session_id = f"test_pronouns_{uuid.uuid4().hex[:8]}"
    cleanup_test_sessions(session_id)

    print(f"\n{'='*60}")
    print(f"PRONOUN REFERENCE TEST")
    print(f"{'='*60}")

    # Conversation with pronouns
    conversation = [
        "Compré unos zapatos hace una semana",
        "¿Puedo devolverlos?",  # "los" refers to zapatos
        "¿Cuánto me tarda?",  # "me" - personal reference, "tarda" refers to devolución
    ]

    for i, question in enumerate(conversation, 1):
        result = conversational_rag(session_id, question)
        response = result['response']

        print(f"\nTurn {i}:")
        print(f"  User: {question}")
        print(f"  Agent: {response[:100]}...")

    # Get final conversation
    from app.models.models import SessionLocal, Conversacion
    db = SessionLocal()
    history = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
    history_list = [{'role': h.role, 'contenido': h.contenido} for h in history]
    db.close()

    # Evaluate context awareness
    latest_response = history_list[-1]['contenido']
    judgment = evaluate_context_awareness(
        conversation_history=history_list,
        latest_response=latest_response,
        model=judge_llm
    )

    print(f"\nContext Score: {judgment['context_score']:.2f}")
    print(f"Maintains Context: {judgment['maintains_context']}")
    print(f"{'='*60}\n")

    # Should handle pronouns correctly
    assert judgment['maintains_context'], (
        f"Failed to resolve pronoun references. "
        f"Explanation: {judgment['explanation']}"
    )


@pytest.mark.conversational
def test_topic_switch_handling(conversational_rag, cleanup_test_sessions, judge_llm):
    """
    Test that system handles topic switches gracefully.
    """
    session_id = f"test_topic_switch_{uuid.uuid4().hex[:8]}"
    cleanup_test_sessions(session_id)

    print(f"\n{'='*60}")
    print(f"TOPIC SWITCH TEST")
    print(f"{'='*60}")

    # Conversation with topic switch
    conversation = [
        "¿Cómo devuelvo un producto?",  # Topic: returns
        "Gracias. Ahora, ¿cuánto cuesta el envío?",  # Topic switch to shipping
        "¿Y cuánto tarda?",  # Should refer to shipping, not returns
    ]

    for i, question in enumerate(conversation, 1):
        result = conversational_rag(session_id, question)
        response = result['response']

        print(f"\nTurn {i}:")
        print(f"  User: {question}")
        print(f"  Agent: {response[:100]}...")

    # Evaluate final turn
    from app.models.models import SessionLocal, Conversacion
    db = SessionLocal()
    history = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
    history_list = [{'role': h.role, 'contenido': h.contenido} for h in history]
    db.close()

    latest_response = history_list[-1]['contenido']
    judgment = evaluate_context_awareness(
        conversation_history=history_list,
        latest_response=latest_response,
        model=judge_llm
    )

    print(f"\nContext Score: {judgment['context_score']:.2f}")
    print(f"Explanation: {judgment['explanation']}")
    print(f"{'='*60}\n")

    # Should handle topic switch
    assert judgment['maintains_context'], (
        f"Failed to handle topic switch appropriately."
    )


@pytest.mark.conversational
def test_no_redundant_questions(conversational_rag, cleanup_test_sessions, judge_llm):
    """
    Test that system doesn't ask for information already provided.
    """
    session_id = f"test_no_redundant_{uuid.uuid4().hex[:8]}"
    cleanup_test_sessions(session_id)

    print(f"\n{'='*60}")
    print(f"NO REDUNDANT QUESTIONS TEST")
    print(f"{'='*60}")

    # User provides information upfront
    conversation = [
        "Compré una camisa hace 25 días en la tienda de Bogotá",
        "¿Puedo devolverla?",
    ]

    for i, question in enumerate(conversation, 1):
        result = conversational_rag(session_id, question)
        response = result['response']

        print(f"\nTurn {i}:")
        print(f"  User: {question}")
        print(f"  Agent: {response}")

    # Check if second response asks redundant questions
    final_response = result['response'].lower()

    # These would be redundant since user already provided the info
    redundant_patterns = [
        "¿cuándo compraste",
        "¿cuántos días hace",
        "¿qué producto",
        "¿dónde compraste"
    ]

    redundant_found = [p for p in redundant_patterns if p in final_response]

    if redundant_found:
        print(f"\n⚠️  Redundant questions detected: {redundant_found}")

    print(f"{'='*60}\n")

    # Evaluate context awareness
    from app.models.models import SessionLocal, Conversacion
    db = SessionLocal()
    history = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
    history_list = [{'role': h.role, 'contenido': h.contenido} for h in history]
    db.close()

    judgment = evaluate_context_awareness(
        conversation_history=history_list,
        latest_response=result['response'],
        model=judge_llm
    )

    # Should maintain context and not ask redundant questions
    assert judgment['maintains_context'], (
        f"System asked redundant questions despite user providing context. "
        f"Found: {redundant_found}"
    )


@pytest.mark.conversational
def test_session_isolation(conversational_rag, cleanup_test_sessions):
    """
    Test that different sessions don't leak context between them.
    """
    session1 = f"test_session1_{uuid.uuid4().hex[:8]}"
    session2 = f"test_session2_{uuid.uuid4().hex[:8]}"
    cleanup_test_sessions(session1)
    cleanup_test_sessions(session2)

    print(f"\n{'='*60}")
    print(f"SESSION ISOLATION TEST")
    print(f"{'='*60}")

    # Session 1: Ask about returns
    result1 = conversational_rag(session1, "¿Cómo devuelvo ropa?")
    print(f"\nSession 1, Turn 1:")
    print(f"  User: ¿Cómo devuelvo ropa?")
    print(f"  Agent: {result1['response'][:80]}...")

    # Session 2: Ask about shipping (different topic)
    result2 = conversational_rag(session2, "¿Cuánto cuesta el envío?")
    print(f"\nSession 2, Turn 1:")
    print(f"  User: ¿Cuánto cuesta el envío?")
    print(f"  Agent: {result2['response'][:80]}...")

    # Session 1: Follow-up on returns
    result1_followup = conversational_rag(session1, "¿Necesito el recibo?")
    print(f"\nSession 1, Turn 2:")
    print(f"  User: ¿Necesito el recibo?")
    print(f"  Agent: {result1_followup['response'][:80]}...")

    # Session 1 should talk about returns, not shipping
    response = result1_followup['response'].lower()

    # Should contain return-related terms
    has_return_context = any(word in response for word in ['devolución', 'devolver', 'recibo', 'retorno'])

    # Should NOT contain shipping terms (from session 2)
    has_shipping_leak = any(word in response for word in ['envío', 'entrega', 'transportadora'])

    print(f"\nSession 1 context check:")
    print(f"  Has return context: {has_return_context}")
    print(f"  Has shipping leak: {has_shipping_leak}")
    print(f"{'='*60}\n")

    # Should maintain session 1 context, not leak session 2
    assert has_return_context, "Session lost its own context"
    assert not has_shipping_leak, "Session leaked context from another session"


@pytest.mark.conversational
def test_long_conversation_coherence(conversational_rag, cleanup_test_sessions, judge_llm):
    """
    Test that system maintains coherence over longer conversations.
    """
    session_id = f"test_long_conv_{uuid.uuid4().hex[:8]}"
    cleanup_test_sessions(session_id)

    print(f"\n{'='*60}")
    print(f"LONG CONVERSATION TEST (6 turns)")
    print(f"{'='*60}")

    # Longer conversation covering multiple aspects
    conversation = [
        "Hola, quisiera información sobre devoluciones",
        "¿Cuántos días tengo para devolver ropa?",
        "¿Y para zapatos?",
        "¿Necesito llevar algo además del recibo?",
        "¿Puedo hacer el cambio en cualquier tienda?",
        "Perfecto, gracias por la información",
    ]

    responses = []

    for i, question in enumerate(conversation, 1):
        result = conversational_rag(session_id, question)
        response = result['response']
        responses.append(response)

        print(f"\nTurn {i}:")
        print(f"  User: {question}")
        print(f"  Agent: {response[:80]}...")

    # Evaluate context awareness across the conversation
    from app.models.models import SessionLocal, Conversacion
    db = SessionLocal()
    history = db.query(Conversacion).filter_by(session_id=session_id).order_by(Conversacion.timestamp).all()
    history_list = [{'role': h.role, 'contenido': h.contenido} for h in history]
    db.close()

    # Check mid-conversation turn (turn 4)
    mid_response = responses[3]  # "¿Necesito llevar algo además del recibo?"

    judgment = evaluate_context_awareness(
        conversation_history=history_list[:8],  # Up to turn 4
        latest_response=mid_response,
        model=judge_llm
    )

    print(f"\nMid-conversation (Turn 4) Context Score: {judgment['context_score']:.2f}")
    print(f"Maintains Context: {judgment['maintains_context']}")
    print(f"{'='*60}\n")

    # Should maintain context even in longer conversations
    assert judgment['maintains_context'], (
        f"Lost context in longer conversation. "
        f"Context score: {judgment['context_score']:.2f}"
    )