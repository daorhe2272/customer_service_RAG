"""
Tests for answer faithfulness (grounding in retrieved context).
Uses LLM-as-judge to detect hallucinations.

Run: pytest tests/test_answer_faithfulness.py -v
"""

import pytest
from evaluators.llm_judge import evaluate_faithfulness


@pytest.mark.faithfulness
@pytest.mark.slow
def test_no_hallucination_in_responses(eval_dataset, rag_pipeline, judge_llm):
    """
    Test that responses are grounded in retrieved context (no hallucinations).
    Uses LLM judge to detect invented facts.
    """
    hallucination_cases = []
    faithful_count = 0
    low_confidence_cases = []

    print(f"\n{'='*60}")
    print(f"FAITHFULNESS EVALUATION (LLM Judge)")
    print(f"{'='*60}")

    # Evaluate subset for speed (can increase for thorough testing)
    sample_size = min(20, len(eval_dataset))
    sample = eval_dataset[:sample_size]

    for i, case in enumerate(sample, 1):
        question = case['question']

        # Get RAG response
        result = rag_pipeline(question)
        context = result['context']
        response = result['response']

        # Judge faithfulness
        judgment = evaluate_faithfulness(
            question=question,
            context=context,
            response=response,
            model=judge_llm
        )

        # Track results
        if judgment['is_faithful']:
            faithful_count += 1
        else:
            hallucination_cases.append({
                'question': question,
                'response': response,
                'judgment': judgment
            })

        # Track low confidence judgments
        if judgment['confidence'] < 0.8:
            low_confidence_cases.append({
                'question': question,
                'confidence': judgment['confidence']
            })

        # Progress indicator
        if i % 5 == 0:
            print(f"  Evaluated {i}/{sample_size} cases...")

    # Calculate metrics
    faithfulness_rate = faithful_count / sample_size
    hallucination_rate = len(hallucination_cases) / sample_size

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Faithfulness Rate: {faithfulness_rate:.2%}")
    print(f"Hallucination Rate: {hallucination_rate:.2%}")
    print(f"Threshold: ≤5% hallucinations")
    print(f"Faithful: {faithful_count}/{sample_size}")
    print(f"Hallucinated: {len(hallucination_cases)}/{sample_size}")

    # Show hallucination examples
    if hallucination_cases:
        print(f"\n{'='*60}")
        print(f"HALLUCINATION CASES:")
        print(f"{'='*60}")
        for i, case in enumerate(hallucination_cases[:3], 1):
            print(f"\n{i}. Question: {case['question']}")
            print(f"   Reason: {case['judgment']['reason']}")
            if case['judgment']['hallucinated_claims']:
                print(f"   Invented: {case['judgment']['hallucinated_claims']}")

    # Show low confidence cases
    if low_confidence_cases:
        print(f"\n{'='*60}")
        print(f"LOW CONFIDENCE JUDGMENTS: {len(low_confidence_cases)}")
        print(f"{'='*60}")
        for case in low_confidence_cases[:3]:
            print(f"  - {case['question'][:60]}... (confidence: {case['confidence']:.2f})")

    print(f"{'='*60}\n")

    # Assert threshold: max 5% hallucination rate
    assert hallucination_rate <= 0.05, (
        f"Hallucination rate {hallucination_rate:.2%} exceeds 5% threshold. "
        f"Found {len(hallucination_cases)} cases with invented facts. "
        f"Review: {[c['question'] for c in hallucination_cases[:3]]}"
    )


@pytest.mark.faithfulness
@pytest.mark.slow
def test_critical_facts_not_invented(rag_pipeline, judge_llm):
    """
    Test that critical facts (numbers, dates, contact info) are not invented.
    """
    critical_questions = [
        "¿Cuántos días tengo para devolver ropa?",
        "¿Cuántos días para devolver calzado?",
        "¿Cuál es el teléfono de atención al cliente?",
        "¿Cuál es el horario de atención?",
        "¿Cuánto tiempo tarda la entrega?",
    ]

    print(f"\n{'='*60}")
    print(f"CRITICAL FACTS FAITHFULNESS TEST")
    print(f"{'='*60}")

    failures = []

    for question in critical_questions:
        result = rag_pipeline(question)

        judgment = evaluate_faithfulness(
            question=question,
            context=result['context'],
            response=result['response'],
            model=judge_llm
        )

        print(f"\n{question}")
        print(f"  Faithful: {judgment['is_faithful']} (confidence: {judgment['confidence']:.2f})")

        if not judgment['is_faithful']:
            failures.append({
                'question': question,
                'response': result['response'],
                'reason': judgment['reason'],
                'hallucinations': judgment['hallucinated_claims']
            })

    print(f"{'='*60}\n")

    # Critical facts must be 100% faithful
    assert len(failures) == 0, (
        f"Critical facts were invented in {len(failures)} cases. "
        f"This is unacceptable for customer service. "
        f"Failed: {[f['question'] for f in failures]}"
    )


@pytest.mark.faithfulness
def test_out_of_context_questions_dont_hallucinate(rag_pipeline, judge_llm):
    """
    Test that when context doesn't contain answer, system doesn't invent one.
    """
    # Questions likely to have no good context match
    edge_questions = [
        "¿Ofrecen productos para mascotas?",
        "¿Venden electrónicos?",
        "¿Tienen servicio de sastrería?",
    ]

    print(f"\n{'='*60}")
    print(f"OUT-OF-CONTEXT QUESTIONS TEST")
    print(f"{'='*60}")

    for question in edge_questions:
        result = rag_pipeline(question)

        judgment = evaluate_faithfulness(
            question=question,
            context=result['context'],
            response=result['response'],
            model=judge_llm
        )

        print(f"\n{question}")
        print(f"  Faithful: {judgment['is_faithful']}")
        print(f"  Response: {result['response'][:100]}...")

        # Should either be faithful (saying "I don't know") or explicitly uncertain
        # Not inventing product categories
        assert judgment['is_faithful'] or "no" in result['response'].lower() or "información" in result['response'].lower(), (
            f"System invented answer to out-of-scope question: {question}"
        )

    print(f"{'='*60}\n")


@pytest.mark.faithfulness
@pytest.mark.slow
def test_faithfulness_by_topic(eval_dataset, rag_pipeline, judge_llm):
    """
    Test faithfulness broken down by topic.
    Helps identify which areas are prone to hallucination.
    """
    # Group by topic
    topics = {}
    for case in eval_dataset:
        topic = case.get('topic', 'unknown')
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(case)

    print(f"\n{'='*60}")
    print(f"FAITHFULNESS BY TOPIC")
    print(f"{'='*60}")

    topic_results = {}

    for topic, cases in topics.items():
        # Sample up to 5 cases per topic for speed
        sample = cases[:5]
        faithful_count = 0

        for case in sample:
            result = rag_pipeline(case['question'])

            judgment = evaluate_faithfulness(
                question=case['question'],
                context=result['context'],
                response=result['response'],
                model=judge_llm
            )

            if judgment['is_faithful']:
                faithful_count += 1

        faithfulness_rate = faithful_count / len(sample)
        topic_results[topic] = {
            'rate': faithfulness_rate,
            'count': len(sample)
        }

        print(f"\n{topic.upper()} ({len(sample)} cases tested):")
        print(f"  Faithfulness: {faithfulness_rate:.2%}")

    print(f"{'='*60}\n")

    # All topics should have >90% faithfulness
    for topic, metrics in topic_results.items():
        assert metrics['rate'] >= 0.90, (
            f"Topic '{topic}' has low faithfulness {metrics['rate']:.2%}. "
            f"May be hallucinating facts in this domain."
        )