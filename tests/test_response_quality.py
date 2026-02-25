"""
Tests for response quality (tone, style, completeness).
Uses LLM-as-judge to evaluate subjective quality criteria.

Run: pytest tests/test_response_quality.py -v
"""

import pytest
import numpy as np
from evaluators.llm_judge import evaluate_tone, evaluate_completeness


@pytest.mark.quality
@pytest.mark.slow
def test_tone_quality_threshold(eval_dataset, rag_pipeline, judge_llm):
    """
    Test that responses have appropriate tone for customer service.
    Evaluates: professional, empathetic, clear, structured.
    """
    tone_scores = []
    low_quality_cases = []

    print(f"\n{'='*60}")
    print(f"TONE QUALITY EVALUATION (LLM Judge)")
    print(f"{'='*60}")

    # Sample for speed
    sample_size = min(20, len(eval_dataset))
    sample = eval_dataset[:sample_size]

    # Track criteria breakdown
    criteria_counts = {
        'is_professional': 0,
        'is_empathetic': 0,
        'is_clear': 0,
        'is_structured': 0
    }

    for i, case in enumerate(sample, 1):
        question = case['question']
        result = rag_pipeline(question)
        response = result['response']

        # Evaluate tone
        judgment = evaluate_tone(
            response=response,
            model=judge_llm
        )

        score = judgment['tone_score']
        tone_scores.append(score)

        # Track criteria
        for criterion in criteria_counts.keys():
            if judgment.get(criterion, False):
                criteria_counts[criterion] += 1

        # Track low quality
        if score < 0.75:
            low_quality_cases.append({
                'question': question,
                'response': response,
                'score': score,
                'feedback': judgment.get('feedback', '')
            })

        # Progress
        if i % 5 == 0:
            print(f"  Evaluated {i}/{sample_size} cases...")

    # Calculate metrics
    avg_tone = np.mean(tone_scores)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Average Tone Score: {avg_tone:.2%}")
    print(f"Threshold: ≥80%")
    print(f"\nCriteria Breakdown:")
    for criterion, count in criteria_counts.items():
        pct = count / sample_size
        print(f"  {criterion}: {count}/{sample_size} ({pct:.1%})")

    # Show low quality cases
    if low_quality_cases:
        print(f"\n{'='*60}")
        print(f"LOW QUALITY CASES ({len(low_quality_cases)}):")
        print(f"{'='*60}")
        for i, case in enumerate(low_quality_cases[:3], 1):
            print(f"\n{i}. Question: {case['question']}")
            print(f"   Score: {case['score']:.2f}")
            print(f"   Feedback: {case['feedback']}")
            print(f"   Response: {case['response'][:100]}...")

    print(f"{'='*60}\n")

    # Assert threshold
    assert avg_tone >= 0.80, (
        f"Average tone quality {avg_tone:.2%} below 80% threshold. "
        f"Found {len(low_quality_cases)} cases with poor tone/style."
    )


@pytest.mark.quality
@pytest.mark.slow
def test_response_completeness(eval_dataset, rag_pipeline, judge_llm):
    """
    Test that responses include all required information elements.
    """
    completeness_scores = []
    incomplete_cases = []

    print(f"\n{'='*60}")
    print(f"RESPONSE COMPLETENESS EVALUATION (LLM Judge)")
    print(f"{'='*60}")

    # Sample cases that have required_elements defined
    sample = [case for case in eval_dataset if case.get('required_elements')][:20]

    for i, case in enumerate(sample, 1):
        question = case['question']
        required_elements = case['required_elements']

        result = rag_pipeline(question)
        response = result['response']

        # Evaluate completeness
        judgment = evaluate_completeness(
            question=question,
            response=response,
            required_elements=required_elements,
            model=judge_llm
        )

        score = judgment['completeness_score']
        completeness_scores.append(score)

        # Track incomplete
        if not judgment['is_complete']:
            incomplete_cases.append({
                'question': question,
                'response': response,
                'score': score,
                'missing': judgment.get('missing_elements', []),
                'explanation': judgment.get('explanation', '')
            })

        # Progress
        if i % 5 == 0:
            print(f"  Evaluated {i}/{len(sample)} cases...")

    # Calculate metrics
    avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
    complete_count = sum(1 for s in completeness_scores if s >= 1.0)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Average Completeness: {avg_completeness:.2%}")
    print(f"Threshold: ≥85%")
    print(f"Fully Complete: {complete_count}/{len(sample)} ({complete_count/len(sample):.1%})")
    print(f"Incomplete: {len(incomplete_cases)}/{len(sample)}")

    # Show incomplete cases
    if incomplete_cases:
        print(f"\n{'='*60}")
        print(f"INCOMPLETE CASES ({len(incomplete_cases)}):")
        print(f"{'='*60}")
        for i, case in enumerate(incomplete_cases[:3], 1):
            print(f"\n{i}. Question: {case['question']}")
            print(f"   Score: {case['score']:.2f}")
            print(f"   Missing: {case['missing']}")
            print(f"   Explanation: {case['explanation']}")

    print(f"{'='*60}\n")

    # Assert threshold
    assert avg_completeness >= 0.85, (
        f"Average completeness {avg_completeness:.2%} below 85% threshold. "
        f"{len(incomplete_cases)} responses missing required information."
    )


@pytest.mark.quality
def test_professional_language(rag_pipeline, judge_llm):
    """
    Test that responses use professional language (no slang, no informal).
    """
    test_questions = [
        "¿Cómo devuelvo un producto?",
        "Mi pedido llegó mal",
        "¿Qué métodos de pago aceptan?",
        "No me llegó el paquete",
    ]

    print(f"\n{'='*60}")
    print(f"PROFESSIONAL LANGUAGE TEST")
    print(f"{'='*60}")

    failures = []

    for question in test_questions:
        result = rag_pipeline(question)
        response = result['response']

        judgment = evaluate_tone(response=response, model=judge_llm)

        print(f"\n{question}")
        print(f"  Professional: {judgment.get('is_professional', False)}")
        print(f"  Tone Score: {judgment['tone_score']:.2f}")

        if not judgment.get('is_professional', False):
            failures.append({
                'question': question,
                'response': response,
                'feedback': judgment.get('feedback', '')
            })

    if failures:
        print(f"\n{'='*60}")
        print(f"UNPROFESSIONAL RESPONSES ({len(failures)}):")
        print(f"{'='*60}")
        for fail in failures:
            print(f"\n{fail['question']}")
            print(f"  Response: {fail['response'][:100]}...")
            print(f"  Issue: {fail['feedback']}")

    print(f"{'='*60}\n")

    # Should be professional
    assert len(failures) == 0, (
        f"{len(failures)} responses lacked professional tone. "
        f"Customer service requires formal language."
    )


@pytest.mark.quality
def test_empathy_in_problem_scenarios(rag_pipeline, judge_llm):
    """
    Test that responses show empathy when customer has a problem.
    """
    problem_scenarios = [
        "Mi pedido llegó dañado",
        "No me llegó el paquete y ya pasó una semana",
        "El producto tiene un defecto",
        "Quiero devolver algo pero perdí el recibo",
    ]

    print(f"\n{'='*60}")
    print(f"EMPATHY TEST (Problem Scenarios)")
    print(f"{'='*60}")

    low_empathy_cases = []

    for question in problem_scenarios:
        result = rag_pipeline(question)
        response = result['response']

        judgment = evaluate_tone(response=response, model=judge_llm)

        print(f"\n{question}")
        print(f"  Empathetic: {judgment.get('is_empathetic', False)}")
        print(f"  Response: {response[:80]}...")

        if not judgment.get('is_empathetic', False):
            low_empathy_cases.append({
                'question': question,
                'response': response
            })

    print(f"{'='*60}\n")

    # At least 75% should show empathy (some flexibility for edge cases)
    empathy_rate = 1 - (len(low_empathy_cases) / len(problem_scenarios))
    assert empathy_rate >= 0.75, (
        f"Only {empathy_rate:.1%} of problem scenarios showed empathy. "
        f"Customer service should be empathetic, especially for problems."
    )


@pytest.mark.quality
def test_clarity_and_conciseness(rag_pipeline, judge_llm):
    """
    Test that responses are clear and concise (not overly verbose).
    """
    test_questions = [
        "¿Cuántos días tengo para devolver?",
        "¿Cuánto cuesta el envío?",
        "¿Qué métodos de pago aceptan?",
    ]

    print(f"\n{'='*60}")
    print(f"CLARITY & CONCISENESS TEST")
    print(f"{'='*60}")

    for question in test_questions:
        result = rag_pipeline(question)
        response = result['response']

        judgment = evaluate_tone(response=response, model=judge_llm)

        print(f"\n{question}")
        print(f"  Clear: {judgment.get('is_clear', False)}")
        print(f"  Length: {len(response)} chars")
        print(f"  Response: {response[:100]}...")

        # Should be clear
        assert judgment.get('is_clear', False), (
            f"Response to '{question}' lacks clarity. "
            f"Customer service answers should be easy to understand."
        )

        # Should not be excessively long for simple questions
        # Allow reasonable length but flag extremely verbose responses
        if len(response) > 1000:
            print(f"  ⚠️  Warning: Response may be too verbose ({len(response)} chars)")

    print(f"{'='*60}\n")


@pytest.mark.quality
def test_structured_formatting(rag_pipeline, judge_llm):
    """
    Test that complex responses use structured formatting (lists, bullets).
    """
    complex_questions = [
        "¿Cómo compro por internet?",
        "¿Qué necesito para devolver un producto?",
        "¿Cuáles son los métodos de pago disponibles?",
    ]

    print(f"\n{'='*60}")
    print(f"STRUCTURED FORMATTING TEST")
    print(f"{'='*60}")

    for question in complex_questions:
        result = rag_pipeline(question)
        response = result['response']

        judgment = evaluate_tone(response=response, model=judge_llm)

        # Check for structure
        is_structured = judgment.get('is_structured', False)

        # Also check for formatting markers (bullets, numbers, etc.)
        has_bullets = '-' in response or '•' in response or '*' in response
        has_numbers = any(f"{i}." in response or f"{i})" in response for i in range(1, 6))
        has_formatting = has_bullets or has_numbers

        print(f"\n{question}")
        print(f"  Structured (judge): {is_structured}")
        print(f"  Has formatting: {has_formatting}")

        # Complex questions should ideally have structure
        # This is a soft check - not all complex answers need bullets
        if not (is_structured or has_formatting):
            print(f"  ⚠️  Note: Complex answer could benefit from structure")

    print(f"{'='*60}\n")


@pytest.mark.quality
@pytest.mark.slow
def test_quality_by_topic(eval_dataset, rag_pipeline, judge_llm):
    """
    Test response quality broken down by topic.
    """
    # Group by topic
    topics = {}
    for case in eval_dataset:
        topic = case.get('topic', 'unknown')
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(case)

    print(f"\n{'='*60}")
    print(f"QUALITY BY TOPIC")
    print(f"{'='*60}")

    for topic, cases in topics.items():
        # Sample up to 5 per topic
        sample = cases[:5]
        scores = []

        for case in sample:
            result = rag_pipeline(case['question'])
            judgment = evaluate_tone(response=result['response'], model=judge_llm)
            scores.append(judgment['tone_score'])

        avg_quality = np.mean(scores)
        print(f"\n{topic.upper()} ({len(sample)} cases):")
        print(f"  Average Quality: {avg_quality:.2%}")

        # All topics should maintain quality
        assert avg_quality >= 0.75, (
            f"Topic '{topic}' has low quality {avg_quality:.2%}"
        )

    print(f"{'='*60}\n")