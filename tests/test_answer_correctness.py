"""
Tests for answer correctness (factual accuracy vs ground truth).
Uses LLM-as-judge to evaluate semantic correctness.

Run: pytest tests/test_answer_correctness.py -v
"""

import pytest
import numpy as np
from evaluators.llm_judge import evaluate_correctness


@pytest.mark.correctness
@pytest.mark.slow
def test_answer_correctness_threshold(eval_dataset, rag_pipeline, judge_llm):
    """
    Test that answers are factually correct compared to expected answers.
    Uses LLM judge for semantic evaluation.
    """
    scores = []
    incorrect_cases = []
    partial_correct_cases = []

    print(f"\n{'='*60}")
    print(f"ANSWER CORRECTNESS EVALUATION (LLM Judge)")
    print(f"{'='*60}")

    # Evaluate subset for speed
    sample_size = min(25, len(eval_dataset))
    sample = eval_dataset[:sample_size]

    for i, case in enumerate(sample, 1):
        question = case['question']
        expected_answer = case['expected_answer']

        # Get RAG response
        result = rag_pipeline(question)
        actual_answer = result['response']

        # Judge correctness
        judgment = evaluate_correctness(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            model=judge_llm
        )

        score = judgment['correctness_score']
        scores.append(score)

        # Track problematic cases
        if score < 0.7:
            incorrect_cases.append({
                'question': question,
                'expected': expected_answer,
                'actual': actual_answer,
                'score': score,
                'explanation': judgment['explanation'],
                'differences': judgment.get('key_differences', [])
            })
        elif score < 0.9:
            partial_correct_cases.append({
                'question': question,
                'score': score
            })

        # Progress indicator
        if i % 5 == 0:
            print(f"  Evaluated {i}/{sample_size} cases...")

    # Calculate metrics
    avg_correctness = np.mean(scores)
    fully_correct = sum(1 for s in scores if s >= 0.9)
    partially_correct = sum(1 for s in scores if 0.7 <= s < 0.9)
    incorrect = sum(1 for s in scores if s < 0.7)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Average Correctness: {avg_correctness:.2%}")
    print(f"Threshold: ≥85%")
    print(f"\nScore Distribution:")
    print(f"  Fully Correct (≥0.9): {fully_correct}/{sample_size} ({fully_correct/sample_size:.1%})")
    print(f"  Partial (0.7-0.9): {partially_correct}/{sample_size} ({partially_correct/sample_size:.1%})")
    print(f"  Incorrect (<0.7): {incorrect}/{sample_size} ({incorrect/sample_size:.1%})")

    # Show incorrect cases
    if incorrect_cases:
        print(f"\n{'='*60}")
        print(f"INCORRECT CASES ({len(incorrect_cases)}):")
        print(f"{'='*60}")
        for i, case in enumerate(incorrect_cases[:3], 1):
            print(f"\n{i}. Question: {case['question']}")
            print(f"   Score: {case['score']:.2f}")
            print(f"   Explanation: {case['explanation']}")
            if case['differences']:
                print(f"   Differences: {case['differences']}")

    # Show partial cases
    if partial_correct_cases and not incorrect_cases:
        print(f"\n{'='*60}")
        print(f"PARTIALLY CORRECT CASES: {len(partial_correct_cases)}")
        print(f"{'='*60}")
        for case in partial_correct_cases[:3]:
            print(f"  - {case['question'][:60]}... (score: {case['score']:.2f})")

    print(f"{'='*60}\n")

    # Assert threshold: average ≥85%
    assert avg_correctness >= 0.85, (
        f"Average correctness {avg_correctness:.2%} below 85% threshold. "
        f"Found {incorrect} incorrect and {partially_correct} partially correct cases. "
        f"Review: {[c['question'] for c in incorrect_cases[:3]]}"
    )


@pytest.mark.correctness
@pytest.mark.slow
def test_critical_policy_facts(rag_pipeline, judge_llm):
    """
    Test that critical policy facts are stated correctly.
    These are non-negotiable facts that must be accurate.
    """
    critical_tests = [
        {
            'question': "¿Cuántos días tengo para devolver ropa?",
            'expected': "30 días para prendas de vestir",
            'min_score': 0.95
        },
        {
            'question': "¿Cuál es el plazo para devolver calzado?",
            'expected': "15 días para calzado",
            'min_score': 0.95
        },
        {
            'question': "¿Cuánto tiempo tarda el reembolso?",
            'expected': "5-10 días hábiles al método de pago original",
            'min_score': 0.90
        },
    ]

    print(f"\n{'='*60}")
    print(f"CRITICAL POLICY FACTS TEST")
    print(f"{'='*60}")

    failures = []

    for test in critical_tests:
        result = rag_pipeline(test['question'])

        judgment = evaluate_correctness(
            question=test['question'],
            expected_answer=test['expected'],
            actual_answer=result['response'],
            model=judge_llm
        )

        score = judgment['correctness_score']
        print(f"\n{test['question']}")
        print(f"  Score: {score:.2%} (min: {test['min_score']:.0%})")
        print(f"  Status: {'✓ PASS' if score >= test['min_score'] else '✗ FAIL'}")

        if score < test['min_score']:
            failures.append({
                'question': test['question'],
                'score': score,
                'expected': test['expected'],
                'actual': result['response'],
                'explanation': judgment['explanation']
            })

    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURES:")
        print(f"{'='*60}")
        for fail in failures:
            print(f"\n{fail['question']}")
            print(f"  Expected: {fail['expected']}")
            print(f"  Actual: {fail['actual'][:100]}...")
            print(f"  Reason: {fail['explanation']}")

    print(f"{'='*60}\n")

    # Critical facts must be highly accurate
    assert len(failures) == 0, (
        f"Critical policy facts were incorrect in {len(failures)} cases. "
        f"This is unacceptable for customer service."
    )


@pytest.mark.correctness
@pytest.mark.slow
def test_correctness_by_difficulty(eval_dataset, rag_pipeline, judge_llm):
    """
    Test answer correctness by question difficulty level.
    """
    # Group by difficulty
    difficulties = {}
    for case in eval_dataset:
        diff = case.get('difficulty', 'unknown')
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(case)

    print(f"\n{'='*60}")
    print(f"CORRECTNESS BY DIFFICULTY")
    print(f"{'='*60}")

    difficulty_results = {}

    for difficulty, cases in difficulties.items():
        if len(cases) < 3:
            continue

        # Sample cases
        sample = cases[:min(8, len(cases))]
        scores = []

        for case in sample:
            result = rag_pipeline(case['question'])

            judgment = evaluate_correctness(
                question=case['question'],
                expected_answer=case['expected_answer'],
                actual_answer=result['response'],
                model=judge_llm
            )

            scores.append(judgment['correctness_score'])

        avg_score = np.mean(scores)
        difficulty_results[difficulty] = {
            'avg_score': avg_score,
            'count': len(sample)
        }

        print(f"\n{difficulty.upper()} ({len(sample)} cases tested):")
        print(f"  Average Correctness: {avg_score:.2%}")

    print(f"{'='*60}\n")

    # Even hard questions should have ≥75% correctness
    for difficulty, metrics in difficulty_results.items():
        min_threshold = 0.75 if difficulty == 'hard' else 0.85
        assert metrics['avg_score'] >= min_threshold, (
            f"Difficulty '{difficulty}' has low correctness {metrics['avg_score']:.2%}. "
            f"Expected ≥{min_threshold:.0%}."
        )


@pytest.mark.correctness
def test_numerical_accuracy(rag_pipeline, judge_llm):
    """
    Test that numerical values (days, hours, percentages) are accurate.
    """
    numerical_questions = [
        {
            'question': "¿Cuántos días tengo para devolver ropa?",
            'expected_contains': ["30"]
        },
        {
            'question': "¿Cuál es el plazo para calzado?",
            'expected_contains': ["15"]
        },
        {
            'question': "¿En cuánto tiempo llega el pedido en ciudades principales?",
            'expected_contains': ["2", "4"]  # 2-4 días
        },
    ]

    print(f"\n{'='*60}")
    print(f"NUMERICAL ACCURACY TEST")
    print(f"{'='*60}")

    for test in numerical_questions:
        result = rag_pipeline(test['question'])
        response = result['response']

        print(f"\n{test['question']}")
        print(f"  Response: {response[:100]}...")

        # Check if expected numbers appear
        found_numbers = [num for num in test['expected_contains']
                        if num in response]

        print(f"  Expected numbers: {test['expected_contains']}")
        print(f"  Found: {found_numbers}")

        # At least one expected number should appear
        assert len(found_numbers) > 0, (
            f"Response missing critical numbers {test['expected_contains']} "
            f"for question: {test['question']}"
        )

    print(f"{'='*60}\n")


@pytest.mark.correctness
@pytest.mark.slow
def test_correctness_by_topic(eval_dataset, rag_pipeline, judge_llm):
    """
    Test correctness broken down by topic to identify weak areas.
    """
    # Group by topic
    topics = {}
    for case in eval_dataset:
        topic = case.get('topic', 'unknown')
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(case)

    print(f"\n{'='*60}")
    print(f"CORRECTNESS BY TOPIC")
    print(f"{'='*60}")

    topic_results = {}

    for topic, cases in topics.items():
        # Sample up to 5 cases per topic
        sample = cases[:5]
        scores = []

        for case in sample:
            result = rag_pipeline(case['question'])

            judgment = evaluate_correctness(
                question=case['question'],
                expected_answer=case['expected_answer'],
                actual_answer=result['response'],
                model=judge_llm
            )

            scores.append(judgment['correctness_score'])

        avg_score = np.mean(scores)
        topic_results[topic] = {
            'avg_score': avg_score,
            'count': len(sample)
        }

        print(f"\n{topic.upper()} ({len(sample)} cases tested):")
        print(f"  Average Correctness: {avg_score:.2%}")

    print(f"{'='*60}\n")

    # All topics should have ≥80% correctness
    for topic, metrics in topic_results.items():
        assert metrics['avg_score'] >= 0.80, (
            f"Topic '{topic}' has low correctness {metrics['avg_score']:.2%}. "
            f"May need better policy documentation or examples."
        )