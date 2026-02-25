"""
Tests for retrieval quality (deterministic metrics).
These tests evaluate the vector search component of the RAG system.

Run: pytest tests/test_retrieval_quality.py -v
"""

import pytest
from evaluators.retrieval_metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_all_retrieval_metrics
)


@pytest.mark.retrieval
def test_hit_rate_threshold(eval_dataset, retrieval_function):
    """
    Test that retrieval hit rate meets minimum threshold.
    Hit rate measures: does the correct document appear in top-k?
    """
    # Calculate hit rate for all eval cases
    hit_rate, detailed_results = calculate_hit_rate(
        eval_dataset,
        retrieval_function,
        k=5
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"HIT RATE EVALUATION")
    print(f"{'='*60}")
    print(f"Hit Rate: {hit_rate:.2%}")
    print(f"Threshold: 85%")
    print(f"Cases: {len(eval_dataset)}")
    print(f"Hits: {sum(1 for r in detailed_results if r['hit'])}")
    print(f"Misses: {sum(1 for r in detailed_results if not r['hit'])}")

    # Print failures
    failures = [r for r in detailed_results if not r['hit']]
    if failures:
        print(f"\nFailed Cases ({len(failures)}):")
        for fail in failures[:5]:  # Show first 5
            print(f"  ❌ {fail['question'][:60]}...")
            print(f"     Expected: {fail['expected_source']}")
            print(f"     Got: {fail['retrieved_chunks'][:2]}")

    print(f"{'='*60}\n")

    # Assert threshold
    assert hit_rate >= 0.85, (
        f"Hit rate {hit_rate:.2%} below 85% threshold. "
        f"{len(failures)} cases failed to retrieve correct document."
    )


@pytest.mark.retrieval
def test_mrr_threshold(eval_dataset, retrieval_function):
    """
    Test that Mean Reciprocal Rank meets minimum threshold.
    MRR measures: how high does the correct document rank?
    """
    # Calculate MRR
    mrr, detailed_results = calculate_mrr(
        eval_dataset,
        retrieval_function,
        k=5
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"MEAN RECIPROCAL RANK EVALUATION")
    print(f"{'='*60}")
    print(f"MRR: {mrr:.2%}")
    print(f"Threshold: 70%")
    print(f"Cases: {len(eval_dataset)}")

    # Analyze rank distribution
    ranks = [r['rank'] for r in detailed_results if r['rank']]
    if ranks:
        print(f"\nRank Distribution:")
        for rank in range(1, 6):
            count = sum(1 for r in ranks if r == rank)
            print(f"  Rank {rank}: {count} cases ({count/len(eval_dataset):.1%})")

    # Show low-ranking cases
    low_rank = [r for r in detailed_results if r['rank'] and r['rank'] > 2]
    if low_rank:
        print(f"\nLow-Ranking Cases (rank > 2): {len(low_rank)}")
        for case in low_rank[:3]:
            print(f"  ⚠️  Rank {case['rank']}: {case['question'][:60]}...")

    print(f"{'='*60}\n")

    # Assert threshold
    assert mrr >= 0.70, (
        f"MRR {mrr:.2%} below 70% threshold. "
        f"Correct documents are ranking too low."
    )


@pytest.mark.retrieval
def test_precision_at_k(eval_dataset, retrieval_function):
    """
    Test average precision at k.
    Measures: what proportion of top-k results are relevant?
    """
    # Calculate precision@5
    precision, detailed_results = calculate_precision_at_k(
        eval_dataset,
        retrieval_function,
        k=5
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"PRECISION@5 EVALUATION")
    print(f"{'='*60}")
    print(f"Precision@5: {precision:.2%}")
    print(f"Threshold: 40%")
    print(f"Cases: {len(eval_dataset)}")

    # Show precision distribution
    precisions = [r['precision'] for r in detailed_results]
    print(f"\nPrecision Distribution:")
    print(f"  Perfect (5/5): {sum(1 for p in precisions if p == 1.0)} cases")
    print(f"  High (3-4/5): {sum(1 for p in precisions if 0.6 <= p < 1.0)} cases")
    print(f"  Medium (2/5): {sum(1 for p in precisions if 0.4 <= p < 0.6)} cases")
    print(f"  Low (0-1/5): {sum(1 for p in precisions if p < 0.4)} cases")

    print(f"{'='*60}\n")

    # Assert threshold (40% = at least 2 out of 5 are relevant on average)
    assert precision >= 0.40, (
        f"Precision@5 {precision:.2%} below 40% threshold. "
        f"Too many irrelevant chunks in top results."
    )


@pytest.mark.retrieval
def test_retrieval_by_topic(eval_dataset, retrieval_function):
    """
    Test retrieval quality broken down by topic.
    Helps identify which policy areas need improvement.
    """
    # Group by topic
    topics = {}
    for case in eval_dataset:
        topic = case.get('topic', 'unknown')
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(case)

    print(f"\n{'='*60}")
    print(f"RETRIEVAL QUALITY BY TOPIC")
    print(f"{'='*60}")

    topic_results = {}
    for topic, cases in topics.items():
        hit_rate, _ = calculate_hit_rate(cases, retrieval_function, k=5)
        mrr, _ = calculate_mrr(cases, retrieval_function, k=5)

        topic_results[topic] = {
            'hit_rate': hit_rate,
            'mrr': mrr,
            'count': len(cases)
        }

        print(f"\n{topic.upper()} ({len(cases)} cases):")
        print(f"  Hit Rate: {hit_rate:.2%}")
        print(f"  MRR: {mrr:.2%}")

    print(f"{'='*60}\n")

    # Check that no topic is drastically underperforming
    for topic, metrics in topic_results.items():
        assert metrics['hit_rate'] >= 0.70, (
            f"Topic '{topic}' has low hit rate {metrics['hit_rate']:.2%}. "
            f"May need more diverse training examples or better chunking."
        )


@pytest.mark.retrieval
def test_retrieval_by_difficulty(eval_dataset, retrieval_function):
    """
    Test retrieval quality broken down by difficulty level.
    """
    # Group by difficulty
    difficulties = {}
    for case in eval_dataset:
        diff = case.get('difficulty', 'unknown')
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(case)

    print(f"\n{'='*60}")
    print(f"RETRIEVAL QUALITY BY DIFFICULTY")
    print(f"{'='*60}")

    for difficulty, cases in difficulties.items():
        if len(cases) < 3:  # Skip if too few cases
            continue

        hit_rate, _ = calculate_hit_rate(cases, retrieval_function, k=5)
        mrr, _ = calculate_mrr(cases, retrieval_function, k=5)

        print(f"\n{difficulty.upper()} ({len(cases)} cases):")
        print(f"  Hit Rate: {hit_rate:.2%}")
        print(f"  MRR: {mrr:.2%}")

    print(f"{'='*60}\n")


@pytest.mark.retrieval
def test_comprehensive_retrieval_metrics(eval_dataset, retrieval_function, eval_results_dir):
    """
    Calculate all retrieval metrics and save detailed report.
    """
    import json
    from datetime import datetime

    # Calculate all metrics
    results = calculate_all_retrieval_metrics(
        eval_dataset,
        retrieval_function,
        k=5
    )

    # Add metadata
    results['timestamp'] = datetime.utcnow().isoformat()
    results['test_name'] = 'comprehensive_retrieval_metrics'

    # Save to file
    output_file = f"{eval_results_dir}/retrieval_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed results saved to: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE RETRIEVAL METRICS")
    print(f"{'='*60}")
    print(f"Hit Rate: {results['hit_rate']:.2%}")
    print(f"MRR: {results['mrr']:.2%}")
    print(f"Precision@{results['k']}: {results['precision_at_k']:.2%}")
    print(f"Total Cases: {results['num_cases']}")
    print(f"{'='*60}\n")

    # All metrics should meet thresholds
    assert results['hit_rate'] >= 0.85, "Hit rate below threshold"
    assert results['mrr'] >= 0.70, "MRR below threshold"
    assert results['precision_at_k'] >= 0.40, "Precision@k below threshold"