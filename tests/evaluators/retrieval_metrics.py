"""
Retrieval quality metrics for RAG evaluation.
These are deterministic metrics that evaluate the vector search component.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def calculate_hit_rate(
    eval_cases: List[Dict[str, Any]],
    retrieval_fn,
    k: int = 5
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate hit rate: percentage of queries where correct document appears in top-k.

    Args:
        eval_cases: List of evaluation cases with 'question' and 'expected_source'
        retrieval_fn: Function that takes (question, k) and returns list of chunk_ids
        k: Number of top results to consider

    Returns:
        Tuple of (hit_rate, detailed_results)
        - hit_rate: Float between 0 and 1
        - detailed_results: List of dicts with per-case results
    """
    hits = 0
    detailed_results = []

    for case in eval_cases:
        question = case['question']
        expected_source = case['expected_source']

        # Get retrieval results
        chunk_ids = retrieval_fn(question, k)

        # Check if any chunk is from the expected source
        hit = any(expected_source in chunk_id for chunk_id in chunk_ids)

        if hit:
            hits += 1

        detailed_results.append({
            'question': question,
            'expected_source': expected_source,
            'retrieved_chunks': chunk_ids,
            'hit': hit
        })

    hit_rate = hits / len(eval_cases) if eval_cases else 0.0

    return hit_rate, detailed_results


def calculate_mrr(
    eval_cases: List[Dict[str, Any]],
    retrieval_fn,
    k: int = 5
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Mean Reciprocal Rank: average of 1/rank for first correct result.

    Args:
        eval_cases: List of evaluation cases with 'question' and 'expected_source'
        retrieval_fn: Function that takes (question, k) and returns list of chunk_ids
        k: Number of top results to consider

    Returns:
        Tuple of (mrr, detailed_results)
        - mrr: Float between 0 and 1
        - detailed_results: List of dicts with per-case results
    """
    reciprocal_ranks = []
    detailed_results = []

    for case in eval_cases:
        question = case['question']
        expected_source = case['expected_source']

        # Get retrieval results
        chunk_ids = retrieval_fn(question, k)

        # Find rank of first correct chunk
        rank = None
        for i, chunk_id in enumerate(chunk_ids, start=1):
            if expected_source in chunk_id:
                rank = i
                break

        rr = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(rr)

        detailed_results.append({
            'question': question,
            'expected_source': expected_source,
            'retrieved_chunks': chunk_ids,
            'rank': rank,
            'reciprocal_rank': rr
        })

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return mrr, detailed_results


def calculate_precision_at_k(
    eval_cases: List[Dict[str, Any]],
    retrieval_fn,
    k: int = 5
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Precision@K: proportion of relevant documents in top-k results.

    Args:
        eval_cases: List of evaluation cases with 'question' and 'expected_source'
        retrieval_fn: Function that takes (question, k) and returns list of chunk_ids
        k: Number of top results to consider

    Returns:
        Tuple of (precision_at_k, detailed_results)
        - precision_at_k: Float between 0 and 1
        - detailed_results: List of dicts with per-case results
    """
    precision_scores = []
    detailed_results = []

    for case in eval_cases:
        question = case['question']
        expected_source = case['expected_source']

        # Get retrieval results
        chunk_ids = retrieval_fn(question, k)

        # Count how many chunks are from the expected source
        relevant_count = sum(1 for chunk_id in chunk_ids if expected_source in chunk_id)

        # Precision = relevant / total
        precision = relevant_count / k if k > 0 else 0.0
        precision_scores.append(precision)

        detailed_results.append({
            'question': question,
            'expected_source': expected_source,
            'retrieved_chunks': chunk_ids,
            'relevant_count': relevant_count,
            'precision': precision
        })

    avg_precision = np.mean(precision_scores) if precision_scores else 0.0

    return avg_precision, detailed_results


def calculate_all_retrieval_metrics(
    eval_cases: List[Dict[str, Any]],
    retrieval_fn,
    k: int = 5
) -> Dict[str, Any]:
    """
    Calculate all retrieval metrics at once for efficiency.

    Args:
        eval_cases: List of evaluation cases
        retrieval_fn: Retrieval function
        k: Number of top results

    Returns:
        Dict with all metrics and detailed results
    """
    hit_rate, hit_details = calculate_hit_rate(eval_cases, retrieval_fn, k)
    mrr, mrr_details = calculate_mrr(eval_cases, retrieval_fn, k)
    precision, precision_details = calculate_precision_at_k(eval_cases, retrieval_fn, k)

    return {
        'hit_rate': hit_rate,
        'mrr': mrr,
        'precision_at_k': precision,
        'k': k,
        'num_cases': len(eval_cases),
        'detailed_results': {
            'hit_rate': hit_details,
            'mrr': mrr_details,
            'precision': precision_details
        }
    }