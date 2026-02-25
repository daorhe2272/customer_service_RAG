"""
Master evaluation runner for the RAG system.
Runs all tests and generates a comprehensive evaluation report.

Usage:
    python tests/run_full_eval_suite.py [--quick] [--full]

Options:
    --quick: Run quick evaluation (smaller sample sizes)
    --full: Run full evaluation (all test cases, slower)
    (default): Run standard evaluation (balanced)
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def run_evaluation(mode='standard'):
    """
    Run the full evaluation suite and generate reports.

    Args:
        mode: 'quick', 'standard', or 'full'
    """
    print("\n" + "=" * 70)
    print("RAG SYSTEM EVALUATION SUITE")
    print("=" * 70)
    print(f"Mode: {mode.upper()}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Create results directory
    results_dir = Path(__file__).parent.parent / 'eval_results'
    results_dir.mkdir(exist_ok=True)

    # Prepare pytest arguments
    pytest_args = [
        'tests/',
        '-v',
        '--tb=short',
        f'--html={results_dir}/report.html',
        '--self-contained-html',
    ]

    # Adjust based on mode
    if mode == 'quick':
        # Skip slow tests
        pytest_args.extend(['-m', 'not slow'])
        print("Quick mode: Skipping slow tests\n")
    elif mode == 'full':
        # Run everything including slow tests
        print("Full mode: Running all tests including slow ones\n")
    else:  # standard
        # Run all tests but with default settings
        print("Standard mode: Running all tests\n")

    # Run pytest
    exit_code = pytest.main(pytest_args)

    # Generate summary report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    generate_summary_report(results_dir, timestamp, mode, exit_code)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit Code: {exit_code}")
    print(f"Results: {results_dir}")
    print("=" * 70 + "\n")

    return exit_code


def generate_summary_report(results_dir, timestamp, mode, exit_code):
    """
    Generate a markdown summary report.
    """
    report_path = results_dir / f'eval_summary_{timestamp}.md'

    summary = f"""# RAG System Evaluation Summary

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Mode:** {mode.upper()}
**Overall Status:** {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}

---

## Test Categories

### 1. Retrieval Quality (Deterministic)
- **Purpose:** Evaluate vector search quality
- **Metrics:** Hit Rate, MRR, Precision@k
- **Method:** Deterministic metrics on retrieval results
- **Tests:**
  - `test_retrieval_quality.py`

### 2. Answer Faithfulness (LLM Judge)
- **Purpose:** Detect hallucinations and ensure grounding
- **Metrics:** Faithfulness rate, hallucination detection
- **Method:** LLM-as-judge evaluation
- **Tests:**
  - `test_answer_faithfulness.py`

### 3. Answer Correctness (LLM Judge)
- **Purpose:** Verify factual accuracy vs ground truth
- **Metrics:** Correctness score, critical fact accuracy
- **Method:** LLM-as-judge semantic comparison
- **Tests:**
  - `test_answer_correctness.py`

### 4. Scope Enforcement (LLM Judge)
- **Purpose:** Test in-scope vs out-of-scope handling
- **Metrics:** Rejection rate, proper handling
- **Method:** LLM-as-judge evaluation
- **Tests:**
  - `test_scope_enforcement.py`

### 5. Response Quality (LLM Judge)
- **Purpose:** Evaluate tone, style, completeness
- **Metrics:** Tone score, completeness, professionalism
- **Method:** LLM-as-judge subjective evaluation
- **Tests:**
  - `test_response_quality.py`

### 6. Conversational Coherence (LLM Judge)
- **Purpose:** Test multi-turn context awareness
- **Metrics:** Context maintenance, coherence
- **Method:** LLM-as-judge conversation evaluation
- **Tests:**
  - `test_conversational_coherence.py`

---

## Thresholds

| Metric | Threshold | Category |
|--------|-----------|----------|
| Hit Rate | ≥85% | Retrieval |
| MRR | ≥70% | Retrieval |
| Precision@5 | ≥40% | Retrieval |
| Faithfulness | ≥95% (≤5% hallucination) | Faithfulness |
| Correctness | ≥85% | Correctness |
| Scope Handling | 100% proper handling | Scope |
| Tone Quality | ≥80% | Quality |
| Completeness | ≥85% | Quality |
| Context Awareness | ≥90% | Conversational |

---

## Key Features of This Evaluation

### Hybrid Approach
- **Deterministic tests** for retrieval (fast, reproducible)
- **LLM-as-judge** for semantic evaluation (handles paraphrasing)

### Production-Grade Metrics
- **Retrieval metrics** (Hit Rate, MRR, Precision) - standard IR metrics
- **Faithfulness** - critical for RAG systems (no hallucinations)
- **Correctness** - semantic accuracy vs ground truth
- **Quality** - customer service specific (tone, empathy, clarity)
- **Conversational** - multi-turn coherence and context

### Comprehensive Coverage
- 30+ evaluation cases across 6 policy topics
- Multiple difficulty levels (easy, medium, hard)
- Edge cases and adversarial prompts
- Topic-specific and difficulty-specific breakdowns

---

## Files Generated

- `report.html` - Full pytest HTML report
- `eval_summary_{timestamp}.md` - This summary (markdown)
- `retrieval_metrics_{timestamp}.json` - Detailed retrieval metrics
- Test-specific outputs in individual test files

---

## Judge LLM Configuration

The evaluation uses the LLM specified in the `JUDGE_LLM` environment variable.
- **Default:** `gemini-2.5-flash`
- **For production:** Set `JUDGE_LLM=gemini-2.0-pro` or similar advanced model

---

## Next Steps

1. **Review detailed report:** Open `report.html` in a browser
2. **Check failures:** Investigate any failed tests
3. **Analyze patterns:** Look for topic-specific or difficulty-specific issues
4. **Iterate:** Improve prompts, chunking, or retrieval based on results
5. **Regression testing:** Run this suite regularly to catch degradations

---

## Running Individual Test Categories

```bash
# Retrieval only (fast, deterministic)
pytest tests/test_retrieval_quality.py -v

# Faithfulness only
pytest tests/test_answer_faithfulness.py -v

# Correctness only
pytest tests/test_answer_correctness.py -v

# Scope enforcement only
pytest tests/test_scope_enforcement.py -v

# Quality only
pytest tests/test_response_quality.py -v

# Conversational only
pytest tests/test_conversational_coherence.py -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Run only slow tests
pytest tests/ -m "slow" -v
```

---

## Cost Considerations

**LLM Judge API Calls:**
- Quick mode: ~50-100 judge calls
- Standard mode: ~100-200 judge calls
- Full mode: ~200-300 judge calls

With Gemini Flash (very cheap), full evaluation costs < $0.10.

---

## Evaluation Philosophy

This test suite follows modern RAG evaluation best practices:

1. **Layer-specific testing:** Test retrieval and generation separately
2. **Hybrid metrics:** Deterministic where possible, LLM-judge for semantics
3. **Domain-specific:** Customer service specific criteria (tone, empathy)
4. **Production-realistic:** Real questions, edge cases, adversarial prompts
5. **Actionable:** Clear failures with explanations for debugging

---

*Generated by run_full_eval_suite.py*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\n✓ Summary report saved: {report_path}")


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description='Run RAG system evaluation suite'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (skip slow tests)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full mode (all tests including slow)'
    )

    args = parser.parse_args()

    # Determine mode
    if args.quick:
        mode = 'quick'
    elif args.full:
        mode = 'full'
    else:
        mode = 'standard'

    # Run evaluation
    exit_code = run_evaluation(mode)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()