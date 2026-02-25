# Testing Guide - RAG Evaluation Suite

This guide explains how to use the comprehensive evaluation suite for the customer service RAG system.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Architecture](#test-architecture)
3. [Running Tests](#running-tests)
4. [Understanding Results](#understanding-results)
5. [Adding New Tests](#adding-new-tests)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **ChromaDB running** on `localhost:8001`
2. **Documents indexed** (run `python scripts/indexar.py`)
3. **Environment variables** set in `.env`:
   ```
   GOOGLE_API_KEY=your_api_key
   JUDGE_LLM=gemini-2.5-flash  # Optional, defaults to this
   ```

### Run Full Evaluation

```bash
# Standard mode (recommended)
python tests/run_full_eval_suite.py

# Quick mode (faster, skips slow tests)
python tests/run_full_eval_suite.py --quick

# Full mode (comprehensive, slower)
python tests/run_full_eval_suite.py --full
```

### View Results

Results are saved to `eval_results/`:
- **`report.html`** - Open in browser for interactive results
- **`eval_summary_[timestamp].md`** - Methodology and summary
- **`retrieval_metrics_[timestamp].json`** - Detailed metrics

---

## Test Architecture

### Hybrid Approach

The suite uses **two complementary testing strategies**:

#### 1. Deterministic Tests (Retrieval)
- **What:** Measure vector search quality
- **Why:** Retrieval is deterministic (same input → same output)
- **Metrics:** Hit Rate, MRR, Precision@k
- **Speed:** Fast (~10 seconds)

#### 2. LLM-as-Judge Tests (Generation)
- **What:** Evaluate semantic quality of responses
- **Why:** Handles paraphrasing, tone, context awareness
- **Metrics:** Faithfulness, correctness, quality
- **Speed:** Slower (~2-5 minutes depending on sample size)

### Test Categories

| Test File | Category | Method | What It Tests |
|-----------|----------|--------|---------------|
| `test_retrieval_quality.py` | Retrieval | Deterministic | Vector search accuracy |
| `test_answer_faithfulness.py` | Faithfulness | LLM Judge | Hallucination detection |
| `test_answer_correctness.py` | Correctness | LLM Judge | Factual accuracy |
| `test_scope_enforcement.py` | Scope | LLM Judge | In/out-of-scope handling |
| `test_response_quality.py` | Quality | LLM Judge | Tone, style, completeness |
| `test_conversational_coherence.py` | Conversational | LLM Judge | Multi-turn context |

---

## Running Tests

### By Category

```bash
# Retrieval only (fast)
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
```

### By Speed

```bash
# Only fast tests (deterministic retrieval tests)
pytest tests/ -m "not slow" -v

# Only slow tests (LLM judge tests)
pytest tests/ -m "slow" -v
```

### Individual Tests

```bash
# Run a specific test function
pytest tests/test_retrieval_quality.py::test_hit_rate_threshold -v

# Run with detailed output
pytest tests/test_answer_faithfulness.py -v -s

# Run with specific markers
pytest tests/ -m "retrieval or faithfulness" -v
```

### Debugging

```bash
# Stop on first failure
pytest tests/ -x

# Print all output (including print statements)
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -v -l

# Run last failed tests only
pytest tests/ --lf
```

---

## Understanding Results

### Retrieval Metrics

**Hit Rate (Target: ≥85%)**
- Percentage of queries where correct document appears in top-k
- Low hit rate = poor retrieval, wrong chunks returned

**MRR - Mean Reciprocal Rank (Target: ≥70%)**
- How high the correct document ranks
- Low MRR = correct chunks rank too low

**Precision@k (Target: ≥40%)**
- Proportion of top-k results that are relevant
- Low precision = many irrelevant chunks in results

### LLM Judge Scores

**Faithfulness (Target: ≥95%, ≤5% hallucination)**
- Are responses grounded in retrieved context?
- Critical for RAG systems - NO invented facts

**Correctness (Target: ≥85%)**
- Semantic accuracy vs ground truth answers
- Handles paraphrasing (e.g., "30 días" = "un mes")

**Scope Enforcement (Target: 100%)**
- Properly reject out-of-scope questions
- Properly answer in-scope questions

**Tone Quality (Target: ≥80%)**
- Professional, empathetic, clear, structured

**Completeness (Target: ≥85%)**
- Includes all required information elements

**Context Awareness (Target: ≥90%)**
- Maintains context across conversation turns

### Reading Test Output

```
RETRIEVAL QUALITY BY TOPIC
============================================================

RETURNS (8 cases):
  Hit Rate: 87.50%    ← Good!
  MRR: 0.85           ← Excellent ranking

SHIPPING (5 cases):
  Hit Rate: 60.00%    ← ⚠️ Below threshold!
  MRR: 0.45           ← ⚠️ Poor ranking
```

This tells you: Shipping queries need better retrieval. Possible fixes:
- Add more shipping policy examples
- Adjust chunking strategy
- Check if shipping docs are properly indexed

---

## Adding New Tests

### Adding Test Cases

Edit `tests/fixtures/eval_dataset.json`:

```json
{
  "question": "Your new test question?",
  "expected_answer": "The correct answer",
  "expected_source": "source_file.txt",
  "required_elements": ["key", "facts", "to", "check"],
  "topic": "returns",
  "difficulty": "medium"
}
```

### Creating New Test Functions

```python
# tests/test_my_new_test.py

import pytest

@pytest.mark.yourmarker
def test_my_new_check(eval_dataset, rag_pipeline, judge_llm):
    """
    Describe what this test validates.
    """
    for case in eval_dataset:
        result = rag_pipeline(case['question'])
        # Your assertions here
        assert some_condition, "Helpful error message"
```

### Adding New Evaluators

```python
# tests/evaluators/llm_judge.py

def evaluate_your_criterion(
    question: str,
    response: str,
    model: Optional[GenerativeModel] = None
) -> Dict[str, Any]:
    """
    Evaluate a new criterion using LLM judge.
    """
    model = model or judge_model

    prompt = f"""
    You are evaluating...

    **Question:** {question}
    **Response:** {response}

    **Output ONLY valid JSON:**
    {{
      "score": 0.0-1.0,
      "explanation": "..."
    }}
    """

    result = model.generate_content(prompt)
    return _parse_json_response(result.text)
```

---

## Troubleshooting

### Common Issues

**"ChromaDB connection refused"**
```bash
# Start ChromaDB
chroma run --host localhost --port 8001 --path ./chroma
```

**"No documents found / Hit rate 0%"**
```bash
# Index documents first
python scripts/indexar.py
```

**"Google API key error"**
```bash
# Check .env file
cat .env  # or type .env on Windows
# Should contain: GOOGLE_API_KEY=your_key_here
```

**Tests are slow**
```bash
# Use quick mode
python tests/run_full_eval_suite.py --quick

# Or skip slow tests
pytest tests/ -m "not slow" -v
```

**JSON parsing errors from LLM judge**
- Judge LLM didn't output valid JSON
- Try using a more capable model: `JUDGE_LLM=gemini-3.1-pro`
- Check the error message for what the LLM actually returned

**Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Debugging Failed Tests

1. **Run single test with verbose output:**
   ```bash
   pytest tests/test_answer_faithfulness.py::test_no_hallucination_in_responses -v -s
   ```

2. **Check what the RAG system returned:**
   - Tests print question, response, and judgment
   - Look for patterns in failures

3. **Inspect evaluation dataset:**
   - Check `tests/fixtures/eval_dataset.json`
   - Ensure questions match your indexed documents

4. **Review LLM judge reasoning:**
   - Judge explanations are printed in test output
   - Look for "Reason:" and "Explanation:" fields

5. **Test retrieval separately:**
   ```python
   from app.services.rag_service import buscar_contexto
   context = buscar_contexto("your question")
   print(context)
   ```

### Performance Tuning

**Reduce sample sizes** (in conftest.py or individual tests):
```python
# In test file
sample_size = min(10, len(eval_dataset))  # Instead of 20 or 30
sample = eval_dataset[:sample_size]
```

**Run only changed tests:**
```bash
# After fixing retrieval
pytest tests/test_retrieval_quality.py -v

# After fixing prompts
pytest tests/test_answer_correctness.py -v
```

**Parallelize tests** (experimental):
```bash
pip install pytest-xdist
pytest tests/ -n auto  # Use all CPU cores
```

---

## Best Practices

### When to Run Tests

- **Before committing:** Run quick mode
- **Before deploying:** Run full evaluation
- **After changing prompts:** Run faithfulness + correctness
- **After re-indexing:** Run retrieval quality
- **Weekly/monthly:** Full evaluation for regression detection

### Interpreting Failures

**Retrieval failures:**
- Problem with indexing or chunking
- Need better document coverage

**Faithfulness failures:**
- LLM hallucinating facts
- Prompt needs stronger grounding instructions

**Correctness failures:**
- Retrieved context is correct but LLM misinterprets
- Prompt engineering issue

**Scope failures:**
- System prompt needs clearer boundaries
- Need more explicit rejection instructions

**Quality failures:**
- Tone/style issues in prompt
- Need more examples of good responses

**Conversational failures:**
- Session management bug
- History not properly passed to LLM

---

## Advanced Usage

### Custom Judge Model

```bash
# In .env
JUDGE_LLM=gemini-3.1-pro-exp  # More capable but slower/pricier
```

### Regression Testing

```bash
# Save baseline results
python tests/run_full_eval_suite.py > baseline_results.txt

# After changes, compare
python tests/run_full_eval_suite.py > new_results.txt
diff baseline_results.txt new_results.txt
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: RAG Evaluation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run quick evaluation
        run: |
          python tests/run_full_eval_suite.py --quick
```

---

## Questions?

For issues or questions about the test suite:
1. Check this guide
2. Review test output and error messages
3. Inspect `eval_results/report.html` for detailed results
4. Check individual test files for documentation

---

*Last updated: 2025-02-25*