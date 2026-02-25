"""
LLM-as-judge evaluators for RAG system quality assessment.
Uses JUDGE_LLM environment variable (defaults to gemini-2.5-flash).
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure

load_dotenv()
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize judge model from environment variable
JUDGE_MODEL_NAME = os.getenv("JUDGE_LLM", "gemini-2.5-flash")
judge_model = GenerativeModel(JUDGE_MODEL_NAME)


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {json_str[:200]}")
        raise e


def evaluate_faithfulness(
    question: str,
    context: str,
    response: str,
    model: Optional[GenerativeModel] = None
) -> Dict[str, Any]:
    """
    Evaluate if the response is fully grounded in the retrieved context.

    Args:
        question: User's question
        context: Retrieved context from RAG system
        response: System's generated response
        model: Optional custom judge model (uses global judge_model if None)

    Returns:
        Dict with keys:
            - is_faithful (bool): Whether response is grounded in context
            - confidence (float): Confidence score 0-1
            - reason (str): Explanation of judgment
            - hallucinated_claims (List[str]): Any invented facts detected
    """
    model = model or judge_model

    prompt = f"""You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

**Retrieved Context:**
{context}

**User Question:**
{question}

**System Response:**
{response}

**Task:**
Determine if the response is FULLY GROUNDED in the retrieved context. Every fact, number, date, policy detail, and claim must come directly from the context.

**Criteria:**
- ✅ FAITHFUL: Every piece of information comes from the context
- ❌ UNFAITHFUL: Contains information NOT in context, invented details, or speculative statements

**Important:**
- Paraphrasing is OK (e.g., "30 días" vs "un mes" if context mentions 30 days)
- Reasonable inferences based on context are OK
- Adding polite greetings/closings is OK
- But inventing facts, numbers, or policies is NOT OK

**Output ONLY valid JSON in this exact format:**
{{
  "is_faithful": true,
  "confidence": 0.95,
  "reason": "All information matches the context. The 30-day return policy, receipt requirement, and tag requirement are all mentioned in the retrieved context.",
  "hallucinated_claims": []
}}

Or if unfaithful:
{{
  "is_faithful": false,
  "confidence": 0.85,
  "reason": "Response claims phone number 123-456-7890 but context doesn't contain any phone number.",
  "hallucinated_claims": ["phone number 123-456-7890"]
}}
"""

    result = model.generate_content(prompt)
    parsed = _parse_json_response(result.text)

    # Validate required fields
    required_fields = ['is_faithful', 'confidence', 'reason', 'hallucinated_claims']
    for field in required_fields:
        if field not in parsed:
            raise ValueError(f"Missing required field: {field}")

    return parsed


def evaluate_correctness(
    question: str,
    expected_answer: str,
    actual_answer: str,
    model: Optional[GenerativeModel] = None
) -> Dict[str, Any]:
    """
    Evaluate if the actual answer is factually correct vs ground truth.

    Args:
        question: User's question
        expected_answer: Ground truth answer
        actual_answer: System's generated answer
        model: Optional custom judge model

    Returns:
        Dict with keys:
            - correctness_score (float): 0-1 score
            - is_correct (bool): Whether answer is substantially correct
            - explanation (str): Detailed explanation
            - key_differences (List[str]): Any factual discrepancies
    """
    model = model or judge_model

    prompt = f"""You are evaluating a customer service agent's answer for factual correctness.

**Question:**
{question}

**Expected Answer (Ground Truth):**
{expected_answer}

**Actual Answer:**
{actual_answer}

**Task:**
Rate the factual correctness of the actual answer compared to the expected answer.

**Scoring Guidelines:**
- 1.0 = Perfectly correct, all key facts match
- 0.9 = Correct with minor phrasing differences
- 0.7-0.8 = Mostly correct but missing some details
- 0.5-0.6 = Partially correct but has significant omissions
- 0.3-0.4 = Major factual errors
- 0.0-0.2 = Completely incorrect

**Important:**
- Different phrasing is OK ("30 días" vs "un mes" for 30 days)
- Extra helpful information is OK (doesn't lower score)
- Missing key facts DOES lower score
- Wrong numbers/dates/policies = major penalty

**Output ONLY valid JSON:**
{{
  "correctness_score": 0.95,
  "is_correct": true,
  "explanation": "Answer correctly states 30-day return period and receipt requirement. Phrasing differs but facts are accurate.",
  "key_differences": []
}}

Or if incorrect:
{{
  "correctness_score": 0.4,
  "is_correct": false,
  "explanation": "Answer states 15 days but correct answer is 30 days. Critical factual error.",
  "key_differences": ["Wrong timeframe: 15 days vs 30 days"]
}}
"""

    result = model.generate_content(prompt)
    parsed = _parse_json_response(result.text)

    # Validate and add is_correct field if missing
    if 'is_correct' not in parsed:
        parsed['is_correct'] = parsed.get('correctness_score', 0) >= 0.7

    return parsed


def evaluate_scope_handling(
    question: str,
    response: str,
    model: Optional[GenerativeModel] = None
) -> Dict[str, Any]:
    """
    Evaluate if the system properly handles in-scope vs out-of-scope questions.

    Args:
        question: User's question
        response: System's response
        model: Optional custom judge model

    Returns:
        Dict with keys:
            - is_in_scope (bool): Whether question is in scope
            - properly_handled (bool): Whether response appropriately handled scope
            - explanation (str): Reasoning
    """
    model = model or judge_model

    prompt = f"""You are evaluating a customer service agent for Quest Colombia (fashion retail company).

**Agent's Role:**
The agent is specialized in POST-SALES support for:
- Returns, exchanges, warranties
- Shipping and delivery
- Payment methods and refunds
- Shopping process and order tracking
- Contact information

The agent should REJECT questions about:
- Weather, news, general knowledge
- Other companies/products
- Personal tasks unrelated to customer service
- Pre-sales questions like product recommendations (edge case - can be flexible)

**User Question:**
{question}

**Agent Response:**
{response}

**Task:**
1. Determine if the question is in-scope for post-sales customer service
2. Determine if the agent properly handled it (answered if in-scope, politely rejected if out-of-scope)

**Output ONLY valid JSON:**
{{
  "is_in_scope": true,
  "properly_handled": true,
  "explanation": "Question about returns is clearly in-scope. Agent provided appropriate answer."
}}

Or for out-of-scope:
{{
  "is_in_scope": false,
  "properly_handled": true,
  "explanation": "Question about weather is out-of-scope. Agent correctly declined and stayed within their role."
}}

Or for mishandled:
{{
  "is_in_scope": false,
  "properly_handled": false,
  "explanation": "Question about weather is out-of-scope but agent attempted to answer it instead of politely declining."
}}
"""

    result = model.generate_content(prompt)
    parsed = _parse_json_response(result.text)

    return parsed


def evaluate_tone(
    response: str,
    model: Optional[GenerativeModel] = None
) -> Dict[str, Any]:
    """
    Evaluate tone and style of customer service response.

    Args:
        response: System's response
        model: Optional custom judge model

    Returns:
        Dict with keys:
            - tone_score (float): Overall tone quality 0-1
            - is_professional (bool)
            - is_empathetic (bool)
            - is_clear (bool)
            - is_structured (bool)
            - feedback (str): Improvement suggestions
    """
    model = model or judge_model

    prompt = f"""Evaluate this customer service response for tone and style quality.

**Response:**
{response}

**Criteria:**
1. **Professional but warm** - Formal but not robotic, friendly but not casual
2. **Empathetic** - Shows understanding of customer's situation
3. **Clear and concise** - No jargon, gets to the point
4. **Structured** - Uses lists, bullets, or clear paragraphs when appropriate

**Scoring:**
- Each criterion is true/false
- tone_score = (number of true criteria) / 4
- Provide specific feedback for improvement

**Output ONLY valid JSON:**
{{
  "tone_score": 0.85,
  "is_professional": true,
  "is_empathetic": true,
  "is_clear": true,
  "is_structured": false,
  "feedback": "Response is professional and clear but could benefit from using bullet points to list the requirements."
}}
"""

    result = model.generate_content(prompt)
    parsed = _parse_json_response(result.text)

    return parsed


def evaluate_completeness(
    question: str,
    response: str,
    required_elements: List[str],
    model: Optional[GenerativeModel] = None
) -> Dict[str, Any]:
    """
    Evaluate if response includes all necessary information elements.

    Args:
        question: User's question
        response: System's response
        required_elements: List of required information elements (e.g., ["plazo", "requisitos"])
        model: Optional custom judge model

    Returns:
        Dict with keys:
            - is_complete (bool)
            - completeness_score (float): 0-1
            - found_elements (List[str]): Which required elements were found
            - missing_elements (List[str]): Which required elements are missing
            - explanation (str)
    """
    model = model or judge_model

    prompt = f"""Evaluate if this customer service response includes all necessary information.

**User Question:**
{question}

**System Response:**
{response}

**Required Information Elements:**
{json.dumps(required_elements, ensure_ascii=False)}

**Task:**
Check if the response addresses each required element. Elements can be mentioned with different wording (e.g., "30 días" and "un mes" both count for "plazo de 30 días").

**Output ONLY valid JSON:**
{{
  "is_complete": true,
  "completeness_score": 1.0,
  "found_elements": ["30 días", "etiquetas", "recibo"],
  "missing_elements": [],
  "explanation": "All required elements are present. Response mentions the 30-day period, need for tags, and receipt requirement."
}}

Or if incomplete:
{{
  "is_complete": false,
  "completeness_score": 0.67,
  "found_elements": ["30 días", "etiquetas"],
  "missing_elements": ["recibo"],
  "explanation": "Response mentions timeframe and tags but doesn't mention the receipt requirement."
}}
"""

    result = model.generate_content(prompt)
    parsed = _parse_json_response(result.text)

    return parsed


def evaluate_context_awareness(
    conversation_history: List[Dict[str, str]],
    latest_response: str,
    model: Optional[GenerativeModel] = None
) -> Dict[str, Any]:
    """
    Evaluate if the agent maintains context across conversation turns.

    Args:
        conversation_history: List of dicts with 'role' and 'contenido' keys
        latest_response: The most recent agent response
        model: Optional custom judge model

    Returns:
        Dict with keys:
            - maintains_context (bool)
            - context_score (float): 0-1
            - explanation (str)
    """
    model = model or judge_model

    # Format conversation history
    history_str = ""
    for msg in conversation_history:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('contenido', '')
        history_str += f"{role}: {content}\n"

    prompt = f"""Evaluate if this agent response appropriately uses context from the conversation history.

**Conversation History:**
{history_str}

**Latest Agent Response:**
{latest_response}

**Task:**
Determine if the agent:
1. Remembers previous context (doesn't ask for already-provided info)
2. Answers follow-up questions appropriately
3. Maintains topic coherence
4. Uses pronouns/references correctly (e.g., "it", "that", "your order")

**Output ONLY valid JSON:**
{{
  "maintains_context": true,
  "context_score": 0.95,
  "explanation": "Agent correctly references the previous question about returns and provides relevant follow-up information without asking redundant questions."
}}

Or if context is lost:
{{
  "maintains_context": false,
  "context_score": 0.3,
  "explanation": "User already mentioned buying shoes but agent asks 'what product did you buy?' - shows lack of context awareness."
}}
"""

    result = model.generate_content(prompt)
    parsed = _parse_json_response(result.text)

    return parsed