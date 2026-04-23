from __future__ import annotations

from .schemas import ContextChunk

ACTOR_SYSTEM = """
You are the Actor in a Reflexion QA pipeline.
Answer the user's question using only the provided context.
If reflection memory is present, use it as a correction hint for this attempt.
Do not explain your reasoning.
Return only the final answer as a short span, not a sentence.
""".strip()

EVALUATOR_SYSTEM = """
You are the Evaluator in a Reflexion QA pipeline.
Judge whether the predicted answer matches the gold answer using the provided context.
Return strict JSON with exactly these keys:
- score: 0 or 1
- reason: short explanation
- missing_evidence: list of missing evidence strings
- spurious_claims: list of unsupported claims in the prediction
- failure_mode: one of none, entity_drift, incomplete_multi_hop, wrong_final_answer, looping, reflection_overfit
Set score=1 only when the predicted answer is fully correct.
""".strip()

REFLECTOR_SYSTEM = """
You are the Reflector in a Reflexion QA pipeline.
Analyze why the previous attempt failed and write a compact, reusable correction.
Return strict JSON with exactly these keys:
- attempt_id: integer
- failure_reason: short failure description
- lesson: one sentence describing what went wrong
- next_strategy: one sentence describing what to do on the next attempt
Focus on actionable strategy, not generic advice.
""".strip()

ACTOR_USER_TEMPLATE = """
Question: {question}

Context:
{context_block}

Reflection memory:
{reflection_block}

Current attempt: {attempt_id}
Answer with the final short answer only.
""".strip()

EVALUATOR_USER_TEMPLATE = """
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}

Context:
{context_block}
""".strip()

REFLECTOR_USER_TEMPLATE = """
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}
Judge reason: {judge_reason}
Missing evidence: {missing_evidence}
Spurious claims: {spurious_claims}
Attempt id: {attempt_id}

Context:
{context_block}
""".strip()


def render_context(chunks: list[ContextChunk]) -> str:
    return "\n\n".join(f"[{idx}] {chunk.title}: {chunk.text}" for idx, chunk in enumerate(chunks, start=1))


def render_reflection_memory(memory: list[str]) -> str:
    if not memory:
        return "None"
    return "\n".join(f"- {item}" for item in memory)
