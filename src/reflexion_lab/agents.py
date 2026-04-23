from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .mock_runtime import FAILURE_MODE_BY_QID
from .runtime import AgentRuntime, MockRuntime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: AgentRuntime = field(default_factory=MockRuntime)

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_failure_mode = "wrong_final_answer"

        for attempt_id in range(1, self.max_attempts + 1):
            actor_turn = self.runtime.actor(example, attempt_id, self.agent_type, reflection_memory)
            judge_turn = self.runtime.evaluate(example, actor_turn.answer)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=actor_turn.answer,
                score=judge_turn.result.score,
                reason=judge_turn.result.reason,
                token_estimate=actor_turn.token_estimate + judge_turn.token_estimate,
                latency_ms=actor_turn.latency_ms + judge_turn.latency_ms,
            )

            final_answer = actor_turn.answer
            final_score = judge_turn.result.score
            final_failure_mode = judge_turn.result.failure_mode

            if judge_turn.result.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection_turn = self.runtime.reflect(example, attempt_id, actor_turn.answer, judge_turn.result)
                reflection = reflection_turn.reflection
                reflections.append(reflection)
                reflection_memory.append(f"Attempt {attempt_id}: {reflection.next_strategy}")
                trace.reflection = reflection
                trace.token_estimate += reflection_turn.token_estimate
                trace.latency_ms += reflection_turn.latency_ms

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else final_failure_mode
        if self.runtime.mode == "mock" and failure_mode == "wrong_final_answer":
            failure_mode = FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime or MockRuntime())


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: AgentRuntime | None = None) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime or MockRuntime())
