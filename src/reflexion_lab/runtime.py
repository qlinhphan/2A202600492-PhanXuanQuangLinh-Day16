from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from . import mock_runtime
from .prompts import (
    ACTOR_SYSTEM,
    ACTOR_USER_TEMPLATE,
    EVALUATOR_SYSTEM,
    EVALUATOR_USER_TEMPLATE,
    REFLECTOR_SYSTEM,
    REFLECTOR_USER_TEMPLATE,
    render_context,
    render_reflection_memory,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry


@dataclass
class ActorTurn:
    answer: str
    token_estimate: int
    latency_ms: int


@dataclass
class JudgeTurn:
    result: JudgeResult
    token_estimate: int
    latency_ms: int


@dataclass
class ReflectionTurn:
    reflection: ReflectionEntry
    token_estimate: int
    latency_ms: int


class AgentRuntime(Protocol):
    mode: str

    def actor(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> ActorTurn:
        ...

    def evaluate(self, example: QAExample, answer: str) -> JudgeTurn:
        ...

    def reflect(self, example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> ReflectionTurn:
        ...


class MockRuntime:
    mode = "mock"

    def actor(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> ActorTurn:
        answer = mock_runtime.actor_answer(example, attempt_id, agent_type, reflection_memory)
        token_estimate = 320 + (attempt_id * 65) + (120 if agent_type == "reflexion" else 0)
        latency_ms = 160 + (attempt_id * 40) + (90 if agent_type == "reflexion" else 0)
        return ActorTurn(answer=answer, token_estimate=token_estimate, latency_ms=latency_ms)

    def evaluate(self, example: QAExample, answer: str) -> JudgeTurn:
        result = mock_runtime.evaluator(example, answer)
        return JudgeTurn(result=result, token_estimate=90, latency_ms=40)

    def reflect(self, example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> ReflectionTurn:
        reflection = mock_runtime.reflector(example, attempt_id, judge)
        return ReflectionTurn(reflection=reflection, token_estimate=140, latency_ms=60)


class OpenAICompatibleRuntime:
    mode = "live"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "OpenAICompatibleRuntime":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("MODEL_CHAT", "").strip() or "gpt-4o-mini"
        base_url = os.getenv("BASE_URL", "").strip() or "https://api.openai.com/v1"
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment or .env.")
        return cls(api_key=api_key, model=model, base_url=base_url)

    def actor(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> ActorTurn:
        user_prompt = ACTOR_USER_TEMPLATE.format(
            question=example.question,
            context_block=render_context(example.context),
            reflection_block=render_reflection_memory(reflection_memory),
            attempt_id=attempt_id,
        )
        response = self._chat(
            [
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=128,
        )
        return ActorTurn(
            answer=self._clean_actor_answer(response.content),
            token_estimate=response.total_tokens,
            latency_ms=response.latency_ms,
        )

    def evaluate(self, example: QAExample, answer: str) -> JudgeTurn:
        user_prompt = EVALUATOR_USER_TEMPLATE.format(
            question=example.question,
            gold_answer=example.gold_answer,
            predicted_answer=answer,
            context_block=render_context(example.context),
        )
        parsed, response = self._structured_chat(
            [
                {"role": "system", "content": EVALUATOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            schema=JudgeResult,
            max_tokens=256,
        )
        return JudgeTurn(result=parsed, token_estimate=response.total_tokens, latency_ms=response.latency_ms)

    def reflect(self, example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> ReflectionTurn:
        user_prompt = REFLECTOR_USER_TEMPLATE.format(
            question=example.question,
            gold_answer=example.gold_answer,
            predicted_answer=answer,
            judge_reason=judge.reason,
            missing_evidence=", ".join(judge.missing_evidence) or "None",
            spurious_claims=", ".join(judge.spurious_claims) or "None",
            attempt_id=attempt_id,
            context_block=render_context(example.context),
        )
        parsed, response = self._structured_chat(
            [
                {"role": "system", "content": REFLECTOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            schema=ReflectionEntry,
            max_tokens=256,
        )
        parsed = parsed.model_copy(update={"attempt_id": attempt_id})
        return ReflectionTurn(reflection=parsed, token_estimate=response.total_tokens, latency_ms=response.latency_ms)

    @dataclass
    class _ChatResponse:
        content: str
        total_tokens: int
        latency_ms: int

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> _ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        started = time.perf_counter()
        raw = self._post_json("/chat/completions", payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = raw["choices"][0]["message"]["content"]
        content = self._coerce_content(choice)
        usage = raw.get("usage", {})
        total_tokens = int(usage.get("total_tokens") or self._estimate_message_tokens(messages, content))
        return self._ChatResponse(content=content, total_tokens=total_tokens, latency_ms=latency_ms)

    def _structured_chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: type[JudgeResult] | type[ReflectionEntry],
        max_tokens: int,
    ) -> tuple[Any, _ChatResponse]:
        try:
            response = self._chat(
                messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except RuntimeError:
            response = self._chat(messages, temperature=0.0, max_tokens=max_tokens)
        parsed_json = self._extract_json_object(response.content)
        return schema.model_validate(parsed_json), response

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse JSON object from response: {text}")
            return json.loads(match.group(0))

    @staticmethod
    def _clean_actor_answer(text: str) -> str:
        text = text.strip()
        lowered = text.lower()
        if lowered.startswith("final answer:"):
            return text.split(":", 1)[1].strip()
        return text

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, str]], content: str) -> int:
        text = "".join(message["content"] for message in messages) + content
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))
