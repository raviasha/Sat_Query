"""Bounded natural-language routing over SatQuery's deterministic spatial tools."""

from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

from ..prediction_data import class_schema
from .tools import TASKS, execute_task

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_QUESTION_LENGTH = 1000
_UNSET = object()


class ProviderError(RuntimeError):
    """A safe, user-actionable language-provider failure."""


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOL_REGISTRY = (
    {
        "type": "function",
        "name": "coverage",
        "description": "Estimate area and fraction for one supported land-cover class or alias.",
        "strict": True,
        "parameters": _object_schema({"class_name": {"type": "string"}}, ["class_name"]),
    },
    {
        "type": "function",
        "name": "presence",
        "description": "Apply an explicit coarse-cell coverage threshold for a supported class.",
        "strict": True,
        "parameters": _object_schema(
            {
                "class_name": {"type": "string"},
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            },
            ["class_name", "threshold"],
        ),
    },
    {
        "type": "function",
        "name": "describe",
        "description": "Describe the three largest estimated land-cover proportions.",
        "strict": True,
        "parameters": _object_schema({}, []),
    },
    {
        "type": "function",
        "name": "locate",
        "description": "Locate coarse 80 metre cells meeting a class-coverage threshold.",
        "strict": True,
        "parameters": _object_schema(
            {
                "class_name": {"type": "string"},
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            },
            ["class_name", "threshold"],
        ),
    },
    {
        "type": "function",
        "name": "change",
        "description": "Compare two aligned, ordered dates; class_name may be null for all classes.",
        "strict": True,
        "parameters": _object_schema({"class_name": {"type": ["string", "null"]}}, ["class_name"]),
    },
)

_ROUTER_INSTRUCTIONS = """You route one satellite question to exactly one function.
Only coverage, presence, describe, locate, and change are available. The measurements are
19-class land-cover fractions on coarse cells. They cannot count buildings, roads, vehicles,
ships, trees, or other objects. They do not support RGB, Cartosat, RISAT, benchmark scoring,
weather, or object detection. Never write or execute code. Choose a function only when its
documented measurement can answer the question. Use threshold 0.5 unless the user explicitly
provides another fraction from 0 through 1."""

_WORDING_INSTRUCTIONS = """Write at most 120 words using only the deterministic tool output.
Do not add measurements, objects, confidence, causation, or certainty. Preserve the stated
units, qualifiers, and limitations. If the tool abstained, explain the supported alternative."""


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _abstention(question: str, reason: str, *, provider: str, model: str | None) -> dict[str, Any]:
    answer = f"I cannot answer this request: {reason}"
    return {
        "provider": provider,
        "language_model": model,
        "question": question,
        "abstained": True,
        "answer": answer,
        "deterministic_answer": answer,
        "llm_wording": None,
        "measurements": [],
        "evidence": [],
        "limitations": [
            "SatQuery currently measures only the fixed 19-class land-cover coverage taxonomy."
        ],
        "trace": {
            "selected_tool": None,
            "provider": provider,
            "language_model": model,
            "parameters": {},
            "function_calls": [],
        },
    }


def _validate_arguments(name: str, arguments: Any) -> dict[str, Any] | None:
    if not isinstance(arguments, dict):
        return None
    expected = {
        "coverage": {"class_name"},
        "presence": {"class_name", "threshold"},
        "describe": set(),
        "locate": {"class_name", "threshold"},
        "change": {"class_name"},
    }[name]
    if set(arguments) != expected:
        return None
    if "class_name" in arguments:
        value = arguments["class_name"]
        if name == "change" and value is None:
            pass
        elif not isinstance(value, str) or not value.strip() or len(value) > 100:
            return None
    if "threshold" in arguments:
        threshold = arguments["threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            return None
    return arguments


def _compact_tool_output(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the language model grounded without sending dense grids or imagery."""
    return {
        "abstained": result["abstained"],
        "answer": result["answer"],
        "measurements": result["measurements"],
        "limitations": result["limitations"],
    }


def _numeric_values(value: Any) -> set[Decimal]:
    tokens = re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?", json.dumps(value))
    values = set()
    for token in tokens:
        try:
            values.add(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return values


def _wording_uses_only_computed_numbers(wording: str, compact: dict[str, Any]) -> bool:
    allowed = _numeric_values(compact)
    return _numeric_values(wording).issubset(allowed)


def _extract_class(question: str) -> str | None:
    aliases = ["built-up", "built up", "builtup", "forest", "farmland", "water"]
    names = [item["name"] for item in class_schema()]
    lowered = question.casefold()
    for name in sorted(aliases + names, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", lowered):
            return name
    return None


def _local_threshold(question: str) -> float:
    percent = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%", question)
    if percent:
        value = float(percent.group(1)) / 100
        if 0 <= value <= 1:
            return value
    fraction = re.search(r"\bthreshold\s+(0(?:\.\d+)?|1(?:\.0+)?)\b", question.casefold())
    return float(fraction.group(1)) if fraction else 0.5


def _local_route(question: str) -> tuple[str, dict[str, Any]] | None:
    lowered = question.casefold()
    if re.search(r"\b(how many|count|number of)\b", lowered):
        return None
    class_name = _extract_class(question)
    if re.search(r"\b(change|changed|difference|compare|before|after|between dates?)\b", lowered):
        return "change", {"class_name": class_name}
    if re.search(r"\b(where|locate|location|map|which cells?)\b", lowered):
        return "locate", {"class_name": class_name, "threshold": _local_threshold(question)}
    if re.search(r"\b(coverage|area|how much|percent|percentage|proportion)\b", lowered):
        return "coverage", {"class_name": class_name}
    if re.search(
        r"\b(present|presence|any|whether|is there|does .* contain|do you see)\b", lowered
    ):
        return "presence", {"class_name": class_name, "threshold": _local_threshold(question)}
    if re.search(
        r"\b(describe|summary|summarize|land cover|scene contents?)\b", lowered
    ) or re.search(r"\bwhat (?:does|is in) (?:this )?(?:scene|image)\b", lowered):
        return "describe", {}
    return None


class AssistantController:
    """Select one fixed measurement tool and optionally ask OpenAI to word its result."""

    def __init__(self, *, openai_client: Any = _UNSET, model: str | None = None):
        self.model = model or os.environ.get("SATQUERY_OPENAI_MODEL", DEFAULT_MODEL)
        self._client = openai_client

    def _openai_client(self) -> Any:
        if self._client is _UNSET:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception:  # noqa: BLE001 - SDK configuration errors stay behind a safe boundary
                self._client = None
        if self._client is None:
            raise ProviderError(
                "OpenAI is not configured; set OPENAI_API_KEY on the server or choose local mode."
            )
        return self._client

    @property
    def openai_configured(self) -> bool:
        try:
            self._openai_client()
        except ProviderError:
            return False
        return True

    @staticmethod
    def _validate_request(question: str, provider: str) -> str:
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > MAX_QUESTION_LENGTH
        ):
            raise ValueError(f"question must contain 1 to {MAX_QUESTION_LENGTH} characters")
        if provider not in ("local", "openai"):
            raise ValueError("provider must be local or openai")
        return question.strip()

    @staticmethod
    def _execute(
        question: str,
        scenes: list[Any],
        provider: str,
        model: str | None,
        task: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            result = execute_task(task, scenes, **arguments)
        except (TypeError, ValueError) as exc:
            return _abstention(question, str(exc), provider=provider, model=model)
        output = deepcopy(result)
        output.update(
            {
                "provider": provider,
                "language_model": model,
                "question": question,
                "deterministic_answer": result["answer"],
                "llm_wording": None,
            }
        )
        output["trace"] = {
            **result["trace"],
            "provider": provider,
            "language_model": model,
            "function_calls": [],
        }
        return output

    def _local_answer(self, question: str, scenes: list[Any]) -> dict[str, Any]:
        route = _local_route(question)
        if route is None:
            if re.search(r"\b(how many|count|number of)\b", question.casefold()):
                reason = (
                    "object counts cannot be derived from the land-cover coverage taxonomy. "
                    "Ask for estimated coverage, presence, or coarse locations instead."
                )
            else:
                reason = (
                    "the question does not map to coverage, presence, description, coarse "
                    "location, or two-date change."
                )
            return _abstention(question, reason, provider="local", model=None)
        task, arguments = route
        return self._execute(question, scenes, "local", None, task, arguments)

    @staticmethod
    def _provider_failure(exc: Exception) -> ProviderError:
        status = getattr(exc, "status_code", None)
        lowered = str(exc).casefold()
        if status in (401, 403) or "401" in lowered or "authentication" in lowered:
            return ProviderError(
                "OpenAI authentication failed; verify OPENAI_API_KEY on the server or choose local mode."
            )
        if status == 429 or "rate limit" in lowered:
            return ProviderError("OpenAI rate limit reached; retry later or choose local mode.")
        if "timeout" in lowered:
            return ProviderError("OpenAI timed out; retry or choose local mode.")
        return ProviderError("OpenAI request failed; retry or choose local mode.")

    def _openai_answer(
        self, question: str, scenes: list[Any], input_summary: dict[str, Any]
    ) -> dict[str, Any]:
        client = self._openai_client()
        base_input = [
            {"role": "developer", "content": _ROUTER_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "input_summary": input_summary},
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            response = client.responses.create(
                model=self.model,
                input=base_input,
                tools=[deepcopy(tool) for tool in TOOL_REGISTRY],
                tool_choice="required",
                store=False,
                max_output_tokens=160,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors must be sanitized
            raise self._provider_failure(exc) from None

        calls = [
            item
            for item in _field(response, "output", [])
            if _field(item, "type") == "function_call"
        ]
        if len(calls) != 1:
            return _abstention(
                question,
                "OpenAI must select exactly one bounded measurement tool.",
                provider="openai",
                model=self.model,
            )
        call = calls[0]
        name = _field(call, "name")
        if name not in TASKS:
            return _abstention(
                question,
                f"OpenAI selected unsupported tool {name!r}.",
                provider="openai",
                model=self.model,
            )
        raw_arguments = _field(call, "arguments")
        try:
            decoded = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            decoded = None
        arguments = _validate_arguments(name, decoded)
        if arguments is None:
            return _abstention(
                question,
                f"OpenAI returned invalid arguments for {name}.",
                provider="openai",
                model=self.model,
            )
        result = self._execute(question, scenes, "openai", self.model, name, arguments)
        compact = _compact_tool_output(result)
        trace_call = {"name": name, "arguments": arguments, "output": compact}
        result["trace"]["function_calls"] = [trace_call]
        if result["abstained"]:
            return result

        call_id = _field(call, "call_id")
        if not isinstance(call_id, str) or not call_id:
            return _abstention(
                question,
                "OpenAI returned a function call without a call identifier.",
                provider="openai",
                model=self.model,
            )
        continuation = base_input + [
            {
                "type": "function_call",
                "name": name,
                "arguments": raw_arguments,
                "call_id": call_id,
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(compact, allow_nan=False, separators=(",", ":")),
            },
        ]
        try:
            wording = client.responses.create(
                model=self.model,
                input=continuation,
                instructions=_WORDING_INSTRUCTIONS,
                store=False,
                max_output_tokens=200,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors must be sanitized
            raise self._provider_failure(exc) from None
        text = _field(wording, "output_text", "")
        if isinstance(text, str) and text.strip():
            candidate = text.strip()[:2000]
            if _wording_uses_only_computed_numbers(candidate, compact):
                result["llm_wording"] = candidate
            else:
                result["limitations"].append(
                    "Optional OpenAI wording was omitted because it introduced an ungrounded numeric value."
                )
        return result

    def answer(
        self,
        question: str,
        scenes: list[Any],
        input_summary: dict[str, Any],
        *,
        provider: str = "local",
    ) -> dict[str, Any]:
        """Answer one bounded question; imagery and dense grids never reach OpenAI."""
        normalized = self._validate_request(question, provider)
        if not isinstance(input_summary, dict):
            raise TypeError("input_summary must be a dictionary")
        if re.search(r"\b(how many|count|number of)\b", normalized.casefold()):
            model = self.model if provider == "openai" else None
            return _abstention(
                normalized,
                "object counts cannot be derived from the land-cover coverage taxonomy. "
                "Ask for estimated coverage, presence, or coarse locations instead.",
                provider=provider,
                model=model,
            )
        if provider == "local":
            return self._local_answer(normalized, scenes)
        return self._openai_answer(normalized, scenes, input_summary)
