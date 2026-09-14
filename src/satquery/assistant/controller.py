"""Bounded natural-language routing over SatQuery's deterministic spatial tools."""

from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
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
provides another fraction from 0 through 1. If you include companion text with the function
call, keep it general and include no numbers or numeric claims; all measurements are produced
locally after routing."""

_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|token)", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"(?i)(?:bearer\s+\S+|\b(?:sk|rk|pk)-[A-Za-z0-9_-]{4,}\b)")
_ABSOLUTE_PATH = re.compile(
    r"(?<![\w.])(?:/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+|[A-Za-z]:\\[^\s'\"]+)"
)
_NUMBER_WORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|half|quarter)\b",
    re.IGNORECASE,
)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _abstention(
    question: str,
    reason: str,
    *,
    provider: str,
    model: str | None,
    function_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
            "function_calls": function_calls or [],
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


def _sanitize_string(value: str) -> str:
    sanitized = _SECRET_VALUE.sub("[redacted]", value)
    sanitized = _ABSOLUTE_PATH.sub("[path]", sanitized)
    return sanitized[:256]


def _sanitize_trace_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "[truncated]"
    if isinstance(value, dict):
        sanitized = {}
        for key, item in list(value.items())[:12]:
            safe_key = _sanitize_string(str(key))
            sanitized[safe_key] = (
                "[redacted]"
                if _SECRET_KEY.search(safe_key)
                else _sanitize_trace_value(item, depth=depth + 1)
            )
        return sanitized
    if isinstance(value, list):
        return [_sanitize_trace_value(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "[invalid number]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_string(str(value))


def _decoded_arguments(raw_arguments: Any) -> Any:
    try:
        return json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return None


def _call_record(call: Any) -> dict[str, Any]:
    raw_arguments = _field(call, "arguments")
    decoded = _decoded_arguments(raw_arguments)
    arguments = (
        _sanitize_trace_value(decoded)
        if decoded is not None
        else {"unparsed": _sanitize_string(raw_arguments if isinstance(raw_arguments, str) else "")}
    )
    raw_name = _field(call, "name")
    raw_call_id = _field(call, "call_id")
    return {
        "name": _sanitize_string(raw_name if isinstance(raw_name, str) else "[invalid]"),
        "call_id": _sanitize_string(raw_call_id if isinstance(raw_call_id, str) else ""),
        "arguments": arguments,
    }


def _rejected_calls(calls: list[Any], reason: str) -> list[dict[str, Any]]:
    return [{**_call_record(call), "accepted": False, "rejection_reason": reason} for call in calls]


def _contains_numeric_claim(wording: str) -> bool:
    return bool(re.search(r"\d", wording) or _NUMBER_WORD.search(wording))


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
                function_calls=_rejected_calls(calls, "multiple function calls returned"),
            )
        call = calls[0]
        name = _field(call, "name")
        if name not in TASKS:
            records = _rejected_calls([call], "unsupported tool")
            return _abstention(
                question,
                f"OpenAI selected unsupported tool {records[0]['name']!r}.",
                provider="openai",
                model=self.model,
                function_calls=records,
            )
        raw_arguments = _field(call, "arguments")
        decoded = _decoded_arguments(raw_arguments)
        arguments = _validate_arguments(name, decoded)
        if arguments is None:
            records = _rejected_calls([call], "invalid function arguments")
            return _abstention(
                question,
                f"OpenAI returned invalid arguments for {name}.",
                provider="openai",
                model=self.model,
                function_calls=records,
            )
        call_id = _field(call, "call_id")
        if not isinstance(call_id, str) or not call_id:
            records = _rejected_calls([call], "missing function call identifier")
            return _abstention(
                question,
                "OpenAI returned a function call without a call identifier.",
                provider="openai",
                model=self.model,
                function_calls=records,
            )
        result = self._execute(question, scenes, "openai", self.model, name, arguments)
        compact = _compact_tool_output(result)
        trace_call = {
            **_call_record(call),
            "accepted": True,
            "rejection_reason": None,
            "output": compact,
        }
        result["trace"]["function_calls"] = [trace_call]
        if result["abstained"]:
            return result

        text = _field(response, "output_text", "")
        if isinstance(text, str) and text.strip():
            candidate = text.strip()[:2000]
            if not _contains_numeric_claim(candidate):
                result["llm_wording"] = candidate
            else:
                result["limitations"].append(
                    "Optional OpenAI wording was omitted because it contained a numeric claim; "
                    "all measurements remain in the deterministic answer and evidence."
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
