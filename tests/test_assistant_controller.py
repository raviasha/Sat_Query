import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

GRID = {
    "crs": "EPSG:32633",
    "transform": [10.0, 0.0, 100.0, 0.0, -10.0, 1300.0],
    "bounds": [100.0, 100.0, 1300.0, 1300.0],
    "width": 120,
    "height": 120,
    "pixel_size_metres": 10.0,
}


def _scene():
    from satquery.assistant.runtime import SceneResult

    coverage = np.zeros((15, 15, 19), dtype=np.float32)
    coverage[..., 2] = 0.7
    coverage[..., 8] = 0.3
    return SceneResult(
        id="scene-1",
        modality="optical",
        acquired=date(2026, 1, 2),
        coverage=coverage,
        preview=np.zeros((120, 120, 3), dtype=np.uint8),
        grid=dict(GRID),
        provenance={"feature_key": "optical_encodings"},
    )


class _FakeResponses:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def _client(*responses):
    api = _FakeResponses(responses)
    return SimpleNamespace(responses=api), api


def _tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id
    )


def _raw_tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id)


def test_local_mode_routes_supported_question_without_openai():
    from satquery.assistant.controller import AssistantController

    controller = AssistantController(openai_client=None)
    result = controller.answer(
        "How much forest coverage is present?",
        [_scene()],
        {"kind": "single", "observations": [{"id": "scene-1"}]},
        provider="local",
    )

    assert result["provider"] == "local"
    assert result["trace"]["selected_tool"] == "coverage"
    assert result["measurements"][0]["class_name"] == "forest"
    assert result["deterministic_answer"] == result["answer"]
    assert result["llm_wording"] is None


def test_local_mode_prefers_location_intent_when_question_mentions_coverage():
    from satquery.assistant.controller import AssistantController

    result = AssistantController(openai_client=None).answer(
        "Where is water coverage above 50%?",
        [_scene()],
        {"kind": "single"},
        provider="local",
    )

    assert result["trace"]["selected_tool"] == "locate"
    assert result["trace"]["parameters"]["threshold"] == 0.5


def test_local_mode_routes_presence_and_abstains_from_unrelated_questions():
    from satquery.assistant.controller import AssistantController

    controller = AssistantController(openai_client=None)
    present = controller.answer("Is there water?", [_scene()], {"kind": "single"}, provider="local")
    weather = controller.answer(
        "What is the weather tomorrow?", [_scene()], {"kind": "single"}, provider="local"
    )

    assert present["trace"]["selected_tool"] == "presence"
    assert weather["abstained"] is True
    assert weather["trace"]["selected_tool"] is None


@pytest.mark.parametrize(
    "question",
    [
        "How many buildings are visible?",
        "Count the roads in this image",
        "How many ships are there?",
    ],
)
def test_local_mode_declines_object_counts_the_coverage_taxonomy_cannot_answer(question):
    from satquery.assistant.controller import AssistantController

    result = AssistantController(openai_client=None).answer(
        question, [_scene()], {"kind": "single"}, provider="local"
    )

    assert result["abstained"] is True
    assert result["trace"]["selected_tool"] is None
    assert "count" in result["answer"].lower()
    assert "coverage taxonomy" in result["answer"].lower()


def test_object_count_is_declined_before_any_openai_request():
    from satquery.assistant.controller import AssistantController

    client, api = _client()
    result = AssistantController(openai_client=client).answer(
        "Count the buildings", [_scene()], {"kind": "single"}, provider="openai"
    )

    assert result["abstained"] is True
    assert "coverage taxonomy" in result["answer"].lower()
    assert api.calls == []


def test_openai_uses_one_strict_allowlisted_call_and_keeps_nonnumeric_wording():
    from satquery.assistant.controller import AssistantController

    routing = SimpleNamespace(
        output=[_tool_call("coverage", {"class_name": "forest"})],
        output_text="A deterministic forest coverage measurement follows.",
    )
    client, api = _client(routing)
    controller = AssistantController(openai_client=client)
    summary = {"kind": "single", "observations": [{"id": "scene-1"}]}

    result = controller.answer("How much forest is there?", [_scene()], summary, provider="openai")

    assert result["provider"] == "openai"
    assert result["llm_wording"] == "A deterministic forest coverage measurement follows."
    assert result["deterministic_answer"].startswith("Estimated forest coverage")
    assert len(api.calls) == 1
    first = api.calls[0]
    assert first["model"] == "gpt-4.1-mini-2025-04-14"
    assert first["store"] is False
    assert first["tool_choice"] == "required"
    assert {tool["name"] for tool in first["tools"]} == {
        "coverage",
        "presence",
        "describe",
        "locate",
        "change",
    }
    assert all(tool["type"] == "function" and tool["strict"] is True for tool in first["tools"])
    assert all(tool["parameters"]["additionalProperties"] is False for tool in first["tools"])
    assert all(
        set(tool["parameters"]["properties"]) == set(tool["parameters"]["required"])
        for tool in first["tools"]
    )
    serialized_calls = json.dumps(api.calls)
    assert "normalized" not in serialized_calls
    assert "/tmp/" not in serialized_calls
    assert result["trace"]["function_calls"][0]["arguments"] == {"class_name": "forest"}
    assert result["trace"]["function_calls"][0]["accepted"] is True
    assert "output" in result["trace"]["function_calls"][0]


@pytest.mark.parametrize(
    "wording",
    [
        "Forest has 0.3% coverage.",
        "Class 8 is dominant.",
    ],
)
def test_openai_wording_with_any_numeric_claim_is_omitted(wording):
    from satquery.assistant.controller import AssistantController

    routing = SimpleNamespace(
        output=[_tool_call("coverage", {"class_name": "forest"})], output_text=wording
    )
    client, api = _client(routing)

    result = AssistantController(openai_client=client).answer(
        "How much forest is there?", [_scene()], {"kind": "single"}, provider="openai"
    )

    assert result["llm_wording"] is None
    assert any("numeric claim" in item for item in result["limitations"])
    assert len(api.calls) == 1


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (
            _tool_call(
                "python",
                {"code": "open('/tmp/internal/result','w')", "api_key": "sk-test-secret"},
            ),
            "unsupported tool",
        ),
        (_tool_call("presence", {"class_name": "forest", "threshold": 2}), "invalid arguments"),
        (_tool_call("describe", {"extra": "field"}), "invalid arguments"),
        (_raw_tool_call("coverage", "{not-json"), "invalid arguments"),
    ],
)
def test_openai_rejects_unknown_tools_and_invalid_arguments_without_executing(call, reason):
    from satquery.assistant.controller import AssistantController

    client, api = _client(SimpleNamespace(output=[call]))
    result = AssistantController(openai_client=client).answer(
        "Please inspect the image", [_scene()], {"kind": "single"}, provider="openai"
    )

    assert result["abstained"] is True
    assert reason in result["answer"].lower()
    assert result["trace"]["selected_tool"] is None
    assert len(api.calls) == 1
    assert len(result["trace"]["function_calls"]) == 1
    attempted = result["trace"]["function_calls"][0]
    assert attempted["accepted"] is False
    assert attempted["rejection_reason"]
    serialized_trace = json.dumps(attempted)
    assert "/tmp/internal" not in serialized_trace
    assert "sk-test-secret" not in serialized_trace
    json.dumps(result, allow_nan=False)


def test_openai_rejects_multiple_function_calls_as_unbounded():
    from satquery.assistant.controller import AssistantController

    routing = SimpleNamespace(
        output=[
            _tool_call("describe", {}, "call-1"),
            _tool_call("coverage", {"class_name": "forest"}, "call-2"),
        ]
    )
    client, api = _client(routing)

    result = AssistantController(openai_client=client).answer(
        "Describe and measure forest", [_scene()], {"kind": "single"}, provider="openai"
    )

    assert result["abstained"] is True
    assert "exactly one" in result["answer"].lower()
    assert len(api.calls) == 1
    assert [item["name"] for item in result["trace"]["function_calls"]] == [
        "describe",
        "coverage",
    ]
    assert all(
        item["rejection_reason"] == "multiple function calls returned"
        for item in result["trace"]["function_calls"]
    )


def test_openai_missing_call_id_preserves_rejected_attempt_without_running_wording_call():
    from satquery.assistant.controller import AssistantController

    routing = SimpleNamespace(
        output=[_tool_call("coverage", {"class_name": "forest"}, call_id="")],
        output_text="Coverage routed.",
    )
    client, api = _client(routing)

    result = AssistantController(openai_client=client).answer(
        "How much forest?", [_scene()], {"kind": "single"}, provider="openai"
    )

    assert result["abstained"] is True
    assert len(api.calls) == 1
    assert result["trace"]["function_calls"] == [
        {
            "name": "coverage",
            "call_id": "",
            "arguments": {"class_name": "forest"},
            "accepted": False,
            "rejection_reason": "missing function call identifier",
        }
    ]


def test_openai_error_is_actionable_and_does_not_leak_credentials():
    from satquery.assistant.controller import AssistantController, ProviderError

    client, _ = _client(RuntimeError("401 Authorization Bearer sk-test-secret"))
    with pytest.raises(ProviderError, match="authentication failed") as caught:
        AssistantController(openai_client=client).answer(
            "Describe this scene", [_scene()], {"kind": "single"}, provider="openai"
        )

    assert "sk-test-secret" not in str(caught.value)


def test_question_and_provider_are_bounded():
    from satquery.assistant.controller import AssistantController

    controller = AssistantController(openai_client=None)
    with pytest.raises(ValueError, match="question"):
        controller.answer("x" * 1001, [_scene()], {"kind": "single"}, provider="local")
    with pytest.raises(ValueError, match="provider"):
        controller.answer("Describe", [_scene()], {"kind": "single"}, provider="automatic")


def test_model_is_configurable_without_reading_or_accepting_an_api_key(monkeypatch):
    from satquery.assistant.controller import AssistantController

    monkeypatch.setenv("SATQUERY_OPENAI_MODEL", "gpt-test-model")
    controller = AssistantController(openai_client=None)

    assert controller.model == "gpt-test-model"
    with pytest.raises(TypeError):
        AssistantController(api_key="must-not-be-accepted")
