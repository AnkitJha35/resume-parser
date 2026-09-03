"""Offline tests for Ollama provider adapter, schema handling, and availability checks."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import httpx

from app.domain.resume import Resume
from app.domain.semantic_contract import (
    DocumentArchetype,
    SemanticInput,
    SemanticOutput,
)
from app.extractors.providers.ollama import OllamaSemanticExtractor
from app.extractors.semantic_extractor import SemanticExtractionError
from app.pipeline.semantic_pipeline import parse_document_semantically
from tests.test_semantic_llm_contract import _make_test_document


def test_ollama_config_defaults_and_override():
    # Test default configuration
    extractor = OllamaSemanticExtractor()
    base_url, model, timeout, num_threads, think = extractor._resolve_config()
    assert base_url == "http://localhost:11434"
    assert model == "qwen2.5-coder:7b"
    assert timeout == 120.0
    assert num_threads == 8
    assert think is False

    # Test explicit override
    extractor2 = OllamaSemanticExtractor(
        base_url="http://custom-host:8000/",
        model="custom-model:latest",
        timeout=45.0,
        num_threads=12,
        think=True,
    )
    b2, m2, t2, th2, thk2 = extractor2._resolve_config()
    assert b2 == "http://custom-host:8000"
    assert m2 == "custom-model:latest"
    assert t2 == 45.0
    assert th2 == 12
    assert thk2 is True


def test_ollama_endpoint_url():
    url = OllamaSemanticExtractor.get_chat_endpoint_url("http://localhost:11434/")
    assert url == "http://localhost:11434/api/chat"


def test_ollama_availability_check_success():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3.2:latest"}]},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    ok, err = OllamaSemanticExtractor.check_availability(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
        client=client,
    )
    assert ok is True
    assert err is None


def test_ollama_availability_check_model_missing():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3.2:latest"}]},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    ok, err = OllamaSemanticExtractor.check_availability(
        base_url="http://localhost:11434",
        model="non-existent-model",
        client=client,
    )
    assert ok is False
    assert "not available in Ollama" in err
    assert "ollama pull non-existent-model" in err


def test_ollama_availability_check_service_unreachable():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    ok, err = OllamaSemanticExtractor.check_availability(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
        client=client,
    )
    assert ok is False
    assert "not reachable" in err


def test_ollama_request_construction_and_structured_response():
    captured_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        llm_payload = {
            "document_archetype": "standard_cv",
            "personal": {
                "name": {"value": "John Doe", "raw_value": "John Doe", "source_block_ids": ["b_p1_0"]},
                "email": {"value": "john.doe@example.com", "source_block_ids": ["b_p1_1"]},
            },
            "experience": [
                {
                    "company": {"value": "Acme Corporation", "source_block_ids": ["b_p1_4"]},
                    "designation": {"value": "Senior Software Engineer", "source_block_ids": ["b_p1_5"]},
                    "source_block_ids": ["b_p1_4", "b_p1_5"],
                }
            ],
        }
        envelope = {
            "model": "qwen2.5-coder:7b",
            "message": {
                "role": "assistant",
                "content": json.dumps(llm_payload),
            },
            "done": True,
            "prompt_eval_count": 210,
            "eval_count": 85,
        }
        return httpx.Response(200, json=envelope, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = OllamaSemanticExtractor(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
        client=client,
    )

    doc = _make_test_document()
    resume = parse_document_semantically(doc, extractor, document_id="ollama-test-1")

    # 1. Assert request payload
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert str(req.url) == "http://localhost:11434/api/chat"
    req_body = json.loads(req.content.decode("utf-8"))
    assert req_body["model"] == "qwen2.5-coder:7b"
    assert req_body["stream"] is False
    assert req_body["think"] is False
    assert "format" in req_body
    assert req_body["format"]["type"] == "object"
    assert "personal" in req_body["format"]["properties"]
    assert "DOCUMENT BLOCKS (JSON):" in req_body["messages"][0]["content"]
    assert req_body["options"] == {
        "temperature": 0.0,
        "num_thread": 8,
    }

    # 2. Assert usage metadata captured
    assert extractor.last_usage_metadata == {
        "prompt_tokens": 210,
        "output_tokens": 85,
        "total_tokens": 295,
        "num_threads": 8,
        "think": False,
    }

    # 3. Assert resulting Resume
    assert isinstance(resume, Resume)
    assert resume.personal.name == "John Doe"
    assert resume.personal.email == "john.doe@example.com"
    assert len(resume.experience) == 1
    assert resume.experience[0].company == "Acme Corporation"


def test_ollama_think_resolution(monkeypatch):
    # 1. Default think is False
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    extractor = OllamaSemanticExtractor()
    assert extractor._resolve_config()[4] is False

    # 2. Explicit constructor argument overrides everything
    extractor_explicit_true = OllamaSemanticExtractor(think=True)
    assert extractor_explicit_true._resolve_config()[4] is True

    extractor_explicit_false = OllamaSemanticExtractor(think=False)
    assert extractor_explicit_false._resolve_config()[4] is False

    # 3. Environment variable OLLAMA_THINK="true" / "1" / "yes"
    monkeypatch.setenv("OLLAMA_THINK", "true")
    extractor_env_true = OllamaSemanticExtractor()
    assert extractor_env_true._resolve_config()[4] is True

    monkeypatch.setenv("OLLAMA_THINK", "1")
    assert OllamaSemanticExtractor()._resolve_config()[4] is True

    monkeypatch.setenv("OLLAMA_THINK", "yes")
    assert OllamaSemanticExtractor()._resolve_config()[4] is True

    # 4. Environment variable OLLAMA_THINK="false" / "0" / "no"
    monkeypatch.setenv("OLLAMA_THINK", "false")
    assert OllamaSemanticExtractor()._resolve_config()[4] is False

    monkeypatch.setenv("OLLAMA_THINK", "0")
    assert OllamaSemanticExtractor()._resolve_config()[4] is False

    # 5. Invalid env var falls back safely to False
    monkeypatch.setenv("OLLAMA_THINK", "invalid_value")
    assert OllamaSemanticExtractor()._resolve_config()[4] is False


def test_ollama_custom_think_in_request():
    captured_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": json.dumps({"personal": {"name": {"value": "Jane", "source_block_ids": ["b1"]}}})},
                "prompt_eval_count": 100,
                "eval_count": 50,
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = OllamaSemanticExtractor(think=True, client=client)
    sem_input = SemanticInput(document_id="d1", page_count=1, pages=[], blocks=[])

    extractor.extract(sem_input)

    assert len(captured_requests) == 1
    req_body = json.loads(captured_requests[0].content.decode("utf-8"))
    # think must be top-level, NOT inside options
    assert req_body["think"] is True
    assert "think" not in req_body["options"]
    assert extractor.last_usage_metadata["think"] is True


def test_ollama_num_threads_resolution(monkeypatch):
    # 1. Default thread count is 8
    monkeypatch.delenv("OLLAMA_NUM_THREADS", raising=False)
    extractor = OllamaSemanticExtractor()
    assert extractor._resolve_config()[3] == 8

    # 2. Explicit constructor argument overrides everything
    extractor_explicit = OllamaSemanticExtractor(num_threads=4)
    assert extractor_explicit._resolve_config()[3] == 4

    # 3. Non-positive constructor argument falls back to default 8
    extractor_non_positive = OllamaSemanticExtractor(num_threads=-2)
    assert extractor_non_positive._resolve_config()[3] == 8

    # 4. Environment variable OLLAMA_NUM_THREADS
    monkeypatch.setenv("OLLAMA_NUM_THREADS", "12")
    extractor_env = OllamaSemanticExtractor()
    assert extractor_env._resolve_config()[3] == 12

    # 5. Non-positive or invalid env var falls back to default 8
    monkeypatch.setenv("OLLAMA_NUM_THREADS", "0")
    extractor_zero = OllamaSemanticExtractor()
    assert extractor_zero._resolve_config()[3] == 8

    monkeypatch.setenv("OLLAMA_NUM_THREADS", "-4")
    extractor_neg = OllamaSemanticExtractor()
    assert extractor_neg._resolve_config()[3] == 8

    monkeypatch.setenv("OLLAMA_NUM_THREADS", "invalid_number")
    extractor_invalid = OllamaSemanticExtractor()
    assert extractor_invalid._resolve_config()[3] == 8


def test_ollama_custom_num_threads_in_request():
    captured_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": json.dumps({"personal": {"name": {"value": "Jane", "source_block_ids": ["b1"]}}})},
                "prompt_eval_count": 100,
                "eval_count": 50,
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = OllamaSemanticExtractor(num_threads=16, client=client)
    sem_input = SemanticInput(document_id="d1", page_count=1, pages=[], blocks=[])

    extractor.extract(sem_input)

    assert len(captured_requests) == 1
    req_body = json.loads(captured_requests[0].content.decode("utf-8"))
    assert req_body["options"]["num_thread"] == 16
    assert extractor.last_usage_metadata["num_threads"] == 16


def test_ollama_connection_error_handling():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to connect", request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = OllamaSemanticExtractor(client=client)
    sem_input = SemanticInput(document_id="d1", page_count=1, pages=[], blocks=[])

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(sem_input)
    assert "Ollama connection failed" in str(exc_info.value)


def test_ollama_malformed_envelope_handling():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        # Envelope missing 'message.content'
        return httpx.Response(200, json={"done": True}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    extractor = OllamaSemanticExtractor(client=client)
    sem_input = SemanticInput(document_id="d1", page_count=1, pages=[], blocks=[])

    with pytest.raises(SemanticExtractionError) as exc_info:
        extractor.extract(sem_input)
    assert "missing 'message.content'" in str(exc_info.value)


def test_serialize_ollama_compact_input():
    from app.domain.semantic_contract import SemanticBlockInput
    from app.extractors.providers.ollama import serialize_ollama_compact_input

    blocks = [
        SemanticBlockInput(
            block_id="b_p1_0",
            text="John Doe",
            page=1,
            bbox=[0.0, 0.0, 100.0, 20.0],
            region_id="r1",
            region_kind="header",
            reading_order=0,
            suggested_role="HEADER",
        ),
        SemanticBlockInput(
            block_id="b_p2_1",
            text="Ship Name",
            page=2,
            bbox=[10.0, 10.0, 50.0, 20.0],
            region_id="r2",
            region_kind="table",
            reading_order=1,
            suggested_role="SECTION_HEADING",
            is_bold=True,
            table_id="tbl_1",
            row_index=0,
            column_index=0,
            cell_role="HEADER",
        ),
    ]
    sem_input = SemanticInput(
        document_id="test_doc",
        page_count=2,
        pages=[],
        blocks=blocks,
    )

    serialized = serialize_ollama_compact_input(sem_input)
    parsed = json.loads(serialized)

    assert parsed["doc_id"] == "test_doc"
    assert len(parsed["blocks"]) == 2

    # Block 1 (page 1, not bold, no table)
    b1 = parsed["blocks"][0]
    assert b1["id"] == "b_p1_0"
    assert b1["text"] == "John Doe"
    assert b1["role"] == "HEADER"
    assert "page" not in b1  # page 1 omitted
    assert "bbox" not in b1  # bbox omitted
    assert "bold" not in b1  # bold False omitted
    assert "table" not in b1  # null table omitted

    # Block 2 (page 2, bold, table)
    b2 = parsed["blocks"][1]
    assert b2["id"] == "b_p2_1"
    assert b2["text"] == "Ship Name"
    assert b2["role"] == "SECTION_HEADING"
    assert b2["page"] == 2
    assert b2["bold"] is True
    assert b2["table"] == "tbl_1"
    assert b2["row"] == 0
    assert b2["col"] == 0
    assert b2["cell_role"] == "HEADER"
    assert "bbox" not in b2


def test_get_ollama_compact_schema():
    from app.domain.semantic_contract import SemanticOutput
    from app.extractors.providers.ollama import get_ollama_compact_schema

    schema = get_ollama_compact_schema(SemanticOutput)
    assert schema["type"] == "object"
    assert "personal" in schema["properties"]
    # Verify titles and descriptions are stripped
    assert "title" not in schema
    assert "description" not in schema
    assert "title" not in schema["properties"]["personal"]
