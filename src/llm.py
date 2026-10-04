"""Pluggable LLM providers. All return a parsed dict for the AOP body."""
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List


SYSTEM_PROMPT = """You are an expert at writing Agent Operation Procedures (AOPs).
Given 1-2 source documents, produce ONE consolidated machine-readable AOP as strict JSON.

Rules:
- Output ONLY valid JSON, no markdown fences, no commentary.
- Follow this schema exactly:
{
  "title": string,
  "description": string,
  "preconditions": [string],
  "inputs": [{"name": string, "type": string, "description": string, "required": boolean}],
  "outputs": [{"name": string, "type": string, "description": string, "required": boolean}],
  "steps": [{
    "id": "step_1",
    "name": string,
    "description": string,
    "action_type": "data_extraction|api_call|decision|validation|manual|notification|file_operation",
    "tool": string or null,
    "parameters": {},
    "expected_output": string,
    "validation_criteria": string or null,
    "on_failure": string or null
  }],
  "error_handling": [string],
  "notes": string
}
- Merge both documents into one coherent procedure, deduplicate steps.
- Steps must be ordered, executable by an AI agent, each with expected_output.
- If info is missing, make reasonable defaults and note assumptions in "notes".
"""


def build_user_prompt(docs: List[Dict[str, str]]) -> str:
    parts = []
    for d in docs:
        parts.append(f"===== DOCUMENT: {d['filename']} =====\n{d['text']}\n===== END {d['filename']} =====")
    return (
        "Generate one consolidated AOP from these source documents:\n\n"
        + "\n\n".join(parts)
        + "\n\nReturn ONLY the JSON object."
    )


def extract_json(text: str) -> dict:
    """Tolerantly extract JSON from LLM output (fences, preamble)."""
    text = text.strip()
    # strip ```json fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_aop_json(self, docs: List[Dict[str, str]]) -> dict:
        raise NotImplementedError


class MockProvider(BaseLLMProvider):
    """Deterministic offline provider: headings/lines -> steps. No API key needed."""

    def generate_aop_json(self, docs: List[Dict[str, str]]) -> dict:
        combined = "\n".join(d["text"] for d in docs)
        lines = [l.strip(" #-*\t") for l in combined.splitlines() if len(l.strip()) > 30]
        # dedupe while preserving order
        seen, uniq = set(), []
        for l in lines:
            key = l.lower()[:80]
            if key not in seen:
                seen.add(key)
                uniq.append(l)
            if len(uniq) >= 10:
                break
        if not uniq:
            uniq = ["Follow the source documents in order and complete all required actions."]

        steps = []
        for i, line in enumerate(uniq, start=1):
            action = "manual"
            low = line.lower()
            if any(k in low for k in ("api", "request", "http", "endpoint", "call")):
                action = "api_call"
            elif any(k in low for k in ("check", "verify", "validate", "confirm", "ensure")):
                action = "validation"
            elif any(k in low for k in ("if ", "decide", "approve", "choose")):
                action = "decision"
            elif any(k in low for k in ("extract", "read", "parse", "collect", "gather")):
                action = "data_extraction"
            elif any(k in low for k in ("notify", "send", "email", "alert", "slack")):
                action = "notification"
            steps.append(
                {
                    "id": f"step_{i}",
                    "name": line[:80],
                    "description": line[:500],
                    "action_type": action,
                    "tool": None,
                    "parameters": {},
                    "expected_output": f"Completion of: {line[:120]}",
                    "validation_criteria": "Step output is recorded and matches expected format",
                    "on_failure": "Log error, retry once, then escalate to human operator",
                }
            )

        title = "Generated AOP from " + ", ".join(d["filename"] for d in docs)
        return {
            "title": title[:150],
            "description": f"Auto-generated (mock provider) consolidated procedure from {len(docs)} document(s).",
            "preconditions": ["Source documents are available", "Agent has required tool access"],
            "inputs": [{"name": "source_documents", "type": "string[]", "description": "Input docs", "required": True}],
            "outputs": [{"name": "completion_report", "type": "string", "description": "Result of procedure", "required": True}],
            "steps": steps,
            "error_handling": ["On any step failure: log, retry once, escalate to human"],
            "notes": "Generated offline with MockProvider. Configure LLM_PROVIDER=openai/anthropic/ollama for LLM-quality output.",
        }


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_aop_json(self, docs: List[Dict[str, str]]) -> dict:
        import httpx

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(docs)},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return extract_json(content)


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_aop_json(self, docs: List[Dict[str, str]]) -> dict:
        import httpx

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4000,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": build_user_prompt(docs)}],
                "temperature": 0.2,
            },
            timeout=120,
        )
        resp.raise_for_status()
        blocks = resp.json()["content"]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return extract_json(text)


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_aop_json(self, docs: List[Dict[str, str]]) -> dict:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(docs)},
                ],
            },
            timeout=180,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return extract_json(content)


def get_provider(name: str) -> BaseLLMProvider:
    from . import config

    name = (name or "mock").lower()
    if name == "openai":
        return OpenAIProvider(config.OPENAI_API_KEY, config.OPENAI_MODEL)
    if name == "anthropic":
        return AnthropicProvider(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
    if name == "ollama":
        return OllamaProvider(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    return MockProvider()
