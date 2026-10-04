"""Pluggable LLM providers with stage prompts for the 4-stage pipeline."""
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List

# --- Stage prompts: minimal, high-signal ---

EXTRACT_PROMPT = """Extract executable procedure from ONE document.
Output ONLY JSON: {"actions": [{"do": "imperative <=140 chars", "check": "verify or null", "tool": "tool or null"}],
"inputs": ["max 5"], "guards": ["MUST NOT rules"], "trigger": "when to use, 1 line"}.
Rules: imperative MUST style, drop background/prose, max 12 actions, dedupe."""

MERGE_PROMPT = """Merge 1-2 extracted procedures into ONE deduplicated draft.
Output ONLY JSON: {"actions": [{"do": str, "check": str|null, "tool": str|null}], "inputs": [], "guards": [], "trigger": str}.
Rules: merge overlaps, order by execution, max 9 actions, resolve conflicts (stricter wins)."""

COMPILE_PROMPT = """Compile merged draft into minimal runtime AOP.
Output ONLY JSON: {"title": str, "trigger_when": str, "inputs": ["max 5"],
"steps": [{"id": "s1", "do": str, "check": str, "tool": str|null, "on_fail": str}],
"must_not": ["min 1 guard"]}.
HARD RULES: 5-9 steps (fewer than 5 is INVALID), each do <=140 chars imperative,
each step MUST have check and on_fail (never null), keep at least 1 must_not guard."""


def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r"[\x00-\x1f]", " ", text)
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # last resort: pull first 0.x number as score (judge prompts are score-only)
            sm = re.search(r"0?\.\d+|[01](?:\.0+)?", fixed)
            if sm:
                return {"score": float(sm.group(0))}
            import hashlib
            import os
            import time

            fn = "/tmp/llm_fail_%s_%d.txt" % (hashlib.md5(text.encode()[:500]).hexdigest()[:8], int(time.time()))
            with open(fn, "w") as f:
                f.write(text[:4000])
            raise ValueError(f"Unparseable LLM JSON (saved {fn}): {text[:200]!r}")


class BaseLLMProvider(ABC):
    @abstractmethod
    def chat_json(self, system: str, user: str) -> dict:
        raise NotImplementedError

    def generate_aop_json(self, docs: List[Dict[str, str]]) -> dict:
        # legacy single-shot fallback
        from .pipeline import run_pipeline_dict

        return run_pipeline_dict(self, docs)


def _clean_line(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line)  # strip html
    line = re.sub(r"^[\d\.\-\*#>\s_]+", "", line).strip()
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _infer_tool(low: str):
    if any(k in low for k in ("slack", "channel", "email", "notify", "pagerduty", "opsgenie")):
        return "notify"
    if any(k in low for k in ("api", "gateway", "quarantine", "block", "siem", "splunk")):
        return "security-tools"
    if any(k in low for k in ("jira", "ticket", "postmortem")):
        return "ticketing"
    if any(k in low for k in ("zoom", "hangout", "bridge", "call")):
        return "video-bridge"
    return None


class MockProvider(BaseLLMProvider):
    """Offline deterministic: implements all stages with heuristics."""

    def chat_json(self, system: str, user: str) -> dict:
        low = system.lower()
        if "compile" in low:
            payload = json.loads(user[user.find("{"):])
            return self._mock_compile(payload)
        if "merge" in low:
            payload = json.loads(user[user.find("{"):])
            return self._mock_merge(payload)
        return self._mock_extract(user)

    def _mock_extract(self, user: str) -> dict:
        verbs = ("collect", "verify", "check", "quarantine", "block", "reset", "notify",
                 "escalate", "contain", "isolate", "revoke", "enforce", "create", "capture",
                 "determine", "assess", "declare", "spin", "bring", "perform", "rollout",
                 "close", "file", "schedule", "monitor", "remove", "disable", "must", "ensure")
        lines = [l.strip() for l in user.splitlines() if len(l.strip()) > 30]
        scored = []
        for l in lines:
            if l.startswith("DOCUMENT:") or l.startswith("===== DOCUMENT"):
                continue
            c = _clean_line(l)
            if len(c) < 30 or c.lower().startswith("document:"):
                continue
            low = c.lower()
            score = sum(1 for v in verbs if v in low)
            if re.match(r"\s*\d+[\.\)]", l):
                score += 2
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        seen, actions = set(), []
        for score, c in scored:
            key = c.lower()[:70]
            if key in seen:
                continue
            seen.add(key)
            low = c.lower()
            check = "Verify recorded" if any(k in low for k in ("verify", "check", "confirm", "ensure", "validate")) else None
            actions.append({"do": c[:140], "check": check, "tool": _infer_tool(low)})
            if len(actions) >= 12:
                break
        if not actions:
            actions = [{"do": "Follow source procedure in order", "check": None, "tool": None}]
        return {
            "actions": actions,
            "inputs": ["message_id", "reporter"],
            "guards": ["Do not delete evidence before snapshot"],
            "trigger": "Suspected security incident or phishing report",
        }

    def _mock_merge(self, payload: dict) -> dict:
        exts = payload.get("extractions", [payload])
        if not isinstance(exts, list):
            exts = [payload]
        seen, actions = set(), []
        inputs, guards = [], []
        for e in exts:
            for i in e.get("inputs", []):
                if i not in inputs:
                    inputs.append(i)
            for g in e.get("guards", []):
                if g not in guards:
                    guards.append(g)
        # round-robin so both docs stay represented (max 9)
        idx = 0
        while len(actions) < 9:
            added = False
            for e in exts:
                acts = e.get("actions", [])
                if idx < len(acts):
                    key = acts[idx].get("do", "").lower()[:70]
                    if key not in seen:
                        seen.add(key)
                        actions.append(acts[idx])
                        added = True
                    if len(actions) >= 9:
                        break
            idx += 1
            if not added:
                break
        return {
            "actions": actions[:9],
            "inputs": inputs[:5],
            "guards": guards[:5],
            "trigger": "Suspected incident or phishing email",
        }

    def _mock_compile(self, payload: dict) -> dict:
        actions = payload.get("actions", [])[:9]
        steps = []
        for i, a in enumerate(actions, 1):
            steps.append(
                {
                    "id": f"s{i}",
                    "do": a.get("do", "")[:140],
                    "check": a.get("check"),
                    "tool": a.get("tool"),
                    "on_fail": "Log, retry once, escalate",
                }
            )
        trigger = payload.get("trigger", "Suspected incident or phishing email")
        title = payload.get("title", f"{trigger} — response")
        return {
            "title": title[:120],
            "trigger_when": trigger,
            "inputs": payload.get("inputs", ["message_id"])[:5],
            "steps": steps,
            "must_not": payload.get("guards", [])[:5],
        }


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def chat_json(self, system: str, user: str) -> dict:
        import httpx

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        r.raise_for_status()
        return extract_json(r.json()["choices"][0]["message"]["content"])


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def chat_json(self, system: str, user: str) -> dict:
        import httpx

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={"model": self.model, "max_tokens": 4000, "system": system,
                  "messages": [{"role": "user", "content": user}], "temperature": 0.2},
            timeout=120,
        )
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        return extract_json(text)


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat_json(self, system: str, user: str) -> dict:
        import httpx

        r = httpx.post(f"{self.base_url}/api/chat",
            json={"model": self.model, "stream": False, "format": "json",
                  "options": {"num_predict": 2000, "num_ctx": 16384},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=180)
        r.raise_for_status()
        return extract_json(r.json()["message"]["content"])


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
