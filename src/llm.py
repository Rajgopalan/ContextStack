"""Pluggable LLM providers with stage prompts for the 4-stage pipeline."""
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List

# --- Stage prompts v3: full contracts (role, schema, taxonomy, rules, example) ---
# Each prompt is self-contained so stages can run on different models/providers.

EXTRACT_PROMPT = """You are a procedure-distillation specialist. Your job: read ONE document chunk and extract ONLY the executable procedure it contains. Ignore marketing prose, background history, and duplicates of earlier context.

OUTPUT — strict JSON object, no other text, exactly these keys:
{
  "actions": [{"do": string, "check": string|null, "tool": string|null}],
  "inputs": [string],
  "guards": [string],
  "trigger": string
}

FIELD DEFINITIONS:
- do: single imperative action, <=140 chars, MUST style ("Quarantine matching emails", NOT "Emails should be quarantined"). One verb, one outcome.
- check: how the agent verifies the action worked, <=100 chars ("Quarantine list contains all matching IDs"). Null only if unverifiable.
- tool: system used, from taxonomy [notify, ticketing, security-tools, identity, email-gateway, video-bridge, forensic, docs] or null if manual/human.
- inputs: data the agent must have BEFORE acting (IDs, reports, credentials), max 5, short noun phrases.
- guards: hard prohibitions as MUST NOT rules ("Do not delete host before forensic snapshot").
- trigger: 1 line describing WHEN this procedure applies.

RULES:
1. Max 12 actions per chunk. Drop background, motivation, and generic advice.
2. Preserve numbers that matter: thresholds ($500), time windows (30 days), severities, SLAs — put them in do or check.
3. If two sentences describe the same action, keep one.
4. Never invent tools, thresholds, or contacts not in the text. Use null rather than guessing.
5. Order actions as they appear in the chunk; cross-chunk ordering happens later.

[STAGE:extract]

EXAMPLE:
chunk: "2. Verify the purchase in the Orders API and check it is within the 30-day refund window. 3. If amount is over $500, request manager approval."
output: {"actions": [{"do": "Verify purchase in Orders API and confirm within 30-day window", "check": "Order exists and purchase date <= 30 days", "tool": "security-tools"}, {"do": "Request manager approval for amounts over $500", "check": "Approval code recorded", "tool": null}], "inputs": ["order_id"], "guards": [], "trigger": "Refund request received"}"""

REDUCE_PROMPT = """You are a procedure editor. Your job: merge several EXTRACTIONS taken from different chunks of the SAME document into one deduplicated per-document procedure. All inputs describe the same source — never treat them as conflicting documents.

OUTPUT — strict JSON object, no other text, exactly these keys:
{"actions": [{"do": string, "check": string|null, "tool": string|null}], "inputs": [], "guards": [], "trigger": string}

RULES:
1. Deduplicate: same action appearing in overlapping chunks (including the [...continued] overlap) MUST appear once. Prefer the more specific wording.
2. Order by procedure phase: detect -> triage/verify -> contain/mitigate -> recover -> notify/document -> review. Do not use chunk order blindly.
3. Max 12 actions. Merge trivially small adjacent actions only if the result stays <=140 chars.
4. Union inputs (max 5, most essential first) and guards (keep all load-bearing prohibitions).
5. Trigger: single line covering the whole document.
6. Never invent. If extractions disagree on a fact, keep the more specific value.

[STAGE:reduce]"""

MERGE_PROMPT = """You are a procedure editor. Your job: merge 1-2 PER-DOCUMENT procedures into ONE consolidated draft covering all sources. Sources may overlap (same incident from two angles) or complement (policy + API reference).

OUTPUT — strict JSON object, no other text, exactly these keys:
{"actions": [{"do": string, "check": string|null, "tool": string|null}], "inputs": [], "guards": [], "trigger": string}

RULES:
1. Deduplicate across documents: one action per real-world step. Prefer the wording with concrete parameters (names, thresholds, tools).
2. Order by procedure phase: detect -> triage/verify -> contain/mitigate -> recover -> notify/document -> review.
3. Max 9 actions. If more remain, fold sub-steps into check fields ("...verify X, Y, Z") rather than dropping load-bearing steps.
4. Conflicts: keep the STRICTER rule (shorter time window, higher approval bar, broader quarantine). Never average thresholds.
5. Inputs: max 5, the minimal set an agent needs to start. Guards: keep every prohibition that prevents data loss, legal exposure, or destructive action.
6. Trigger: one line stating when the merged procedure applies.
7. Never invent tools, thresholds, or contacts. Null beats a guess.

[STAGE:merge]"""

COMPILE_PROMPT = """You are a runtime-context compiler. Your job: compile the merged draft into the MINIMAL artifact an AI agent loads at incident time. Every token must earn its place.

OUTPUT — strict JSON object, no other text, exactly these keys:
{"title": string, "trigger_when": string, "inputs": ["max 5"],
"steps": [{"id": "s1", "do": string, "check": string, "tool": string|null, "on_fail": string}],
"must_not": ["min 1 guard"]}

HARD RULES (output violating any of these is INVALID):
1. 5-9 steps. Fewer than 5 omits load-bearing work; more than 9 does not fit the runtime budget.
2. Each do: imperative, <=140 chars, one verb + outcome. No background clauses.
3. Each check: concrete verification <=100 chars (what to look at, what value counts as done). Never null.
4. Each on_fail: recovery <=100 chars ("Log, retry once, escalate to X"). Never null.
5. must_not: at least 1 guard, each a concrete prohibition.
6. Compress by rewriting, not by dropping: fold details into check fields, keep thresholds and tool names.
7. Title <=120 chars naming the procedure. trigger_when: 1-2 lines so a router can decide relevance without loading steps.

[STAGE:compile]"""


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
        payload = lambda: json.loads(user[user.find("{"):])
        if "[STAGE:compile]" in system:
            return self._mock_compile(payload())
        if "[STAGE:reduce]" in system:
            return self._mock_reduce(payload())
        if "[STAGE:merge]" in system:
            return self._mock_merge(payload())
        return self._mock_extract(user)

    def _mock_extract(self, user: str) -> dict:
        verbs = ("collect", "verify", "check", "quarantine", "block", "reset", "notify",
                 "escalate", "contain", "isolate", "revoke", "enforce", "create", "capture",
                 "determine", "assess", "declare", "spin", "bring", "perform", "rollout",
                 "close", "file", "schedule", "monitor", "remove", "disable", "must", "ensure")
        lines = [l.strip() for l in user.splitlines() if len(l.strip()) > 30]
        scored = []
        for l in lines:
            if l.startswith(("DOCUMENT:", "===== DOCUMENT", "CHUNK ", "SOURCE:", "[...continued]")):
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

    def _mock_reduce(self, payload: dict) -> dict:
        exts = payload.get("chunk_extractions", payload.get("extractions", [payload]))
        if not isinstance(exts, list):
            exts = [payload]
        return self._union(exts, cap=12, trigger="Document procedure")

    def _mock_merge(self, payload: dict) -> dict:
        exts = payload.get("extractions", [payload])
        if not isinstance(exts, list):
            exts = [payload]
        out = self._union(exts, cap=9, trigger="Suspected incident or phishing email")
        out["inputs"] = out["inputs"][:5]
        return out

    def _union(self, exts: list, cap: int, trigger: str) -> dict:
        seen, actions = set(), []
        inputs, guards = [], []
        for e in exts:
            for i in e.get("inputs", []):
                if i not in inputs:
                    inputs.append(i)
            for g in e.get("guards", []):
                if g not in guards:
                    guards.append(g)
        # round-robin so every chunk/doc stays represented
        idx = 0
        while len(actions) < cap:
            added = False
            for e in exts:
                acts = e.get("actions", [])
                if idx < len(acts):
                    key = acts[idx].get("do", "").lower()[:70]
                    if key not in seen:
                        seen.add(key)
                        actions.append(acts[idx])
                        added = True
                    if len(actions) >= cap:
                        break
            idx += 1
            if not added:
                break
        trig = exts[0].get("trigger", trigger) if exts else trigger
        return {
            "actions": actions[:cap],
            "inputs": inputs[:5],
            "guards": guards[:5],
            "trigger": trig,
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
