"""4-stage pipeline: extract per-doc -> merge -> compile/minimize -> validate."""
import json
from typing import Dict, List

from .llm import COMPILE_PROMPT, EXTRACT_PROMPT, MERGE_PROMPT


def run_pipeline_dict(provider, docs: List[Dict[str, str]]) -> dict:
    # 1. extract per doc (keeps per-doc context small, no cross-contamination)
    extractions = []
    for d in docs:
        user = f"DOCUMENT: {d['filename']}\n{d['text'][:15000]}"
        e = provider.chat_json(EXTRACT_PROMPT, user)
        extractions.append(e)

    # 2. merge (dedupe, resolve conflicts, order)
    merged = provider.chat_json(MERGE_PROMPT, json.dumps({"extractions": extractions})[:20000])

    # 3. compile/minimize (enforce 5-9 steps, <=140 chars, token budget)
    compiled = provider.chat_json(COMPILE_PROMPT, json.dumps(merged)[:12000])

    # 4. validate (deterministic, no LLM): enforce 5-9 steps, fill gaps from merged
    steps = compiled.get("steps", [])[:9]
    if len(steps) < 5 and merged.get("actions"):
        have = {s.get("do", "").lower()[:70] for s in steps}
        for a in merged["actions"]:
            if len(steps) >= 5:
                break
            if a.get("do", "").lower()[:70] not in have:
                steps.append({"id": "", "do": a["do"][:140], "check": a.get("check") or "Verify recorded",
                              "tool": a.get("tool"), "on_fail": "Log, retry once, escalate"})
    for i, s in enumerate(steps, 1):
        s["id"] = f"s{i}"
        s["do"] = s.get("do", "")[:140]
        if not s.get("check"):
            s["check"] = "Verify recorded"
        if not s.get("on_fail"):
            s["on_fail"] = "Log, retry once, escalate"
    compiled["steps"] = steps
    compiled["inputs"] = compiled.get("inputs", [])[:5]
    must_not = compiled.get("must_not") or merged.get("guards", [])
    compiled["must_not"] = (must_not or ["Do not delete evidence before snapshot"])[:5]
    return compiled
