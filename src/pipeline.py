"""Chunked map-reduce pipeline. No truncation: every chunk is processed.
extract per chunk -> reduce per doc -> merge docs -> compile -> validate."""
import json
from typing import Dict, List

from .chunking import chunk_text
from .llm import COMPILE_PROMPT, EXTRACT_PROMPT, MERGE_PROMPT, REDUCE_PROMPT


def _extract_user(filename: str, chunk: dict, n_chunks: int) -> str:
    head = f" [{chunk['heading']}]" if chunk.get("heading") else ""
    return (
        f"SOURCE: {filename} — chunk {chunk['index'] + 1}/{n_chunks}{head}\n"
        f"{chunk['text']}"
    )


def run_pipeline_dict(provider, docs: List[Dict[str, str]]) -> dict:
    # 1. map: extract per chunk (full text covered, chunks bounded for context)
    extractions = []
    for d in docs:
        chunks = chunk_text(d["text"])
        chunk_exts = [
            provider.chat_json(EXTRACT_PROMPT, _extract_user(d["filename"], c, len(chunks)))
            for c in chunks
        ]
        # 2. reduce per doc (dedupe overlap, order by phase)
        if len(chunk_exts) == 1:
            doc_ext = chunk_exts[0]
        else:
            doc_ext = provider.chat_json(
                REDUCE_PROMPT,
                json.dumps({"document": d["filename"], "chunk_extractions": chunk_exts}),
            )
        extractions.append(doc_ext)

    # 3. merge docs into one draft
    if len(extractions) == 1:
        merged = {
            "actions": extractions[0].get("actions", [])[:9],
            "inputs": extractions[0].get("inputs", []),
            "guards": extractions[0].get("guards", []),
            "trigger": extractions[0].get("trigger", ""),
        }
    else:
        merged = provider.chat_json(MERGE_PROMPT, json.dumps({"extractions": extractions}))

    # 4. compile to minimal runtime artifact
    compiled = provider.chat_json(COMPILE_PROMPT, json.dumps(merged))

    # 5. validate (deterministic, no LLM): enforce 5-9 steps, fill gaps from merged
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
