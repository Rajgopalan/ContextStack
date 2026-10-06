"""Load 1-2 docs (full text, chunked downstream) -> pipeline -> lean AOP -> save JSON + MD."""
import os
import uuid
from typing import List

from . import config
from .llm import get_provider
from .models import AOP, aop_tokens, to_markdown
from .parsers import extract_text
from .pipeline import run_pipeline_dict


def load_documents(filepaths: List[str]):
    if not 1 <= len(filepaths) <= 2:
        raise ValueError("Provide 1 or 2 documents (got %d)" % len(filepaths))
    docs = []
    for fp in filepaths:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        text, _ = extract_text(fp)
        text = text.strip()
        if not text:
            raise ValueError(f"No extractable text in {fp}")
        if len(text) > config.DOC_HARD_CAP:
            print(f"WARNING: {fp} is {len(text)} chars, hard-capping at {config.DOC_HARD_CAP}")
            text = text[: config.DOC_HARD_CAP]
        docs.append({"filename": os.path.basename(fp), "text": text})
    return docs


def generate_aop(filepaths: List[str], title_override=None, provider_name=None) -> AOP:
    docs = load_documents(filepaths)
    provider = get_provider(provider_name or config.LLM_PROVIDER)
    raw = run_pipeline_dict(provider, docs)
    aop = AOP(
        id=f"aop_{uuid.uuid4().hex[:8]}",
        title=title_override or raw.get("title", "Untitled AOP"),
        trigger_when=raw.get("trigger_when", raw.get("trigger", "On matching incident")),
        inputs=raw.get("inputs", [])[:5],
        steps=raw.get("steps", []),
        must_not=raw.get("must_not", raw.get("guards", []))[:5],
        source_documents=[d["filename"] for d in docs],
        notes=f"lean v2, {aop_tokens_raw(raw)} tok est, from {len(docs)} doc(s)",
    )
    return aop


def aop_tokens_raw(raw: dict) -> int:
    import json as _j

    return max(1, len(_j.dumps(raw)) // 4)


def save_aop(aop: AOP) -> str:
    jpath = os.path.join(config.AOP_DIR, f"{aop.id}.json")
    mpath = os.path.join(config.AOP_DIR, f"{aop.id}.md")
    with open(jpath, "w", encoding="utf-8") as f:
        f.write(aop.model_dump_json(indent=2))
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(to_markdown(aop))
    return jpath
