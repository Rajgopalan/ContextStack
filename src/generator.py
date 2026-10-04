"""Core generation logic: load 1-2 docs -> extract -> truncate -> LLM -> validate."""
import os
import uuid
from typing import List

from . import config
from .llm import get_provider
from .models import AOP
from .parsers import extract_text


def load_documents(filepaths: List[str]):
    if not 1 <= len(filepaths) <= 2:
        raise ValueError("Provide 1 or 2 documents for MVP (got %d)" % len(filepaths))
    docs = []
    for fp in filepaths:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        text, _ = extract_text(fp)
        text = text.strip()
        if not text:
            raise ValueError(f"No extractable text in {fp}")
        # truncate per-doc then globally
        text = text[: config.MAX_CHARS_PER_DOC]
        docs.append({"filename": os.path.basename(fp), "text": text})
    total = sum(len(d["text"]) for d in docs)
    if total > config.MAX_TOTAL_CHARS:
        budget = config.MAX_TOTAL_CHARS // len(docs)
        for d in docs:
            d["text"] = d["text"][:budget]
    return docs


def generate_aop(filepaths: List[str], title_override=None, provider_name=None) -> AOP:
    docs = load_documents(filepaths)
    provider = get_provider(provider_name or config.LLM_PROVIDER)
    raw = provider.generate_aop_json(docs)
    aop = AOP(
        id=f"aop_{uuid.uuid4().hex[:8]}",
        title=title_override or raw.get("title", "Untitled AOP"),
        version="1.0.0",
        description=raw.get("description", ""),
        source_documents=[d["filename"] for d in docs],
        preconditions=raw.get("preconditions", []),
        inputs=raw.get("inputs", []),
        outputs=raw.get("outputs", []),
        steps=raw.get("steps", []),
        error_handling=raw.get("error_handling", []),
        notes=raw.get("notes", ""),
    )
    return aop


def save_aop(aop: AOP) -> str:
    path = os.path.join(config.AOP_DIR, f"{aop.id}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(aop.model_dump_json(indent=2))
    # also save YAML-style via json (avoid extra dep); YAML export in API if pyyaml present
    return path
