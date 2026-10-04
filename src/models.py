"""Minimal high-signal AOP schema. Optimized for runtime loading, not storage."""
from typing import List, Optional
from pydantic import BaseModel, Field


class AOPStep(BaseModel):
    id: str = Field(description="Stable id, e.g. s1")
    do: str = Field(description="Imperative action, <=140 chars, MUST style")
    check: Optional[str] = Field(default=None, description="How to verify, <=100 chars")
    tool: Optional[str] = Field(default=None, description="Tool to use, if any")
    on_fail: Optional[str] = Field(default=None, description="Recovery, short")


class AOP(BaseModel):
    id: str
    title: str
    trigger_when: str = Field(description="When to load this AOP, 1-2 lines (discovery)")
    inputs: List[str] = Field(default_factory=list, description="Required inputs, max 5")
    steps: List[AOPStep] = Field(default_factory=list, description="5-9 ordered steps")
    must_not: List[str] = Field(default_factory=list, description="Guardrails")
    source_documents: List[str] = Field(default_factory=list)
    version: str = "2.0.0"
    notes: str = ""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def aop_tokens(aop: "AOP") -> int:
    return estimate_tokens(to_markdown(aop))


def to_markdown(aop: "AOP") -> str:
    lines = [f"# {aop.title}", ""]
    lines.append(f"USE WHEN: {aop.trigger_when}")
    lines.append("")
    if aop.inputs:
        lines.append(f"INPUTS: {', '.join(aop.inputs)}")
        lines.append("")
    lines.append("STEPS:")
    for i, s in enumerate(aop.steps, 1):
        tool = f" [tool: {s.tool}]" if s.tool else ""
        lines.append(f"{i}. MUST {s.do}{tool}")
        if s.check:
            lines.append(f"   Check: {s.check}")
        if s.on_fail:
            lines.append(f"   On fail: {s.on_fail}")
    if aop.must_not:
        lines.append("")
        lines.append("MUST NOT:")
        for m in aop.must_not:
            lines.append(f"- {m}")
    lines.append("")
    lines.append(f"Sources: {', '.join(aop.source_documents)}")
    return "\n".join(lines)


AOP_JSON_SCHEMA = AOP.model_json_schema()
