"""Rigorous eval: deterministic core (offline) + optional LLM A/B (needs real provider).
Verdict is based on deterministic core. Mock accuracy is reported but never decides PASS.
"""
import argparse
import json
import re
import sys
from typing import Dict, List

from .generator import generate_aop, load_documents
from .llm import get_provider
from .models import AOP, aop_tokens, to_markdown

EVAL_QUESTIONS_PROMPT = """From the SOURCE below, write 5 runtime test questions an agent must answer correctly.
Output ONLY JSON: {"questions": [{"q": str, "keywords": ["2-4 expected answer keywords"]}]}.
Cover: detection, containment, recovery, escalation, guards."""
EVAL_ANSWER_PROMPT = """Answer using ONLY the CONTEXT below. Be brief (1-3 sentences).
Output ONLY JSON: {"answer": str}."""
FAITH_PROMPT = """Score 0.0-1.0 if ANSWER is fully supported by CONTEXT (no hallucinations).
Output ONLY JSON with a number and nothing else: {"score": 0.95}."""
RELEV_PROMPT = """Score 0.0-1.0 if ANSWER addresses QUESTION (0=off-topic, 1=direct).
Output ONLY JSON with a number and nothing else: {"score": 0.9}."""

IMPERATIVES = ("collect", "verify", "check", "quarantine", "block", "reset", "notify",
               "escalate", "contain", "isolate", "revoke", "enforce", "create", "capture",
               "determine", "assess", "declare", "spin", "bring", "perform", "rollout",
               "close", "file", "schedule", "monitor", "remove", "disable")


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!])\s+|\n+", text) if len(s.strip()) > 40]


def source_must_dos(doc_text: str, cap: int = 7) -> List[str]:
    scored = []
    for s in _sentences(doc_text):
        low = s.lower()
        if any(v in low for v in IMPERATIVES) or re.match(r"\s*\d+[\.\)]", s):
            c = re.sub(r"<[^>]+>", "", s).strip()[:200]
            if len(c) < 40:
                continue
            score = sum(1 for v in IMPERATIVES if v in low)
            if re.match(r"\s*\d+[\.\)]", s):
                score += 2
            if len(c) < 80:  # intro fluff penalty
                score -= 1
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, c in scored:
        key = c.lower()[:70]
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= cap:
            break
    return out


def _overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z]{4,}", a.lower()))
    wb = set(re.findall(r"[a-z]{4,}", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa)


def coverage(aop: AOP, must_dos: List[str], thresh: float = 0.3) -> float:
    if not must_dos:
        return 1.0
    steps = [s.do + " " + (s.check or "") for s in aop.steps]
    hit = sum(1 for m in must_dos if any(_overlap(m, s) >= thresh for s in steps))
    return hit / len(must_dos)


def grounding(aop: AOP, docs) -> float:
    src = " ".join(d["text"].lower() for d in docs)
    src_words = set(re.findall(r"[a-z]{4,}", src))
    ok = 0
    for s in aop.steps:
        dw = set(re.findall(r"[a-z]{4,}", s.do.lower()))
        if dw and len(dw & src_words) / len(dw) >= 0.5:
            ok += 1
    return ok / max(1, len(aop.steps))


def executability(aop: AOP) -> float:
    if not aop.steps:
        return 0.0
    pts = 0.0
    for s in aop.steps:
        if s.do and len(s.do) <= 140:
            pts += 0.5
        if s.check or s.tool:
            pts += 0.3
        if s.on_fail:
            pts += 0.2
    return pts / len(aop.steps)


def _mock_questions(docs) -> List[Dict]:
    return [
        {"q": "What triggers this procedure?", "keywords": []},
        {"q": "What is the first containment step?", "keywords": []},
        {"q": "When must you escalate?", "keywords": []},
        {"q": "What must NOT be done?", "keywords": []},
        {"q": "What are recovery steps?", "keywords": []},
    ]


def _with_retry(fn, tries: int = 3):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
    raise last


def gen_questions(docs, provider) -> List[Dict]:
    if type(provider).__name__ == "MockProvider":
        return _mock_questions(docs)
    src = "\n".join(f"[{d['filename']}]\n{d['text'][:8000]}" for d in docs)
    out = _with_retry(lambda: provider.chat_json(EVAL_QUESTIONS_PROMPT, f"SOURCE:\n{src}"))
    return out.get("questions", [])[:5]


def answer(question: str, context: str, provider) -> str:
    out = _with_retry(lambda: provider.chat_json(EVAL_ANSWER_PROMPT, f"CONTEXT:\n{context[:12000]}\n\nQ: {question}"))
    return out.get("answer", "")


def judge(question: str, context: str, ans: str, provider) -> dict:
    """Built-in LLM judge (no new deps): faithfulness + relevancy. Needs real provider."""
    f = _with_retry(lambda: provider.chat_json(FAITH_PROMPT, f"CONTEXT:\n{context[:8000]}\nQ: {question}\nANSWER: {ans}"))
    r = _with_retry(lambda: provider.chat_json(RELEV_PROMPT, f"Q: {question}\nANSWER: {ans}"))
    return {"faith": float(f.get("score", 0)), "relev": float(r.get("score", 0))}


def try_ragas(samples: list):
    """Optional Ragas hook. Returns None if ragas not installed (Python 3.9 safe)."""
    try:
        import importlib.util as _u

        if _u.find_spec("ragas") is None:
            return None
        from datasets import Dataset
        from ragas import evaluate as _ev
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        ds = Dataset.from_list(samples)
        res = _ev(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        df = res.to_pandas()
        return {c: float(df[c].mean()) for c in df.columns if df[c].dtype != object}
    except Exception as e:
        return {"error": str(e)[:200]}


def evaluate(filepaths: List[str], provider_name=None, title=None, real: bool = False) -> dict:
    docs = load_documents(filepaths)
    provider = get_provider(provider_name)
    is_mock = type(provider).__name__ == "MockProvider"
    aop = generate_aop(filepaths, title_override=title, provider_name=provider_name)

    raw_ctx = "\n".join(f"[{d['filename']}]\n{d['text']}" for d in docs)[:30000]
    aop_ctx = to_markdown(aop)
    raw_tok = max(1, len(raw_ctx) // 4)
    aop_tok = aop_tokens(aop)

    # deterministic core: per-doc + overall coverage
    per_doc = {}
    for d in docs:
        md = source_must_dos(d["text"])
        per_doc[d["filename"]] = {"n_must": len(md), "coverage": round(coverage(aop, md), 3)}
    all_must = []
    for d in docs:
        all_must.extend(source_must_dos(d["text"]))
    cov = coverage(aop, all_must)
    grd = grounding(aop, docs)
    exe = executability(aop)
    tok_red = 1 - aop_tok / max(1, raw_tok)

    # real judge (needs non-mock provider): A/B answers scored by same LLM
    llm_ab = None
    ragas_scores = None
    if real:
        if is_mock:
            raise RuntimeError("--real needs a real provider (openai/anthropic/ollama), not mock")
        qs = gen_questions(docs, provider)
        rows, r_f, r_r, a_f, a_r = [], [], [], [], []
        samples = []
        for item in qs:
            q = item["q"]
            ra, aa = answer(q, raw_ctx, provider), answer(q, aop_ctx, provider)
            jr, ja = judge(q, raw_ctx, ra, provider), judge(q, aop_ctx, aa, provider)
            r_f.append(jr["faith"])
            r_r.append(jr["relev"])
            a_f.append(ja["faith"])
            a_r.append(ja["relev"])
            rows.append({"q": q, "raw": ra[:200], "aop": aa[:200],
                         "raw_faith": jr["faith"], "aop_faith": ja["faith"],
                         "raw_relev": jr["relev"], "aop_relev": ja["relev"]})
            samples.append({"question": q, "contexts": [[aop_ctx]], "answer": aa, "ground_truth": ra})
        llm_ab = {"raw_faith": round(sum(r_f) / len(r_f), 3), "aop_faith": round(sum(a_f) / len(a_f), 3),
                  "raw_relev": round(sum(r_r) / len(r_r), 3), "aop_relev": round(sum(a_r) / len(a_r), 3),
                  "rows": rows}
        ragas_scores = try_ragas(samples)

    checks = {
        "grounding>=0.8": round(grd, 3) >= 0.8,
        "executability>=0.8": round(exe, 3) >= 0.8,
        "token_reduction>=0.7": round(tok_red, 3) >= 0.7,
    }
    if llm_ab:
        checks["aop_faith>=raw_faith-0.1"] = llm_ab["aop_faith"] >= llm_ab["raw_faith"] - 0.1
        checks["aop_relev>=raw_relev-0.1"] = llm_ab["aop_relev"] >= llm_ab["raw_relev"] - 0.1
        verdict = "PASS" if all(checks.values()) else "FAIL"
    else:
        verdict = "STRUCTURAL-PASS" if all(checks.values()) else "FAIL"

    return {"report": {
        "aop_id": aop.id, "provider": type(provider).__name__, "verdict": verdict, "mode": "real" if real else "structural",
        "raw_tokens_est": raw_tok, "aop_tokens_est": aop_tok,
        "token_reduction": round(tok_red, 3),
        "coverage": round(cov, 3), "grounding": round(grd, 3), "executability": round(exe, 3),
        "per_doc": per_doc, "checks": checks, "llm_ab": llm_ab, "ragas": ragas_scores,
        "note": "Structural verdict is offline. --real runs LLM judge (faith/relev) + optional Ragas; needs API key.",
    }, "aop": aop.model_dump(), "markdown": aop_ctx}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--provider", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--real", action="store_true", help="LLM judge A/B + Ragas if installed (needs real provider)")
    args = p.parse_args()
    try:
        out = evaluate(args.files, provider_name=args.provider, title=args.title, real=args.real)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    r = out["report"]
    print(f"verdict={r['verdict']} mode={r['mode']} provider={r['provider']}")
    print(f"tokens raw~{r['raw_tokens_est']} aop~{r['aop_tokens_est']} reduction={r['token_reduction']*100:.1f}%")
    print(f"coverage={r['coverage']} grounding={r['grounding']} executability={r['executability']}")
    print(f"checks={json.dumps(r['checks'])}")
    if r.get("llm_ab"):
        print(f"judge={json.dumps({k: v for k, v in r['llm_ab'].items() if k != 'rows'})}")
    if r.get("ragas"):
        print(f"ragas={json.dumps(r['ragas'])}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"Saved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
