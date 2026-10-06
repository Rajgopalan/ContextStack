"""Section-aware chunking: pack by headings/paragraphs, never hard-truncate prose."""
import re
from typing import Dict, List


def split_sections(text: str) -> List[Dict[str, str]]:
    """Split markdown-ish text into sections on headings; fallback to paragraph blocks."""
    lines = text.splitlines()
    sections, cur_head, cur_buf = [], "", []
    head_re = re.compile(r"^(#{1,4}\s+|\d+\.\s+[A-Z].{0,80}$)")
    for line in lines:
        if head_re.match(line.strip()) and cur_buf:
            sections.append({"heading": cur_head, "text": "\n".join(cur_buf).strip()})
            cur_head, cur_buf = line.strip(), []
        else:
            if head_re.match(line.strip()) and not cur_buf:
                cur_head = line.strip()
            else:
                cur_buf.append(line)
    if cur_buf or cur_head:
        sections.append({"heading": cur_head, "text": "\n".join(cur_buf).strip()})
    sections = [s for s in sections if s["text"]]
    if not sections:
        return [{"heading": "", "text": text.strip()}]
    return sections


def _split_oversize(text: str, budget: int) -> List[str]:
    """Split an oversized section on paragraph, then sentence, boundaries."""
    paras, out, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= budget:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                out.append(cur)
            if len(p) <= budget:
                cur = p
            else:  # hard split long paragraph on sentences
                sents = re.split(r"(?<=[.!])\s+", p)
                cur = ""
                for s in sents:
                    if len(cur) + len(s) + 1 <= budget:
                        cur = (cur + " " + s).strip()
                    else:
                        if cur:
                            out.append(cur)
                        cur = s[:budget]
                if cur:
                    out.append(cur)
                    cur = ""
    if cur:
        out.append(cur)
    return [o for o in out if o.strip()]


def chunk_text(text: str, budget: int = 6000, overlap: int = 400) -> List[Dict[str, str]]:
    """Greedily pack sections into chunks <= budget; overlap tail chars for continuity."""
    sections = split_sections(text)
    chunks, cur, cur_head = [], "", ""
    for s in sections:
        body = s["text"]
        if len(body) > budget:
            if cur.strip():
                chunks.append({"heading": cur_head, "text": cur.strip()})
                cur, cur_head = "", ""
            for i, piece in enumerate(_split_oversize(body, budget)):
                head = s["heading"] + (f" (part {i+1})" if len(_split_oversize(body, budget)) > 1 else "")
                chunks.append({"heading": head, "text": piece})
            continue
        if len(cur) + len(body) + 2 <= budget:
            cur = (cur + "\n\n" + body).strip()
            cur_head = cur_head or s["heading"]
        else:
            if cur.strip():
                chunks.append({"heading": cur_head, "text": cur.strip()})
            cur, cur_head = body, s["heading"]
    if cur.strip():
        chunks.append({"heading": cur_head, "text": cur.strip()})
    # attach overlap: prefix each chunk (after first) with tail of previous
    for i in range(1, len(chunks)):
        tail = chunks[i - 1]["text"][-overlap:]
        cut = tail.find("\n")
        tail = tail[cut + 1 :] if cut != -1 else tail
        if tail.strip():
            chunks[i]["text"] = "[...continued]\n" + tail.strip() + "\n\n" + chunks[i]["text"]
    for i, c in enumerate(chunks):
        c["index"] = i
    return chunks
