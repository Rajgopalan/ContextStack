import os
import shutil
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .generator import generate_aop, save_aop
from .models import AOP

app = FastAPI(title="AOP Generator", version="0.1.0")

ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}

os.makedirs("static", exist_ok=True)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    idx = os.path.join("static", "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "AOP Generator API. See /docs"}


class GenerateRequest(BaseModel):
    file_ids: List[str]
    title: Optional[str] = None
    provider: Optional[str] = None  # mock | openai | anthropic | ollama


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(400, f"Unsupported type {ext} for {f.filename}")
        fid = f"{uuid.uuid4().hex[:8]}_{f.filename}"
        dest = os.path.join(config.UPLOAD_DIR, fid)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append({"file_id": fid, "filename": f.filename})
    return {"files": saved}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not 1 <= len(req.file_ids) <= 2:
        raise HTTPException(400, "Provide 1 or 2 file_ids")
    paths = []
    for fid in req.file_ids:
        # guard path traversal
        safe = os.path.basename(fid)
        p = os.path.join(config.UPLOAD_DIR, safe)
        if not os.path.exists(p):
            raise HTTPException(404, f"File not found: {fid}")
        paths.append(p)
    try:
        aop = generate_aop(paths, title_override=req.title, provider_name=req.provider)
    except Exception as e:
        raise HTTPException(500, str(e))
    save_aop(aop)
    return aop.model_dump()


@app.get("/api/aops")
def list_aops():
    out = []
    for fn in sorted(os.listdir(config.AOP_DIR)):
        if fn.endswith(".json"):
            out.append(fn[:-5])
    return {"aops": out}


@app.get("/api/aops/{aop_id}")
def get_aop(aop_id: str):
    import json

    safe = os.path.basename(aop_id)
    p = os.path.join(config.AOP_DIR, f"{safe}.json")
    if not os.path.exists(p):
        raise HTTPException(404, "AOP not found")
    with open(p, encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))
