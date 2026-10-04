# ContextStack

Build high-quality agent context from source documents.
Current module: **AOP Generator**. Next: **Skills**, prompts, context packs.

## 1. AOP Generator

Generates one machine-readable **Agent Operation Procedure (AOP)** from 1-2 uploaded documents (PDF, DOCX, TXT, MD).

### Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # default LLM_PROVIDER=mock (works offline)
uvicorn src.main:app --reload
# open http://127.0.0.1:8000
```

API docs: `http://127.0.0.1:8000/docs`

### Usage

1. `POST /api/upload` with 1-2 files → returns `file_id`s
2. `POST /api/generate` with `{ "file_ids": [...], "provider": "mock" }` → returns AOP JSON
3. `GET /api/aops/{id}` to retrieve

CLI test (no server):
```bash
python -m src.cli examples/sample_doc_1_refund_policy.md examples/sample_doc_2_payments_api.md
```

### LLM providers (pluggable)

Set `LLM_PROVIDER` in `.env`: `mock | openai | anthropic | ollama`

- `mock`: offline rule-based, no key, good for testing
- `openai`: needs `OPENAI_API_KEY`
- `anthropic`: needs `ANTHROPIC_API_KEY`
- `ollama`: needs local Ollama at `OLLAMA_BASE_URL`

### Output schema

See `src/models.py` (`AOP` model): title, description, preconditions, inputs, outputs, ordered `steps[]` (action_type, tool, parameters, expected_output, validation, on_failure), error_handling.

AOPs are saved to `aops/{id}.json`, uploads to `uploads/`.

## Roadmap

- [x] AOP generation from 1-2 docs
- [ ] Skill generation (SKILL.md + scripts)
- [ ] Prompt / instruction pack generation
- [ ] Multi-doc merge + eval
