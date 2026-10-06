import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
AOP_DIR = os.path.join(BASE_DIR, "aops")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AOP_DIR, exist_ok=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()  # mock | openai | anthropic | ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

MAX_CHARS_PER_DOC = int(os.getenv("MAX_CHARS_PER_DOC", "15000"))
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_CHARS", "30000"))

# Chunked pipeline budgets (no truncation: full text is chunked and map-reduced)
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "6000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "400"))
DOC_HARD_CAP = int(os.getenv("DOC_HARD_CAP", "200000"))  # safety only, warns instead of cutting
