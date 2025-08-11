from openai import OpenAI
from dotenv import load_dotenv
import os, json, time, math, subprocess, pathlib, sys
import feedparser, requests
from pathlib import Path

# ====== Setup ======
load_dotenv()                     # .env があれば読む（Actionsでは env: から渡す）
client = OpenAI()                 # OPENAI_API_KEY は環境変数から

PROJECT_ID = os.getenv("PROJECT_ID", "podcast日本語訳")
RSS_URL    = os.getenv("RSS_URL")

OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)
STATE_F = Path("state.json")

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
CHAT_MODEL       = os.getenv("CHAT_MODEL",       "gpt-4o-mini")
CHUNK_SECONDS    = int(os.getenv("CHUNK_SECONDS", "1200"))   # 20分。モデル上限1400秒に余裕

PROMPT = (
    "You are an expert EN->JA editor.\n"
    "Input is an English transcript.\n\n"
    "Tasks:\n"
    "1) Summarize in Japanese (200-300 words, bullet points allowed).\n"
    "2) Provide faithful Japanese translation (keep sections/paragraphs).\n"
    "3) Extract >=10 idioms / phrasal verbs / slang with:\n"
    "   - en_phrase\n"
    "   - ja_meaning (1-2 lines)\n"
    "   - example_en + example_ja\n"
    "Return JSON with keys:\n"
    "{summary_ja, translation_ja, idioms:[{en, ja, example_en, example_ja}]}\n"
    "Then append a clean, human-readable Japanese report.\n"
)

def log(*args): print(*args, file=sys.stderr)

# ====== Guards ======
def guard_env():
    ok = True
    if not os.getenv("OPENAI_API_KEY"):
        log("ERROR: OPENAI_API_KEY is missing in Secrets/env.")
        ok = False
    if not RSS_URL:
        log("ERROR: RSS_URL is missing. Set it in workflow env or .env.")
        ok = False
    return ok

# ====== State ======
def load_state():
    if STATE_F.exists():
        return json.loads(STATE_F.read_text())
    return {"done_ids": []}

def save_state(st):
    STATE_F.write_text(json.dumps(st, ensure_ascii=False, indent=2))

# ====== RSS ======
def fetch_new_episodes(rss_url, done_ids):
    feed = feedparser.parse(rss_url)
    items = []
    for e in feed.entries:
        eid =
