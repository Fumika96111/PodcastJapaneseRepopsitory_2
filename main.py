from openai import OpenAI
from dotenv import load_dotenv
import os, json, time, math, subprocess, pathlib, sys
import feedparser, requests
from pathlib import Path

# ====== Setup ======
load_dotenv()
client = OpenAI()

PROJECT_ID = os.getenv("PROJECT_ID", "podcast日本語訳")
RSS_URL    = os.getenv("RSS_URL")
OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)
STATE_F = Path("state.json")

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
CHAT_MODEL       = os.getenv("CHAT_MODEL",       "gpt-4o-mini")
CHUNK_SECONDS    = int(os.getenv("CHUNK_SECONDS", "1200"))  # 20 minutes
VECTOR_STORE_ID  = os.getenv("VECTOR_STORE_ID")  # optional: force a specific VS

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

def guard_env():
    ok = True
    if not os.getenv("OPENAI_API_KEY"):
        log("ERROR: OPENAI_API_KEY is missing.")
        ok = False
    if not RSS_URL:
        log("ERROR: RSS_URL is missing.")
        ok = False
    return ok

def load_state():
    if STATE_F.exists():
        return json.loads(STATE_F.read_text())
    return {"done_ids": []}

def save_state(st):
    STATE_F.write_text(json.dumps(st, ensure_ascii=False, indent=2))

# ====== Vector Store (reuse) ======
def get_or_create_vector_store(project_id):
    global VECTOR_STORE_ID
    st = load_state()
    VECTOR_STORE_ID = VECTOR_STORE_ID or st.get("vector_store_id")
    if VECTOR_STORE_ID:
        return VECTOR_STORE_ID
    vs = client.vector_stores.create(name=project_id)
    VECTOR_STORE_ID = vs.id
    st["vector_store_id"] = vs.id
    save_state(st)
    return vs.id

def upload_to_vector_store(files, project_id):
    vs_id = get_or_create_vector_store(project_id)
    uploaded = []
    for p in files:
        f = client.files.create(purpose="assistants", file=p)
        client.vector_stores.files.create(vector_store_id=vs_id, file_id=f.id)
        uploaded.append(f.id)
    return vs_id, uploaded

# ====== RSS ======
def fetch_new_episodes(rss_url, done_ids):
    feed = feedparser.parse(rss_url)
    items = []
    for e in feed.entries:
        eid = e.get('id') or e.get('link') or e.get('guid') or e.get('title')
        if not eid or eid in done_ids:
            continue
        audio_url = None
        if getattr(e, "enclosures", None):
            audio_url = e.enclosures[0].get("href")
        items.append({
            "id": eid,
            "title": e.get("title", "untitled"),
            "url": audio_url,
            "published": e.get("published") or e.get("updated"),
        })
    return items

def safe_filename(name: str) -> str:
    keep = "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in name)[:80]

def download_audio(url, dest):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest

# ====== Split (ffmpeg) ======
def split_audio_ffmpeg(src_path: str, chunk_seconds: int = CHUNK_SECONDS):
    src = Path(src_path)
    # total duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)],
        capture_output=True, text=True, check=True
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    import math
    parts = []
    n = math.ceil(duration / chunk_seconds)
    for i in range(n):
        start = i * chunk_seconds
        out = src.with_name(f"{src.stem}_part{i+1:02d}{src.suffix}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-t", str(chunk_seconds), "-c", "copy", str(out)],
            check=True
        )
        parts.append(out)
    return parts

# ====== OpenAI Calls ======
def transcribe(audio_path, retries=3, backoff=5):
    for attempt in range(1, retries+1):
        try:
            with open(audio_path, "rb") as f:
                tr = client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f)
            return tr.text
        except Exception as e:
            if attempt == retries:
                raise
            log(f"[WARN] transcribe retry {attempt}/{retries} after error: {e}")
            time.sleep(backoff * attempt)

def postprocess_with_gpt(transcript_text, retries=3, backoff=5):
    for attempt in range(1, retries+1):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise bilingual editor."},
                    {"role": "user", "content": f"{PROMPT}\n\n---\nTRANSCRIPT:\n{transcript_text}"}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries:
                raise
            log(f"[WARN] chat retry {attempt}/{retries} after error: {e}")
            time.sleep(backoff * attempt)

# ====== Main ======
def run_once():
    if not guard_env():
        sys.exit(1)

    st = load_state()
    items = fetch_new_episodes(RSS_URL, st["done_ids"])
    if not items:
        log("No new episodes.")
        return

    for it in items:
        if not it["url"]:
            log(f"[SKIP] No audio URL: {it['title']}")
            st["done_ids"].append(it["id"]); save_state(st)
            continue

        base = safe_filename(it["id"] or it["title"])
        audio_path = OUT_DIR / f"{base}.mp3"
        txt_path   = OUT_DIR / f"{base}.en.txt"
        rep_path   = OUT_DIR / f"{base}.ja_report.txt"

        log(f"[DL] {it['title']}")
        download_audio(it["url"], audio_path)

        log("[STT] splitting for model max duration…")
        parts = split_audio_ffmpeg(str(audio_path), chunk_seconds=CHUNK_SECONDS)
        log(f"[STT] transcribing {len(parts)} part(s)…")

        texts = []
        for idx, part in enumerate(parts, 1):
            log(f"  - part {idx}/{len(parts)}: {part.name}")
            t = transcribe(str(part))
            texts.append(t)

        en_text = "\n\n".join(texts)
        txt_path.write_text(en_text, encoding="utf-8")

        log("[GPT] post-processing…")
        ja_report = postprocess_with_gpt(en_text)
        rep_path.write_text(ja_report, encoding="utf-8")

        log("[STORE] uploading to vector store…")
        vs_id, file_ids = upload_to_vector_store([txt_path, rep_path], PROJECT_ID)
        log(f"Done: {it['title']}  VS={vs_id}  files={file_ids}")

        st["done_ids"].append(it["id"])
        save_state(st)

if __name__ == "__main__":
    run_once()

