from openai import OpenAI
from dotenv import load_dotenv
import os, json, feedparser, requests, pathlib, sys

load_dotenv()
client = OpenAI()

PROJECT_ID = os.getenv("PROJECT_ID")
RSS_URL    = os.getenv("RSS_URL")
OUT_DIR    = pathlib.Path("out"); OUT_DIR.mkdir(exist_ok=True)
STATE_F    = pathlib.Path("state.json")

TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
CHAT_MODEL       = "gpt-4o-mini"

PROMPT = (
    "You are an expert EN->JA editor.\\n"
    "Input is an English transcript.\\n\\n"
    "Tasks:\\n"
    "1) Summarize in Japanese (200-300 words, bullet points allowed).\\n"
    "2) Provide faithful Japanese translation (keep sections/paragraphs).\\n"
    "3) Extract >=10 idioms / phrasal verbs / slang with:\\n"
    "   - en_phrase\\n"
    "   - ja_meaning (1-2 lines)\\n"
    "   - example_en + example_ja\\n"
    "Return JSON with keys:\\n"
    "{summary_ja, translation_ja, idioms:[{en, ja, example_en, example_ja}]}\\n"
    "Then append a clean, human-readable Japanese report.\\n"
)

def log(*args): print(*args, file=sys.stderr)

def load_state():
    if STATE_F.exists():
        return json.loads(STATE_F.read_text())
    return {"done_ids": []}

def save_state(st):
    STATE_F.write_text(json.dumps(st, ensure_ascii=False, indent=2))

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
        items.append({"id": eid, "title": e.get("title", "untitled"), "url": audio_url})
    return items

def safe_filename(name: str) -> str:
    keep = "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in name)[:80]

def download_audio(url, dest):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    return dest

def transcribe(audio_path):
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f)
    return tr.text

def postprocess_with_gpt(transcript_text):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise bilingual editor."},
            {"role": "user", "content": f"{PROMPT}\\n\\n---\\nTRANSCRIPT:\\n{transcript_text}"}
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def upload_to_vector_store(files, project_id):
    vs = client.vector_stores.create(name=project_id)
    for p in files:
        f = client.files.create(purpose="assistants", file=p)
        client.vector_stores.files.create(vector_store_id=vs.id, file_id=f.id)
    return vs.id

def run_once():
    st = load_state()
    new_items = fetch_new_episodes(RSS_URL, st["done_ids"])
    if not new_items:
        log("No new episodes.")
        return
    for it in new_items:
        if not it["url"]:
            continue
        base = safe_filename(it["id"] or it["title"])
        audio_path = OUT_DIR / f"{base}.mp3"
        txt_path   = OUT_DIR / f"{base}.en.txt"
        rep_path   = OUT_DIR / f"{base}.ja_report.txt"
        log(f"[DL] {it['title']}")
        download_audio(it["url"], audio_path)
        log("[STT] transcribing…")
        en_text = transcribe(str(audio_path))
        txt_path.write_text(en_text, encoding="utf-8")
        log("[GPT] post-processing…")
        ja_report = postprocess_with_gpt(en_text)
        rep_path.write_text(ja_report, encoding="utf-8")
        log("[STORE] uploading…")
        upload_to_vector_store([txt_path, rep_path], PROJECT_ID)
        st["done_ids"].append(it["id"])
        save_state(st)

if __name__ == "__main__":
    run_once()
