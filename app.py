import os, json, base64, time, requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BASE = "https://api.hindsight.vectorize.io/v1/default"
HEADERS = {"Authorization": f"Bearer {os.getenv('HINDSIGHT_API_KEY')}",
           "Content-Type": "application/json"}

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# Set LLM_PROVIDER=groq in .env to force Groq even when a Gemini key exists
USE_GEMINI = bool(GEMINI_KEY) and os.getenv("LLM_PROVIDER", "gemini").lower() != "groq"
if USE_GEMINI:
    from openai import OpenAI
    client = OpenAI(api_key=GEMINI_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
else:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def call_llm(**kwargs):
    """Call the model, retrying automatically when the server is busy."""
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code in (500, 502, 503, 504) and attempt < 5:
                time.sleep(3 * (attempt + 1))
                continue
            if code == 429:
                st.error("Daily free request limit reached for this model. "
                         "Use a different network with Groq, a new Gemini key, "
                         "or set GEMINI_MODEL in .env to another model.")
                st.stop()
            if code == 403:
                st.error("Groq blocked by this network (403). Try a phone "
                         "hotspot, or use Gemini instead.")
                st.stop()
            raise


def retain(bank, text):
    r = requests.post(f"{BASE}/banks/{bank}/memories",
                      json={"items": [{"content": text}]},
                      headers=HEADERS, timeout=60)
    if r.status_code != 200:
        st.error(f"Retain failed: {r.status_code} {r.text[:200]}")
    return r.status_code == 200


def recall(bank, query):
    r = requests.post(f"{BASE}/banks/{bank}/memories/recall",
                      json={"query": query}, headers=HEADERS, timeout=60)
    if r.status_code == 404:
        return []  # bank doesn't exist yet: no memories stored so far
    if r.status_code != 200:
        st.error(f"Recall failed: {r.status_code} {r.text[:200]}")
        return []
    return [m.get("text", "") for m in r.json().get("results", [])]


def ask(prompt, as_json=False):
    kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
    res = call_llm(
        model=MODEL, temperature=0.2,
        messages=[{"role": "user", "content": prompt}], **kwargs)
    return res.choices[0].message.content


def transcribe(audio_bytes, filename="meeting.wav"):
    fmt = "mp3" if filename.lower().endswith(".mp3") else "wav"
    if USE_GEMINI:
        b64 = base64.b64encode(audio_bytes).decode()
        res = call_llm(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text",
                 "text": "Transcribe this meeting audio word for word. "
                         "Label speakers as Speaker 1, Speaker 2 if you can "
                         "tell them apart. Return only the transcript."},
                {"type": "input_audio",
                 "input_audio": {"data": b64, "format": fmt}},
            ]}])
        return res.choices[0].message.content
    res = client.audio.transcriptions.create(
        file=(filename, audio_bytes), model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3"),
        response_format="text")
    return res if isinstance(res, str) else res.text


st.set_page_config(page_title="Promise-Keeper", layout="wide")
st.title("🤝 Promise-Keeper")
st.caption("AI meeting prep agent with persistent Hindsight memory")

me = st.sidebar.text_input("Your name", "Tony")
name = st.sidebar.text_input("Contact name", "Priya Sharma")
bank = "contact_" + name.lower().replace(" ", "_")
menu = st.sidebar.radio("Mode", ["Add Meeting Notes", "Pre-Meeting Brief"])

main, inspector = st.columns([1.2, 0.8])

with inspector:
    st.subheader("🧠 Live Memory Inspector")
    st.info(f"Memory bank: {bank}")
    for i, m in enumerate(st.session_state.get("recalled", []), 1):
        st.warning(f"Memory {i}: {m}")

with main:
    if menu == "Add Meeting Notes":
        date = st.date_input("Meeting date")

        source = st.radio("How do you want to add the meeting?",
                          ["🎙️ Record live", "📁 Upload audio", "⌨️ Paste text"],
                          horizontal=True)

        audio = None
        if source.startswith("🎙️"):
            audio = st.audio_input("Click the mic, talk, then click stop")
        elif source.startswith("📁"):
            audio = st.file_uploader("Upload a meeting recording",
                                     type=["wav", "mp3"])

        if audio is not None and st.button("📝 Transcribe audio"):
            with st.spinner("Transcribing..."):
                try:
                    st.session_state["transcript"] = transcribe(
                        audio.getvalue(), getattr(audio, "name", "meeting.wav"))
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

        transcript = st.text_area("Transcript (you can edit it before saving)",
                                  key="transcript", height=200)

        if st.button("Process & Save to Memory") and transcript:
            with st.spinner("Working..."):
                open_mems = recall(bank, "What promises are open or overdue?")
                st.session_state["recalled"] = open_mems
                open_text = "\n".join(open_mems) if open_mems else "(none)"
                out = ask(
                    f"The user is {me}. The other person is {name}. "
                    f"The meeting date is {date}; convert relative dates like "
                    f"'Friday' into YYYY-MM-DD based on it.\n\n"
                    f"Previously open promises:\n{open_text}\n\n"
                    f"New transcript:\n{transcript}\n\n"
                    "Return ONLY JSON with this shape: "
                    '{"fulfilled": [{"promise": "...", "evidence": "..."}], '
                    '"commitments": [{"owner": "", "recipient": "", "task": "", '
                    '"due_date": "YYYY-MM-DD or Unspecified"}]}. '
                    "'fulfilled' lists only earlier promises clearly completed "
                    "in this transcript. 'commitments' lists NEW promises made "
                    "in this transcript; use real names instead of 'I'.",
                    as_json=True)
                data = json.loads(out)
                for f in data.get("fulfilled", []):
                    st.success(f"✅ Fulfilled: {f['promise']}")
                    retain(bank, f"On {date}, this promise was FULFILLED: "
                                 f"{f['promise']}. Evidence: {f['evidence']}")
                commitments = data.get("commitments", [])
                for c in commitments:
                    retain(bank, f"On {date}, {c['owner']} promised {c['recipient']} "
                                 f"to {c['task']} (due: {c['due_date']}). Status: OPEN.")
                st.subheader("📋 New commitments saved")
                st.json(commitments)

    else:
        purpose = st.text_input("Upcoming meeting purpose", "Q4 strategy")
        if st.button("Generate Brief"):
            with st.spinner("Recalling memory..."):
                mems = recall(bank, "open promises, overdue items, recent interactions")
                st.session_state["recalled"] = mems
                brief = ask(
                    f"Write a concise pre-meeting brief in Markdown for {name}. "
                    f"Meeting purpose: {purpose}. Today is 2026-09-29.\n"
                    "Put OPEN and OVERDUE promises at the top, then recent highlights.\n\n"
                    "Memories:\n" + "\n".join(mems))
                st.markdown(brief)
