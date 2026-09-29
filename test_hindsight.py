import os, json, requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
BASE = "https://api.hindsight.vectorize.io/v1/default"
HEADERS = {"Authorization": f"Bearer {os.getenv('HINDSIGHT_API_KEY')}",
           "Content-Type": "application/json"}
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


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
    if r.status_code != 200:
        st.error(f"Recall failed: {r.status_code} {r.text[:200]}")
        return []
    return [m.get("text", "") for m in r.json().get("results", [])]


def ask(prompt, as_json=False):
    kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
    res = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role": "user", "content": prompt}], **kwargs)
    return res.choices[0].message.content


st.set_page_config(page_title="Promise-Keeper", layout="wide")
st.title("🤝 Promise-Keeper")
st.caption("AI meeting prep agent with persistent Hindsight memory")

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
        transcript = st.text_area("Paste notes / transcript", height=200)
        if st.button("Process & Save to Memory") and transcript:
            with st.spinner("Working..."):
                # 1. Check whether old promises were fulfilled
                open_mems = recall(bank, "What promises are open or overdue?")
                st.session_state["recalled"] = open_mems
                if open_mems:
                    out = ask(
                        "Open promises:\n" + "\n".join(open_mems) +
                        f"\n\nNew transcript:\n{transcript}\n\n"
                        'Return JSON: {"fulfilled": [{"promise": "...", "evidence": "..."}]}. '
                        "Only include promises clearly completed in the transcript.",
                        as_json=True)
                    for f in json.loads(out).get("fulfilled", []):
                        st.success(f"✅ Fulfilled: {f['promise']}")
                        retain(bank, f"On {date}, this promise was FULFILLED: "
                                     f"{f['promise']}. Evidence: {f['evidence']}")
                # 2. Extract new promises
                out = ask(
                    f"Transcript:\n{transcript}\n\n"
                    'Return JSON: {"commitments": [{"owner": "", "recipient": "", '
                    '"task": "", "due_date": "YYYY-MM-DD or Unspecified"}]}',
                    as_json=True)
                commitments = json.loads(out).get("commitments", [])
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