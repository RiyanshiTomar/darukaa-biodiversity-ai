"""
ui/app.py — Streamlit demo UI
--------------------------------
Talks to the FastAPI backend (must be running: uvicorn backend.main:app --port 8000).

Run:
    streamlit run ui/app.py
"""

import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Darukaa.Earth Biodiversity AI", page_icon="🌱", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #f3f6ef;
        --panel: #ffffff;
        --primary: #1f7a4d;
        --primary-soft: #eaf7f0;
        --accent: #d39247;
        --text: #1f2a1f;
        --muted: #5a645d;
        --border: #dfe8dc;
    }
    .stApp {
        background: linear-gradient(180deg, #f4f7f1 0%, #eef4ee 100%);
        color: var(--text);
    }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.78);
        border-right: 1px solid var(--border);
    }
    .hero {
        text-align: center;
        padding: 1.5rem 1rem 0.8rem 1rem;
    }
    .hero-title {
        font-size: 3rem;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #1b1f1c;
    }
    .hero-badge {
        background: var(--primary-soft);
        color: var(--primary);
        border: 1px solid #cfe6d5;
        border-radius: 999px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 1.08rem;
        max-width: 760px;
        margin: 0 auto;
        padding-top: 0.5rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.8);
        box-shadow: 0 7px 18px rgba(31, 52, 37, 0.03);
    }
    .stChatInput {
        border-top: 1px solid var(--border);
        padding-top: 0.8rem;
        background: rgba(255,255,255,0.38);
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background: #f7faf6;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    .metric-card {
        background: linear-gradient(180deg, rgba(28,122,77,0.05), rgba(28,122,77,0.01));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
        <div class='hero-badge'>DARUKAA.EARTH</div>
        <div class='hero-title'>Biodiversity Intelligence Chatbot</div>
        <div class='hero-sub'>Evidence-backed, multi-metric ecosystem recommendations — grounded in biodiversity knowledge, not generic AI advice.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Optional: structured input")
    st.caption("Add exact numbers when available to improve grounding.")
    soc = st.text_input("Soil organic carbon (%)", "")
    rainfall = st.selectbox("Rainfall", ["", "low", "moderate", "high"])
    crop = st.text_input("Current crop / land cover", "")
    region = st.selectbox("Region type", ["", "semi-arid", "tropical", "temperate", "arid"])
    st.divider()
    if st.button("Reset conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe your land / ecosystem concern...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    structured = {}
    if soc:
        try:
            structured["soil_organic_carbon_pct"] = float(soc)
        except ValueError:
            pass
    if rainfall:
        structured["rainfall"] = rainfall
    if crop:
        structured["crop"] = crop
    if region:
        structured["region"] = region

    with st.spinner("Reasoning over the biodiversity knowledge base..."):
        resp = requests.post(
            f"{BACKEND_URL}/chat/structured",
            json={
                "session_id": st.session_state.session_id,
                "message": user_input,
                "structured_data": structured,
            },
            timeout=60,
        )
        data = resp.json()

    reply = data.get("message", "Something went wrong.")
    with st.chat_message("assistant"):
        st.markdown(reply)
        if data.get("type") == "recommendation":
            st.caption(f"Grounded on KB entries: {', '.join(data.get('used_kb_ids', []))}")

    st.session_state.messages.append({"role": "assistant", "content": reply})
