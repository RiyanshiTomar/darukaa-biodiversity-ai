"""
reasoning_engine.py
---------------------
This is the "brain" of the system. It does three jobs:

1. SLOT FILLING — tracks which of the 5 required environmental variable
   categories (soil, land use, biodiversity, climate, human impact) are known
   for the current conversation. If fewer than 3 are known, it asks a
   clarifying question instead of guessing.

2. RETRIEVAL — once enough is known, it queries the KnowledgeRetriever
   (semantic + structured) to pull grounded, citable KB entries.

3. GENERATION — it builds a strict prompt that forces the LLM to only
   reason on top of the retrieved KB entries (not invent new facts), and to
   output in the mandatory structured format: Recommendation / Why /
   Metrics impacted / Time horizon / Confidence / Source.

The LLM used is Google Gemini (gemini-1.5-flash), called directly via the
google-generativeai SDK. Swap `call_llm()` for any other provider if needed.
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from rag.retriever import KnowledgeRetriever

load_dotenv()

# ---- Required variable categories (the "multi-metric" backbone) ----
REQUIRED_SLOTS = {
    "soil_health": ["soil organic carbon", "soc", "ph", "moisture", "soil"],
    "land_use": ["land use", "monoculture", "crop", "intercropping", "agroforestry", "pasture", "forest"],
    "biodiversity": ["species", "biodiversity", "habitat", "pollinator", "wildlife"],
    "climate": ["rainfall", "temperature", "climate", "drought", "semi-arid", "monsoon"],
    "human_impact": ["pollution", "deforestation", "pesticide", "irrigation", "grazing", "runoff"],
}

MIN_SLOTS_BEFORE_ANSWERING = 3  # rubric: "must handle at least 3 environmental variables together"


@dataclass
class ConversationState:
    """Per-session memory: what we know so far about this user's land/query."""
    session_id: str
    known_slots: dict = field(default_factory=dict)   # slot_name -> user-provided value/text
    history: list = field(default_factory=list)        # [{"role": "user"/"assistant", "content": str}]

    def filled_slot_count(self) -> int:
        return len(self.known_slots)

    def missing_slots(self) -> list:
        return [s for s in REQUIRED_SLOTS if s not in self.known_slots]


def detect_slots_in_text(text: str) -> dict:
    """Very lightweight keyword-based slot detection for free-text input."""
    text_lower = text.lower()
    found = {}
    for slot, keywords in REQUIRED_SLOTS.items():
        for kw in keywords:
            if kw in text_lower:
                found[slot] = text  # store the raw sentence as evidence; refine with structured input when available
                break
    return found


def merge_structured_input(state: ConversationState, structured: dict):
    """
    structured example:
    {
        "soil_organic_carbon_pct": 0.3,
        "rainfall": "low",
        "crop": "monoculture wheat",
        "region": "semi-arid"
    }
    Maps structured JSON fields onto the 5 slot categories deterministically.
    """
    if not structured:
        return
    if any(k in structured for k in ["soil_organic_carbon_pct", "soil_ph", "soil_moisture"]):
        state.known_slots["soil_health"] = structured
    if any(k in structured for k in ["crop", "land_cover", "region_land_use"]):
        state.known_slots["land_use"] = structured
    if any(k in structured for k in ["species_observed", "habitat_type"]):
        state.known_slots["biodiversity"] = structured
    if any(k in structured for k in ["rainfall", "temperature", "region"]):
        state.known_slots["climate"] = structured
    if any(k in structured for k in ["pollution_level", "deforestation", "pesticide_use"]):
        state.known_slots["human_impact"] = structured


def next_clarifying_question(state: ConversationState) -> str:
    """Ask for ONE missing slot at a time, prioritized in rubric order."""
    priority = ["soil_health", "land_use", "climate", "biodiversity", "human_impact"]
    for slot in priority:
        if slot not in state.known_slots:
            questions = {
                "soil_health": "Can you share your soil organic carbon % (or a rough idea — low/medium/high), and soil pH if known?",
                "land_use": "What's currently growing on this land — is it a monoculture, mixed cropping, pasture, or something else?",
                "climate": "What's the rainfall like in your region — low, moderate, high — and is it a semi-arid, tropical, or temperate zone?",
                "biodiversity": "Have you noticed any specific decline — fewer pollinators/birds, less soil life — or is it a general concern?",
                "human_impact": "Any pesticide use, irrigation source, or nearby pollution/deforestation worth knowing about?",
            }
            return questions[slot]
    return None


def build_grounded_prompt(user_query: str, kb_hits: list, known_slots: dict) -> str:
    """
    Builds the strict, KB-grounded prompt. The LLM is instructed to ONLY use
    the provided KB context — this is what prevents hallucinated statistics.
    """
    context_blocks = []
    for hit in kb_hits:
        r = hit["record"] if "record" in hit else hit
        context_blocks.append(
            f"- [{r['id']}] {r['topic']}\n"
            f"  What it does: {r['content']}\n"
            f"  Quantified evidence: {r.get('quantified_impact', 'N/A')}\n"
            f"  Metrics affected: {', '.join(r['metrics_affected'])}\n"
            f"  Source: {r['source']}\n"
            f"  Time horizon: {r.get('time_horizon', 'N/A')} | Confidence: {r.get('confidence', 'N/A')}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No strong KB matches found."

    known_summary = json.dumps(known_slots, indent=2, ensure_ascii=False)

    prompt = f"""You are an AI environmental scientist embedded in a biodiversity intelligence system.
You must reason like a soil/ecosystem scientist, not a generic chatbot.

STRICT RULES:
1. Base every factual/quantified claim ONLY on the KNOWLEDGE BASE CONTEXT below. Do not invent statistics or sources.
2. Connect AT LEAST 3 environmental variables in your reasoning (e.g. soil health <-> biodiversity <-> land use), never a single-variable answer.
3. Every recommendation must be non-obvious and specific — never say vague things like "use sustainable practices".
4. For EACH recommendation, use this exact label structure:
   - Recommendation
   - Why it works
   - Metrics impacted
   - Quantified expected improvement
   - Time horizon
   - Confidence level
   - Source
5. If a number is not explicitly present in the KB context, do NOT invent it. Instead write: "The KB does not provide a quantified uplift range for this practice in this context."
6. If the evidence is directional, say directional rather than claiming a precise percentage.
7. Be honest about confidence and uncertainty. Prefer mechanistic explanation when KB numbers are limited.

USER'S SITUATION (known variables so far):
{known_summary}

USER'S QUERY:
{user_query}

KNOWLEDGE BASE CONTEXT (retrieved, grounded evidence — use ONLY this for facts/numbers):
{context}

Now produce 2-3 evidence-backed, multi-metric recommendations in the structured format above.
"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Calls the configured LLM provider. Vertex remains available for later,
    while Mistral can be used when Vertex model access is unavailable.
    """
    provider = os.environ.get("LLM_PROVIDER", "vertex").strip().lower()
    if provider == "mistral":
        return call_mistral(prompt)
    if provider == "anthropic":
        return call_anthropic(prompt)
    if provider == "vertex":
        return call_vertex(prompt)
    raise RuntimeError(
        f"Unsupported LLM_PROVIDER={provider!r}. Use 'vertex', 'mistral', or 'anthropic'."
    )


def call_vertex(prompt: str) -> str:
    """Call Gemini through Vertex AI using Application Default Credentials."""
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    model = os.environ.get("LLM_MODEL", "gemini-2.0-flash-001")

    if not project:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT in .env.")
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("Set GOOGLE_APPLICATION_CREDENTIALS in .env.")

    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    if not response.text:
        raise RuntimeError("Vertex Gemini returned an empty response.")
    return response.text


def call_mistral(prompt: str) -> str:
    """Call Mistral's chat API without exposing the key to the frontend."""
    import requests

    api_key = os.environ.get("MISTRAL_API_KEY")
    model = os.environ.get("LLM_MODEL", "mistral-small-latest")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY in .env when LLM_PROVIDER=mistral.")

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-grounded environmental scientist. "
                        "Follow the user's prompt exactly and do not invent sources."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=90,
    )
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(f"Mistral API error ({response.status_code}): {detail}")

    payload = response.json()
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Mistral returned an unexpected response shape.") from exc
    if not text:
        raise RuntimeError("Mistral returned an empty response.")
    return text


def call_anthropic(prompt: str) -> str:
    """Call Anthropic's Messages API using a server-side API key."""
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("LLM_MODEL", "claude-3-5-haiku-latest")
    url = os.environ.get(
        "ANTHROPIC_API_URL",
        "https://api.anthropic.com/v1/messages",
    )
    if not url.rstrip("/").endswith("/messages"):
        url = f"{url.rstrip('/')}/v1/messages"
    if not api_key:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY in .env when LLM_PROVIDER=anthropic."
        )

    response = requests.post(
        url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1400,
            "temperature": 0.2,
            "system": (
                "You are an evidence-grounded environmental scientist. "
                "Follow the user's prompt exactly and do not invent sources."
            ),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(
            f"Anthropic API error ({response.status_code}): {detail}"
        )

    payload = response.json()
    try:
        text = "".join(
            block["text"]
            for block in payload["content"]
            if block.get("type") == "text"
        )
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Anthropic returned an unexpected response shape.") from exc
    if not text:
        raise RuntimeError("Anthropic returned an empty response.")
    return text


class BiodiversityReasoningEngine:
    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.sessions: dict[str, ConversationState] = {}

    def get_or_create_session(self, session_id: str) -> ConversationState:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState(session_id=session_id)
        return self.sessions[session_id]

    def handle_message(self, session_id: str, user_text: str, structured: dict = None) -> dict:
        state = self.get_or_create_session(session_id)
        state.history.append({"role": "user", "content": user_text})

        # 1. Update known slots from free text + structured input
        state.known_slots.update(detect_slots_in_text(user_text))
        merge_structured_input(state, structured)

        # 2. If not enough is known, ask a clarifying question (conversational intelligence requirement)
        if state.filled_slot_count() < MIN_SLOTS_BEFORE_ANSWERING:
            question = next_clarifying_question(state)
            state.history.append({"role": "assistant", "content": question})
            return {
                "type": "clarifying_question",
                "message": question,
                "known_slots": list(state.known_slots.keys()),
                "missing_slots": state.missing_slots(),
            }

        # 3. Retrieve grounded KB evidence — both semantic (query-based) and
        #    structured (metric-based) so we don't miss directly-relevant entries.
        semantic_hits = self.retriever.retrieve(user_text, k=4)
        metric_hits = self.retriever.retrieve_by_metrics(
            ["soil_organic_carbon", "species_richness", "habitat_diversity", "soil_moisture"]
        )
        # de-duplicate by id
        seen_ids = set()
        merged_hits = []
        for h in semantic_hits + [{"record": r} for r in metric_hits]:
            rid = h["record"]["id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                merged_hits.append(h)
        merged_hits = merged_hits[:5]

        # 4. Build grounded prompt + call LLM
        prompt = build_grounded_prompt(user_text, merged_hits, state.known_slots)
        answer = call_llm(prompt)

        state.history.append({"role": "assistant", "content": answer})

        return {
            "type": "recommendation",
            "message": answer,
            "used_kb_ids": [h["record"]["id"] for h in merged_hits],
            "known_slots": list(state.known_slots.keys()),
        }
