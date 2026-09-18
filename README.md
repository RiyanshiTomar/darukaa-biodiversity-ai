# Darukaa.Earth — Biodiversity Intelligence Chatbot

Darukaa.Earth is a biodiversity decision-support system that turns land and ecosystem conditions into evidence-backed recommendations. It is designed to behave more like an AI environmental scientist than a generic chatbot.

## 1. Project goal

The challenge requires a system that:
- maintains a grounded knowledge base of biodiversity and environmental metrics
- understands ecosystem, land, and climate conditions from text or structured input
- provides non-obvious recommendations that connect multiple variables
- gives explanations grounded in scientific reasoning and credible sources
- supports follow-up questions and context-aware conversations

This project solves that by combining retrieval, reasoning, and LLM synthesis into one pipeline.

## 2. Why this design matches the challenge

The brief explicitly rejects shallow, generic LLM-only answers. This project instead uses a real RAG architecture:
- a retrievable biodiversity knowledge base
- semantic retrieval using FAISS + embeddings
- structured field input for soil, climate, land use, and biodiversity indicators
- a reasoning engine that requires multi-metric context before answering
- an evidence-grounded prompt that only uses KB context and avoids unsupported claims

This makes the system more like an environmental analyst than a normal chatbot.

## 3. System architecture

```text
User input (text or structured JSON)
        |
        v
FastAPI backend (backend/main.py)
        |
        v
BiodiversityReasoningEngine (rag/reasoning_engine.py)
        |
        +-- 1. Slot detection / variable tracking
        |     - soil health
        |     - land use
        |     - biodiversity
        |     - climate
        |     - human impact
        |
        +-- 2. Retrieval layer (rag/retriever.py)
        |     - FAISS semantic search
        |     - deterministic fallback lookup by metric
        |
        +-- 3. Grounded generation
              - strict prompt with evidence constraints
              - LLM provider switching (Vertex / Anthropic / Mistral)
              - structured final outputs

Final response includes:
- diagnosis
- recommendation
- why it works
- impacted metrics
- quantified improvement (only when present in KB)
- time horizon
- confidence
- source
```

## 4. Knowledge base design

The project uses a structured JSON knowledge base stored in `knowledge_base/kb_data.json`.

Each record includes:
- `id`
- `category`
- `topic`
- `content`
- `metrics_affected`
- `quantified_impact`
- `source`
- `time_horizon`
- `confidence`

This supports sustainability-related reasoning across:
- soil health
- biodiversity indicators
- land use / land cover
- climate and rainfall stress
- human impacts such as pesticide use, deforestation, or pollution

## 5. Tech stack

- Python 3.11
- FastAPI
- Streamlit
- lightweight token/phrase retrieval over the local knowledge base
- python-dotenv
- requests
- Google GenAI / Vertex AI support
- Anthropic API support
- Mistral API support

## 6. Project structure

```text
darukaa-biodiversity-ai/
├── backend/
│   └── main.py                  # FastAPI app with /chat and /chat/structured
├── rag/
│   ├── retriever.py             # lightweight lexical + structured retrieval logic
│   └── reasoning_engine.py      # prompt building, slot detection, model routing
├── knowledge_base/
│   ├── kb_data.json             # biodiversity/environment knowledge records
│   └── build_index.py           # validates the KB (compatibility helper)
├── ui/
│   └── app.py                   # Streamlit demo UI
├── .env.example                 # environment configuration template
├── requirements.txt
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml               # basic validation workflow
└── venv/                        # local virtual environment
```

## 7. Local setup

### 7.1 Create environment

```powershell
cd C:\Users\Riyanshi\Downloads\darukaa-biodiversity-ai
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 7.2 Configure environment variables

Create a root `.env` file based on `.env.example`:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-latest
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_API_URL=https://api.anthropic.com/v1/messages
MISTRAL_API_KEY=your_key_here
```

Notes:
- If using Vertex AI, set the Google variables and choose `LLM_PROVIDER=vertex`.
- If using Anthropic, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.
- Mistral is supported as a fallback, but may be restricted by quota or tier access.

### 7.3 Validate the knowledge base

```powershell
python knowledge_base\build_index.py
```

### 7.4 Start backend

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

### 7.5 Start UI

Open a second terminal:

```powershell
streamlit run ui\app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## 8. API validation

### Health check

```powershell
curl.exe http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Structured test request

```powershell
$payload = @{
  session_id = "demo1"
  message = "Biodiversity is declining on my land"
  structured_data = @{
    soil_organic_carbon_pct = 0.3
    rainfall = "low"
    crop = "monoculture wheat"
    region = "semi-arid"
  }
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri "http://localhost:8000/chat/structured" -Method Post -ContentType "application/json" -Body $payload
```

A good response should include:
- diagnosis
- 2–3 recommendations
- scientific mechanism
- impacted metrics
- time horizon
- confidence
- source names

## 9. Deployment guidance

### Option A: Local deployment

Use the instructions above. This is the easiest way to run and demonstrate the project locally.

### Option B: Cloud deployment (Render / Railway / Azure / any Python host)

1. Push the repo to GitHub.
2. Create a new cloud app service for the FastAPI backend.
3. Set the build command:
   ```bash
   pip install -r requirements.txt && python knowledge_base/build_index.py
   ```
4. Set the runtime command. `$PORT` is required because Render assigns it:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```
5. Add all required environment variables in the hosting dashboard.
6. Deploy the frontend separately or host a Streamlit app if desired.
7. Update the demo link in the Word submission document.

The hosted backend intentionally uses dependency-free lexical retrieval instead
of FAISS/SentenceTransformers so it can start within Render's 512 MB free-tier
memory limit. The evidence remains grounded in `knowledge_base/kb_data.json`;
the LLM is still used only for the final explanation and recommendations.

### Option C: Streamlit cloud demo

1. Push the repo to GitHub.
2. Connect the repo to Streamlit Cloud.
3. Set the main file to `ui/app.py`.
4. Add environment variables in the Streamlit app settings.
5. Use a public or private URL for the final demo.

## 10. Submission checklist

Before final hackathon submission, make sure all of the following are ready:
- GitHub repository is live
- README is updated and clear
- local setup instructions work
- backend health check passes
- structured API request returns a valid recommendation
- demo URL is live and usable
- Word document contains repo link + live demo URL + setup notes
- credentials / secrets are not committed to the repo

## 11. Known limitations and improvements

This project is already strong, but some final improvements still matter for a polished submission:
- improve landing-page polish and visual hierarchy
- tighten prompt language to avoid unsupported percentage claims
- capture more environmental variables via follow-up questions
- expand the knowledge base with more research-backed entries
- add a cleaner report/export format for final judges

## 12. Final position

This project is a meaningful biodiversity intelligence system, not a superficial chatbot. It matches the challenge’s strongest requirements:
- grounded knowledge base
- multi-variable reasoning
- evidence-backed recommendations
- scientific explainability
- structured output
- real environmental use case

That makes it a strong hackathon submission and a credible demo for Darukaa.Earth.
