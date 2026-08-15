# VoyageVoice 🎙️✈️

**A production-grade real-time AI voice agent for travel visa support.**

Built as a demonstration of end-to-end voice AI engineering for the Atlys Voice AI Intern role. VoyageVoice implements the complete pipeline from raw audio to intelligent response — including interruption handling, RAG-grounded answers, tool calling, and a metrics-driven evaluation dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Client)                         │
│  ┌──────────┐  ┌────────────────┐  ┌──────────────────────┐   │
│  │ Mic (raw │  │ Web Audio API  │  │ AudioContext Queue    │   │
│  │ audio)   │  │ VAD (RMS loop) │  │ (MP3 playback)       │   │
│  └────┬─────┘  └───────┬────────┘  └──────────┬───────────┘   │
│       │ binary          │ speech_start/end       │ audio bytes   │
└───────┼─────────────────┼───────────────────────┼───────────────┘
        │ WebSocket       │ JSON control           │
┌───────▼─────────────────▼───────────────────────▼───────────────┐
│                    FastAPI WebSocket Server                       │
│  ┌──────────┐  ┌───────────┐  ┌────────────────────────────┐   │
│  │ Deepgram │  │ VADState  │  │ InterruptionHandler         │   │
│  │ STT      │  │ (P3)      │  │ cancel TTS task on barge-in │   │
│  └────┬─────┘  └───────────┘  └────────────────────────────┘   │
│       │ transcript                                                │
│  ┌────▼──────────────────────────────────────────────────────┐  │
│  │              ConversationOrchestrator                      │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ State   │  │ RAG      │  │ Groq LLM │  │ Tool     │  │  │
│  │  │ Machine │  │ ChromaDB │  │ Llama 3  │  │ Calling  │  │  │
│  │  └─────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                            │ sentences                            │
│  ┌─────────────────────────▼──────────────────────────────────┐ │
│  │ Edge TTS (free Microsoft TTS) → MP3 bytes                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Evaluation Pipeline (async, never blocks E2E)              │  │
│  │  TurnMetrics → LLM Judge (Groq) → JSON logs → Dashboard   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Features

### Phase 1 — Core Voice Pipeline
- 🎙️ **Deepgram Streaming STT** — Real-time transcription with interim results
- 🤖 **Groq LLM (Llama 3.1)** — Ultra-fast inference via OpenAI-compatible API
- 🔊 **Edge TTS** — Free Microsoft neural TTS, no API key needed
- ⚡ **Sentence-chunked streaming** — TTS starts speaking before LLM finishes
- 📡 **Full-duplex WebSocket** — Binary audio + JSON control messages

### Phase 2 — Intelligence Layer
- 📚 **RAG with ChromaDB** — Local vector search over 10+ country visa data
- 🔧 **Groq Tool Calling** — 4 tools: eligibility, documents, processing time, escalation
- 🗺️ **Conversation State Machine** — 6-state flow with entity extraction
- 🧠 **Dynamic system prompt** — State instructions + RAG context injected per turn

### Phase 3 — Interruption Handling
- ✂️ **Client-side VAD** — Web Audio AnalyserNode RMS energy detection
- 🛑 **Barge-in detection** — User speaking mid-response → TTS cancelled immediately
- 🔄 **Graceful recovery** — LLM told what it was saying, doesn't repeat itself
- 💚 **Visual feedback** — Green mic glow (speaking) + red ring flash (barge-in)

### Phase 4 — Evaluation Dashboard
- 📊 **Latency metrics** — STT, LLM TTFT, TTS, and E2E per turn + P95
- 🎯 **LLM-as-judge** — Groq scores every turn: relevance (1–5), naturalness (1–5), hallucination (bool)
- 🔍 **Failure analysis** — Detects high latency, hallucinations, RAG misses, interruption spikes
- 📋 **Session drill-down** — Click any session to see every turn's full metrics

### Phase 5 — Polish & Demo
- 🎬 **Demo scenarios** — 5 predefined scripts to showcase the agent without a microphone
- ✍️ **Prompt v2** — Anti-hallucination guardrails, brevity enforcement, tool-calling preference
- 🧪 **35+ integration tests** — Full test suite across all 5 phases with regression coverage

---

## Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| STT | Deepgram Streaming | Free ($200 credit) |
| LLM | Groq (Llama 3.1 8B Instant) | Free tier |
| TTS | Microsoft Edge TTS | **100% Free** |
| Vector DB | ChromaDB (local) | **Free** |
| Embeddings | all-MiniLM-L6-v2 (local) | **Free** |
| Backend | FastAPI + Python 3.13 | **Free** |
| Frontend | Vanilla HTML/CSS/JS | **Free** |

**Total API cost for a full 2-week development cycle: $0**

---

## Setup

### Prerequisites
- Python 3.11+
- A Deepgram API key ([console.deepgram.com](https://console.deepgram.com) — free $200 credit)
- A Groq API key ([console.groq.com/keys](https://console.groq.com/keys) — free)

### Installation

```bash
git clone https://github.com/AkhilK004/VoyageVoice.git
cd VoyageVoice

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY and DEEPGRAM_API_KEY
```

### Run

```bash
# Start the server
python -m uvicorn backend.main:app --reload --port 8000

# Open the call UI
# → http://localhost:8000

# Open the evaluation dashboard
# → http://localhost:8000/dashboard
```

---

## Running Tests

```bash
# Full test suite (all 5 phases, ~35 tests)
python -m pytest tests/ -v -s

# Individual phases
python -m pytest tests/test_phase1.py -v -s   # Core pipeline
python -m pytest tests/test_phase2.py -v -s   # RAG + tools
python -m pytest tests/test_phase3.py -v -s   # Interruption
python -m pytest tests/test_phase4.py -v -s   # Evaluation
python -m pytest tests/test_phase5.py -v -s   # Final integration
```

> **Note:** First run will download `all-MiniLM-L6-v2` (~80MB). Every subsequent run is instant.

---

## Demo Scenarios (No Microphone Needed)

Run predefined conversations to generate evaluation data and showcase the agent:

```bash
# Japan visa enquiry (recommended first demo)
python demo_scenarios.py --scenario japan_visa

# All 5 scenarios
python demo_scenarios.py --all
```

**Available scenarios:**
| Key | Name | What it tests |
|-----|------|---------------|
| `japan_visa` | India → Japan | Full RAG + tool calling |
| `multi_destination` | Europe + USA | Multi-turn state memory |
| `escalation_test` | Complex rejection | Escalation trigger |
| `voa_query` | Thailand VoA | Visa-on-arrival query |
| `unknown_country` | Bhutan | Honest "I don't know" |

---

## REST API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Call UI |
| `GET /dashboard` | Evaluation dashboard |
| `GET /health` | Health check |
| `GET /api/conversations` | List all sessions |
| `GET /api/conversations/{id}` | Session detail (all turns) |
| `GET /api/metrics` | Global aggregate metrics |
| `GET /api/analysis` | Failure pattern analysis |
| `WS /ws/call` | WebSocket voice call |

---

## Project Structure

```
VoyageVoice/
├── backend/
│   ├── config.py                    # Settings loader
│   ├── main.py                      # FastAPI app + WebSocket + REST API
│   ├── voice/
│   │   ├── stt.py                   # Deepgram streaming STT
│   │   ├── tts.py                   # Edge TTS streaming
│   │   ├── vad.py                   # VAD state tracker
│   │   └── interruption.py          # Barge-in detection + recovery
│   ├── intelligence/
│   │   ├── orchestrator.py          # Full pipeline: RAG + tools + state + LLM
│   │   ├── prompts.py               # System prompt v2
│   │   ├── rag.py                   # RAG retrieval wrapper
│   │   ├── tools.py                 # Tool definitions + Groq schemas
│   │   └── state.py                 # Conversation state machine
│   ├── knowledge/
│   │   ├── loader.py                # ChromaDB ingestion
│   │   └── data/visa_data.json      # 10+ countries, 15+ visa types
│   └── evaluation/
│       ├── metrics.py               # TurnMetrics + ConversationMetrics
│       ├── logger.py                # Session persistence + aggregation
│       ├── judge.py                 # LLM-as-judge (Groq)
│       └── analyzer.py             # Failure pattern detection
├── frontend/
│   ├── index.html                   # Call UI
│   ├── dashboard.html               # Evaluation dashboard
│   ├── css/
│   │   ├── main.css                 # Dark glassmorphic call UI
│   │   └── dashboard.css            # Dashboard layout + charts
│   └── js/
│       ├── audio.js                 # Mic capture + VAD + WebSocket + playback
│       ├── ui.js                    # UI state management
│       └── dashboard.js             # Chart.js rendering + REST API calls
├── tests/
│   ├── test_phase1.py               # Core pipeline tests
│   ├── test_phase2.py               # RAG + tool tests
│   ├── test_phase3.py               # Interruption + VAD tests
│   ├── test_phase4.py               # Metrics + judge + API tests
│   └── test_phase5.py               # Final integration tests
├── demo_scenarios.py                # Demo runner (no mic needed)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Design Decisions

**Why FastAPI over MERN?** Python has significantly more mature ML/voice SDKs. The asyncio model maps perfectly to concurrent voice streams (STT + TTS + WebSocket).

**Why Groq over OpenAI?** Groq's LPU inference is 10–25x faster than OpenAI for the same models (often <300ms TTFT), which is critical for real-time voice. And it's free.

**Why Edge TTS over ElevenLabs?** For a portfolio project, $0 cost matters. Edge TTS neural voices (Aria, Guy, Sonia) are high quality and have <400ms latency.

**Why sentence-chunking?** Streaming TTS sentence-by-sentence reduces perceived E2E latency from ~2s to ~800ms — the user hears the first word while the LLM is still generating the rest.

**Why async judge scoring?** The LLM judge adds ~1–2s of latency. Running it as a background `asyncio.Task` means it never touches the real-time E2E path.

---

## License

MIT
