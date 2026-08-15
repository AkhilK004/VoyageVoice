"""
main.py — FastAPI application (Phase 4: Metrics + Logging + Eval Dashboard API).

WebSocket: ws://localhost:8000/ws/call

REST API (for dashboard):
  GET  /api/conversations          → list of all sessions (summary)
  GET  /api/conversations/{id}     → full session detail with per-turn data
  GET  /api/metrics                → global aggregate metrics + chart data
  GET  /api/analysis               → failure pattern analysis
"""
import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import settings
from backend.voice.stt import STTSession
from backend.voice.tts import synthesise_speech
from backend.voice.vad import VADState
from backend.voice.interruption import InterruptionHandler
from backend.intelligence.orchestrator import ConversationOrchestrator
from backend.intelligence.rag import build_rag_context
from backend.evaluation.metrics import ConversationMetrics, TurnMetrics
from backend.evaluation.logger import (
    new_session_id, save_session, load_all_sessions,
    load_session, compute_global_aggregate
)
from backend.evaluation.judge import score_turn_background
from backend.evaluation.analyzer import analyze_sessions

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="VoyageVoice", version="4.0.0")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Page Routes ───────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(str(FRONTEND_DIR / "dashboard.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "4.0.0"}


# ── REST API — Evaluation Dashboard ──────────────────────────────────────────
@app.get("/api/conversations")
async def get_conversations():
    """List all conversation sessions (summary only)."""
    sessions = load_all_sessions()
    summaries = []
    for s in sessions:
        agg = s.get("aggregate", {})
        summaries.append({
            "session_id": s.get("session_id"),
            "start_time": s.get("start_time"),
            "duration_seconds": agg.get("duration_seconds"),
            "turn_count": agg.get("turn_count", len(s.get("turns", []))),
            "e2e_avg_ms": agg.get("latency", {}).get("e2e_avg_ms"),
            "avg_relevance": agg.get("quality", {}).get("avg_relevance"),
            "total_interruptions": agg.get("total_interruptions", 0),
            "escalated": agg.get("escalated", False),
        })
    return JSONResponse({"sessions": summaries, "total": len(summaries)})


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Get full detail for a single session."""
    data = load_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return JSONResponse(data)


@app.get("/api/metrics")
async def get_metrics():
    """Global aggregate metrics across all sessions."""
    sessions = load_all_sessions()
    return JSONResponse(compute_global_aggregate(sessions))


@app.get("/api/analysis")
async def get_analysis():
    """Failure pattern analysis across all sessions."""
    sessions = load_all_sessions()
    return JSONResponse(analyze_sessions(sessions))


# ── WebSocket Call Handler ────────────────────────────────────────────────────
@app.websocket("/ws/call")
async def websocket_call(websocket: WebSocket):
    await websocket.accept()

    # Initialise all Phase 3 + Phase 4 components
    session_id = new_session_id()
    logger.info(f"New call: {session_id}")

    orchestrator = ConversationOrchestrator()
    vad = VADState()
    conv_metrics = ConversationMetrics(session_id=session_id)
    current_turn: TurnMetrics = None
    tts_task: asyncio.Task = None
    user_speech_end_time: float = None

    async def send_json(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def send_status(value: str):
        await send_json({"type": "status", "value": value})

    async def stop_audio():
        await send_json({"type": "stop_audio"})

    interruption = InterruptionHandler(send_stop_fn=stop_audio)

    # ── TTS Pipeline with metric instrumentation ───────────────────────────────
    _queued_sentences: list[str] = []

    async def run_tts_pipeline(response_text_hint: str = ""):
        nonlocal current_turn
        interruption.set_agent_speaking(True, response_text_hint)
        await send_status("speaking")

        tts_started = False
        full_agent_text = []

        try:
            for sentence in _queued_sentences:
                if not interruption.is_agent_speaking:
                    break

                await send_json({"type": "agent_text", "text": sentence})
                interruption.on_sentence_spoken(sentence)
                full_agent_text.append(sentence)

                sent_start = time.monotonic()
                first_byte = True
                async for audio_chunk in synthesise_speech(sentence):
                    if not interruption.is_agent_speaking:
                        return
                    # Track TTS first byte latency (first sentence only)
                    if first_byte and not tts_started:
                        tts_lat = (time.monotonic() - sent_start) * 1000
                        if current_turn:
                            current_turn.tts_first_byte_ms = tts_lat
                        tts_started = True
                        first_byte = False
                        # E2E latency: from when user stopped speaking to agent audio starts
                        if user_speech_end_time and current_turn:
                            current_turn.e2e_latency_ms = (time.monotonic() - user_speech_end_time) * 1000
                    try:
                        await websocket.send_bytes(audio_chunk)
                    except Exception:
                        return
        finally:
            interruption.set_agent_speaking(False)
            await send_status("listening")

            # Update turn with full agent text
            if current_turn and full_agent_text:
                current_turn.agent_text = " ".join(full_agent_text)

    async def handle_final_transcript(text: str, latency_ms: float):
        nonlocal tts_task, _queued_sentences, current_turn, user_speech_end_time

        # Cancel any running TTS
        if tts_task and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass

        # Start a new turn in metrics
        current_turn = conv_metrics.start_turn(conv_metrics.turns.__len__() + 1 if conv_metrics.turns else 1)
        current_turn.turn_id = len(conv_metrics.turns)
        current_turn.user_text = text
        current_turn.stt_latency_ms = latency_ms
        current_turn.conversation_state = orchestrator.context.state.value
        current_turn.was_interrupted = interruption.get_stats()["total_interruptions"] > 0

        # RAG context for this turn (used for judge scoring)
        rag_ctx = build_rag_context(text)
        current_turn.rag_context_chars = len(rag_ctx)

        # Recovery context if barge-in
        recovery_context = interruption.get_recovery_context()
        interruption.complete_last_interruption(text)
        if recovery_context:
            orchestrator.inject_interruption_context(recovery_context)

        await send_json({
            "type": "transcript",
            "text": text,
            "is_final": True,
            "latency_ms": round(latency_ms or 0, 1),
        })
        await send_status("thinking")

        # LLM timing
        llm_start = time.monotonic()
        llm_first_token_logged = False
        _queued_sentences = []

        async for sentence in orchestrator.respond(text):
            if not llm_first_token_logged:
                current_turn.llm_ttft_ms = (time.monotonic() - llm_start) * 1000
                llm_first_token_logged = True
            _queued_sentences.append(sentence)

        current_turn.llm_total_ms = (time.monotonic() - llm_start) * 1000
        current_turn.tools_called = orchestrator.context.tools_called.copy()

        if not _queued_sentences:
            return

        # Start TTS as cancellable task
        tts_task = asyncio.create_task(
            run_tts_pipeline(" ".join(_queued_sentences))
        )

        # Judge scoring (fire-and-forget, does NOT block the call)
        asyncio.create_task(
            score_turn_background(current_turn, text, " ".join(_queued_sentences), rag_ctx)
        )

    async def on_transcript(text: str, is_final: bool, latency_ms: float):
        if is_final:
            await handle_final_transcript(text, latency_ms)
        else:
            await send_json({"type": "transcript", "text": text, "is_final": False, "latency_ms": None})

    # ── STT startup ───────────────────────────────────────────────────────────
    stt_session = STTSession(on_transcript=on_transcript)
    try:
        await stt_session.start()
    except Exception as e:
        await send_json({"type": "error", "message": f"STT failed: {e}"})
        await websocket.close()
        return

    await send_status("listening")
    await send_json({"type": "session_id", "value": session_id})

    # Greeting
    greeting = "Hi, I'm Aria, your Atlys travel assistant. How can I help you today?"
    _queued_sentences = [greeting]
    tts_task = asyncio.create_task(run_tts_pipeline(greeting))

    # ── Main Loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                await stt_session.send_audio(data["bytes"])

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    msg_type = msg.get("type")

                    if msg_type == "speech_start":
                        vad.on_speech_start()
                        was_barge_in = await interruption.check_barge_in()
                        if was_barge_in:
                            conv_metrics.total_interruptions += 1
                            if current_turn:
                                current_turn.was_interrupted = True
                            await send_json({"type": "interruption_ack"})
                            if tts_task and not tts_task.done():
                                tts_task.cancel()
                                try:
                                    await tts_task
                                except asyncio.CancelledError:
                                    pass

                    elif msg_type == "speech_end":
                        vad.on_speech_end()
                        user_speech_end_time = time.monotonic()

                    elif msg_type == "end_call":
                        break

                    elif msg_type == "ping":
                        await send_json({"type": "pong"})

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await stt_session.finish()
        if tts_task and not tts_task.done():
            tts_task.cancel()

        conv_metrics.finish()
        conv_metrics.escalated = orchestrator.context.escalated
        # Wait a moment for any in-flight judge tasks to complete
        await asyncio.sleep(0.5)
        save_session(conv_metrics)

        logger.info(
            f"Session {session_id} complete — "
            f"{len(conv_metrics.turns)} turns, "
            f"{conv_metrics.total_interruptions} interruptions, "
            f"{conv_metrics.duration_seconds():.1f}s"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
