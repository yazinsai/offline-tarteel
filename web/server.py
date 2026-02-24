"""
WebSocket backend for real-time Quran recitation tracking.

Accepts binary 16kHz mono Float32 PCM audio from the browser,
transcribes it with mlx-whisper, and tracks verse position.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "streaming-asr"))

from offline_tarteel.quran_db import QuranDB
from verse_position_tracker import VersePositionTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)  # 48000
MLX_MODEL = "mlx-community/whisper-base-mlx"
CONFIDENCE_DISPLAY_THRESHOLD = 0.4
ADVANCE_COVERAGE_THRESHOLD = 0.90
ADVANCE_CONFIDENCE_THRESHOLD = 0.6
SILENCE_RMS_THRESHOLD = 0.005
PORT = 8765

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tarteel-ws")

# ---------------------------------------------------------------------------
# Globals (loaded once at startup)
# ---------------------------------------------------------------------------
quran_db: QuranDB | None = None
mlx_whisper_module = None  # lazy-loaded on first transcription


def _ensure_model():
    """Lazy-load mlx_whisper on first use (triggers model download if needed)."""
    global mlx_whisper_module
    if mlx_whisper_module is None:
        import mlx_whisper
        mlx_whisper_module = mlx_whisper
        log.info("mlx-whisper model loaded: %s", MLX_MODEL)


def _transcribe(audio: np.ndarray) -> str:
    """Transcribe a Float32 audio array and return the text."""
    _ensure_model()
    result = mlx_whisper_module.transcribe(
        audio,
        path_or_hf_repo=MLX_MODEL,
        language="ar",
        word_timestamps=False,
        condition_on_previous_text=False,
        hallucination_silence_threshold=1.0,
        no_speech_threshold=0.5,
    )
    text = result.get("text", "").strip()
    # Filter hallucinations: excessive repetition or very long output
    words = text.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return ""
    if len(words) > 30:
        return ""
    return text


def _is_silence(audio: np.ndarray) -> bool:
    """Return True if the audio chunk is mostly silence."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < SILENCE_RMS_THRESHOLD


def _verse_match_to_dict(match) -> dict:
    """Convert a VerseMatch dataclass to the JSON-friendly dict for the client."""
    return {
        "surah": match.surah,
        "ayah": match.ayah,
        "surah_name_en": match.surah_name_en,
        "text_uthmani": match.verse_text_uthmani,
        "words": match.verse_text_uthmani.split(),
        "matched_indices": match.matched_word_indices,
        "word_position": match.word_position,
        "total_words": match.total_words,
        "confidence": round(match.confidence, 4),
        "progress_pct": round(match.progress_pct, 2),
    }


def _completed_entry(match) -> dict:
    """Build a compact completed-verse dict."""
    return {
        "surah": match.surah,
        "ayah": match.ayah,
        "text_uthmani": match.verse_text_uthmani,
        "surah_name_en": match.surah_name_en,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Offline Tarteel", docs_url=None, redoc_url=None)


@app.on_event("startup")
async def _startup():
    global quran_db
    quran_db = QuranDB()
    log.info("QuranDB loaded: %d verses", quran_db.total_verses)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected: %s", ws.client)

    tracker = VersePositionTracker(quran_db)
    completed: list[dict] = []
    audio_buffer = np.empty(0, dtype=np.float32)
    last_status: str | None = None

    async def send_status(status: str):
        nonlocal last_status
        if status != last_status:
            await ws.send_json({"type": "status", "status": status})
            last_status = status

    async def send_verse_update(match):
        payload = {
            "type": "verse_update",
            "current": _verse_match_to_dict(match),
            "completed": list(completed),
        }
        await ws.send_json(payload)

    try:
        while True:
            data = await ws.receive_bytes()

            # Decode incoming Float32 PCM
            samples = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate([audio_buffer, samples])

            # Process whenever we have enough samples
            while len(audio_buffer) >= CHUNK_SAMPLES:
                chunk = audio_buffer[:CHUNK_SAMPLES]
                audio_buffer = audio_buffer[CHUNK_SAMPLES:]

                # Silence detection
                if _is_silence(chunk):
                    await send_status("silence")
                    continue

                await send_status("listening")

                # Transcribe (CPU-bound, run in thread pool)
                text = await asyncio.get_event_loop().run_in_executor(
                    None, _transcribe, chunk
                )

                if not text:
                    continue

                words = text.split()
                log.info("Transcribed %d words: %s", len(words), text[:80])

                # Feed words into the verse tracker
                match = tracker.update(words)
                if match is None:
                    continue

                # Below display threshold — skip sending
                if match.confidence < CONFIDENCE_DISPLAY_THRESHOLD:
                    continue

                # Check auto-advance
                coverage = match.word_position / match.total_words if match.total_words else 0
                if (
                    coverage >= ADVANCE_COVERAGE_THRESHOLD
                    and match.confidence >= ADVANCE_CONFIDENCE_THRESHOLD
                ):
                    log.info(
                        "Auto-advance: %s %d:%d (coverage=%.0f%%, conf=%.2f)",
                        match.surah_name_en, match.surah, match.ayah,
                        coverage * 100, match.confidence,
                    )
                    completed.append(_completed_entry(match))
                    # Send final update for this verse before resetting
                    await send_verse_update(match)
                    # Reset tracker for the next verse
                    tracker.reset()
                    continue

                await send_verse_update(match)

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", ws.client)
    except Exception:
        log.exception("WebSocket error")


# ---------------------------------------------------------------------------
# Serve React frontend (if built)
# ---------------------------------------------------------------------------
_frontend_dist = PROJECT_ROOT / "web" / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
    log.info("Serving frontend from %s", _frontend_dist)
else:
    @app.get("/")
    async def _root():
        return {"status": "ok", "message": "Offline Tarteel backend. Frontend not built yet."}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
