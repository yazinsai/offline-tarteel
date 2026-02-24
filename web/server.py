"""
WebSocket backend for real-time Quran recitation tracking.

Accepts binary 16kHz mono Float32 PCM audio from the browser,
transcribes it with the LoRA-finetuned Whisper model, and tracks
verse position.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
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
BASE_MODEL = "openai/whisper-small"
LORA_ADAPTER = str(PROJECT_ROOT / "data" / "lora-adapter-small")
CONFIDENCE_DISPLAY_THRESHOLD = 0.6
ADVANCE_COVERAGE_THRESHOLD = 0.85
ADVANCE_CONFIDENCE_THRESHOLD = 0.6
MIN_WORDS_FOR_MATCH = 2  # don't show matches until we have at least this many words
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
whisper_model = None
whisper_processor = None
device = None


def _load_model():
    """Load whisper-small + LoRA adapter for Quran transcription."""
    global whisper_model, whisper_processor, device
    if whisper_model is not None:
        return

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel

    # Use MPS on Apple Silicon, CUDA if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info("Loading whisper-small + LoRA adapter on %s...", device)
    whisper_processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="arabic", task="transcribe"
    )
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    whisper_model = PeftModel.from_pretrained(base, LORA_ADAPTER)
    whisper_model = whisper_model.to(device)
    whisper_model.eval()
    log.info("Model loaded: %s + %s", BASE_MODEL, LORA_ADAPTER)


def _transcribe(audio: np.ndarray) -> str:
    """Transcribe a Float32 audio array using the finetuned model."""
    _load_model()

    # Pad short chunks to at least 1 second
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))

    inputs = whisper_processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            input_features,
            max_new_tokens=225,
            repetition_penalty=1.2,
            language="ar",
            task="transcribe",
        )

    text = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()

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
@asynccontextmanager
async def lifespan(application: FastAPI):
    global quran_db
    quran_db = QuranDB()
    log.info("QuranDB loaded: %d verses", quran_db.total_verses)
    yield


app = FastAPI(title="Offline Tarteel", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected: %s", ws.client)

    tracker = VersePositionTracker(quran_db)
    completed: list[dict] = []
    audio_buffer = np.empty(0, dtype=np.float32)
    last_status: str | None = None
    last_sent_match: dict | None = None  # keep showing last good match
    words_since_reset = 0  # track words accumulated since last reset

    async def send_status(status: str):
        nonlocal last_status
        if status != last_status:
            await ws.send_json({"type": "status", "status": status})
            last_status = status

    async def send_verse_update(current_dict: dict | None):
        nonlocal last_sent_match
        if current_dict:
            last_sent_match = current_dict
        payload = {
            "type": "verse_update",
            "current": current_dict or last_sent_match,
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
                words_since_reset += len(words)
                log.info("Transcribed %d words: %s", len(words), text[:120])

                # Feed words into the verse tracker
                match = tracker.update(words)
                if match is None:
                    log.info("No match found")
                    continue

                log.info(
                    "Match: %s %d:%d conf=%.3f pos=%d/%d words_since_reset=%d accumulated='%s'",
                    match.surah_name_en, match.surah, match.ayah,
                    match.confidence, match.word_position, match.total_words,
                    words_since_reset,
                    tracker.accumulated_text[:100],
                )

                # Don't show matches until we have enough accumulated words
                # (prevents garbage matches on fragments after reset)
                if words_since_reset < MIN_WORDS_FOR_MATCH:
                    log.info("Skipping — not enough words yet (%d < %d)", words_since_reset, MIN_WORDS_FOR_MATCH)
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
                    await send_verse_update(_verse_match_to_dict(match))
                    # Reset tracker for the next verse
                    tracker.reset()
                    words_since_reset = 0
                    last_sent_match = None
                    continue

                await send_verse_update(_verse_match_to_dict(match))

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
