"""
WebSocket backend for real-time Quran recitation tracking.

Uses a growing-window approach: keeps all audio and re-transcribes
a larger window each time, giving Whisper more context for better
Arabic accuracy. Matches verses sequentially within a locked surah.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from Levenshtein import ratio as lev_ratio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
TRIGGER_SECONDS = 3.0           # process every N seconds of new audio
TRIGGER_SAMPLES = int(SAMPLE_RATE * TRIGGER_SECONDS)
MAX_WINDOW_SECONDS = 30.0       # max audio to keep in window
MAX_WINDOW_SAMPLES = int(SAMPLE_RATE * MAX_WINDOW_SECONDS)
BASE_MODEL = "openai/whisper-small"
LORA_ADAPTER = str(PROJECT_ROOT / "data" / "lora-adapter-small")
SILENCE_RMS_THRESHOLD = 0.005
PORT = 8765

# Matching thresholds
LOCK_CONFIDENCE = 0.55
DISPLAY_CONFIDENCE = 0.45
ADVANCE_COVERAGE = 0.65
ADVANCE_CONFIDENCE = 0.50
MIN_WORDS_FOR_LOCK = 5          # need 5+ input words before locking to a surah
LOCK_STREAK_REQUIRED = 3        # same surah must win N consecutive global searches
MAX_SKIP_AYAHS = 3              # max ayahs to skip-complete on a jump

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tarteel-ws")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
quran_db: QuranDB | None = None
whisper_model = None
whisper_processor = None
device = None


def _load_model():
    global whisper_model, whisper_processor, device
    if whisper_model is not None:
        return

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel

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
    """Transcribe audio using the finetuned model."""
    _load_model()

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

    # Filter hallucinations
    words = text.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return ""
    if len(words) > 50:
        return ""
    return text


def _is_silence(audio: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < SILENCE_RMS_THRESHOLD


# ---------------------------------------------------------------------------
# Sequential Recitation Tracker
# ---------------------------------------------------------------------------
class SequentialTracker:
    """
    Track Quran recitation sequentially within a surah.

    Phase 1 (unlocked): Global search every cycle. The same surah must win
                         LOCK_STREAK_REQUIRED consecutive searches before locking.
    Phase 2 (locked):   Only match against current + next few ayahs.
                         Never switches surah. Stays locked until silence reset.
    """

    def __init__(self, db: QuranDB):
        self.db = db
        self.locked_surah: int | None = None
        self.current_ayah: int = 1
        self.current_text: str = ""
        # Track consecutive wins for the same surah before locking
        self._candidate_surah: int | None = None
        self._candidate_streak: int = 0

    def reset_verse(self):
        self.current_text = ""

    def reset_all(self):
        self.locked_surah = None
        self.current_ayah = 1
        self.current_text = ""
        self._candidate_surah = None
        self._candidate_streak = 0

    def update_from_text(self, full_text: str) -> dict | None:
        """Replace accumulated text with full transcription and match."""
        self.current_text = normalize_arabic(full_text)
        words = self.current_text.split()

        if len(words) < 2:
            return None

        if self.locked_surah:
            return self._match_in_surah(self.current_text)
        else:
            return self._match_global(self.current_text)

    def _score_verse(self, text: str, verse: dict) -> float:
        """Score a verse against input text using prefix/full blend."""
        input_words = text.split()
        n = len(input_words)
        verse_words = verse["text_clean"].split()

        prefix_len = min(n, len(verse_words))
        verse_prefix = " ".join(verse_words[:prefix_len])
        prefix_score = lev_ratio(text, verse_prefix)
        full_score = lev_ratio(text, verse["text_clean"])

        coverage = n / max(len(verse_words), 1)
        if coverage > 0.7:
            return 0.3 * prefix_score + 0.7 * full_score
        else:
            return 0.7 * prefix_score + 0.3 * full_score

    def _match_global(self, text: str) -> dict | None:
        best = None
        best_score = 0.0

        input_words = text.split()
        n = len(input_words)

        for v in self.db.verses:
            score = self._score_verse(text, v)
            if score > best_score:
                best_score = score
                best = {**v, "score": score}

        if not best or best_score < DISPLAY_CONFIDENCE:
            return None

        # Track consecutive wins for locking
        if best["surah"] == self._candidate_surah:
            self._candidate_streak += 1
        else:
            self._candidate_surah = best["surah"]
            self._candidate_streak = 1
            log.info(
                "New candidate surah %d (%s) ayah %d (conf=%.3f, words=%d)",
                best["surah"], best["surah_name_en"], best["ayah"],
                best_score, n,
            )

        # Lock only after enough consecutive wins AND enough words
        can_lock = (
            n >= MIN_WORDS_FOR_LOCK
            and best_score >= LOCK_CONFIDENCE
            and self._candidate_streak >= LOCK_STREAK_REQUIRED
        )
        if can_lock:
            self.locked_surah = best["surah"]
            self.current_ayah = best["ayah"]
            log.info(
                "LOCKED to surah %d (%s) ayah %d "
                "(conf=%.3f, words=%d, streak=%d)",
                self.locked_surah, best["surah_name_en"],
                self.current_ayah, best_score, n,
                self._candidate_streak,
            )

        return self._build_result(best, text)

    def _match_in_surah(self, text: str) -> dict | None:
        surah_verses = self.db.get_surah(self.locked_surah)
        if not surah_verses:
            return None

        input_words = text.split()
        n = len(input_words)

        start_idx = max(0, self.current_ayah - 1)
        end_idx = min(start_idx + 4, len(surah_verses))
        candidates = surah_verses[start_idx:end_idx]

        best = None
        best_score = 0.0

        for v in candidates:
            verse_words = v["text_clean"].split()
            prefix_len = min(n, len(verse_words))
            verse_prefix = " ".join(verse_words[:prefix_len])
            prefix_score = lev_ratio(text, verse_prefix)
            full_score = lev_ratio(text, v["text_clean"])

            coverage = n / max(len(verse_words), 1)
            if coverage > 0.7:
                score = 0.3 * prefix_score + 0.7 * full_score
            else:
                score = 0.8 * prefix_score + 0.2 * full_score

            if score > best_score:
                best_score = score
                best = {**v, "score": score}

        if best and best_score >= DISPLAY_CONFIDENCE:
            self.current_ayah = best["ayah"]
            return self._build_result(best, text)

        return None

    def _build_result(self, verse: dict, text: str) -> dict:
        verse_words_clean = verse["text_clean"].split()
        verse_words_uthmani = verse["text_uthmani"].split()
        input_words = text.split()

        position = self._find_position(input_words, verse_words_clean)
        total = len(verse_words_uthmani)
        coverage = position / total if total > 0 else 0

        return {
            "surah": verse["surah"],
            "ayah": verse["ayah"],
            "surah_name_en": verse["surah_name_en"],
            "text_uthmani": verse["text_uthmani"],
            "words": verse_words_uthmani,
            "word_position": position,
            "total_words": total,
            "confidence": round(verse["score"], 4),
            "progress_pct": round(coverage * 100, 1),
        }

    @staticmethod
    def _find_position(input_words: list[str], verse_words: list[str]) -> int:
        verse_idx = 0
        for inp_word in input_words:
            if verse_idx >= len(verse_words):
                break
            if lev_ratio(inp_word, verse_words[verse_idx]) >= 0.5:
                verse_idx += 1
            elif verse_idx + 1 < len(verse_words) and lev_ratio(inp_word, verse_words[verse_idx + 1]) >= 0.5:
                verse_idx += 2
        return verse_idx


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

    tracker = SequentialTracker(quran_db)
    completed: list[dict] = []
    last_status: str | None = None
    last_completed_ayah = 0

    # Growing audio window
    full_audio = np.empty(0, dtype=np.float32)     # all audio for current verse
    new_audio_count = 0                             # samples since last transcription
    prev_transcription = ""                         # to detect when transcription changes

    async def send_status(status: str):
        nonlocal last_status
        if status != last_status:
            await ws.send_json({"type": "status", "status": status})
            last_status = status

    def complete_ayah(surah: int, ayah: int):
        nonlocal last_completed_ayah
        if ayah <= last_completed_ayah:
            return
        v = quran_db.get_verse(surah, ayah)
        if v:
            completed.append({
                "surah": surah,
                "ayah": ayah,
                "text_uthmani": v["text_uthmani"],
                "surah_name_en": v["surah_name_en"],
            })
            last_completed_ayah = ayah
            log.info("Completed: %s %d:%d", v["surah_name_en"], surah, ayah)

    try:
        while True:
            data = await ws.receive_bytes()
            samples = np.frombuffer(data, dtype=np.float32)
            full_audio = np.concatenate([full_audio, samples])
            new_audio_count += len(samples)

            # Trim window to max size (keep latest audio)
            if len(full_audio) > MAX_WINDOW_SAMPLES:
                full_audio = full_audio[-MAX_WINDOW_SAMPLES:]

            # Only process when we have enough new audio
            if new_audio_count < TRIGGER_SAMPLES:
                continue
            new_audio_count = 0

            # Check if latest chunk is silence
            tail = full_audio[-TRIGGER_SAMPLES:]
            if _is_silence(tail):
                await send_status("silence")
                continue

            await send_status("listening")

            # Transcribe the FULL audio window (growing window approach)
            text = await asyncio.get_event_loop().run_in_executor(
                None, _transcribe, full_audio.copy()
            )

            if not text:
                continue

            log.info(
                "Transcribed (%.1fs audio): %s",
                len(full_audio) / SAMPLE_RATE,
                text[:150],
            )

            # Remember expected ayah before update
            expected_ayah = tracker.current_ayah if tracker.locked_surah else None

            # Feed FULL transcription to tracker (replaces, not appends)
            result = tracker.update_from_text(text)
            if result is None:
                log.info("No match")
                continue

            log.info(
                "Match: %s %d:%d conf=%.3f pos=%d/%d locked=%s",
                result["surah_name_en"], result["surah"], result["ayah"],
                result["confidence"], result["word_position"], result["total_words"],
                tracker.locked_surah,
            )

            matched_ayah = result["ayah"]
            surah = result["surah"]

            # If match jumped ahead, auto-complete skipped ayahs (max MAX_SKIP_AYAHS)
            skip_count = matched_ayah - expected_ayah if expected_ayah else 0
            if (
                tracker.locked_surah
                and expected_ayah is not None
                and 0 < skip_count <= MAX_SKIP_AYAHS
            ):
                for skip_ayah in range(expected_ayah, matched_ayah):
                    log.info("Skip-completing %d:%d (jumped to %d)", surah, skip_ayah, matched_ayah)
                    complete_ayah(surah, skip_ayah)
                # Reset audio window for new verse
                full_audio = tail.copy()  # keep just the latest chunk
                tracker.reset_verse()
                # Re-match with just the latest audio's text
                text = await asyncio.get_event_loop().run_in_executor(
                    None, _transcribe, full_audio.copy()
                )
                if text:
                    result = tracker.update_from_text(text)
                if not result:
                    await ws.send_json({
                        "type": "verse_update",
                        "current": None,
                        "completed": list(completed),
                    })
                    continue

            # Check auto-advance
            coverage = result["word_position"] / result["total_words"] if result["total_words"] else 0
            if (
                coverage >= ADVANCE_COVERAGE
                and result["confidence"] >= ADVANCE_CONFIDENCE
            ):
                log.info(
                    "Auto-advance: %s %d:%d → next ayah %d",
                    result["surah_name_en"], result["surah"], result["ayah"],
                    result["ayah"] + 1,
                )
                complete_ayah(result["surah"], result["ayah"])
                tracker.current_ayah = result["ayah"] + 1
                tracker.reset_verse()
                # Reset audio window for next verse
                full_audio = np.empty(0, dtype=np.float32)
                new_audio_count = 0

                await ws.send_json({
                    "type": "verse_update",
                    "current": result,
                    "completed": list(completed),
                })
                continue

            await ws.send_json({
                "type": "verse_update",
                "current": result,
                "completed": list(completed),
            })

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


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
