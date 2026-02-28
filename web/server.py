"""
WebSocket backend for real-time Quran verse recognition.

Uses NVIDIA FastConformer (NeMo) for Arabic ASR with a rolling
audio window. Matches transcripts against QuranDB and sends
verse_match or raw_transcript messages to the frontend.
"""

import asyncio
import logging
import os
import sys
import tempfile
import types
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB, partial_ratio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
TRIGGER_SECONDS = 2.0
TRIGGER_SAMPLES = int(SAMPLE_RATE * TRIGGER_SECONDS)
MAX_WINDOW_SECONDS = 10.0
MAX_WINDOW_SAMPLES = int(SAMPLE_RATE * MAX_WINDOW_SECONDS)
SILENCE_RMS_THRESHOLD = 0.005
PORT = 8000

# Matching thresholds
VERSE_MATCH_THRESHOLD = 0.45
FIRST_MATCH_THRESHOLD = 0.75  # higher bar before any verse is locked on
RAW_TRANSCRIPT_THRESHOLD = 0.25
SURROUNDING_CONTEXT = 2  # verses before/after current

# Model config
NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
LOCAL_MODEL_DIR = Path(
    os.getenv(
        "NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR",
        str(PROJECT_ROOT / "data" / "nvidia-fastconformer-ar"),
    )
)
DECODER_TYPE = os.getenv("NVIDIA_FASTCONFORMER_DECODER", "ctc")

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
_nemo_model = None


# ---------------------------------------------------------------------------
# FastConformer model loading
# ---------------------------------------------------------------------------
def _install_kaldialign_fallback() -> None:
    """Install a tiny kaldialign-compatible fallback when package is absent.

    NeMo imports kaldialign in context-biasing utilities, even for inference
    flows that do not use those codepaths. This fallback unblocks model import.
    """
    try:
        import kaldialign  # noqa: F401
        return
    except Exception:
        pass

    def align(ref, hyp, eps="<eps>"):
        ref = list(ref)
        hyp = list(hyp)
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
            bt[i][0] = "D"
        for j in range(1, m + 1):
            dp[0][j] = j
            bt[0][j] = "I"
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                sub = dp[i - 1][j - 1] + cost
                ins = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                best = min(sub, ins, delete)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = "S"
                elif best == ins:
                    bt[i][j] = "I"
                else:
                    bt[i][j] = "D"
        out = []
        i, j = n, m
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "S":
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif move == "I":
                out.append((eps, hyp[j - 1]))
                j -= 1
            else:
                out.append((ref[i - 1], eps))
                i -= 1
        out.reverse()
        return out

    mod = types.ModuleType("kaldialign")
    mod.align = align
    sys.modules["kaldialign"] = mod


def _extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _load_fastconformer():
    global _nemo_model
    if _nemo_model is not None:
        return

    _install_kaldialign_fallback()
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from nemo.utils import logging as nemo_logging
    except ImportError as exc:
        raise ImportError(
            "NeMo ASR dependencies required. Install with: "
            "pip install 'nemo_toolkit[asr]'"
        ) from exc

    nemo_logging.set_verbosity(nemo_logging.ERROR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else NVIDIA_MODEL_ID
    log.info("Loading FastConformer from %s on %s...", source, device)

    try:
        _nemo_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=source,
            map_location=device,
        )
    except Exception:
        if LOCAL_MODEL_DIR.exists():
            nemo_files = sorted(LOCAL_MODEL_DIR.glob("*.nemo"))
            if not nemo_files:
                raise
            _nemo_model = EncDecHybridRNNTCTCBPEModel.restore_from(
                str(nemo_files[0]),
                map_location=device,
            )
        else:
            raise

    _nemo_model.eval()
    try:
        _nemo_model.change_decoding_strategy(decoder_type=DECODER_TYPE)
    except Exception:
        pass

    log.info("FastConformer loaded successfully")


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
def _transcribe(audio: np.ndarray) -> str:
    """Transcribe audio array using FastConformer."""
    _load_fastconformer()

    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(str(tmp_path), audio, SAMPLE_RATE)

        try:
            outputs = _nemo_model.transcribe(
                audio=[str(tmp_path)],
                batch_size=1,
                return_hypotheses=True,
                verbose=False,
            )
        except TypeError:
            outputs = _nemo_model.transcribe(
                paths2audio_files=[str(tmp_path)],
                batch_size=1,
                return_hypotheses=True,
            )

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, list) and outputs:
            transcript = _extract_text(outputs[0])
        else:
            transcript = _extract_text(outputs)

        return normalize_arabic(transcript)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _is_silence(audio: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms < SILENCE_RMS_THRESHOLD


def _get_surrounding_verses(db: QuranDB, surah: int, ayah: int) -> list[dict]:
    """Get surrounding verses for context display."""
    verses = db.get_surah(surah)
    result = []
    for v in verses:
        if abs(v["ayah"] - ayah) <= SURROUNDING_CONTEXT:
            result.append(
                {
                    "surah": v["surah"],
                    "ayah": v["ayah"],
                    "text": v["text_uthmani"],
                    "is_current": v["ayah"] == ayah,
                }
            )
    return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    global quran_db
    quran_db = QuranDB()
    log.info("QuranDB loaded: %d verses", quran_db.total_verses)
    _load_fastconformer()
    yield


app = FastAPI(
    title="Offline Tarteel",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected: %s", ws.client)

    full_audio = np.empty(0, dtype=np.float32)
    new_audio_count = 0
    last_emitted_ref: tuple[int, int] | None = None
    last_emitted_text: str = ""

    try:
        while True:
            data = await ws.receive_bytes()
            samples = np.frombuffer(data, dtype=np.float32)
            full_audio = np.concatenate([full_audio, samples])
            new_audio_count += len(samples)

            # Trim to max window (keep latest audio)
            if len(full_audio) > MAX_WINDOW_SAMPLES:
                full_audio = full_audio[-MAX_WINDOW_SAMPLES:]

            # Only process when enough new audio has accumulated
            if new_audio_count < TRIGGER_SAMPLES:
                continue
            new_audio_count = 0

            # Skip silent chunks
            tail = full_audio[-TRIGGER_SAMPLES:]
            if _is_silence(tail):
                continue

            # Transcribe the full audio window
            text = await asyncio.get_event_loop().run_in_executor(
                None, _transcribe, full_audio.copy()
            )

            if not text or len(text.strip()) < 5:
                continue

            audio_len = len(full_audio) / SAMPLE_RATE
            log.info(
                "Transcribed (%.1fs): %s",
                audio_len,
                text[:120],
            )

            # Skip if transcription is mostly residual from the last emitted verse
            if last_emitted_text:
                residual = partial_ratio(text, last_emitted_text)
                if residual > 0.70:
                    log.info(
                        "  (residual overlap %.2f with last emitted, skipping)",
                        residual,
                    )
                    continue

            # Match against QuranDB (span-aware, with continuation bias)
            match = quran_db.match_verse(
                text,
                threshold=RAW_TRANSCRIPT_THRESHOLD,
                max_span=4,
                hint=last_emitted_ref,
                return_top_k=5,
            )

            # --- Debug log: full prediction table ---
            hint_str = (
                f"{last_emitted_ref[0]}:{last_emitted_ref[1]}"
                if last_emitted_ref
                else "none"
            )
            if match:
                ayah_end = match.get("ayah_end", "")
                end_str = f"-{ayah_end}" if ayah_end else ""
                log.info(
                    "MATCH  %s:%s%s  score=%.3f (raw=%.3f +bonus=%.3f)  hint=%s",
                    match["surah"],
                    match["ayah"],
                    end_str,
                    match["score"],
                    match.get("raw_score", match["score"]),
                    match.get("bonus", 0.0),
                    hint_str,
                )
                for i, r in enumerate(match.get("runners_up", []), 1):
                    tag = " <<<" if r.get("bonus", 0) > 0 else ""
                    log.info(
                        "  #%d  %s:%s  score=%.3f (raw=%.3f +%.3f)  %s%s",
                        i,
                        r["surah"],
                        r["ayah"],
                        r["score"],
                        r["raw_score"],
                        r["bonus"],
                        r["text_clean"][:40],
                        tag,
                    )
            else:
                log.info("NO MATCH (below %.2f)  hint=%s", RAW_TRANSCRIPT_THRESHOLD, hint_str)

            effective_threshold = FIRST_MATCH_THRESHOLD if last_emitted_ref is None else VERSE_MATCH_THRESHOLD
            if match and match["score"] >= effective_threshold:
                ref = (match["surah"], match["ayah"])

                # Dedup: skip if same verse was just sent
                if ref == last_emitted_ref:
                    log.info("  (dedup — same as last emitted, skipping)")
                    continue

                verse = quran_db.get_verse(match["surah"], match["ayah"])
                surrounding = _get_surrounding_verses(
                    quran_db, match["surah"], match["ayah"]
                )

                await ws.send_json(
                    {
                        "type": "verse_match",
                        "surah": match["surah"],
                        "ayah": match["ayah"],
                        "verse_text": (
                            verse["text_uthmani"] if verse else match.get("text", "")
                        ),
                        "surah_name": verse["surah_name"] if verse else "",
                        "confidence": round(match["score"], 2),
                        "surrounding_verses": surrounding,
                    }
                )

                # For multi-verse spans, advance hint to the last verse
                ayah_end = match.get("ayah_end")
                effective_ref = (match["surah"], ayah_end) if ayah_end else ref
                log.info(
                    ">>> EMITTED verse_match %s:%s%s (was %s)",
                    match["surah"],
                    match["ayah"],
                    f"-{ayah_end}" if ayah_end else "",
                    hint_str,
                )
                last_emitted_ref = effective_ref
                last_emitted_text = normalize_arabic(
                    match.get("text_clean", "")
                    or (verse["text_clean"] if verse else "")
                )

                # Reset window for next verse detection
                full_audio = tail.copy()
            else:
                score = round(match["score"], 2) if match else 0.0
                log.info("  (below threshold %.2f — sending raw_transcript, score=%.2f)", effective_threshold, score)
                await ws.send_json(
                    {
                        "type": "raw_transcript",
                        "text": text,
                        "confidence": score,
                    }
                )

    except WebSocketDisconnect:
        log.info("Client disconnected: %s", ws.client)
    except Exception:
        log.exception("WebSocket error")


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
@app.get("/api/surah/{surah_num}")
async def get_surah(surah_num: int):
    verses = quran_db.get_surah(surah_num)
    if not verses:
        raise HTTPException(status_code=404, detail="Surah not found")
    return {
        "surah": surah_num,
        "surah_name": verses[0]["surah_name"],
        "surah_name_en": verses[0]["surah_name_en"],
        "verses": [
            {
                "ayah": v["ayah"],
                "text_uthmani": v["text_uthmani"],
            }
            for v in verses
        ],
    }


# ---------------------------------------------------------------------------
# Serve frontend (if built)
# ---------------------------------------------------------------------------
_frontend_dist = PROJECT_ROOT / "web" / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_frontend_dist), html=True),
        name="frontend",
    )
    log.info("Serving frontend from %s", _frontend_dist)
else:

    @app.get("/")
    async def _root():
        return {
            "status": "ok",
            "message": "Offline Tarteel backend. Frontend not built yet.",
        }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
