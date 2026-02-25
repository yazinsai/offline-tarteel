"""CTC forced-alignment experiment.

Uses a pre-trained Arabic wav2vec2 CTC model to produce frame-level
character logits, then scores candidate Quran verses via the CTC
forward algorithm. Falls back to a fine-tuned local model if available.

Flow:
  1. Encode audio -> frame-level logits
  2. Greedy decode -> rough text for candidate pruning
  3. Levenshtein top-K candidates from QuranDB
  4. CTC re-score each candidate against frame logits
  5. Best score wins
"""
import sys
import math
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic

# Import ctc_scorer from sibling file (hyphenated directory)
_scorer_path = Path(__file__).parent / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

from Levenshtein import ratio as lev_ratio

LOCAL_MODEL = PROJECT_ROOT / "data" / "ctc-model"
PRETRAINED_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
TOP_K = 50  # candidates for CTC re-scoring

_model = None
_processor = None
_db = None
_device = None


def _ensure_loaded():
    global _model, _processor, _db, _device
    if _model is not None:
        return
    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else PRETRAINED_MODEL
    print(f"Loading CTC model from {model_path}...")
    _processor = Wav2Vec2Processor.from_pretrained(model_path)
    _model = Wav2Vec2ForCTC.from_pretrained(model_path)
    _model.eval()
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model.to(_device)
    _db = QuranDB()


def transcribe(audio_path: str) -> str:
    """Return raw greedy-decoded transcript."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    text = _processor.decode(pred_ids.cpu().numpy())
    return normalize_arabic(text)


def _spaceless_search(text: str, top_k: int = 50) -> list[dict]:
    """Search QuranDB with spaces stripped from both query and candidates.
    CTC models may not produce word delimiters, so spaceless matching
    gives better candidate pruning.
    """
    query = text.replace(" ", "")
    scored = []
    for v in _db.verses:
        verse_spaceless = v["text_clean"].replace(" ", "")
        score = lev_ratio(query, verse_spaceless)
        scored.append({**v, "score": score, "text": v["text_uthmani"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _tokenize_for_ctc(text: str) -> list[int]:
    """Convert Arabic text to token IDs using model's tokenizer."""
    encoded = _processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)

    # 1. Encode audio -> frame-level logits
    inputs = _processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits  # (1, T, V)

    # 2. Greedy decode -> rough text for candidate pruning
    pred_ids = torch.argmax(logits, dim=-1)[0]
    rough_text = _processor.decode(pred_ids.cpu().numpy())
    rough_text_normalized = normalize_arabic(rough_text)

    if not rough_text_normalized.strip():
        return {
            "surah": 0, "ayah": 0, "ayah_end": None,
            "score": 0.0, "transcript": "",
        }

    # 3. Prune: Levenshtein top-K candidates
    candidates = _db.search(rough_text_normalized, top_k=TOP_K)

    # 4. CTC re-score each candidate
    blank_id = _processor.tokenizer.pad_token_id
    scored = score_candidates(
        logits.cpu(),
        candidates,
        _tokenize_for_ctc,
        blank_id=blank_id,
    )

    if not scored:
        return {
            "surah": 0, "ayah": 0, "ayah_end": None,
            "score": 0.0, "transcript": rough_text_normalized,
        }

    best_candidate, best_loss = scored[0]

    # Convert normalized CTC loss to 0-1 confidence
    confidence = math.exp(-best_loss)

    return {
        "surah": best_candidate["surah"],
        "ayah": best_candidate["ayah"],
        "ayah_end": best_candidate.get("ayah_end"),
        "score": round(confidence, 4),
        "transcript": rough_text_normalized,
    }


def model_size() -> int:
    """wav2vec2-large-xlsr-53-arabic ~1.2GB."""
    return 1_200 * 1024 * 1024
