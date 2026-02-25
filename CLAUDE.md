# Offline Tarteel

Offline Quran verse recognition — given an audio clip of someone reciting, identify the surah and ayah.

## Project Structure

```
shared/              # Common utilities (audio, normalizer, quran_db)
experiments/         # Each approach gets its own directory with a run.py
benchmark/           # Runner, test corpus, results
  runner.py          # CLI: python -m benchmark.runner
  test_corpus/       # Audio files + manifest.json (37 samples)
  results/           # Timestamped JSON output
data/                # quran.json, reference audio, LoRA adapters
src/offline_tarteel/ # Legacy package (kept for compatibility)
scripts/             # One-off training/eval scripts
web/                 # FastAPI + React frontend for live demo
```

## Experiment Convention

Every experiment lives in `experiments/<name>/` with a `run.py` that exports:

### Required functions

```python
def predict(audio_path: str) -> dict:
    """Run inference on an audio file.

    Returns:
        {"surah": int, "ayah": int, "ayah_end": int|None, "score": float, "transcript": str}

    On failure/no match: surah=0, ayah=0, score=0.0
    transcript="" is fine for non-ASR approaches (embedding search, contrastive).
    """

def model_size() -> int:
    """Total model size in bytes (estimate is fine)."""
```

### Pattern

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.quran_db import QuranDB

# Lazy-load globals
_model = None

def _ensure_loaded():
    global _model
    if _model is not None:
        return
    # ... load model, processor, etc.

def predict(audio_path: str) -> dict:
    _ensure_loaded()
    # ... inference logic
    # For ASR approaches: transcribe -> match against QuranDB
    # For embedding approaches: encode -> find nearest verse

def model_size() -> int:
    return 461 * 1024 * 1024  # estimate in bytes
```

### Multi-model experiments

If one experiment wraps multiple models (like `new-models/`), also export:

```python
def list_models() -> list[str]: ...
def predict(audio_path: str, model_name: str = "default") -> dict: ...
def model_size(model_name: str = "default") -> int: ...
```

The benchmark runner expands these into separate entries automatically.

### Registering a new experiment

Add it to `EXPERIMENT_REGISTRY` in `benchmark/runner.py`:

```python
EXPERIMENT_REGISTRY = {
    "whisper-lora": EXPERIMENTS_DIR / "whisper-lora" / "run.py",
    "your-new-experiment": EXPERIMENTS_DIR / "your-new-experiment" / "run.py",
    # ...
}
```

Directory names use hyphens (e.g. `whisper-lora`). The runner uses `importlib.util.spec_from_file_location` so hyphens work fine.

## Shared Utilities

Use `from shared.X import ...` (not `from offline_tarteel.X`):

- `shared.audio.load_audio(path, sr=16000)` — returns float32 numpy array at 16kHz
- `shared.normalizer.normalize_arabic(text)` — strips diacritics, normalizes alef/taa marbuta
- `shared.quran_db.QuranDB` — loads `data/quran.json` (6,236 verses), provides:
  - `match_verse(text)` — fuzzy match with multi-ayah span support
  - `search(text, top_k=5)` — top-k Levenshtein matches
  - `get_verse(surah, ayah)`, `get_surah(surah)`, `get_next_verse(surah, ayah)`
- `shared.verse_tracker.VerseTracker` — streaming verse detection with continuation bias
- `shared.streaming.StreamingPipeline` — connects ASR backends to verse tracker

## Running Benchmarks

```bash
.venv/bin/python -m benchmark.runner                          # all experiments
.venv/bin/python -m benchmark.runner --experiment whisper-lora # single experiment
.venv/bin/python -m benchmark.runner --category short          # filter by category
```

Results go to `benchmark/results/<timestamp>.json`.

### Scoring

The benchmark uses **streaming sequence evaluation**. Each experiment's `transcribe()` output is fed through `VerseTracker` which emits an ordered sequence of `(surah, ayah)` detections. Metrics:

- **Recall**: fraction of expected verses detected in the correct order
- **Precision**: fraction of predicted verses that are correct
- **Sequence Accuracy**: 1.0 only if the full ordered sequence matches exactly

### Experiment Interface

Each experiment must export:

```python
def transcribe(audio_path: str) -> str:  # raw transcript (required for benchmark)
def predict(audio_path: str) -> dict:     # full prediction with verse match
def model_size() -> int:                  # model size in bytes
```

## Test Corpus

`benchmark/test_corpus/manifest.json` — 55 samples:
- 2 user recordings (.m4a)
- 23 EveryAyah reference (Alafasy, includes 8 long single-ayah + 8 multi-ayah concatenated)
- 30 RetaSy crowdsourced (curated via `benchmark/curate_corpus.py`)
- Categories: short (17), medium (20), long (9), multi (9)

## Current Experiments

| Experiment | Approach | Recall | Precision | SeqAcc | Latency | Size |
|---|---|---|---|---|---|---|
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran | 72% | 72% | 62% | ~1.6s | 290 MB |
| whisper-lora | Whisper-small + LoRA | 64% | 65% | 58% | ~1.3s | 485 MB |
| ctc-alignment | Arabic CTC wav2vec2 | 69% | 65% | 56% | ~1.0s | 1.2 GB |
| embedding-search | HuBERT → FAISS index | — | — | — | — | — |
| contrastive | QuranCLAP (HuBERT+AraBERT) | — | — | — | — | — |
| streaming-asr | mlx-whisper chunked | — | — | — | — | — |
| new-models/* | Various HF models | — | — | — | — | — |

## Python Environment

Always use `.venv/bin/python` — the system Python lacks project dependencies.
