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
  - `get_verse(surah, ayah)`, `get_surah(surah)`

## Running Benchmarks

```bash
.venv/bin/python -m benchmark.runner                          # all experiments
.venv/bin/python -m benchmark.runner --experiment whisper-lora # single experiment
.venv/bin/python -m benchmark.runner --category short          # filter by category
```

Results go to `benchmark/results/<timestamp>.json`.

## Test Corpus

`benchmark/test_corpus/manifest.json` — 37 samples:
- 7 baseline (2 user recordings + 5 EveryAyah reference)
- 30 RetaSy crowdsourced (curated via `benchmark/curate_corpus.py`)
- Categories: short (1-5 words), medium (6-15), long (15+), multi (multi-ayah)

## Current Experiments

| Experiment | Approach | Accuracy | Latency |
|---|---|---|---|
| whisper-lora | Whisper-small + LoRA → QuranDB match | 78% | ~1s |
| embedding-search | HuBERT → FAISS index | — | — |
| contrastive | QuranCLAP (HuBERT+AraBERT contrastive) | — | — |
| streaming-asr | mlx-whisper chunked → QuranDB match | — | — |
| new-models/* | Various HF models (6 models) | — | — |

## Python Environment

Always use `.venv/bin/python` — the system Python lacks project dependencies.
