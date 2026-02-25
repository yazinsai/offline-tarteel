# Offline Tarteel

Offline Quran verse recognition. Record someone reciting, identify the surah and ayah -- no internet required.

This repo is a research workbench for evaluating different approaches to the problem: ASR-based transcription + fuzzy matching, audio embedding search, contrastive audio-text models, CTC forced alignment, and streaming ASR. Each approach lives in its own experiment directory with a standardized `run.py` interface, and a shared benchmark runner evaluates them all against the same test corpus.

## Current results

Benchmarked on 37 samples (user recordings, professional reference audio, crowdsourced recordings). Results from `benchmark/results/latest.json`:

| Experiment | Approach | Accuracy | Avg Latency | Model Size |
|---|---|---|---|---|
| **ctc-alignment** | wav2vec2 CTC forced alignment + QuranDB re-scoring | **81%** (30/37) | 0.25s | 1.2 GB |
| whisper-lora | Whisper-small + LoRA fine-tune + QuranDB match | 78% (29/37) | 0.96s | 485 MB |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran + QuranDB match | 78% (29/37) | 1.04s | 290 MB |

Other experiments (embedding-search, contrastive, streaming-asr, new-models) have been evaluated qualitatively but aren't yet wired into the benchmark runner with the full 37-sample corpus. See [Experiments](#experiments) below.

## Project structure

```
shared/                  # Common utilities used by all experiments
  audio.py               # load_audio(path, sr=16000) -> float32 numpy array
  normalizer.py          # normalize_arabic(text) - strip diacritics, normalize alef/taa
  quran_db.py            # QuranDB - 6,236 verses, fuzzy match, multi-ayah spans

experiments/             # Each approach gets its own directory
  ctc-alignment/         # CTC forced alignment (best accuracy so far)
  whisper-lora/          # Whisper-small + LoRA adapter
  tarteel-whisper-base/  # Tarteel's whisper-base-ar-quran
  embedding-search/      # HuBERT + FAISS nearest-neighbor
  contrastive/           # QuranCLAP (HuBERT + AraBERT contrastive)
  streaming-asr/         # mlx-whisper chunked streaming
  new-models/            # Multi-model benchmark (8 ASR models)

benchmark/               # Evaluation framework
  runner.py              # CLI benchmark runner
  test_corpus/           # 37 audio samples + manifest.json
  results/               # Timestamped JSON results + latest.json

data/                    # Reference data
  quran.json             # 6,236 verses (uthmani + cleaned text)
  lora-adapter-small/    # Trained LoRA adapter (~21 MB)
  reference_audio/       # EveryAyah samples (Alafasy)
  test_audio/            # User recordings

web/                     # Live demo
  server.py              # FastAPI backend
  frontend/              # React frontend

scripts/                 # One-off training/eval scripts
  train_modal.py         # LoRA training on Modal (A10G GPU)
  train_lora.py          # Local training script (MPS/CUDA)

docs/plans/              # Design docs and experiment plans
REPORT.md                # Full experiment report with cross-comparison
RESEARCH-audio-to-verse.md  # Research notes on approaches
```

## Experiments

### ctc-alignment (best)

CTC forced alignment using a pre-trained Arabic wav2vec2 model. Instead of decoding to text and then matching, this approach scores candidate verses directly against the model's frame-level character logits using the CTC forward algorithm.

**Flow:** audio -> wav2vec2 frame logits -> greedy decode for candidate pruning -> top-K candidates from QuranDB -> CTC re-score each candidate -> best score wins

- **Accuracy:** 81% (30/37)
- **Latency:** ~0.25s
- **Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB)

### whisper-lora

Whisper-small fine-tuned with a LoRA adapter on Quranic audio (EveryAyah + RetaSy datasets). Transcribes audio to Arabic text, then fuzzy-matches against QuranDB.

- **Accuracy:** 78% (29/37)
- **Latency:** ~0.96s
- **Model:** openai/whisper-small (461 MB) + LoRA adapter (21 MB)
- **Training:** 3,000 steps on A10G GPU (Modal), ~53 minutes

### tarteel-whisper-base

Tarteel's Whisper-base model fine-tuned specifically for Quranic Arabic. Same transcribe-then-match pipeline as whisper-lora but using a purpose-built model.

- **Accuracy:** 78% (29/37)
- **Latency:** ~1.04s
- **Model:** `tarteel-ai/whisper-base-ar-quran` (290 MB)

### embedding-search

Audio fingerprinting via HuBERT embeddings + FAISS index. Pre-computes embeddings for all 6,236 verses (Alafasy recitation), then finds nearest neighbor at inference time.

- **Same-reciter accuracy:** 100% (5/5 reference audio)
- **Cross-speaker accuracy:** 0% (0/2 user recordings)
- **Latency:** ~377ms (embedding extraction dominates, FAISS search is <1ms)
- **Verdict:** Not viable standalone -- HuBERT embeddings encode speaker identity more than linguistic content. Could work as a supplementary signal for known reciters.

### contrastive (QuranCLAP)

CLIP-style contrastive model mapping Quran audio (HuBERT) and Arabic text (AraBERT) into a shared 256-dim embedding space. Proof of concept with Phase 1 training only (frozen encoders, projection heads).

- **Retrieval accuracy:** ~1.6% top-10 on 789 candidates
- **Why it doesn't work yet:** English HuBERT on Arabic audio, batch size 4, single reciter, minimal training data
- **What would fix it:** Arabic speech encoder, multi-reciter data (Quran-MD), GPU training with batch size 256+, 100+ epochs

### streaming-asr

Processes audio in 1-3 second chunks with mlx-whisper, accumulates transcription progressively, matches verses in real-time with prefix-aware Levenshtein scoring.

- **Best accuracy:** 43% (3/7, stream_2s mode, smaller test set)
- **Time to first match:** 0.19s (stream_3s) vs 1.17s (batch)
- **Key finding:** Streaming is 6x faster to first match. Accuracy is limited by whisper-base Arabic quality, not the chunking approach. 2-3 second chunks are optimal.

### new-models

Benchmarks 8 different ASR models head-to-head on the same test set:

| Model | Accuracy (7 samples) | Avg Latency | Size |
|---|---|---|---|
| Moonshine Tiny Arabic | 3/7 | 1.26s | 103 MB |
| Whisper Large-v3-Turbo | 3/7 | 2.04s | 3.1 GB |
| Tarteel Whisper Base | 3/7 | 2.08s | 277 MB |
| MMS-1B-All (Arabic) | 3/7 | 4.01s | 3.7 GB |
| Distil-Whisper Large-v3 | 0/7 | 1.66s | 2.9 GB |
| SeamlessM4T-v2 Large | 0/7 | 17.32s | 5.7 GB |

Moonshine Tiny Arabic is the efficiency winner -- 30x smaller than Whisper-turbo, fastest inference, same accuracy.

## Test corpus

`benchmark/test_corpus/manifest.json` contains 37 samples across three sources:

| Source | Count | Description |
|---|---|---|
| User recordings | 2 | Phone recordings (.m4a), ambient noise, non-professional |
| EveryAyah (Alafasy) | 23 | Professional studio recordings, includes long single-ayah and multi-ayah concatenated |
| RetaSy crowdsourced | 30 | Curated subset from 1,287 speakers across 81 countries |

**Categories:** short (17), medium (20), long (9), multi (9). Some samples span multiple categories.

## Running benchmarks

```bash
# Activate the project venv
source .venv/bin/activate

# Run all registered experiments
.venv/bin/python -m benchmark.runner

# Run a single experiment
.venv/bin/python -m benchmark.runner --experiment ctc-alignment

# Filter by audio category
.venv/bin/python -m benchmark.runner --category short
```

Results are saved to `benchmark/results/<timestamp>.json`. The runner also maintains `benchmark/results/latest.json` with the best result per experiment.

## Adding a new experiment

1. Create `experiments/<name>/run.py` with two required functions:

```python
def predict(audio_path: str) -> dict:
    """Returns {"surah": int, "ayah": int, "ayah_end": int|None, "score": float, "transcript": str}"""

def model_size() -> int:
    """Total model size in bytes."""
```

2. Register it in `benchmark/runner.py`:

```python
EXPERIMENT_REGISTRY = {
    ...
    "your-experiment": EXPERIMENTS_DIR / "your-experiment" / "run.py",
}
```

3. Run the benchmark: `.venv/bin/python -m benchmark.runner --experiment your-experiment`

For multi-model experiments, also export `list_models()` and accept `model_name` as an optional parameter in `predict()` and `model_size()`.

## Shared utilities

All experiments use `from shared.X import ...`:

- `shared.audio.load_audio(path, sr=16000)` -- returns float32 numpy array at 16kHz
- `shared.normalizer.normalize_arabic(text)` -- strips diacritics, normalizes alef/taa marbuta/alef maqsura
- `shared.quran_db.QuranDB` -- loads `data/quran.json`, provides `match_verse(text)`, `search(text, top_k)`, `get_verse(surah, ayah)`

## Web demo

FastAPI backend + React frontend for live recitation and verse identification.

```bash
# Backend
.venv/bin/python web/server.py

# Frontend (separate terminal)
cd web/frontend && npm run dev
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Some experiments have additional dependencies (faiss-cpu, moonshine, mlx-whisper). Check individual experiment READMEs.

## Key findings

1. **CTC forced alignment is the best approach so far** -- scoring candidates directly against frame logits avoids the information loss of greedy decoding, giving 81% accuracy at 0.25s latency.

2. **ASR quality is the bottleneck, not matching.** All ASR-based approaches fail on the same samples. Better Arabic transcription (larger models, Quran-specific fine-tuning) would improve all of them.

3. **Embedding search is speaker-dependent.** 100% for same-reciter, 0% for different speakers. Not viable standalone, but could supplement ASR for known voices.

4. **Streaming works for UX, not accuracy.** The chunked approach gets a verse match 6x faster but doesn't improve correctness. Pair with a better model for production use.

5. **Short verses are hard across all approaches.** Verses under 3-4 words don't provide enough signal for any method. Needs minimum-length gating or surah-context bias.

## Further reading

- `REPORT.md` -- Full experiment report with per-sample breakdowns, failure analysis, and recommendations
- `RESEARCH-audio-to-verse.md` -- Research survey of approaches (WavLink, Moonshine v2, WhisperKit, contrastive learning, audio fingerprinting)
- `docs/plans/` -- Design documents for individual experiments
- Individual experiment `README.md` files for reproduction instructions
