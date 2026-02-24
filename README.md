# Offline Tarteel

Offline Quran verse recognition. Record a recitation, get back the surah and ayah number -- no internet required.

## How it works

The system has two stages:

1. **Transcribe** -- A LoRA-finetuned Whisper model converts audio to Arabic text
2. **Match** -- Levenshtein fuzzy matching identifies the verse(s) from the transcription

### Stage 1: Transcription

We finetune [openai/whisper-small](https://huggingface.co/openai/whisper-small) (244M params) with a LoRA adapter (5.3M trainable params, ~21MB on disk). The adapter targets the attention projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`) with rank 32 and alpha 64.

**Training data** (interleaved 85/15):

| Dataset | Samples | Type |
|---------|---------|------|
| [tarteel-ai/everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) | ~127K | Professional recitations, studio quality |
| [RetaSy/quranic_audio_dataset](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset) | ~6.1K (after filtering) | Crowdsourced phone recordings from 1,287 people across 81 countries |

The RetaSy data adds variety in recording conditions, accents, and background noise. Samples labeled as incorrect, unrelated, or mismatched are filtered out.

**Training config:**

- 3,000 steps on an A10G GPU (Modal.com), ~53 minutes
- Batch size 16, learning rate 1e-5, 100 warmup steps, fp16
- Final training loss: 0.25

**Inference:**

At inference time we use `repetition_penalty=1.2` to prevent the decoder from getting stuck in loops (a known Whisper issue with repetitive text like Quranic verses).

### Stage 2: Verse matching

The transcription is matched against all 6,236 Quran verses using Levenshtein similarity ratio. Arabic text is normalized before comparison:

- Strip diacritics (tashkeel)
- Normalize alef variants (hamza forms to bare alef)
- Normalize taa marbuta to haa
- Normalize alef maqsura to yaa

**Multi-ayah matching:** People often recite across verse boundaries. A two-pass approach handles this:

1. Score all 6,236 single verses (fast)
2. Take the top 20 candidates, expand to 2-3 consecutive verse spans within those surahs
3. Return the highest-scoring match (single verse or span)

This runs in <0.1s on a laptop.

## Results

Tested on two personal phone recordings:

| Recording | Expected | Match | Score |
|-----------|----------|-------|-------|
| Aal-i-Imraan, verse 23 | 3:23 | 3:23 | 0.679 |
| Al-Ikhlas, verses 2-3 | 112:2-3 | 112:2-3 | 0.846 |

## Project structure

```
src/offline_tarteel/
    audio.py          # Audio loading (librosa, 16kHz mono)
    normalizer.py     # Arabic text normalization
    quran_db.py       # Verse database + fuzzy matching

scripts/
    train_modal.py    # LoRA training on Modal (A10G GPU)
    train_lora.py     # Local training script (MPS/CUDA)
    benchmark.py      # Evaluation benchmarks

data/
    quran.json        # 6,236 verses (uthmani + cleaned text)
    lora-adapter-small/   # Trained LoRA adapter (~21MB)
    test_audio/       # Test recordings
```

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from offline_tarteel.audio import load_audio
from offline_tarteel.quran_db import QuranDB

# Load model + adapter
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="arabic", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(model, "data/lora-adapter-small")
model.eval()

# Transcribe
audio = load_audio("recording.m4a")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
ids = model.generate(inputs.input_features, max_new_tokens=225, repetition_penalty=1.2, language="ar", task="transcribe")
text = processor.batch_decode(ids, skip_special_tokens=True)[0]

# Match
db = QuranDB()
match = db.match_verse(text)
# {'surah': 112, 'ayah': 2, 'ayah_end': 3, 'score': 0.846, ...}
```

## Training

Requires a [Modal](https://modal.com) account:

```bash
pip install modal
modal run scripts/train_modal.py
```

This trains on an A10G GPU and downloads the adapter to `data/lora-adapter-small/` when done.

## Model sizes

| Component | Size |
|-----------|------|
| whisper-small (base) | ~461MB |
| LoRA adapter | ~21MB |
| quran.json | ~2.5MB |
| **Total additional** | **~24MB** |

The base Whisper model can be bundled with the app or downloaded once. The LoRA adapter and verse database are the only project-specific assets.

## Future work

- Integrate [tarteel-ai/tlog](https://huggingface.co/datasets/tarteel-ai/tlog) dataset (gated, access requested) for more crowdsourced training data
- Audio augmentation (room reverb, background noise, codec compression) via audiomentations
- Larger evaluation set beyond two test recordings
- Optimize for on-device inference (CoreML export for iOS)
