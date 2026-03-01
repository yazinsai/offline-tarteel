import {
  mel_filter_bank,
  spectrogram,
  window_function,
} from "@huggingface/transformers";

const SAMPLE_RATE = 16000;
const N_FFT = 512;
const HOP_LENGTH = 160; // window_stride=0.01 * 16000
const WIN_LENGTH = 400; // window_size=0.025 * 16000
const N_MELS = 80;
const PREEMPH = 0.97;
const DITHER = 1e-5;
const LOG_GUARD = 1e-5;

// Pre-computed mel filterbank and window (reused across calls)
let _melFilters: number[][] | null = null;
let _window: Float64Array | null = null;

function getMelFilters(): number[][] {
  if (!_melFilters) {
    // num_frequency_bins = n_fft/2 + 1 = 257
    _melFilters = mel_filter_bank(
      N_FFT / 2 + 1, // num_frequency_bins
      N_MELS, // num_mel_filters
      0, // min_frequency
      8000, // max_frequency
      SAMPLE_RATE,
      "slaney", // norm
      "htk", // mel_scale (NeMo uses HTK)
    );
  }
  return _melFilters;
}

function getWindow(): Float64Array {
  if (!_window) {
    _window = window_function(WIN_LENGTH, "hann", { periodic: true });
  }
  return _window;
}

/**
 * Compute NeMo-compatible mel spectrogram from raw audio.
 * Returns flat Float32Array in [n_mels, time] layout (row-major).
 */
export async function computeMelSpectrogram(
  audio: Float32Array,
): Promise<{ features: Float32Array; timeFrames: number }> {
  // 1. Dither: add small random noise to prevent log(0)
  const dithered = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) {
    dithered[i] = audio[i] + DITHER * (Math.random() * 2 - 1);
  }

  // 2. Get mel filters and window
  const melFilters = getMelFilters();
  const win = getWindow();

  // 3. Compute spectrogram using transformers.js
  // The spectrogram function handles: preemphasis, STFT, power, mel application
  // NeMo uses center=false (no padding), power=2.0 (power spectrum)
  const spec = await spectrogram(dithered, win, WIN_LENGTH, HOP_LENGTH, {
    fft_length: N_FFT,
    power: 2.0,
    center: false,
    pad_mode: "reflect",
    onesided: true,
    preemphasis: PREEMPH,
    mel_filters: melFilters,
    mel_floor: 1e-10,
    log_mel: null, // We'll apply log manually for NeMo compatibility
    transpose: false, // Keep [n_mels, time] layout
  });

  // spec is Float32Array in [n_mels, time] layout
  const timeFrames = spec.length / N_MELS;

  // 4. Log with guard: ln(mel + guard)
  // NeMo uses natural log with additive guard
  const logged = new Float32Array(spec.length);
  for (let i = 0; i < spec.length; i++) {
    logged[i] = Math.log(spec[i] + LOG_GUARD);
  }

  // 5. Per-feature normalization (mean/std per mel bin across time)
  // NeMo's "per_feature" normalize: for each mel bin, subtract mean and divide by std
  for (let m = 0; m < N_MELS; m++) {
    // Compute mean
    let sum = 0;
    for (let t = 0; t < timeFrames; t++) {
      sum += logged[m * timeFrames + t];
    }
    const mean = sum / timeFrames;

    // Compute std
    let sumSq = 0;
    for (let t = 0; t < timeFrames; t++) {
      const diff = logged[m * timeFrames + t] - mean;
      sumSq += diff * diff;
    }
    const std = Math.sqrt(sumSq / timeFrames) || 1e-10;

    // Normalize
    for (let t = 0; t < timeFrames; t++) {
      logged[m * timeFrames + t] = (logged[m * timeFrames + t] - mean) / std;
    }
  }

  return { features: logged, timeFrames };
}
