import { loadModel } from "./model-cache";
import { computeMelSpectrogram } from "./mel";
import { CTCDecoder } from "./ctc-decode";
import { createSession, runInference } from "./session";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import { normalizeArabic } from "../lib/normalizer";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";

const MODEL_URL = "/fastconformer_ar_ctc_q8.onnx";

let tracker: RecitationTracker | null = null;
let decoder: CTCDecoder | null = null;
let db: QuranDB | null = null;

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

async function transcribe(audio: Float32Array): Promise<string> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const numMels = 80;
  const { logprobs, timeSteps, vocabSize } = await runInference(
    features,
    numMels,
    timeFrames,
  );
  const text = decoder!.decode(logprobs, timeSteps, vocabSize);
  return normalizeArabic(text);
}

async function init() {
  try {
    // Load vocab
    post({ type: "loading_status", message: "Loading vocabulary..." });
    const vocabRes = await fetch("/vocab.json");
    if (!vocabRes.ok) throw new Error(`vocab.json fetch failed: ${vocabRes.status}`);
    const vocabJson = await vocabRes.json();
    decoder = new CTCDecoder(vocabJson);

    // Load ONNX model
    post({ type: "loading_status", message: "Downloading model..." });
    const modelBuffer = await loadModel(
      MODEL_URL,
      (loaded, total) => {
        post({
          type: "loading",
          percent: total ? Math.round((loaded / total) * 100) : 0,
        });
      },
    );

    post({ type: "loading_status", message: "Creating inference session..." });
    await createSession(modelBuffer);

    // Load QuranDB
    post({ type: "loading_status", message: "Loading Quran data..." });
    const quranRes = await fetch("/quran.json");
    if (!quranRes.ok) throw new Error(`quran.json fetch failed: ${quranRes.status}`);
    const quranData = await quranRes.json();
    db = new QuranDB(quranData);

    // Create tracker
    tracker = new RecitationTracker(db, transcribe);
    post({ type: "ready" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Worker init failed:", message);
    post({ type: "error", message });
  }
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === "init") {
    await init();
  } else if (msg.type === "reset") {
    if (db) {
      tracker = new RecitationTracker(db, transcribe);
    }
  } else if (msg.type === "audio") {
    if (!tracker) return;
    const messages = await tracker.feed(msg.samples);
    for (const m of messages) {
      post(m);
    }
  }
};
