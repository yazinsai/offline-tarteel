import { loadModel } from "./model-cache";
import { computeMelSpectrogram } from "./mel";
import { CTCDecoder } from "./ctc-decode";
import { createSession, runInference } from "./session";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import { normalizeArabic } from "../lib/normalizer";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";

let tracker: RecitationTracker | null = null;
let decoder: CTCDecoder | null = null;

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
  // Load vocab
  const vocabRes = await fetch("/vocab.json");
  const vocabJson = await vocabRes.json();
  decoder = new CTCDecoder(vocabJson);

  // Load ONNX model
  const modelBuffer = await loadModel(
    "/fastconformer_ar_ctc.onnx",
    (loaded, total) => {
      post({
        type: "loading",
        percent: total ? Math.round((loaded / total) * 100) : 0,
      });
    },
  );
  await createSession(modelBuffer);

  // Load QuranDB
  const quranRes = await fetch("/quran.json");
  const quranData = await quranRes.json();
  const db = new QuranDB(quranData);

  // Create tracker
  tracker = new RecitationTracker(db, transcribe);
  post({ type: "ready" });
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === "init") {
    await init();
  } else if (msg.type === "audio") {
    if (!tracker) return;
    const messages = await tracker.feed(msg.samples);
    for (const m of messages) {
      post(m);
    }
  }
};
