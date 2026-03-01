import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import "./style.css";

import type {
  VerseMatchMessage,
  RawTranscriptMessage,
  WordProgressMessage,
  WorkerOutbound,
  QuranVerse,
} from "./lib/types";

// ---------------------------------------------------------------------------
// Types (UI-only)
// ---------------------------------------------------------------------------
interface SurahVerse {
  ayah: number;
  text_uthmani: string;
}

interface SurahData {
  surah: number;
  surah_name: string;
  surah_name_en: string;
  verses: SurahVerse[];
}

interface VerseGroup {
  surah: number;
  surahName: string;
  surahNameEn: string;
  currentAyah: number;
  verses: SurahVerse[];
  element: HTMLElement;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  groups: [] as VerseGroup[],
  worker: null as Worker | null,
  audioCtx: null as AudioContext | null,
  stream: null as MediaStream | null,
  isActive: false,
  hasFirstMatch: false,
  surahCache: new Map<number, SurahData>(),
  quranData: null as QuranVerse[] | null,
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $verses = document.getElementById("verses")!;
const $rawTranscript = document.getElementById("raw-transcript")!;
const $indicator = document.getElementById("listening-indicator")!;
const $permissionPrompt = document.getElementById("permission-prompt")!;
const $listeningStatus = document.getElementById("listening-status")!;
const $modelStatus = document.getElementById("model-status")!;

// ---------------------------------------------------------------------------
// Arabic numeral converter
// ---------------------------------------------------------------------------
const arabicNumerals = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];
function toArabicNum(n: number): string {
  return String(n)
    .split("")
    .map((d) => arabicNumerals[parseInt(d)])
    .join("");
}

// ---------------------------------------------------------------------------
// Surah data (loaded from quran.json, no server needed)
// ---------------------------------------------------------------------------
async function loadQuranData(): Promise<void> {
  if (state.quranData) return;
  const res = await fetch("/quran.json");
  state.quranData = await res.json();
}

async function fetchSurah(surahNum: number): Promise<SurahData> {
  const cached = state.surahCache.get(surahNum);
  if (cached) return cached;

  await loadQuranData();
  const verses = state.quranData!.filter((v) => v.surah === surahNum);
  if (!verses.length) throw new Error(`Surah ${surahNum} not found`);

  const data: SurahData = {
    surah: surahNum,
    surah_name: verses[0].surah_name,
    surah_name_en: verses[0].surah_name_en,
    verses: verses.map((v) => ({
      ayah: v.ayah,
      text_uthmani: v.text_uthmani,
    })),
  };
  state.surahCache.set(surahNum, data);
  return data;
}

// ---------------------------------------------------------------------------
// Verse rendering
// ---------------------------------------------------------------------------
const WAQF_MARKS = new Set([
  "\u06D6", "\u06D7", "\u06D8", "\u06D9", "\u06DA", "\u06DB", "\u06DC",
]);

function isWaqfToken(token: string): boolean {
  return token.length <= 2 && [...token].every((c) => WAQF_MARKS.has(c));
}

interface WordToken {
  text: string;
  isRealWord: boolean;
}

function splitUthmaniWords(text: string): WordToken[] {
  const raw = text.split(/\s+/).filter((w) => w.length > 0);
  const result: WordToken[] = [];

  for (const token of raw) {
    if (isWaqfToken(token) && result.length > 0) {
      result[result.length - 1].text += " " + token;
    } else {
      result.push({ text: token, isRealWord: true });
    }
  }

  return result;
}

const BISMILLAH_WORD_COUNT = 4;
const BISMILLAH_BASE = "بسم الله الرحمن الرحيم";

function stripDiacritics(s: string): string {
  return s.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]/g, "");
}

function startsWithBismillah(text: string): boolean {
  const stripped = stripDiacritics(text);
  return stripped.startsWith(BISMILLAH_BASE) || stripped.startsWith(stripDiacritics(BISMILLAH_BASE));
}

function createVerseGroupElement(group: VerseGroup): HTMLElement {
  const el = document.createElement("div");
  el.className = "verse-group";
  el.setAttribute("data-surah", String(group.surah));

  const header = document.createElement("div");
  header.className = "surah-header";
  header.textContent = group.surahNameEn;
  el.appendChild(header);

  const hasBismillah =
    group.surah !== 1 &&
    group.surah !== 9 &&
    startsWithBismillah(group.verses[0]?.text_uthmani ?? "");
  if (hasBismillah) {
    const words = group.verses[0].text_uthmani.split(/\s+/);
    const bsmText = words.slice(0, BISMILLAH_WORD_COUNT).join(" ");
    const bsmEl = document.createElement("div");
    bsmEl.className = "bismillah";
    bsmEl.dir = "rtl";
    bsmEl.lang = "ar";
    bsmEl.textContent = bsmText;
    el.appendChild(bsmEl);
  }

  const body = document.createElement("div");
  body.className = "verse-body";
  body.dir = "rtl";
  body.lang = "ar";

  for (const v of group.verses) {
    const verseEl = document.createElement("span");
    verseEl.className = "verse verse--upcoming";
    verseEl.setAttribute("data-ayah", String(v.ayah));

    const allWords = splitUthmaniWords(v.text_uthmani);
    const skipBsm = hasBismillah && v.ayah === 1;
    const startIdx = skipBsm ? BISMILLAH_WORD_COUNT : 0;

    const textEl = document.createElement("span");
    textEl.className = "verse-text";
    for (let i = startIdx; i < allWords.length; i++) {
      const wordEl = document.createElement("span");
      wordEl.className = "word";
      wordEl.setAttribute("data-word-idx", String(i));
      wordEl.textContent = allWords[i].text;
      textEl.appendChild(wordEl);
      if (i < allWords.length - 1) {
        textEl.appendChild(document.createTextNode(" "));
      }
    }
    verseEl.appendChild(textEl);

    const markerEl = document.createElement("span");
    markerEl.className = "verse-marker";
    markerEl.textContent = ` \u06DD${toArabicNum(v.ayah)} `;
    verseEl.appendChild(markerEl);

    body.appendChild(verseEl);
  }

  el.appendChild(body);
  return el;
}

function updateVerseHighlight(group: VerseGroup, newAyah: number): void {
  const el = group.element;
  const oldAyah = group.currentAyah;

  const verses = el.querySelectorAll<HTMLElement>(".verse");
  for (const verseEl of verses) {
    const ayah = parseInt(verseEl.getAttribute("data-ayah") || "0");
    if (ayah === newAyah) {
      verseEl.className = "verse verse--active";
    } else if (ayah <= newAyah && (ayah >= oldAyah || ayah < oldAyah)) {
      if (
        verseEl.classList.contains("verse--active") ||
        (ayah > oldAyah && ayah < newAyah) ||
        ayah <= oldAyah
      ) {
        verseEl.className = "verse verse--recited";
      }
    }
  }

  group.currentAyah = newAyah;
  scrollToActiveVerse();
}

function scrollToActiveVerse(): void {
  const active = document.querySelector(".verse--active");
  if (active) {
    active.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------
async function handleVerseMatch(msg: VerseMatchMessage): Promise<void> {
  $rawTranscript.textContent = "";
  $rawTranscript.classList.remove("visible");

  if (!state.hasFirstMatch) {
    state.hasFirstMatch = true;
    $listeningStatus.hidden = true;
    $indicator.classList.add("has-verses");
  }

  const lastGroup = state.groups[state.groups.length - 1];

  if (lastGroup && lastGroup.surah === msg.surah) {
    updateVerseHighlight(lastGroup, msg.ayah);
    return;
  }

  if (lastGroup) {
    lastGroup.element.classList.add("verse-group--exiting");
    const oldEl = lastGroup.element;
    setTimeout(() => oldEl.remove(), 400);
  }

  const surahData = await fetchSurah(msg.surah);

  const group: VerseGroup = {
    surah: msg.surah,
    surahName: surahData.surah_name,
    surahNameEn: surahData.surah_name_en,
    currentAyah: 0,
    verses: surahData.verses,
    element: document.createElement("div"),
  };
  group.element = createVerseGroupElement(group);
  state.groups.push(group);
  $verses.appendChild(group.element);

  updateVerseHighlight(group, msg.ayah);
}

let _matchedWordIndices = new Set<number>();
let _trackingKey = "";

function handleWordProgress(msg: WordProgressMessage): void {
  const lastGroup = state.groups[state.groups.length - 1];
  if (!lastGroup || lastGroup.surah !== msg.surah) return;

  const verseEl = lastGroup.element.querySelector<HTMLElement>(
    `.verse[data-ayah="${msg.ayah}"]`,
  );
  if (!verseEl) return;

  if (!verseEl.classList.contains("verse--active")) {
    updateVerseHighlight(lastGroup, msg.ayah);
  }

  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _trackingKey) {
    _matchedWordIndices = new Set<number>();
    _trackingKey = key;
  }

  for (const idx of msg.matched_indices) {
    _matchedWordIndices.add(idx);
  }

  let contiguousMax = -1;
  for (let i = 0; i <= msg.total_words; i++) {
    if (_matchedWordIndices.has(i)) {
      contiguousMax = i;
    } else {
      break;
    }
  }

  const wordEls = verseEl.querySelectorAll<HTMLElement>(".word");
  for (const wordEl of wordEls) {
    const idx = parseInt(wordEl.getAttribute("data-word-idx") || "-1");
    if (idx <= contiguousMax) {
      wordEl.classList.add("word--spoken");
    }
  }
}

function handleRawTranscript(msg: RawTranscriptMessage): void {
  $rawTranscript.textContent = msg.text;
  $rawTranscript.classList.add("visible");
}

// ---------------------------------------------------------------------------
// Worker message handler
// ---------------------------------------------------------------------------
function handleWorkerMessage(msg: WorkerOutbound): void {
  if (msg.type === "loading") {
    $modelStatus.textContent = `Loading model... ${msg.percent}%`;
    $modelStatus.classList.remove("ready");
  } else if (msg.type === "ready") {
    $modelStatus.textContent = "Model ready";
    $modelStatus.classList.add("ready");
  } else if (msg.type === "verse_match") {
    handleVerseMatch(msg);
  } else if (msg.type === "word_progress") {
    handleWordProgress(msg);
  } else if (msg.type === "raw_transcript") {
    handleRawTranscript(msg);
  }
}

// ---------------------------------------------------------------------------
// Audio capture
// ---------------------------------------------------------------------------
async function startAudio(): Promise<void> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    state.stream = stream;
    $permissionPrompt.hidden = true;
    $listeningStatus.hidden = false;

    const audioCtx = new AudioContext();
    state.audioCtx = audioCtx;

    await audioCtx.audioWorklet.addModule("/audio-processor.js");
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioCtx, "audio-stream-processor");

    processor.port.onmessage = (e: MessageEvent) => {
      if (state.worker) {
        const samples = new Float32Array(e.data as ArrayBuffer);
        state.worker.postMessage(
          { type: "audio", samples },
          [samples.buffer],
        );
      }
    };

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    source.connect(processor);

    const levelBuf = new Float32Array(analyser.fftSize);
    const checkLevel = () => {
      if (!state.isActive) return;
      analyser.getFloatTimeDomainData(levelBuf);
      let sum = 0;
      for (let i = 0; i < levelBuf.length; i++) {
        sum += levelBuf[i] * levelBuf[i];
      }
      const rms = Math.sqrt(sum / levelBuf.length);
      if (rms > 0.01) {
        $indicator.classList.add("audio-detected");
        $indicator.classList.remove("silence");
      } else {
        $indicator.classList.remove("audio-detected");
        $indicator.classList.add("silence");
      }
      requestAnimationFrame(checkLevel);
    };
    checkLevel();

    state.isActive = true;
    $indicator.classList.add("active");
  } catch (err) {
    console.error("Failed to start audio:", err);
    $permissionPrompt.hidden = false;
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Create inference worker
  const worker = new Worker(
    new URL("./worker/inference.ts", import.meta.url),
    { type: "module" },
  );
  state.worker = worker;

  worker.onmessage = (e: MessageEvent<WorkerOutbound>) => {
    handleWorkerMessage(e.data);
  };

  // Initialize worker (loads model, vocab, quranDB)
  worker.postMessage({ type: "init" });

  // Start audio capture
  startAudio();
});
