import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import "./style.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface VerseMatchMessage {
  type: "verse_match";
  surah: number;
  ayah: number;
  verse_text: string;
  surah_name: string;
  confidence: number;
  surrounding_verses: { surah: number; ayah: number; text: string; is_current: boolean }[];
}

interface RawTranscriptMessage {
  type: "raw_transcript";
  text: string;
  confidence: number;
}

interface WordProgressMessage {
  type: "word_progress";
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  matched_indices: number[];
}

type ServerMessage = VerseMatchMessage | RawTranscriptMessage | WordProgressMessage;

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
  ws: null as WebSocket | null,
  audioCtx: null as AudioContext | null,
  stream: null as MediaStream | null,
  isActive: false,
  hasFirstMatch: false,
  surahCache: new Map<number, SurahData>(),
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $verses = document.getElementById("verses")!;
const $rawTranscript = document.getElementById("raw-transcript")!;
const $indicator = document.getElementById("listening-indicator")!;
const $permissionPrompt = document.getElementById("permission-prompt")!;
const $listeningStatus = document.getElementById("listening-status")!;

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
// Surah data fetching
// ---------------------------------------------------------------------------
async function fetchSurah(surahNum: number): Promise<SurahData> {
  const cached = state.surahCache.get(surahNum);
  if (cached) return cached;

  const res = await fetch(`/api/surah/${surahNum}`);
  const data: SurahData = await res.json();
  state.surahCache.set(surahNum, data);
  return data;
}

// ---------------------------------------------------------------------------
// Verse rendering
// ---------------------------------------------------------------------------
// Quranic stop/waqf marks that appear as standalone tokens in Uthmani text
// but are stripped by the normalizer (so they don't exist in text_clean)
const WAQF_MARKS = new Set([
  "\u06D6", "\u06D7", "\u06D8", "\u06D9", "\u06DA", "\u06DB", "\u06DC",
]);

function isWaqfToken(token: string): boolean {
  return token.length <= 2 && [...token].every((c) => WAQF_MARKS.has(c));
}

interface WordToken {
  text: string;       // display text (may include trailing waqf mark)
  isRealWord: boolean; // false for standalone waqf marks (shouldn't happen after merge)
}

/**
 * Split Uthmani text into word tokens, merging standalone waqf marks
 * with the preceding word so that word indices align with text_clean.
 */
function splitUthmaniWords(text: string): WordToken[] {
  const raw = text.split(/\s+/).filter((w) => w.length > 0);
  const result: WordToken[] = [];

  for (const token of raw) {
    if (isWaqfToken(token) && result.length > 0) {
      // Merge with preceding word
      result[result.length - 1].text += " " + token;
    } else {
      result.push({ text: token, isRealWord: true });
    }
  }

  return result;
}

// Bismillah: 4 words "بسم الله الرحمن الرحيم"
// We detect it by stripping diacritics and comparing base letters,
// since the Uthmani text may have different diacritic orderings.
const BISMILLAH_WORD_COUNT = 4;
const BISMILLAH_BASE = "بسم الله الرحمن الرحيم";

/** Strip Arabic diacritics (tashkeel) for comparison */
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

  // Surah header
  const header = document.createElement("div");
  header.className = "surah-header";
  header.textContent = group.surahNameEn;
  el.appendChild(header);

  // Bismillah line (separate from verse text) for surahs that have it
  const hasBismillah =
    group.surah !== 1 &&
    group.surah !== 9 &&
    startsWithBismillah(group.verses[0]?.text_uthmani ?? "");
  if (hasBismillah) {
    // Extract the actual bismillah text (first 4 words) from the verse
    const words = group.verses[0].text_uthmani.split(/\s+/);
    const bsmText = words.slice(0, BISMILLAH_WORD_COUNT).join(" ");
    const bsmEl = document.createElement("div");
    bsmEl.className = "bismillah";
    bsmEl.dir = "rtl";
    bsmEl.lang = "ar";
    bsmEl.textContent = bsmText;
    el.appendChild(bsmEl);
  }

  // Flowing verse body — all verses as inline spans
  const body = document.createElement("div");
  body.className = "verse-body";
  body.dir = "rtl";
  body.lang = "ar";

  for (const v of group.verses) {
    const verseEl = document.createElement("span");
    verseEl.className = "verse verse--upcoming";
    verseEl.setAttribute("data-ayah", String(v.ayah));

    // Split verse text into individual word spans (merging waqf marks)
    const allWords = splitUthmaniWords(v.text_uthmani);

    // For ayah 1, skip the bismillah words (already shown in header)
    // but keep the data-word-idx aligned with the server's text_clean
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
      // Add space between words (except after the last)
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

  // Mark all ayahs between old and new (inclusive of old) as recited
  const verses = el.querySelectorAll<HTMLElement>(".verse");
  for (const verseEl of verses) {
    const ayah = parseInt(verseEl.getAttribute("data-ayah") || "0");
    if (ayah === newAyah) {
      verseEl.className = "verse verse--active";
    } else if (ayah <= newAyah && (ayah >= oldAyah || ayah < oldAyah)) {
      // Mark as recited if it was active or skipped
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

  // First match: hide listening status, show indicator with has-verses
  if (!state.hasFirstMatch) {
    state.hasFirstMatch = true;
    $listeningStatus.hidden = true;
    $indicator.classList.add("has-verses");
  }

  const lastGroup = state.groups[state.groups.length - 1];

  // Same surah as current group — just move the highlight
  if (lastGroup && lastGroup.surah === msg.surah) {
    updateVerseHighlight(lastGroup, msg.ayah);
    return;
  }

  // New surah — fade out old group, fetch full surah, render
  if (lastGroup) {
    lastGroup.element.classList.add("verse-group--exiting");
    // Remove after animation completes
    const oldEl = lastGroup.element;
    setTimeout(() => oldEl.remove(), 400);
  }

  const surahData = await fetchSurah(msg.surah);

  const group: VerseGroup = {
    surah: msg.surah,
    surahName: surahData.surah_name,
    surahNameEn: surahData.surah_name_en,
    currentAyah: 0, // will be set by updateVerseHighlight
    verses: surahData.verses,
    element: document.createElement("div"),
  };
  group.element = createVerseGroupElement(group);
  state.groups.push(group);
  $verses.appendChild(group.element);

  // Set the active verse highlight
  updateVerseHighlight(group, msg.ayah);
}

// Cumulative set of matched word indices for the current tracking verse
let _matchedWordIndices = new Set<number>();
let _trackingKey = "";  // "surah:ayah" to detect verse changes

function handleWordProgress(msg: WordProgressMessage): void {
  const lastGroup = state.groups[state.groups.length - 1];
  if (!lastGroup || lastGroup.surah !== msg.surah) return;

  const verseEl = lastGroup.element.querySelector<HTMLElement>(
    `.verse[data-ayah="${msg.ayah}"]`,
  );
  if (!verseEl) return;

  // Ensure this verse is active
  if (!verseEl.classList.contains("verse--active")) {
    updateVerseHighlight(lastGroup, msg.ayah);
  }

  // Reset cumulative indices when tracking a new verse
  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _trackingKey) {
    _matchedWordIndices = new Set<number>();
    _trackingKey = key;
  }

  // Add new matched indices to cumulative set
  for (const idx of msg.matched_indices) {
    _matchedWordIndices.add(idx);
  }

  // Find the highest contiguously matched index starting from 0.
  // This prevents false forward jumps from highlighting unread words.
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
// WebSocket
// ---------------------------------------------------------------------------
function connectWebSocket(): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${window.location.host}/ws`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onmessage = (e: MessageEvent) => {
    try {
      const msg: ServerMessage = JSON.parse(e.data);
      if (msg.type === "verse_match") {
        handleVerseMatch(msg);
      } else if (msg.type === "word_progress") {
        handleWordProgress(msg);
      } else if (msg.type === "raw_transcript") {
        handleRawTranscript(msg);
      }
    } catch {
      // ignore non-JSON
    }
  };

  ws.onclose = () => {
    state.ws = null;
    if (state.isActive) {
      setTimeout(() => {
        if (state.isActive) {
          state.ws = connectWebSocket();
        }
      }, 2000);
    }
  };

  return ws;
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

    // Show listening status
    $listeningStatus.hidden = false;

    const ws = connectWebSocket();
    state.ws = ws;

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("WebSocket failed"));
      setTimeout(() => reject(new Error("WebSocket timeout")), 5000);
    });

    const audioCtx = new AudioContext();
    state.audioCtx = audioCtx;

    await audioCtx.audioWorklet.addModule("/audio-processor.js");
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioCtx, "audio-stream-processor");

    processor.port.onmessage = (e: MessageEvent) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(e.data);
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
  startAudio();
});
