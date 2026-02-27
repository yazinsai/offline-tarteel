import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import "./style.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface SurroundingVerse {
  surah: number;
  ayah: number;
  text: string;
  is_current: boolean;
}

interface VerseMatchMessage {
  type: "verse_match";
  surah: number;
  ayah: number;
  verse_text: string;
  surah_name: string;
  confidence: number;
  surrounding_verses: SurroundingVerse[];
}

interface RawTranscriptMessage {
  type: "raw_transcript";
  text: string;
  confidence: number;
}

type ServerMessage = VerseMatchMessage | RawTranscriptMessage;

interface VerseGroup {
  surah: number;
  surahName: string;
  startAyah: number;
  endAyah: number;
  currentAyah: number;
  verses: SurroundingVerse[];
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
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $verses = document.getElementById("verses")!;
const $rawTranscript = document.getElementById("raw-transcript")!;
const $indicator = document.getElementById("listening-indicator")!;
const $permissionPrompt = document.getElementById("permission-prompt")!;

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
// Verse rendering
// ---------------------------------------------------------------------------
function createVerseGroupElement(group: VerseGroup): HTMLElement {
  const el = document.createElement("div");
  el.className = "verse-group";
  el.setAttribute("data-surah", String(group.surah));
  renderVerseGroup(el, group);
  return el;
}

function renderVerseGroup(el: HTMLElement, group: VerseGroup): void {
  el.innerHTML = "";

  for (const v of group.verses) {
    const verseEl = document.createElement("div");
    verseEl.className = `verse ${v.is_current ? "verse--active" : "verse--context"}`;
    verseEl.dir = "rtl";
    verseEl.lang = "ar";

    const textEl = document.createElement("span");
    textEl.className = "verse-text";
    textEl.textContent = v.text;
    verseEl.appendChild(textEl);

    // Ayah end marker
    const markerEl = document.createElement("span");
    markerEl.className = "verse-marker";
    markerEl.textContent = ` \u06DD${toArabicNum(v.ayah)} `;
    verseEl.appendChild(markerEl);

    // Metadata (hidden by default, shown on hover/tap)
    const metaEl = document.createElement("div");
    metaEl.className = "verse-meta";
    metaEl.dir = "rtl";
    metaEl.textContent = `${group.surahName} \u2014 \u0622\u064A\u0629 ${toArabicNum(v.ayah)}`;
    verseEl.appendChild(metaEl);

    // Tap to toggle metadata on mobile
    verseEl.addEventListener("click", () => {
      verseEl.classList.toggle("verse--meta-visible");
    });

    el.appendChild(verseEl);
  }
}

function handleVerseMatch(msg: VerseMatchMessage): void {
  // Clear raw transcript when we get a confident match
  $rawTranscript.textContent = "";
  $rawTranscript.classList.remove("visible");

  // Dim the indicator when verses are showing
  $indicator.classList.add("has-verses");

  const lastGroup = state.groups[state.groups.length - 1];

  // Check if this verse belongs to the current group (same surah, nearby)
  if (
    lastGroup &&
    lastGroup.surah === msg.surah &&
    Math.abs(msg.ayah - lastGroup.currentAyah) <= SURROUNDING_CONTEXT + 1
  ) {
    // Update existing group: expand context
    lastGroup.currentAyah = msg.ayah;
    lastGroup.verses = msg.surrounding_verses;
    lastGroup.startAyah = Math.min(
      lastGroup.startAyah,
      msg.surrounding_verses[0]?.ayah ?? msg.ayah
    );
    lastGroup.endAyah = Math.max(
      lastGroup.endAyah,
      msg.surrounding_verses[msg.surrounding_verses.length - 1]?.ayah ?? msg.ayah
    );
    renderVerseGroup(lastGroup.element, lastGroup);
  } else {
    // Deactivate previous group's active verse
    if (lastGroup) {
      lastGroup.element.querySelectorAll(".verse--active").forEach((el) => {
        el.classList.remove("verse--active");
        el.classList.add("verse--context");
      });
    }

    // Create new group
    const group: VerseGroup = {
      surah: msg.surah,
      surahName: msg.surah_name,
      startAyah: msg.surrounding_verses[0]?.ayah ?? msg.ayah,
      endAyah:
        msg.surrounding_verses[msg.surrounding_verses.length - 1]?.ayah ?? msg.ayah,
      currentAyah: msg.ayah,
      verses: msg.surrounding_verses,
      element: document.createElement("div"),
    };
    group.element = createVerseGroupElement(group);
    state.groups.push(group);
    $verses.appendChild(group.element);

    // Scroll to the new group smoothly
    group.element.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

const SURROUNDING_CONTEXT = 2;

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
      } else if (msg.type === "raw_transcript") {
        handleRawTranscript(msg);
      }
    } catch {
      // ignore non-JSON
    }
  };

  ws.onclose = () => {
    state.ws = null;
    // Reconnect after a short delay
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

    // Connect WebSocket first
    const ws = connectWebSocket();
    state.ws = ws;

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("WebSocket failed"));
      // Timeout after 5s
      setTimeout(() => reject(new Error("WebSocket timeout")), 5000);
    });

    // Set up AudioContext + WorkletNode
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

    // Audio level detection for indicator
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
