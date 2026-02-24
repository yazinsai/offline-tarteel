import { useState, useEffect, useCallback } from "react";
import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import { useAudioStream } from "./hooks/useAudioStream";
import { CurrentVerse } from "./components/CurrentVerse";
import { CompletedVerses } from "./components/CompletedVerses";
import type { VerseState, CompletedVerse, ServerMessage } from "./types";
import "./App.css";

const WS_URL = `ws://${window.location.host}/ws`;

function App() {
  const { status, start, stop, ws } = useAudioStream(WS_URL);
  const [current, setCurrent] = useState<VerseState | null>(null);
  const [completed, setCompleted] = useState<CompletedVerse[]>([]);

  const handleMessage = useCallback((e: MessageEvent) => {
    try {
      const msg: ServerMessage = JSON.parse(e.data);
      if (msg.type === "verse_update") {
        setCurrent(msg.current ?? null);
        if (msg.completed) {
          setCompleted(msg.completed);
        }
      }
    } catch {
      // ignore non-JSON messages
    }
  }, []);

  useEffect(() => {
    const socket = ws.current;
    if (!socket) return;

    socket.addEventListener("message", handleMessage);
    return () => {
      socket.removeEventListener("message", handleMessage);
    };
  }, [status, ws, handleMessage]);

  const handleStop = useCallback(() => {
    stop();
    setCurrent(null);
    setCompleted([]);
  }, [stop]);

  const isStreaming = status === "streaming";

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <span className="title">Tarteel</span>
          <span
            className={`status-dot ${isStreaming ? "active" : ""}`}
          />
        </div>
        {isStreaming && <span className="status-label">Live</span>}
      </header>

      {status === "idle" && (
        <div className="start-screen">
          <div className="start-bismillah">بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ</div>
          <button className="start-btn" onClick={start}>
            Start Reciting
          </button>
        </div>
      )}

      {status === "connecting" && (
        <div className="start-screen">
          <button className="start-btn" disabled>
            Connecting...
          </button>
        </div>
      )}

      {status === "error" && (
        <div className="error-state">
          <span className="error-msg">
            Could not connect. Check mic permissions.
          </span>
          <button className="retry-btn" onClick={start}>
            Try Again
          </button>
        </div>
      )}

      {isStreaming && (
        <>
          {current ? (
            <CurrentVerse verse={current} />
          ) : (
            <div className="listening-state">
              <div className="listening-rings">
                <div className="listening-ring" />
                <div className="listening-ring" />
                <div className="listening-ring" />
              </div>
              <span className="listening-label">Listening</span>
            </div>
          )}

          <CompletedVerses verses={completed} />

          <button className="stop-btn" onClick={handleStop}>
            Stop
          </button>
        </>
      )}
    </div>
  );
}

export default App;
