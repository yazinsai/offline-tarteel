import { useRef, useCallback, useState } from "react";

type Status = "idle" | "connecting" | "streaming" | "error";

export function useAudioStream(wsUrl: string) {
  const [status, setStatus] = useState<Status>("idle");
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const start = useCallback(async () => {
    try {
      setStatus("connecting");

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("WebSocket connection failed"));
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule("/audio-processor.js");
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = new AudioWorkletNode(
        audioCtx,
        "audio-stream-processor"
      );

      processor.port.onmessage = (e: MessageEvent) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
        }
      };

      source.connect(processor);
      setStatus("streaming");
    } catch (err) {
      console.error("Audio stream error:", err);
      setStatus("error");
    }
  }, [wsUrl]);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    wsRef.current?.close();
    wsRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
    setStatus("idle");
  }, []);

  return { status, start, stop, ws: wsRef };
}
