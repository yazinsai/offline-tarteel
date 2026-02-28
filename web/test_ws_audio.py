"""
Test script: send an audio file over WebSocket to the server
and print all responses (verse_match, word_progress, raw_transcript).
"""
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import websockets


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 4800  # ~300ms chunks, matching the frontend


async def main(audio_path: str):
    # Convert to 16kHz mono WAV using ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav",
            tmp_wav,
        ],
        capture_output=True,
    )

    import soundfile as sf
    audio, sr = sf.read(tmp_wav, dtype="float32")
    Path(tmp_wav).unlink(missing_ok=True)

    print(f"Audio: {len(audio)/SAMPLE_RATE:.1f}s, {sr}Hz")

    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # Send audio in chunks (simulating real-time streaming)
        send_task = asyncio.create_task(_send_audio(ws, audio))
        recv_task = asyncio.create_task(_recv_messages(ws))

        await send_task
        # Wait a few more seconds for final processing
        await asyncio.sleep(5)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass


async def _send_audio(ws, audio: np.ndarray):
    """Send audio in chunks with real-time pacing."""
    chunk_duration = CHUNK_SAMPLES / SAMPLE_RATE
    total = len(audio)
    offset = 0
    while offset < total:
        chunk = audio[offset : offset + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        await ws.send(chunk.tobytes())
        offset += CHUNK_SAMPLES
        await asyncio.sleep(chunk_duration)  # real-time pacing
    print(f"\n--- Audio send complete ({total/SAMPLE_RATE:.1f}s) ---\n")


async def _recv_messages(ws):
    """Receive and print all server messages."""
    try:
        async for raw in ws:
            msg = json.loads(raw)
            t = msg["type"]
            if t == "verse_match":
                print(
                    f"VERSE_MATCH  {msg['surah']}:{msg['ayah']}  "
                    f"conf={msg['confidence']}  "
                    f"{msg['verse_text'][:60]}..."
                )
            elif t == "word_progress":
                print(
                    f"WORD_PROG    {msg['surah']}:{msg['ayah']}  "
                    f"word {msg['word_index']}/{msg['total_words']}  "
                    f"indices={msg['matched_indices']}"
                )
            elif t == "raw_transcript":
                print(
                    f"RAW          conf={msg['confidence']}  "
                    f"{msg['text'][:80]}"
                )
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path:
        print("Usage: python test_ws_audio.py <audio_file>")
        sys.exit(1)
    asyncio.run(main(path))
