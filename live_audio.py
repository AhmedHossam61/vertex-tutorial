from pathlib import Path
import asyncio
import os
import re
import time

from google import genai
from google.genai import types
from langsmith import traceable
import numpy as np
import sounddevice as sd


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


# ── LangSmith: trace each individual voice turn ───────────────────────────────
# `wrap_gemini` only covers REST calls; the Live API uses WebSockets,
# so we instrument manually using @traceable on the per-turn coroutine.
# Each turn becomes its own trace in LangSmith with input/output metadata.
@traceable(
    name="gemini_live_turn",
    run_type="llm",
    tags=["live-api", "audio"],
)
async def run_live_turn(
    session,
    user_audio_bytes: bytes,
    input_rate_hz: int,
    default_output_rate_hz: int,
    turn_index: int,
    *,
    # LangSmith uses these kwargs to populate the trace input
    metadata: dict,
) -> dict:
    """
    Send one audio turn and collect the model response.
    Returns a dict so LangSmith can display input + output in its UI.
    """
    t_start = time.perf_counter()

    await session.send_realtime_input(
        audio=types.Blob(data=user_audio_bytes, mime_type=f"audio/pcm;rate={input_rate_hz}"),
    )
    await session.send_realtime_input(audio_stream_end=True)

    model_audio = bytearray()
    output_rate_hz = default_output_rate_hz
    transcript_chunks: list[str] = []

    def parse_rate_from_mime(mime_type: str, fallback: int) -> int:
        match = re.search(r"rate=(\d+)", mime_type)
        return int(match.group(1)) if match else fallback

    async for message in session.receive():
        # Transcript comes via output_transcription when audio transcription is enabled
        if message.server_content and message.server_content.output_transcription:
            chunk = message.server_content.output_transcription.text
            if chunk:
                transcript_chunks.append(chunk)

        if message.server_content and message.server_content.model_turn:
            for part in (message.server_content.model_turn.parts or []):
                if part.inline_data and part.inline_data.data:
                    model_audio.extend(part.inline_data.data)
                    mime = part.inline_data.mime_type or "audio/pcm"
                    output_rate_hz = parse_rate_from_mime(mime, output_rate_hz)

        if message.server_content and message.server_content.turn_complete:
            break

    latency_ms = round((time.perf_counter() - t_start) * 1000)
    transcript = "".join(transcript_chunks).strip()

    # The dict returned here becomes the "output" in LangSmith
    return {
        "turn": turn_index,
        "transcript": transcript or "(audio only — no text transcript)",
        "audio_bytes_received": len(model_audio),
        "output_rate_hz": output_rate_hz,
        "latency_ms": latency_ms,
        # pass raw audio back for playback
        "_audio": bytes(model_audio),
        "_output_rate_hz": output_rate_hz,
    }


def record_audio_pcm(seconds: int, rate_hz: int) -> bytes:
    frames = int(seconds * rate_hz)
    audio = sd.rec(frames, samplerate=rate_hz, channels=1, dtype="int16")
    sd.wait()
    return audio.tobytes()


def play_pcm_audio(pcm_bytes: bytes, sample_rate_hz: int) -> None:
    if not pcm_bytes:
        return
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return
    sd.play(samples, samplerate=sample_rate_hz)
    sd.wait()


async def main() -> None:
    load_env_file(Path(__file__).with_name(".env"))

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west4")
    model_id = "gemini-live-2.5-flash-native-audio"

    # Plain client — no wrap_gemini needed; tracing happens via @traceable
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    input_rate_hz = 16000
    default_output_rate_hz = 24000
    record_seconds = 5
    turn_index = 0

    print(f"Starting live session with {model_id} in {location}...")
    print("LangSmith tracing: each turn is a separate trace in your project.")
    print("Press Enter to record a turn (5 seconds), or type q then Enter to quit.")

    try:
        async with client.aio.live.connect(
            model=model_id,
            config=types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                # Enable text transcripts alongside audio
                output_audio_transcription=types.AudioTranscriptionConfig(),
                input_audio_transcription=types.AudioTranscriptionConfig(),
            ),
        ) as session:
            while True:
                user_cmd = input("\nPress Enter to speak, or q to quit: ").strip().lower()
                if user_cmd in {"q", "quit", "exit"}:
                    break

                turn_index += 1
                print("Recording... speak now.")
                user_audio = record_audio_pcm(record_seconds, input_rate_hz)
                print(f"Sending audio to model (turn {turn_index})...")

                # ── Each turn is now a LangSmith trace ──────────────────────
                result = await run_live_turn(
                    session,
                    user_audio,
                    input_rate_hz,
                    default_output_rate_hz,
                    turn_index,
                    metadata={
                        "model": model_id,
                        "location": location,
                        "input_bytes": len(user_audio),
                        "record_seconds": record_seconds,
                    },
                )

                if result["transcript"]:
                    print(f"Model transcript: {result['transcript']}")

                print(f"Latency: {result['latency_ms']} ms  |  "
                      f"Audio bytes received: {result['audio_bytes_received']}")

                if result["_audio"]:
                    print("Playing model audio reply...")
                    play_pcm_audio(result["_audio"], result["_output_rate_hz"])
                else:
                    print("No audio returned from model for this turn.")

        print("Voice session ended.")
    except Exception as e:
        print(f"Live model test failed: {e}")
        print(
            "If this fails, keep GOOGLE_CLOUD_LOCATION as europe-west4 or us-central1 and verify Vertex Live API access."
        )
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
