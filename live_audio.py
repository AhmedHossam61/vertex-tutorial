from pathlib import Path
import asyncio
import os
import re

from google import genai
from google.genai import types
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


async def main() -> None:
    load_env_file(Path(__file__).with_name(".env"))

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west4")
    model_id = "gemini-live-2.5-flash-native-audio"

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    input_rate_hz = 16000
    default_output_rate_hz = 24000
    record_seconds = 5

    print(f"Starting live session with {model_id} in {location}...")
    print("Press Enter to record a turn (5 seconds), or type q then Enter to quit.")

    def parse_rate_from_mime(mime_type: str, fallback: int) -> int:
        match = re.search(r"rate=(\\d+)", mime_type)
        if not match:
            return fallback
        return int(match.group(1))

    def record_audio_pcm(seconds: int) -> bytes:
        frames = int(seconds * input_rate_hz)
        audio = sd.rec(frames, samplerate=input_rate_hz, channels=1, dtype="int16")
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

    try:
        async with client.aio.live.connect(
            model=model_id,
            config=types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
            ),
        ) as session:
            while True:
                user_cmd = input("\nPress Enter to speak, or q to quit: ").strip().lower()
                if user_cmd in {"q", "quit", "exit"}:
                    break

                print("Recording... speak now.")
                user_audio = record_audio_pcm(record_seconds)
                print("Sending audio to model...")

                await session.send_realtime_input(
                    audio=types.Blob(data=user_audio, mime_type=f"audio/pcm;rate={input_rate_hz}"),
                    audio_stream_end=True,
                )

                model_audio = bytearray()
                output_rate_hz = default_output_rate_hz
                transcript_chunks: list[str] = []

                async for message in session.receive():
                    if message.text:
                        transcript_chunks.append(message.text)

                    if message.server_content and message.server_content.model_turn:
                        for part in (message.server_content.model_turn.parts or []):
                            if part.inline_data and part.inline_data.data:
                                model_audio.extend(part.inline_data.data)
                                mime = part.inline_data.mime_type or "audio/pcm"
                                output_rate_hz = parse_rate_from_mime(mime, output_rate_hz)

                    if message.server_content and message.server_content.turn_complete:
                        break

                if transcript_chunks:
                    print("Model transcript:")
                    print("".join(transcript_chunks).strip())

                if model_audio:
                    print("Playing model audio reply...")
                    play_pcm_audio(bytes(model_audio), output_rate_hz)
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
