# vertex-tutorial

Vertex AI quickstart project for:

1. Basic Gemini text generation
2. Gemini Live native-audio voice interaction (microphone input, speaker output)

## Project files

1. [main.py](main.py): simple text request to `gemini-2.5-flash`
2. [live_audio.py](live_audio.py): interactive voice agent using `gemini-live-2.5-flash-native-audio`
3. [streaming.py](streaming.py): text streaming example
4. [list_models.py](list_models.py): list available models in your project/region

## Prerequisites

1. Google Cloud project with billing enabled
2. Vertex AI API enabled
3. `gcloud` installed
4. Python virtual environment (`.venv`)

## Cloud setup (one time)

### 1. Enable Vertex AI API

In Google Cloud Console, enable the Vertex AI API for your project.

### 2. Authenticate locally with ADC

Run:

```powershell
gcloud auth application-default login
```

Important:

1. Complete browser sign-in fully and accept prompts.
2. This creates local ADC credentials used by the Python SDK.

## Local setup

### 1. Configure environment in [.env](.env)

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=europe-west4
GOOGLE_GENAI_USE_VERTEXAI=true
```

Notes:

1. Use your real project ID (not project display name).
2. `europe-west4` is a good default for users in Egypt (lower latency than many US regions).
3. `us-west1` does not expose Gemini models in this project setup.

### 2. Install dependencies

```powershell
uv pip install google-genai numpy sounddevice
```

`numpy` and `sounddevice` are required for microphone capture and audio playback in [live_audio.py](live_audio.py).

## Verify setup in order

### 1. Check available models

```powershell
.venv\Scripts\python list_models.py
```

Look for:

1. `publishers/google/models/gemini-2.5-flash`
2. `publishers/google/models/gemini-live-2.5-flash-native-audio`

### 2. Run basic text test

```powershell
.venv\Scripts\python main.py
```

If this works, Vertex auth and project config are correct.

### 3. Run Gemini Live voice test

```powershell
.venv\Scripts\python live_audio.py
```

Flow:

1. Press Enter to record one voice turn (5 seconds)
2. Speak while recording
3. Script sends your audio to Gemini Live
4. Model returns audio chunks
5. Script plays model voice reply
6. Type `q` then Enter to exit

## How Gemini Live is implemented here

The working implementation in [live_audio.py](live_audio.py) uses:

1. `client.aio.live.connect(...)` (async Live API path)
2. Model: `gemini-live-2.5-flash-native-audio`
3. `response_modalities=[AUDIO]` (required for native-audio model)
4. `send_realtime_input(audio=...)` then `send_realtime_input(audio_stream_end=True)` as two separate calls

The two-call pattern is required because Live API enforces one-of input fields per message.

## Where to get model IDs

Use any of these:

1. Vertex AI Model Garden in Google Cloud Console
2. Vertex AI Gemini model docs
3. Programmatically via [list_models.py](list_models.py)

Common IDs in this project:

1. `gemini-2.5-flash`
2. `gemini-2.5-pro`
3. `gemini-live-2.5-flash-native-audio`

## Troubleshooting

### `DefaultCredentialsError: Your default credentials were not found`

1. Run `gcloud auth application-default login` again
2. Complete browser consent flow fully
3. Ensure you run script under same Windows user profile used for gcloud login

### `Only one argument can be set, got 2: ['audio', 'audio_stream_end']`

Fix:

1. Send audio and stream-end in separate calls
2. Current [live_audio.py](live_audio.py) already does this correctly

### `Text output is not supported for native audio output model`

Fix:

1. Use `response_modalities=[AUDIO]` for `gemini-live-2.5-flash-native-audio`
2. Current [live_audio.py](live_audio.py) already does this

### `Unsupported model` or model not found

1. Switch to a supported region like `europe-west4` or `us-central1`
2. Re-run [list_models.py](list_models.py) to confirm availability

### No sound from speaker

1. Check default output device in OS audio settings
2. Ensure speaker/headphones are selected and volume is up
3. Confirm `sounddevice` can access audio device

### Microphone not recording

1. Allow microphone permissions for terminal/Python on Windows
2. Check default input device in OS settings
3. Test microphone with another app first

## Next improvements (optional)

1. Replace fixed 5-second turns with continuous streaming microphone mode
2. Save returned PCM audio to WAV files for debugging
3. Add push-to-talk key handling instead of Enter prompt