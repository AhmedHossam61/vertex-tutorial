# vertex-tutorial

Simple Vertex AI text-generation test project using Gemini models.

## What this project does

This repo shows the smallest useful setup for calling a Gemini model on Vertex AI from Python.

The sample in [main.py](main.py) does one thing:

1. Reads local config from [.env](.env)
2. Uses Application Default Credentials from `gcloud`
3. Sends a text prompt to a Gemini model on Vertex AI
4. Prints the response

## Prerequisites

You need:

1. A Google Cloud project with Vertex AI enabled
2. Billing enabled for that project
3. `gcloud` installed locally
4. Python dependencies installed in your virtual environment

## One-time setup

### 1. Enable Vertex AI

In Google Cloud Console, enable the Vertex AI API for your project.

### 2. Sign in for local development

Run:

```powershell
gcloud auth application-default login
```

This creates your local Application Default Credentials file, which Python uses when calling Vertex AI.

### 3. Configure the project locally

Keep local config in [.env](.env). A minimal setup looks like this:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
```

Notes:

1. `GOOGLE_CLOUD_PROJECT` is your Google Cloud project ID, not the project name.
2. `GOOGLE_CLOUD_LOCATION` can be `global` for a simple smoke test, or a region such as `us-central1`.
3. `GOOGLE_GENAI_USE_VERTEXAI` is a flag for Vertex usage. The current sample also sets Vertex mode directly in code.

## Run the sample

After the environment is configured, run:

```powershell
python .\main.py
```

If everything is correct, the script should print a short Gemini response.

## Where to get the model ID

The model ID is the exact string passed to `model=` in `generate_content`.

You can get it from these places:

1. Vertex AI Model Garden in the Google Cloud Console
2. The Vertex AI Gemini documentation for supported model names
3. The Vertex AI API by listing available models

Example model IDs you may see are:

```text
gemini-2.5-flash
gemini-2.5-pro
gemini-2.0-flash
```

The exact model ID depends on what is available in your project and region.

### List available models from Python

If you want to discover model IDs programmatically, you can use:

```python
from google import genai

client = genai.Client(vertexai=True, project="your-project-id", location="global")

for model in client.models.list():
	print(model.name)
```

## Troubleshooting

If you see `DefaultCredentialsError`, it usually means one of these:

1. You did not run `gcloud auth application-default login`
2. You signed in with a different Windows user profile
3. The ADC file was not created successfully

If you see a model or region error, check that:

1. The model exists in your selected location
2. The project has Vertex AI enabled
3. Your account has permission to use Vertex AI

## Current example

The current sample uses `gemini-2.5-flash` for a basic text-generation test.

If you want to change the test model, edit the `model=` value in [main.py](main.py).