from pathlib import Path
import os

from google import genai


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


load_env_file(Path(__file__).with_name(".env"))

project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location,
)

# Example: Streaming response from Gemini
# This provides a real-time streaming experience
print("Streaming response from Gemini:")
print("-" * 60)

prompt = "Explain machine learning in 3 sentences, streaming the response word by word."

try:
    # Use generate_content_stream for streaming responses
    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    
    # Iterate through chunks and print as they arrive
    for chunk in response_stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    
    print("\n" + "-" * 60)
    print("✓ Streaming completed successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
