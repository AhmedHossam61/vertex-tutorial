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
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location,
)

print(f"Listing models in project: {project_id}, location: {location}\n")
print("Available models:")
print("-" * 60)

try:
    for model in client.models.list():
        print(f"Model: {model.name}")
        if hasattr(model, 'display_name'):
            print(f"  Display name: {model.display_name}")
        if hasattr(model, 'supported_methods'):
            print(f"  Supported methods: {model.supported_methods}")
        print()
except Exception as e:
    print(f"Error listing models: {e}")
