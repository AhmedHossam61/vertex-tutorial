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

print("=== client.interactions ===")
print("Type:", type(client.interactions))
print("Methods:")
attrs = [a for a in dir(client.interactions) if not a.startswith('_')]
print('\n'.join(sorted(attrs)))
