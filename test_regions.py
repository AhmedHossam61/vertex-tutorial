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

# Test multiple regions (prioritized for Egypt location)
regions_to_test = [
    "europe-west4",   # Belgium - closest to Egypt in Europe
    "europe-west1",   # Belgium alternative
    "us-central1",    # Central USA - where most Gemini models are
    "us-east4",       # US East
    "global",         # Fallback
]

for location in regions_to_test:
    try:
        print(f"\n{'='*60}")
        print(f"Testing region: {location}")
        print('='*60)
        
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        
        models = []
        for model in client.models.list():
            if "gemini" in model.name.lower() or "live" in model.name.lower():
                models.append(model.name)
        
        if models:
            print(f"✓ Found {len(models)} Gemini/Live models in {location}:")
            for m in models:
                print(f"  - {m}")
        else:
            print(f"✗ No Gemini models in {location}")
            
    except Exception as e:
        print(f"✗ Error in {location}: {str(e)[:100]}")
