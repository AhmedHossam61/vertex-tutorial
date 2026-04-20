import os
from pathlib import Path
from google import genai
from langsmith import wrappers, client

def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

def verify():
    load_env()
    
    if os.environ.get("LANGSMITH_API_KEY") == "your_api_key_here":
        print("[ERROR] You are still using the placeholder API key in .env")
        print("Please replace 'your_api_key_here' with your real key from LangSmith.")
        return

    # Initialize and wrap
    genai_client = genai.Client(vertexai=True)
    wrapped_client = wrappers.wrap_gemini(genai_client)

    print(f"Testing LangSmith project: {os.environ.get('LANGSMITH_PROJECT')}")
    
    try:
        # This call will be traced
        print("Sending test greeting to Gemini...")
        response = wrapped_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'LangSmith is active!'"
        )
        print(f"Model response: {response.text}")
        
        # Check if we can see the run in the LangSmith client
        ls_client = client.Client()
        recent_runs = list(ls_client.list_runs(project_name=os.environ.get("LANGSMITH_PROJECT"), limit=1))
        
        if recent_runs:
            print("\n[SUCCESS] LangSmith captured a run!")
            run = recent_runs[0]
            print(f"Run ID: {run.id}")
            # Note: The URL is usually https://smith.langchain.com/o/<org>/projects/p/<project>/r/<id>
            print(f"Check your dashboard: https://smith.langchain.com/projects/p/{os.environ.get('LANGSMITH_PROJECT')}")
        else:
            print("\n⚠️  No runs found yet. Data might be still uploading (wait a few seconds).")

    except Exception as e:
        print(f"[ERROR] Tracing failed or setup error: {e}")

if __name__ == "__main__":
    verify()
