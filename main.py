from google import genai

# Create client (uses your gcloud auth automatically)
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain AI agents in 2 lines"
)

print(response.text)