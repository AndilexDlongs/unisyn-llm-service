import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

def call_openrouter(model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 200):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "UnisynAI Backend"
    }

    payload = {
        "model": model,
        "messages": messages,  # include previous messages (user + assistant)
        "temperature": temperature,
        "max_tokens": max_tokens
    }


    res = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
    if not res.ok:
        raise Exception(f"OpenRouter error {res.status_code}: {res.text}")
    data = res.json()
    return data["choices"][0]["message"]["content"]
