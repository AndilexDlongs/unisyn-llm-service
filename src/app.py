from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from src import call_openrouter, append_message, get_history

app = FastAPI(title="Unisyn LLM Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODELS = [
    {"id": "openai/gpt-4o-mini", "label": "OpenAI · GPT-4o-mini"},
    {"id": "anthropic/claude-3.5-sonnet", "label": "Anthropic · Claude 3.5 Sonnet"},
    {"id": "meta-llama/llama-3.1-70b-instruct", "label": "Meta · Llama 3.1 70B"},
    {"id": "deepseek/deepseek-chat", "label": "DeepSeek · Chat"},
]

@app.get("/health")
def health():
    return {"ok": True, "models": [m["id"] for m in MODELS]}

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    session_id = body.get("session_id", "default")
    if not prompt:
        return {"error": "Missing prompt"}

    append_message(session_id, "user", prompt)
    history = get_history(session_id)

    async def query_model(m):
        try:
            response = await asyncio.to_thread(call_openrouter, m["id"], history)
            append_message(session_id, "assistant", response)
            return {"model": m["label"], "text": response}
        except Exception as e:
            return {"model": m["label"], "error": str(e)}

    results = await asyncio.gather(*(query_model(m) for m in MODELS))
    return {"results": results, "session_id": session_id}
