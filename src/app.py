from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from .openrouter_client import call_openrouter
from .memory_manager import append_message
from .context_builder import build_context

app = FastAPI(title="Unisyn LLM Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 SHORT KEY -> actual OpenRouter model + human label
MODEL_REGISTRY = {
    # Special pseudo-model for auto-routing
    "unisyn-auto": {
        "id": "microsoft/phi-4",   # default backend choice for now
        "label": "Unisyn Auto",
    },

    # --- OpenAI ---
    "gpt51":        {"id": "openai/gpt-5.1",        "label": "OpenAI · GPT-5.1"},
    "gpt5":         {"id": "openai/gpt-5",          "label": "OpenAI · GPT-5"},
    "gpt41mini":    {"id": "openai/gpt-4.1-mini",   "label": "OpenAI · GPT-4.1 Mini"},
    "gpt4omini":    {"id": "openai/gpt-4o-mini",    "label": "OpenAI · GPT-4o Mini"},
    "gpt35":        {"id": "openai/gpt-3.5-turbo",  "label": "OpenAI · GPT-3.5 Turbo"},

    # --- Anthropic ---
    "claudehaiku45":   {"id": "anthropic/claude-haiku-4.5",   "label": "Claude Haiku 4.5"},
    "claude37sonnet":  {"id": "anthropic/claude-3.7-sonnet",  "label": "Claude 3.7 Sonnet"},

    # --- Google ---
    "gemini25pro":   {"id": "google/gemini-2.5-pro",        "label": "Gemini 2.5 Pro"},
    "gemini25flash": {"id": "google/gemini-2.5-flash",      "label": "Gemini 2.5 Flash"},
    "gemma327b":     {"id": "google/gemma-3-27b-it:free",   "label": "Gemma 3 27B"},

    # --- Meta (Llama) ---
    "llama4mav":     {"id": "meta-llama/llama-4-maverick:free",         "label": "Llama 4 Maverick"},
    "llama33370b":   {"id": "meta-llama/llama-3.3-70b-instruct:free",   "label": "Llama 3.3 70B"},

    # --- DeepSeek ---
    "deepseekv31":   {"id": "deepseek/deepseek-chat-v3.1",         "label": "DeepSeek Chat v3.1"},
    "deepseek0324":  {"id": "deepseek/deepseek-chat-v3-0324:free", "label": "DeepSeek Chat v3-0324"},
    "deepseekr1":    {"id": "deepseek/deepseek-r1:free",           "label": "DeepSeek R1"},

    # --- xAI (Grok) ---
    "grok4fast":     {"id": "x-ai/grok-4-fast",  "label": "Grok-4 Fast"},
    "grok4":         {"id": "x-ai/grok-4",       "label": "Grok-4"},
    "grok3":         {"id": "x-ai/grok-3",       "label": "Grok-3"},
    "grok3mini":     {"id": "x-ai/grok-3-mini",  "label": "Grok-3 Mini"},

    # --- Perplexity ---
    "sonarpro":      {"id": "perplexity/sonar-pro",        "label": "Sonar Pro"},
    "sonar":         {"id": "perplexity/sonar",            "label": "Sonar"},
    "sonarreason":   {"id": "perplexity/sonar-reasoning",  "label": "Sonar Reasoning"},

    # --- Microsoft (Copilot family) ---
    "phi4rp":        {"id": "microsoft/phi-4-reasoning-plus",       "label": "Phi-4 Reasoning Plus"},
    "phi4":          {"id": "microsoft/phi-4",                      "label": "Phi-4"},
    "phi3m":         {"id": "microsoft/phi-3-medium-128k-instruct", "label": "Phi-3 Medium 128k"},

    # --- Mistral ---
    "mistrallarge":  {"id": "mistralai/mistral-large-2407",                   "label": "Mistral Large 2407"},
    "mistralmed":    {"id": "mistralai/mistral-medium-3.1",                   "label": "Mistral Medium 3.1"},
    "mistralsmall":  {"id": "mistralai/mistral-small-3.2-24b-instruct:free",  "label": "Mistral Small 24B"},

    # --- Qwen ---
    "qwen235b":      {"id": "qwen/qwen3-235b-a22b:free",  "label": "Qwen3 235B"},
    "qwencoder":     {"id": "qwen/qwen3-coder:free",      "label": "Qwen3 Coder"},
}

# 🔑 Modular system prompts
SYSTEM_PROMPTS = {
    "solo": (
        "You are a single AI assistant in a one-on-one conversation on Unisyn AI. "
        "Focus on being clear, helpful, concise, and truthful."
    ),
    "multi_isolated": (
        "You are one of several AI assistants in Unisyn AI. "
        "You do NOT see the other assistants' messages. "
        "Provide your own best answer independently."
    ),
    "multi_shared": (
        "You are one of several AI assistants in Unisyn AI. "
        "You CAN see previous responses from other assistants. "
        "You may critique, build on, or contrast them when helpful."
    ),
}


@app.get("/health")
def health():
    return {
        "ok": True,
        "models": [
            {"key": k, "id": v["id"], "label": v["label"]}
            for k, v in MODEL_REGISTRY.items()
        ],
    }


def resolve_models_and_mode(model_keys: list[str] | None, conversation_type: str | None):
    """
    - Map short keys from the frontend to actual model configs.
    - Decide conversation_type: 'solo', 'multi_isolated', 'multi_shared'.
    """
    # 1️⃣ Default models: if nothing selected, treat as ['unisyn-auto']
    if not model_keys:
        model_keys = ["unisyn-auto"]

    # Enforce max 4 on the backend too (defensive)
    model_keys = model_keys[:4]

    models = []
    for key in model_keys:
        cfg = MODEL_REGISTRY.get(key)
        if cfg:
            models.append({"key": key, "id": cfg["id"], "label": cfg["label"]})

    # If mapping failed, fall back
    if not models:
        fallback = MODEL_REGISTRY["unisyn-auto"]
        models = [{"key": "unisyn-auto", "id": fallback["id"], "label": fallback["label"]}]

    # 2️⃣ Decide conversation type
    if not conversation_type:
        # Default: 1 model -> solo, many -> multi_isolated
        if len(models) == 1:
            conversation_type = "solo"
        else:
            conversation_type = "multi_isolated"

    # Safety
    if conversation_type not in ("solo", "multi_isolated", "multi_shared"):
        conversation_type = "solo"

    # 3️⃣ Map conversation_type -> context_mode
    if conversation_type == "multi_shared":
        context_mode = "shared_all"
    else:
        # solo and multi_isolated both use isolated per-model histories
        context_mode = "isolated"

    return models, conversation_type, context_mode


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    session_id = body.get("session_id", "default")
    model_keys = body.get("models", [])
    conversation_type = body.get("conversation_type")  # optional

    if not prompt:
        return {"error": "Missing prompt"}

    # 1️⃣ Log user message once into the global user session
    await append_message(session_id, "user", prompt)

    # 2️⃣ Decide models + context mode
    models, conversation_type, context_mode = resolve_models_and_mode(
        model_keys, conversation_type
    )

    async def query_model(m: dict):
        key = m["key"]
        model_id = m["id"]
        label = m["label"]

        # Per-model session id (for assistant messages)
        # e.g. "web-user_deepseekv31"
        model_session_id = f"{session_id}_{key}"

        # System prompt for this conversation type
        base_prompt = SYSTEM_PROMPTS[conversation_type]
        system_identity = base_prompt

        # Build context for this model (user history + per-model history + shared if enabled)
        context_messages = await build_context(session_id, key, context_mode)

        # DO NOT append the prompt again: it is already in user history
        context = [{"role": "system", "content": system_identity}] + context_messages

        try:
            # Call OpenRouter (blocking -> run in thread)
            response_text = await asyncio.to_thread(call_openrouter, model_id, context)

            # Save assistant reply in its own model-specific session
            await append_message(model_session_id, "assistant", response_text, label)

            return {
                "model": model_id,
                "label": label,
                "key": key,
                "text": response_text,
            }
        except Exception as e:
            return {
                "model": model_id,
                "label": label,
                "key": key,
                "error": str(e),
                "text": "",
            }

    # 🔁 Query one or many models in parallel
    results = await asyncio.gather(*(query_model(m) for m in models))

    return {
        "results": results,
        "session_id": session_id,
        "conversation_type": conversation_type,
        "context_mode": context_mode,
    }
