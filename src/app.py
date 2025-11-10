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

# MODELS = [
#     {"id": "openai/gpt-4o-mini", "label": "OpenAI · GPT-4o-mini"},
#     {"id": "anthropic/claude-3.5-sonnet", "label": "Anthropic · Claude 3.5 Sonnet"},
#     {"id": "meta-llama/llama-3.1-70b-instruct", "label": "Meta · Llama 3.1 70B"},
#     {"id": "deepseek/deepseek-chat", "label": "DeepSeek · Chat"},
# ]

MODELS = [
    {"id": "deepseek/deepseek-chat", "label": "GPT-4o-mini"},
    {"id": "deepseek/deepseek-chat", "label": "Claude"},
    {"id": "deepseek/deepseek-chat", "label": "Llama"},
    {"id": "deepseek/deepseek-chat", "label": "DeepSeek"},
]


SYSTEM_PROMPT = (
    "You are an AI assistant participating in a multi-agent conversation called Unisyn AI. "
    "There are 4 assistants (GPT-4o-mini, Claude, Llama, and DeepSeek). "
    "Each assistant can see the others' responses and should reply naturally, "
    "respectfully, and intelligently while maintaining their unique personality."
)

PERSONALITIES = {
    "GPT-4o-mini": "You are logical, structured, and philosophical — like ChatGPT. Focus on reasoning and balanced insight.",
    "Claude": "You are thoughtful, ethical, and articulate — like Anthropic’s Claude. Be gentle, empathetic, and introspective.",
    "Llama": "You are bold, analytical, and technical — like Meta’s Llama. Focus on structure, technology, and creative synthesis.",
    "DeepSeek": "You are straightforward, precise, and curious — like DeepSeek itself. Be confident and data-driven.",
}

CONTEXT_MODE = {
    "shared_all": "you are able to see all asssistants involved in the conversation",
    "isolated": "you can not see what other assistants have responded. Do not hallucinate and attempt to guess what the other assistants said. Be truthful.",
    "handover": "you are now the main assistant and can see the user and assistant conversation with a different model."
}


@app.get("/health")
def health():
    return {"ok": True, "models": [m["id"] for m in MODELS]}

@app.get("/api/history/{session_id}")
async def show_history(session_id: str):
    history = await get_history(session_id, limit=50)
    return {"session_id": session_id, "history": history}

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    session_id = body.get("session_id", "default")
    context_mode = "isolated" # body.get("context_mode", "shared_all")  
    # options: "shared_all", "isolated", "handover"

    if not prompt:
        return {"error": "Missing prompt"}

    await append_message(session_id, "user", prompt)

    # Retrieve global history
    history = await get_history(session_id)

    async def query_model(m):
        try:
            label = m["label"]
            model_id = m["id"]
            model_session_id = f"{session_id}_{label.lower()}"
            context_instructions = CONTEXT_MODE[context_mode]

            # === SYSTEM IDENTITY ===
            system_identity = (
                f"{SYSTEM_PROMPT}\nYou are {label}.\n"
                f"{PERSONALITIES.get(label, '')}"
                f"\n Currently, all assistants are on '{context_mode}' context mode."
                f"This means {context_instructions}"
            )

            # === CONTEXT MODES ===
            if context_mode == "shared_all":
                # All AIs share same full conversation
                context = [{"role": "system", "content": system_identity}] + history
                context.append({"role": "user", "content": prompt})

            elif context_mode == "isolated":
                # Each AI only gets its personal thread with user
                personal_history = await get_history(model_session_id)
                context = [{"role": "system", "content": system_identity}] + personal_history
                context.append({"role": "user", "content": prompt})

            elif context_mode == "handover":
                # Handover: if user switches from one AI to another,
                # the new AI receives the user’s last conversation (excluding other AIs)
                personal_history = await get_history(model_session_id)
                if not personal_history:
                    # This model hasn't spoken yet, so we give it the user's past messages only
                    user_only_history = [
                        h for h in history if h["role"] == "user"
                    ]
                    context = [{"role": "system", "content": system_identity}] + user_only_history
                else:
                    context = [{"role": "system", "content": system_identity}] + personal_history
                context.append({"role": "user", "content": prompt})

            else:
                return {"model": label, "error": f"Invalid context_mode: {context_mode}"}

            # === MODEL CALL ===
            response = await asyncio.to_thread(call_openrouter, model_id, context)
            await append_message(model_session_id, "assistant", response, label)
            return {"model": label, "text": response}

        except Exception as e:
            return {"model": m["label"], "error": str(e)}

    results = await asyncio.gather(*(query_model(m) for m in MODELS))
    return {
        "results": results,
        "session_id": session_id,
        "context_mode": context_mode,
    }
