import os
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
mongo_db = os.getenv("MONGODB_DB", "unisyn_ai")
MEMORY_LIMIT = int(os.getenv("CHAT_MEMORY_LIMIT", 15))
TOKEN_LIMIT = int(os.getenv("CHAT_TOKEN_LIMIT", 4000))
ENC = tiktoken.get_encoding("cl100k_base")

# create async client
client = AsyncIOMotorClient(mongo_uri)
db = client[mongo_db]
collection = db["chat_messages"]

async def append_message(session_id: str, role: str, content: str, model_name: str = None):
    doc = {
        "session_id": session_id,
        "role": role,
        "model_name": model_name,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    await collection.insert_one(doc)

def count_tokens(text):
    return len(ENC.encode(text))

async def get_history(session_id: str, limit: int = MEMORY_LIMIT):
    """
    Retrieve the last N messages for this session (roughly 3–4 turns if we count user+assistant pairs).
    """
    # cursor = (
    #     collection.find({"session_id": session_id})
    #     .sort("timestamp", -1)
    #     .limit(limit)
    # )
    # docs = await cursor.to_list(length=limit)
    # docs.reverse()  # oldest → newest order for logical flow
    # history = []
    # for d in docs:
    #     role = d["role"]
    #     name = d.get("model_name")
    #     if role == "assistant" and name:
    #         history.append({"role": "assistant", "content": f"{name}: {d['content']}"})
    #     else:
    #         history.append({"role": role, "content": d["content"]})
    # return history

    cursor = collection.find({"session_id": session_id}).sort("timestamp", -1)
    docs = await cursor.to_list(length=MEMORY_LIMIT)
    total_tokens, selected = 0, []
    for d in docs:
        tokens = count_tokens(d["content"])
        if total_tokens + tokens > TOKEN_LIMIT:
            break
        total_tokens += tokens
        selected.append(d)
    selected.reverse()
    return [{"role": d["role"], "content": d["content"]} for d in selected]

async def clear_session(session_id: str):
    """Asynchronously delete all messages for this session"""
    await collection.delete_many({"session_id": session_id})
