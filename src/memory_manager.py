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
        "timestamp": datetime.utcnow().isoformat(),
    }
    await collection.insert_one(doc)


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


async def get_history(session_id: str, limit: int = MEMORY_LIMIT):
    """
    Retrieve recent messages for this session, oldest → newest,
    while respecting a global token limit.
    """
    cursor = collection.find({"session_id": session_id}).sort("timestamp", -1)
    docs = await cursor.to_list(length=limit)

    total_tokens, selected = 0, []
    for d in docs:
        tokens = count_tokens(d["content"])
        if total_tokens + tokens > TOKEN_LIMIT:
            break
        total_tokens += tokens
        selected.append(d)

    # Reverse so we return chronological order (oldest → newest)
    selected.reverse()

    return [{"role": d["role"], "content": d["content"]} for d in selected]


async def clear_session(session_id: str):
    """Asynchronously delete all messages for this session"""
    await collection.delete_many({"session_id": session_id})
