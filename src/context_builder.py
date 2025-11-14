from .memory_manager import get_history, collection


async def build_context(session_id: str, model_key: str, mode: str):
    """
    Build the full context for a given model:
    - user messages are global to all models  (session_id)
    - assistant messages are model-specific  (session_id_modelkey)
    - shared mode merges all assistant histories (session_id_*)
    """
    user_session = session_id
    model_session = f"{session_id}_{model_key}"

    # 1️⃣ Get all user messages (chronological, token-limited)
    user_history = await get_history(user_session)

    # 2️⃣ Model-specific assistant history
    model_history = await get_history(model_session)

    # 3️⃣ Shared mode — merge all assistants' histories
    if mode == "shared_all":
        from_db = await collection.find({
            "session_id": {"$regex": f"^{session_id}_"}
        }).sort("timestamp", 1).to_list(length=500)

        assistant_histories = [
            {"role": m["role"], "content": m["content"], "model": m.get("model_name")}
            for m in from_db
        ]

        return user_history + assistant_histories

    # 4️⃣ Isolated (default for solo + multi_isolated)
    if mode == "isolated":
        return user_history + model_history

    # 5️⃣ Handover (not used yet, but kept for future)
    if mode == "handover":
        if model_history:
            return model_history
        return user_history

    # Fallback: just user history
    return user_history
