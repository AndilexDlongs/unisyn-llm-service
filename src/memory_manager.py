from collections import defaultdict

# session_id -> list of {role, content}
_chat_memory = defaultdict(list)

def get_history(session_id: str):
    return _chat_memory[session_id]

def append_message(session_id: str, role: str, content: str):
    _chat_memory[session_id].append({"role": role, "content": content})
