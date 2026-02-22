import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage


LOG_PATH = "/Users/reza/source/ai-edu/llamaindex/.cursor/debug-15dc20.log"
logger = logging.getLogger(__name__)


# region agent log
def write_debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    """Append a single NDJSON debug log line."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": "15dc20",
            "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Never let logging break the app
        pass


# endregion agent log


# ---- Tools configuration ----

def get_current_datetime() -> str:
    """Returns the current system date and time in a human-readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


datetime_tool = FunctionTool.from_defaults(fn=get_current_datetime)


# Thread-local storage for current thread_id (used by memory tool)
_current_thread_id = None


def store_memory(key: str, value: str) -> str:
    """Store important information in the thread's memory as a key-value pair.
    
    Args:
        key: The key to store (e.g., 'user_name', 'birthday', 'favorite_color')
        value: The value to store
    
    Returns:
        Confirmation message
    """
    global _current_thread_id
    if _current_thread_id is None:
        return "Error: Thread context not available. Cannot store memory."
    
    chat_store.add_to_thread_memory(_current_thread_id, key, value)
    return f"Memory saved: {key} = {value}"


memory_tool = FunctionTool.from_defaults(fn=store_memory)


# ---- Custom Chat Store ----

class CustomChatStore(SimpleChatStore):
    """Custom chat store subclass for future enhancements and customizations."""
    
    metadata: Dict[str, list] = Field(default_factory=dict)
    
    def get_thread_memory(self, thread_id: str) -> list:
        """Get the memory (key-value pairs) for a specific thread."""
        return self.metadata.get(thread_id, [])
    
    def add_to_thread_memory(self, thread_id: str, key: str, value: str) -> None:
        """Add or update a key-value pair in a thread's memory."""
        if thread_id not in self.metadata:
            self.metadata[thread_id] = []
        
        # Check if key already exists and update it
        for item in self.metadata[thread_id]:
            if item.get('key') == key:
                item['value'] = value
                return
        
        # If key doesn't exist, add it
        self.metadata[thread_id].append({'key': key, 'value': value})
    
    def persist(self, persist_path: str) -> None:
        """Persist the store including metadata to file."""
        super().persist(persist_path)
        # Also save metadata to the JSON file
        try:
            with open(persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['metadata'] = self.metadata
            with open(persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to persist metadata: {e}")
    
    @classmethod
    def from_persist_path(cls, persist_path: str) -> "CustomChatStore":
        """Load from persist path including metadata."""
        try:
            # Load the JSON file directly
            with open(persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract and remove metadata so parent can load the store
            metadata = data.get('metadata', {})
            store_data = {k: v for k, v in data.items() if k != 'metadata'}
            
            # Create an instance using the store data (Pydantic model_validate)
            store = super().model_validate(store_data)
            
            # Create custom store and copy the loaded data
            custom_store = cls.__new__(cls)
            custom_store.__dict__.update(store.__dict__)
            custom_store.metadata = metadata
            
            return custom_store
        except Exception as e:
            logger.warning(f"Failed to load from persist path: {e}")
            custom_store = cls()
            return custom_store


# ---- LlamaIndex + chat store configuration ----

Settings.llm = OpenAI(model="gpt-4o-mini")

CHAT_STORE_PATH = "chat_store.json"
SUMMARY_TOKEN_THRESHOLD = 3000  # When to trigger summarization
RECENT_MESSAGES_TO_KEEP = 6  # Keep last N messages, summarize the rest

if os.path.exists(CHAT_STORE_PATH):
    chat_store = CustomChatStore.from_persist_path(CHAT_STORE_PATH)
else:
    chat_store = CustomChatStore()


def get_message_content(msg) -> str:
    """Extract text content from a message, handling both content field and blocks structure."""
    if hasattr(msg, 'content') and msg.content:
        return msg.content
    elif hasattr(msg, 'blocks') and msg.blocks:
        # Handle blocks structure
        text_parts = []
        for block in msg.blocks:
            if hasattr(block, 'text') and block.text:
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get('text'):
                text_parts.append(block['text'])
        return " ".join(text_parts)
    return ""


def summarize_old_messages(messages: list, num_recent_to_keep: int = 6) -> list:
    """Summarize older messages and keep recent ones intact."""
    if len(messages) <= num_recent_to_keep:
        return messages
    
    # Split messages into old and recent
    old_messages = messages[:-num_recent_to_keep]
    recent_messages = messages[-num_recent_to_keep:]
    
    # Create a conversation text from old messages
    conversation_text = "\n".join([
        f"{msg.role}: {get_message_content(msg)}" for msg in old_messages
    ])
    
    # Generate summary using the LLM
    summary_prompt = f"""Summarize the following conversation concisely, preserving key information, context, and any important details the user shared:

{conversation_text}

Provide a brief summary:"""
    
    summary_response = Settings.llm.complete(summary_prompt)
    summary_text = summary_response.text
    
    # Create a system message with the summary
    summary_message = ChatMessage(
        role="system",
        content=f"Previous conversation summary: {summary_text}"
    )
    
    # Return summary + recent messages
    return [summary_message] + recent_messages


def should_summarize(messages: list, threshold: int = SUMMARY_TOKEN_THRESHOLD) -> bool:
    """Check if the conversation should be summarized based on approximate token count."""
    # Rough estimate: 1 token ≈ 4 characters
    total_chars = sum(len(get_message_content(msg)) for msg in messages)
    estimated_tokens = total_chars // 4
    return estimated_tokens > threshold


def get_chat_engine(thread_id: str) -> ReActAgent:
    global _current_thread_id
    _current_thread_id = thread_id
    
    # Get existing messages for this thread
    messages = chat_store.get_messages(thread_id)
    
    # Check if we need to summarize
    if should_summarize(messages):
        summarized_messages = summarize_old_messages(messages, RECENT_MESSAGES_TO_KEEP)
        # Update the chat store with summarized messages
        chat_store.set_messages(thread_id, summarized_messages)
        persist_chat_store()
    
    # Get thread memory
    thread_memory = chat_store.get_thread_memory(thread_id)
    memory_context = ""
    if thread_memory:
        memory_items = [f"{item['key']}: {item['value']}" for item in thread_memory]
        memory_context = f"\n\nImportant information about this user:\n" + "\n".join(memory_items)
    
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=4096,
        chat_store=chat_store,
        chat_store_key=thread_id,
    )
    
    system_prompt = f"You are a helpful AI assistant. Use the available tools when appropriate.{memory_context}"
    
    return ReActAgent.from_tools(
        tools=[datetime_tool, memory_tool],
        llm=Settings.llm,
        memory=memory,
        system_prompt=system_prompt,
        verbose=True,
    )


def persist_chat_store() -> None:
    chat_store.persist(CHAT_STORE_PATH)


# ---- FastAPI app ----

app = FastAPI(title="LlamaIndex Chatbot with Threads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup() -> None:
    # region agent log
    write_debug_log(
        run_id="startup",
        hypothesis_id="H0",
        location="chatbot_backend.py:_on_startup",
        message="FastAPI app startup",
        data={},
    )
    # endregion agent log


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    # region agent log
    write_debug_log(
        run_id="shutdown",
        hypothesis_id="H0",
        location="chatbot_backend.py:_on_shutdown",
        message="FastAPI app shutdown",
        data={},
    )
    # endregion agent log


class MessageIn(BaseModel):
    message: str


class MessageOut(BaseModel):
    thread_id: str
    response: str


@app.post("/threads", response_model=MessageOut)
async def chat_new_thread(payload: MessageIn) -> MessageOut:
    """Start a new thread and return its id + first response."""
    run_id = "initial"
    write_debug_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="chatbot_backend.py:chat_new_thread:entry",
        message="New thread request received",
        data={"payloadLength": len(payload.message)},
    )

    thread_id = str(uuid.uuid4())

    try:
        chat_engine = get_chat_engine(thread_id)
        resp = chat_engine.chat(payload.message)
        persist_chat_store()
    except Exception as exc:
        logger.exception("Unhandled error in /threads")
        write_debug_log(
            run_id=run_id,
            hypothesis_id="H2",
            location="chatbot_backend.py:chat_new_thread:error",
            message="Error during new thread chat",
            data={"errorType": type(exc).__name__, "errorMessage": str(exc)},
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc) if "api key" in str(exc).lower() or "401" in str(exc) else "Internal error when chatting",
        )

    write_debug_log(
        run_id=run_id,
        hypothesis_id="H5",
        location="chatbot_backend.py:chat_new_thread:success",
        message="New thread response generated",
        data={
            "threadIdLength": len(thread_id),
            "responseLength": len(resp.response),
        },
    )

    return MessageOut(thread_id=thread_id, response=resp.response)


@app.post("/threads/{thread_id}", response_model=MessageOut)
async def chat_existing_thread(thread_id: str, payload: MessageIn) -> MessageOut:
    """Continue an existing thread, keeping memory within that thread."""
    run_id = "initial"
    write_debug_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="chatbot_backend.py:chat_existing_thread:entry",
        message="Existing thread request received",
        data={
            "threadIdLength": len(thread_id),
            "payloadLength": len(payload.message),
        },
    )

    try:
        chat_engine = get_chat_engine(thread_id)
        resp = chat_engine.chat(payload.message)
        persist_chat_store()
    except Exception as exc:
        logger.exception("Unhandled error in /threads/{thread_id}")
        write_debug_log(
            run_id=run_id,
            hypothesis_id="H2",
            location="chatbot_backend.py:chat_existing_thread:error",
            message="Error during existing thread chat",
            data={
                "threadIdLength": len(thread_id),
                "errorType": type(exc).__name__,
                "errorMessage": str(exc),
            },
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc) if "api key" in str(exc).lower() or "401" in str(exc) else "Internal error when chatting",
        )

    write_debug_log(
        run_id=run_id,
        hypothesis_id="H5",
        location="chatbot_backend.py:chat_existing_thread:success",
        message="Existing thread response generated",
        data={
            "threadIdLength": len(thread_id),
            "responseLength": len(resp.response),
        },
    )

    return MessageOut(thread_id=thread_id, response=resp.response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chatbot_backend:app", host="0.0.0.0", port=8000, reload=True)
