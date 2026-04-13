import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from gemini_webapi import GeminiClient

API_KEY = os.getenv("API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3-flash-preview")
GEMINI_1PSID = os.getenv("GEMINI_1PSID", "")
GEMINI_1PSIDTS = os.getenv("GEMINI_1PSIDTS", "")

client: GeminiClient | None = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False


def require_api_key(auth_header: str | None):
    if not API_KEY:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    token = auth_header.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(401, "Unauthorized")


def parse_data_url(url: str) -> tuple[bytes, str]:
    header, data = url.split(",", 1)
    mime = header.split(";", 1)[0].split(":", 1)[1]
    return base64.b64decode(data), mime


async def extract_prompt_and_files(messages: list[ChatMessage]) -> tuple[str, list[io.BytesIO]]:
    text_parts: list[str] = []
    files: list[io.BytesIO] = []

    for msg in messages:
        role = msg.role.upper()
        content = msg.content
        if isinstance(content, str):
            text_parts.append(f"{role}: {content}")
            continue

        if isinstance(content, list):
            text_parts.append(f"{role}:")
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        data, mime = parse_data_url(url)
                        ext = mime.split("/")[-1] or "bin"
                        bio = io.BytesIO(data)
                        bio.name = f"upload_{uuid.uuid4().hex}.{ext}"
                        files.append(bio)
                    else:
                        text_parts.append(f"[External file URL: {url}]")

    return "\n".join([p for p in text_parts if p]).strip(), files


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = GeminiClient(GEMINI_1PSID, GEMINI_1PSIDTS or None, proxy=None)
    await client.init(timeout=300, auto_close=True, close_delay=180, auto_refresh=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    assert client is not None
    models = client.list_models() or []
    return {
        "object": "list",
        "data": [
            {
                "id": m.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "google-gemini-web",
            }
            for m in models
            if m.is_available
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    assert client is not None

    model = req.model or DEFAULT_MODEL
    prompt, files = await extract_prompt_and_files(req.messages)

    if not prompt and not files:
        raise HTTPException(400, "Empty prompt")

    if req.stream:
        async def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            async for chunk in client.generate_content_stream(prompt=prompt, files=files or None, model=model):
                delta = chunk.text_delta
                if not delta:
                    continue
                payload = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {JSONResponse(content=payload).body.decode()}\n\n"
            done = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {JSONResponse(content=done).body.decode()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await client.generate_content(prompt=prompt, files=files or None, model=model)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
