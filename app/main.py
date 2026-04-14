import asyncio
import base64
import io
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from gemini_webapi import GeminiClient

API_KEY = os.getenv("API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3-flash-preview")
GATEWAY_VERSION = "2026-04-14c"
GEMINI_1PSID = os.getenv("GEMINI_1PSID", "")
GEMINI_1PSIDTS = os.getenv("GEMINI_1PSIDTS", "")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/gemini-video-uploads"))
UPLOAD_RETENTION_SECONDS = int(os.getenv("UPLOAD_RETENTION_SECONDS", str(3 * 24 * 60 * 60)))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", str(6 * 60 * 60)))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))

client: GeminiClient | None = None
generate_semaphore: asyncio.Semaphore | None = None
cleanup_task: asyncio.Task | None = None


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


def cleanup_old_uploads() -> None:
    if not UPLOAD_DIR.exists():
        return
    cutoff = time.time() - UPLOAD_RETENTION_SECONDS
    for path in UPLOAD_DIR.iterdir():
        if not path.is_file():
            continue
        with suppress(FileNotFoundError):
            if path.stat().st_mtime < cutoff:
                path.unlink()


async def cleanup_loop() -> None:
    while True:
        cleanup_old_uploads()
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


async def extract_prompt_and_files(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    files: list[str] = []

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
                        fd, temp_path = tempfile.mkstemp(prefix="gemini_upload_", suffix=f".{ext}", dir=str(UPLOAD_DIR))
                        os.close(fd)
                        with open(temp_path, "wb") as f:
                            f.write(data)
                        files.append(temp_path)
                    else:
                        text_parts.append(f"[External file URL: {url}]")

    return "\n".join([p for p in text_parts if p]).strip(), files


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, generate_semaphore, cleanup_task
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_uploads()
    generate_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    cleanup_task = asyncio.create_task(cleanup_loop())
    client = GeminiClient(GEMINI_1PSID, GEMINI_1PSIDTS or None, proxy=None)
    await client.init(timeout=300, auto_close=True, close_delay=180, auto_refresh=True)
    try:
        yield
    finally:
        if cleanup_task:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": GATEWAY_VERSION}


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
    assert generate_semaphore is not None

    if req.stream:
        async def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            try:
                async with generate_semaphore:
                    async for chunk in client.generate_content_stream(
                        prompt=prompt,
                        files=files or None,
                        model=model,
                        temporary=True,
                    ):
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
            finally:
                for path in files:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        async with generate_semaphore:
            result = await client.generate_content(prompt=prompt, files=files or None, model=model, temporary=True)
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
    finally:
        for path in files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
