import asyncio
import base64
import json
import os
import re
import tempfile
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from gemini_webapi import GeminiClient
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3-flash-preview")
GATEWAY_VERSION = "2026-04-14e"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/gemini-video-uploads"))
UPLOAD_RETENTION_SECONDS = int(os.getenv("UPLOAD_RETENTION_SECONDS", str(3 * 24 * 60 * 60)))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", str(6 * 60 * 60)))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
PER_ACCOUNT_CONCURRENCY = int(os.getenv("PER_ACCOUNT_CONCURRENCY", "1"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60)))
MAX_CONTINUATIONS = int(os.getenv("MAX_CONTINUATIONS", "2"))
CONTINUATION_MIN_CHARS = int(os.getenv("CONTINUATION_MIN_CHARS", "1200"))

accounts: list["AccountState"] = []
conversations: dict[str, "ConversationState"] = {}
global_semaphore: asyncio.Semaphore | None = None
cleanup_task: asyncio.Task | None = None
selection_lock = asyncio.Lock()
conversation_lock = asyncio.Lock()
started_at = time.time()
request_counter = 0
total_successes = 0
total_failures = 0
total_continuations = 0
model_counter: Counter[str] = Counter()


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    conversation_id: str | None = None
    max_continuations: int | None = None
    user: str | None = None


@dataclass
class AccountState:
    index: int
    name: str
    psid: str
    psidts: str | None = None
    client: GeminiClient | None = None
    semaphore: asyncio.Semaphore | None = None
    models: list[str] = field(default_factory=list)
    status: str = "initializing"
    requests: int = 0
    successes: int = 0
    failures: int = 0
    active_requests: int = 0
    last_used_at: float | None = None
    last_error: str = ""
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.successes if self.successes else 0.0


@dataclass
class ConversationState:
    conversation_id: str
    account_index: int
    chat: Any
    model: str
    created_at: float
    last_used_at: float
    requests: int = 0
    continuations: int = 0
    active_requests: int = 0


def load_account_configs() -> list[dict[str, str | None]]:
    raw = os.getenv("GEMINI_ACCOUNTS_JSON", "").strip()
    if raw:
        data = json.loads(raw)
        return [
            {
                "name": item.get("name") or f"account-{idx + 1}",
                "psid": item.get("psid") or item.get("__Secure-1PSID") or "",
                "psidts": item.get("psidts") or item.get("__Secure-1PSIDTS") or None,
            }
            for idx, item in enumerate(data)
            if item.get("psid") or item.get("__Secure-1PSID")
        ]

    result: list[dict[str, str | None]] = []
    for idx in range(1, 10):
        psid = os.getenv(f"GEMINI_{idx}PSID", "").strip()
        if not psid:
            continue
        result.append(
            {
                "name": os.getenv(f"GEMINI_{idx}_NAME", f"account-{idx}"),
                "psid": psid,
                "psidts": os.getenv(f"GEMINI_{idx}PSIDTS", "").strip() or None,
            }
        )
    return result


def require_api_key(auth_header: str | None):
    if not API_KEY:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    token = auth_header.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(401, "Unauthorized")


def require_dashboard_key(key: str | None):
    if API_KEY and key != API_KEY:
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
        if path.is_file():
            with suppress(FileNotFoundError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()


async def cleanup_expired_conversations() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    async with conversation_lock:
        expired = [
            cid for cid, conv in conversations.items() if conv.active_requests == 0 and conv.last_used_at < cutoff
        ]
        for cid in expired:
            conversations.pop(cid, None)


async def cleanup_loop() -> None:
    while True:
        cleanup_old_uploads()
        await cleanup_expired_conversations()
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
                        fd, temp_path = tempfile.mkstemp(
                            prefix="gemini_upload_",
                            suffix=f".{ext}",
                            dir=str(UPLOAD_DIR),
                        )
                        os.close(fd)
                        with open(temp_path, "wb") as f:
                            f.write(data)
                        files.append(temp_path)
                    else:
                        text_parts.append(f"[External file URL: {url}]")

    return "\n".join([p for p in text_parts if p]).strip(), files


def should_continue(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped == "<DONE>":
        return False
    if stripped.count("```") % 2 == 1:
        return True
    if re.search(r"<[^>\n]*$", stripped):
        return True
    if len(stripped) >= CONTINUATION_MIN_CHARS and stripped[-1] not in ".!?。！？'\"”】）》`":
        return True
    if stripped.endswith((":", ",", "，", "、", "；", ";", " and", " or", " of", " to", "是", "的")):
        return True
    return False


def build_continuation_prompt(existing_text: str) -> str:
    tail = existing_text[-4000:]
    return (
        "Continue your previous answer from exactly where it stopped.\n\n"
        "Rules:\n"
        "- Do not restart from the beginning.\n"
        "- Do not repeat content already written.\n"
        "- Continue with the next sentence or next token only.\n"
        "- Preserve the same language, tone, structure, Markdown, JSON, XML, and code fences.\n"
        "- If the previous answer is already complete, reply exactly with <DONE>.\n\n"
        f"Last answer tail:\n{tail}"
    )


def deduplicate_continuation(existing_text: str, new_text: str) -> str:
    candidate = new_text.lstrip()
    if candidate.strip() == "<DONE>":
        return ""
    tail = existing_text[-8000:]
    max_overlap = min(len(tail), len(candidate))
    for size in range(max_overlap, 20, -1):
        if tail.endswith(candidate[:size]):
            return candidate[size:]
    return candidate


async def reserve_account(preferred_index: int | None = None) -> AccountState:
    async with selection_lock:
        if preferred_index is not None:
            state = accounts[preferred_index]
            if state.status != "ready" or state.semaphore is None or state.client is None:
                raise HTTPException(503, f"Account {state.name} unavailable")
        else:
            ready = [a for a in accounts if a.status == "ready" and a.semaphore and a.client]
            if not ready:
                raise HTTPException(503, "No Gemini accounts configured")
            state = min(ready, key=lambda a: (a.active_requests, a.requests, a.index))
        state.active_requests += 1
        state.requests += 1
        state.last_used_at = time.time()
        return state


async def release_account(state: AccountState):
    state.active_requests = max(0, state.active_requests - 1)


def mark_success(state: AccountState, model: str, latency_ms: float):
    global total_successes
    total_successes += 1
    state.successes += 1
    state.total_latency_ms += latency_ms
    model_counter[model] += 1
    state.last_error = ""


def mark_failure(state: AccountState, error: Exception):
    global total_failures
    total_failures += 1
    state.failures += 1
    state.last_error = str(error)[:500]


async def get_or_create_conversation(conversation_id: str, model: str) -> ConversationState:
    async with conversation_lock:
        conv = conversations.get(conversation_id)
        if conv:
            conv.last_used_at = time.time()
            conv.model = model
            conv.chat.model = model
            return conv

        ready = [a for a in accounts if a.status == "ready" and a.client]
        if not ready:
            raise HTTPException(503, "No ready Gemini account")
        state = min(ready, key=lambda a: (a.active_requests, a.requests, a.index))
        chat = state.client.start_chat(model=model)
        conv = ConversationState(
            conversation_id=conversation_id,
            account_index=state.index,
            chat=chat,
            model=model,
            created_at=time.time(),
            last_used_at=time.time(),
        )
        conversations[conversation_id] = conv
        return conv


async def touch_conversation(conv: ConversationState, continuation_count: int):
    global total_continuations
    async with conversation_lock:
        conv.last_used_at = time.time()
        conv.requests += 1
        conv.continuations += continuation_count
        total_continuations += continuation_count


def get_requested_conversation_id(req: ChatRequest, x_conversation_id: str | None) -> str | None:
    return req.conversation_id or x_conversation_id or req.user


async def run_nonstream_request(
    state: AccountState,
    prompt: str,
    files: list[str],
    model: str,
    conversation: ConversationState | None,
    max_continuations: int,
) -> tuple[str, int]:
    assert state.client is not None
    chat = conversation.chat if conversation else state.client.start_chat(model=model)

    output = await chat.send_message(prompt=prompt, files=files or None, temporary=True)
    full_text = output.text
    continuation_count = 0

    while continuation_count < max_continuations and should_continue(full_text):
        more = await chat.send_message(build_continuation_prompt(full_text), temporary=True)
        piece = deduplicate_continuation(full_text, more.text)
        if not piece.strip():
            break
        full_text += piece
        continuation_count += 1

    return full_text, continuation_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    global accounts, global_semaphore, cleanup_task
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_uploads()
    cleanup_task = asyncio.create_task(cleanup_loop())
    global_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    loaded = load_account_configs()
    accounts = [
        AccountState(index=i, name=item["name"] or f"account-{i+1}", psid=item["psid"] or "", psidts=item.get("psidts"))
        for i, item in enumerate(loaded)
    ]

    for state in accounts:
        try:
            state.client = GeminiClient(state.psid, state.psidts or None, proxy=None)
            state.semaphore = asyncio.Semaphore(PER_ACCOUNT_CONCURRENCY)
            await state.client.init(timeout=300, auto_close=True, close_delay=180, auto_refresh=True)
            state.models = [m.model_name for m in (state.client.list_models() or []) if m.is_available]
            state.status = "ready"
        except Exception as exc:
            state.status = "error"
            state.last_error = str(exc)[:500]

    try:
        yield
    finally:
        if cleanup_task:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def dashboard():
    return HTMLResponse(
        """
<!doctype html>
<html><head><meta charset='utf-8'><title>Gemini Video Dashboard</title>
<style>
body{font-family:system-ui;padding:24px;max-width:1200px;margin:auto;background:#0b1020;color:#e8ecff}
input,button{padding:8px 10px;border-radius:8px;border:1px solid #3a4267;background:#111936;color:#fff}
button{cursor:pointer}.card{background:#111936;border:1px solid #2d3558;border-radius:12px;padding:16px;margin:16px 0}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #2d3558;text-align:left;font-size:14px}
.ok{color:#7ee787}.bad{color:#ff7b72}.muted{color:#9aa4d1}pre{white-space:pre-wrap;word-break:break-word;background:#0b1020;padding:12px;border-radius:8px}
</style></head><body>
<h1>Gemini Video Dashboard</h1>
<div class='card'><label>API Key: <input id='key' type='password' style='min-width:360px'></label> <button onclick='loadData()'>刷新</button></div>
<div id='summary' class='card'></div><div id='accounts' class='card'></div><div id='conversations' class='card'></div>
<script>
async function loadData(){
 const key=document.getElementById('key').value;
 const res=await fetch('/admin/overview?key='+encodeURIComponent(key));
 const data=await res.json();
 document.getElementById('summary').innerHTML=`<h3>概览</h3><pre>${JSON.stringify(data.summary,null,2)}</pre>`;
 const rows=data.accounts.map(a=>`<tr><td>${a.name}</td><td class='${a.status==='ready'?'ok':'bad'}'>${a.status}</td><td>${a.active_requests}</td><td>${a.requests}</td><td>${a.successes}</td><td>${a.failures}</td><td>${a.avg_latency_ms.toFixed(1)}</td><td>${a.models.join(', ')}</td><td>${a.last_error||''}</td></tr>`).join('');
 document.getElementById('accounts').innerHTML=`<h3>账号</h3><table><thead><tr><th>名称</th><th>状态</th><th>处理中</th><th>总请求</th><th>成功</th><th>失败</th><th>均延迟ms</th><th>模型</th><th>最后错误</th></tr></thead><tbody>${rows}</tbody></table>`;
 const convRows=data.conversations.map(c=>`<tr><td>${c.conversation_id}</td><td>${c.account_name}</td><td>${c.model}</td><td>${c.requests}</td><td>${c.continuations}</td><td>${c.active_requests}</td><td>${c.last_used_at||''}</td></tr>`).join('');
 document.getElementById('conversations').innerHTML=`<h3>会话</h3><table><thead><tr><th>ID</th><th>账号</th><th>模型</th><th>请求</th><th>续写</th><th>处理中</th><th>最后使用</th></tr></thead><tbody>${convRows}</tbody></table>`;
}
setInterval(loadData, 5000);
</script></body></html>
        """
    )


@app.get("/healthz")
async def healthz():
    ready = sum(1 for a in accounts if a.status == "ready")
    return {"status": "ok", "version": GATEWAY_VERSION, "ready_accounts": ready}


@app.get("/admin/overview")
async def admin_overview(key: str | None = Query(default=None)):
    require_dashboard_key(key)
    ready = [a for a in accounts if a.status == "ready"]
    async with conversation_lock:
        convs = list(conversations.values())
    return {
        "summary": {
            "version": GATEWAY_VERSION,
            "uptime_sec": int(time.time() - started_at),
            "configured_accounts": len(accounts),
            "ready_accounts": len(ready),
            "total_requests": request_counter,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "total_continuations": total_continuations,
            "active_requests": sum(a.active_requests for a in accounts),
            "active_conversations": len(convs),
            "max_concurrency": MAX_CONCURRENCY,
            "per_account_concurrency": PER_ACCOUNT_CONCURRENCY,
            "session_ttl_sec": SESSION_TTL_SECONDS,
            "upload_retention_days": round(UPLOAD_RETENTION_SECONDS / 86400, 2),
            "model_counts": dict(model_counter),
        },
        "accounts": [
            {
                "name": a.name,
                "status": a.status,
                "requests": a.requests,
                "successes": a.successes,
                "failures": a.failures,
                "active_requests": a.active_requests,
                "avg_latency_ms": a.avg_latency_ms,
                "last_used_at": a.last_used_at,
                "last_error": a.last_error,
                "models": a.models,
            }
            for a in accounts
        ],
        "conversations": [
            {
                "conversation_id": c.conversation_id,
                "account_name": accounts[c.account_index].name,
                "model": c.model,
                "requests": c.requests,
                "continuations": c.continuations,
                "active_requests": c.active_requests,
                "last_used_at": c.last_used_at,
            }
            for c in convs
        ],
    }


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    model_ids = sorted({m for a in accounts if a.status == "ready" for m in a.models})
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "google-gemini-web"}
            for model_id in model_ids
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest,
    authorization: str | None = Header(default=None),
    x_conversation_id: str | None = Header(default=None, alias="X-Conversation-Id"),
):
    global request_counter
    require_api_key(authorization)
    assert global_semaphore is not None

    request_counter += 1
    model = req.model or DEFAULT_MODEL
    max_continuations = req.max_continuations if req.max_continuations is not None else MAX_CONTINUATIONS
    requested_conversation_id = get_requested_conversation_id(req, x_conversation_id)
    prompt, files = await extract_prompt_and_files(req.messages)
    if not prompt and not files:
        raise HTTPException(400, "Empty prompt")

    conversation = await get_or_create_conversation(requested_conversation_id, model) if requested_conversation_id else None
    preferred_index = conversation.account_index if conversation else None
    state = await reserve_account(preferred_index)
    if state.semaphore is None or state.client is None:
        await release_account(state)
        raise HTTPException(503, "No ready Gemini account")

    if conversation:
        async with conversation_lock:
            conversation.active_requests += 1

    if req.stream:
        async def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            full_text = ""
            continuation_count = 0
            started = time.time()
            try:
                chat = conversation.chat if conversation else state.client.start_chat(model=model)
                async with global_semaphore, state.semaphore:
                    async for chunk in chat.send_message_stream(prompt=prompt, files=files or None, temporary=True):
                        delta = chunk.text_delta
                        full_text += delta
                        if not delta:
                            continue
                        payload = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        yield f"data: {JSONResponse(content=payload).body.decode()}\n\n"

                    while continuation_count < max_continuations and should_continue(full_text):
                        more = await chat.send_message(build_continuation_prompt(full_text), temporary=True)
                        piece = deduplicate_continuation(full_text, more.text)
                        if not piece.strip():
                            break
                        full_text += piece
                        continuation_count += 1
                        payload = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                        }
                        yield f"data: {JSONResponse(content=payload).body.decode()}\n\n"

                mark_success(state, model, (time.time() - started) * 1000)
                if conversation:
                    await touch_conversation(conversation, continuation_count)
                done = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {JSONResponse(content=done).body.decode()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                mark_failure(state, exc)
                raise
            finally:
                for path in files:
                    with suppress(FileNotFoundError):
                        os.remove(path)
                if conversation:
                    async with conversation_lock:
                        conversation.active_requests = max(0, conversation.active_requests - 1)
                        conversation.last_used_at = time.time()
                await release_account(state)

        headers = {}
        if requested_conversation_id:
            headers["X-Conversation-Id"] = requested_conversation_id
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    started = time.time()
    try:
        async with global_semaphore, state.semaphore:
            text, continuation_count = await run_nonstream_request(
                state=state,
                prompt=prompt,
                files=files,
                model=model,
                conversation=conversation,
                max_continuations=max_continuations,
            )
        mark_success(state, model, (time.time() - started) * 1000)
        if conversation:
            await touch_conversation(conversation, continuation_count)
        response = JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "conversation_id": requested_conversation_id,
                "continuations": continuation_count,
            }
        )
        if requested_conversation_id:
            response.headers["X-Conversation-Id"] = requested_conversation_id
        return response
    except Exception as exc:
        mark_failure(state, exc)
        raise HTTPException(502, f"Upstream error: {exc}")
    finally:
        for path in files:
            with suppress(FileNotFoundError):
                os.remove(path)
        if conversation:
            async with conversation_lock:
                conversation.active_requests = max(0, conversation.active_requests - 1)
                conversation.last_used_at = time.time()
        await release_account(state)
