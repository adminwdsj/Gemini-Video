import asyncio
import base64
import json
import os
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
from pydantic import BaseModel
from gemini_webapi import GeminiClient

API_KEY = os.getenv("API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3-flash-preview")
GATEWAY_VERSION = "2026-04-14d"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/gemini-video-uploads"))
UPLOAD_RETENTION_SECONDS = int(os.getenv("UPLOAD_RETENTION_SECONDS", str(3 * 24 * 60 * 60)))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", str(6 * 60 * 60)))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "3"))
PER_ACCOUNT_CONCURRENCY = int(os.getenv("PER_ACCOUNT_CONCURRENCY", "1"))

accounts: list["AccountState"] = []
account_queue: asyncio.Queue[int] | None = None
global_semaphore: asyncio.Semaphore | None = None
cleanup_task: asyncio.Task | None = None
started_at = time.time()
request_counter = 0
total_successes = 0
total_failures = 0
model_counter: Counter[str] = Counter()


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False


@dataclass
class AccountState:
    index: int
    name: str
    psid: str
    psidts: str | None = None
    client: GeminiClient | None = None
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


async def acquire_account() -> AccountState:
    if not accounts:
        raise HTTPException(503, "No Gemini accounts configured")
    if account_queue is None:
        raise HTTPException(503, "Account queue unavailable")
    idx = await account_queue.get()
    state = accounts[idx]
    state.active_requests += 1
    state.requests += 1
    state.last_used_at = time.time()
    return state


async def release_account(state: AccountState):
    state.active_requests = max(0, state.active_requests - 1)
    if account_queue is not None and state.status == "ready":
        await account_queue.put(state.index)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global accounts, account_queue, global_semaphore, cleanup_task
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_uploads()
    cleanup_task = asyncio.create_task(cleanup_loop())
    global_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    account_queue = asyncio.Queue()

    loaded = load_account_configs()
    accounts = [
        AccountState(index=i, name=item["name"] or f"account-{i+1}", psid=item["psid"] or "", psidts=item.get("psidts"))
        for i, item in enumerate(loaded)
    ]

    for state in accounts:
        try:
            state.client = GeminiClient(state.psid, state.psidts or None, proxy=None)
            await state.client.init(timeout=300, auto_close=True, close_delay=180, auto_refresh=True)
            state.models = [m.model_name for m in (state.client.list_models() or []) if m.is_available]
            state.status = "ready"
            for _ in range(PER_ACCOUNT_CONCURRENCY):
                await account_queue.put(state.index)
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
body{font-family:system-ui;padding:24px;max-width:1100px;margin:auto;background:#0b1020;color:#e8ecff}
input,button{padding:8px 10px;border-radius:8px;border:1px solid #3a4267;background:#111936;color:#fff}
button{cursor:pointer}
.card{background:#111936;border:1px solid #2d3558;border-radius:12px;padding:16px;margin:16px 0}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #2d3558;text-align:left;font-size:14px}
.ok{color:#7ee787}.bad{color:#ff7b72}.muted{color:#9aa4d1}
pre{white-space:pre-wrap;word-break:break-word;background:#0b1020;padding:12px;border-radius:8px}
</style></head>
<body>
<h1>Gemini Video Dashboard</h1>
<div class='card'>
<label>API Key: <input id='key' type='password' style='min-width:360px'></label>
<button onclick='loadData()'>刷新</button>
<p class='muted'>如果设置了 API_KEY，请填同一个 key。</p>
</div>
<div id='summary' class='card'></div>
<div id='accounts' class='card'></div>
<script>
async function loadData(){
  const key=document.getElementById('key').value;
  const res=await fetch('/admin/overview?key='+encodeURIComponent(key));
  const data=await res.json();
  document.getElementById('summary').innerHTML=`<h3>概览</h3>
  <pre>${JSON.stringify(data.summary,null,2)}</pre>`;
  const rows=data.accounts.map(a=>`<tr>
    <td>${a.name}</td><td class='${a.status==='ready'?'ok':'bad'}'>${a.status}</td>
    <td>${a.active_requests}</td><td>${a.requests}</td><td>${a.successes}</td><td>${a.failures}</td>
    <td>${a.avg_latency_ms.toFixed(1)}</td><td>${a.models.join(', ')}</td><td>${a.last_error||''}</td></tr>`).join('');
  document.getElementById('accounts').innerHTML=`<h3>账号</h3><table><thead><tr>
    <th>名称</th><th>状态</th><th>处理中</th><th>总请求</th><th>成功</th><th>失败</th><th>均延迟ms</th><th>模型</th><th>最后错误</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
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
    return {
        "summary": {
            "version": GATEWAY_VERSION,
            "uptime_sec": int(time.time() - started_at),
            "configured_accounts": len(accounts),
            "ready_accounts": len(ready),
            "total_requests": request_counter,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "active_requests": sum(a.active_requests for a in accounts),
            "max_concurrency": MAX_CONCURRENCY,
            "per_account_concurrency": PER_ACCOUNT_CONCURRENCY,
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
async def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)):
    global request_counter
    require_api_key(authorization)
    assert global_semaphore is not None

    request_counter += 1
    model = req.model or DEFAULT_MODEL
    prompt, files = await extract_prompt_and_files(req.messages)
    if not prompt and not files:
        raise HTTPException(400, "Empty prompt")

    state = await acquire_account()
    if state.client is None:
        await release_account(state)
        raise HTTPException(503, "No ready Gemini account")

    if req.stream:
        async def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            started = time.time()
            try:
                async with global_semaphore:
                    async for chunk in state.client.generate_content_stream(
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
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        yield f"data: {JSONResponse(content=payload).body.decode()}\n\n"
                mark_success(state, model, (time.time() - started) * 1000)
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
                await release_account(state)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    started = time.time()
    try:
        async with global_semaphore:
            result = await state.client.generate_content(
                prompt=prompt,
                files=files or None,
                model=model,
                temporary=True,
            )
        mark_success(state, model, (time.time() - started) * 1000)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    except Exception as exc:
        mark_failure(state, exc)
        raise HTTPException(502, f"Upstream error: {exc}")
    finally:
        for path in files:
            with suppress(FileNotFoundError):
                os.remove(path)
        await release_account(state)
