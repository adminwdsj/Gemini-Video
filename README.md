# Gemini-Video

OpenAI-compatible gateway for Gemini Web, powered by `gemini-webapi`.

## Features
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI-style multimodal input via `messages[].content[]`
- Supports file/video analysis through Gemini Web file uploads
- Railway-ready Docker deployment

## Environment Variables
- `GEMINI_1PSID`
- `GEMINI_1PSIDTS`
- `API_KEY`
- `PORT` (default: `8000`)
- `DEFAULT_MODEL` (default: `gemini-3-flash-preview`)
- `GEMINI_COOKIE_PATH` (default: `/data/cookies`)

## Deploy
Use the included `Dockerfile` on Railway.
