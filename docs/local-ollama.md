# Local Ollama Setup

JAR-63 is **local-first**: Ollama is the primary LLM provider and no cloud API
key is required to run the application. This guide covers installing Ollama,
pulling a model, and verifying JAR-63 can reach it.

## 1. Install Ollama

### Linux / WSL

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS

```bash
brew install ollama
```

Or download the installer from <https://ollama.com/download/mac>.

### Windows

Download the installer from <https://ollama.com/download/windows> and run it.
Ollama runs as a background service listening on `http://localhost:11434`.

## 2. Start the Ollama service

```bash
# Linux / macOS / WSL
ollama serve
```

On Windows, the installer starts the service automatically. Verify it is
running:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response with a `models` array.

## 3. Pull a model

JAR-63 never downloads or modifies models itself. Pull the default model
manually:

```bash
ollama pull qwen2.5-coder:7b
```

Other useful models:

```bash
ollama pull qwen2.5-coder:14b   # larger, more capable (needs ~16 GB RAM)
ollama pull llama3.2:3b         # smaller, faster
```

## 4. Configure JAR-63

The defaults in `.env.example` point to `http://localhost:11434` and
`qwen2.5-coder:7b`. To use a different model or URL, copy `.env.example` to
`.env` and edit:

```bash
cp .env.example .env
# Edit .env:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_DEFAULT_MODEL=qwen2.5-coder:7b
#   LLM_DEFAULT_MODEL=qwen2.5-coder:7b
```

## 5. Verify the connection

### Via the API

```bash
# Start the backend (see README for full setup)
cd backend
uvicorn app.main:app --reload

# Check LLM status
curl http://localhost:8000/api/v1/llm/status
```

Expected response (abridged):

```json
{
  "enabled": true,
  "available": true,
  "status": "available",
  "model": "qwen2.5-coder:7b",
  "provider": "ollama",
  "installed_models": ["qwen2.5-coder:7b"],
  "capabilities": ["chat", "coding", "tool_calling"]
}
```

### Via the Phase 5 demo

```bash
cd backend
python -m app.llm.phase5_demos --real
```

### Via the integration tests

```bash
cd backend
OLLAMA_INTEGRATION=1 pytest -m ollama -v
```

## 6. Remote Ollama (optional)

If Ollama runs on a different machine, set:

```bash
OLLAMA_BASE_URL=http://192.168.1.50:11434
```

Ensure Ollama is configured to listen on all interfaces
(`OLLAMA_HOST=0.0.0.0:11434` on the Ollama server).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `status: unavailable` | Ollama not running | `ollama serve` |
| `model_not_found` | Model not pulled | `ollama pull qwen2.5-coder:7b` |
| `timeout` | Slow model load | Increase `LLM_REQUEST_TIMEOUT` |
| `connection_refused` | Wrong URL / firewall | Check `OLLAMA_BASE_URL` and port |
| `/health` shows LLM degraded | Any of the above | System still works via deterministic fallback |
