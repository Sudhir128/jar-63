# Windows Development Guide

This guide covers running and developing JAR-63 on Windows. The backend runs
natively on Windows via Python; Docker Desktop is used for PostgreSQL and Redis.

## Prerequisites

- **Python 3.13+** — <https://www.python.org/downloads/windows/>
  - During install, check "Add Python to PATH".
- **Docker Desktop** — <https://www.docker.com/products/docker-desktop>
  - Start Docker Desktop before running `docker compose`.
- **Git** — <https://git-scm.com/download/win>
- **Ollama** (optional, for local LLM) — <https://ollama.com/download/windows>

## 1. Clone and enter the repository

```powershell
git clone <repo-url> jar-63
cd jar-63
```

## 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 3. Install dependencies

```powershell
cd backend
pip install -e ".[dev]"
```

## 4. Configure environment

```powershell
cd ..
copy .env.example .env
```

Edit `.env` as needed. For local LLM, ensure Ollama is running (see
`docs/local-ollama.md`).

## 5. Start infrastructure (PostgreSQL + Redis)

Start Docker Desktop, then:

```powershell
docker compose up -d postgres redis
```

Verify:

```powershell
docker compose ps
```

## 6. Run the backend

```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> for the interactive API.

## 7. Run tests

```powershell
cd backend
$env:APP_ENV="testing"
python -m pytest
```

## 8. Run Phase 5 demos

```powershell
cd backend
python -m app.llm.phase5_demos
```

## Windows-specific notes

### Line endings

The repository uses LF line endings. Configure Git to avoid CRLF conversion:

```powershell
git config --global core.autocrlf input
```

### Paths

JAR-63 uses `pathlib.Path` throughout, so Windows backslash paths work
correctly. No hardcoded Unix paths exist in the codebase.

### Docker ports

Docker Compose exposes PostgreSQL on `5432` and Redis on `6379`. If these
ports are already in use, edit `docker-compose.yml` to map different host
ports.

### Ollama on Windows

Ollama runs as a background service on Windows. It listens on
`http://localhost:11434` by default. No WSL is required for Ollama itself.

### WSL2 (alternative)

If you prefer a Linux environment, you can run the entire stack inside WSL2:

```powershell
wsl --install
```

Then follow the Linux instructions in the main README inside WSL.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `docker compose` not found | Start Docker Desktop |
| `Activate.ps1` cannot be loaded | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip install` fails | Upgrade pip: `python -m pip install --upgrade pip` |
| Port 8000 in use | Change `--port` in the uvicorn command |
| Ollama connection refused | Ensure Ollama service is running (check system tray) |
