#!/usr/bin/env bash
# Convenience scripts for local development.
set -euo pipefail

cd "$(dirname "$0")/.."

case "${1:-help}" in
  install)
    pip install -e ".[dev]"
    ;;
  test)
    pytest "${@:2}"
    ;;
  lint)
    ruff check backend scripts
    ruff format --check backend scripts
    ;;
  format)
    ruff format backend scripts
    ruff check --fix backend scripts
    ;;
  run)
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ;;
  up)
    docker compose up --build
    ;;
  down)
    docker compose down
    ;;
  health)
    curl -fsS http://localhost:8000/health | python -m json.tool
    ;;
  *)
    echo "Usage: $0 {install|test|lint|format|run|up|down|health}"
    exit 1
    ;;
esac
