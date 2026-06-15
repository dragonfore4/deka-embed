#!/usr/bin/env bash
#
# setup.sh — provision an Ubuntu EC2 box to run the Thai legal-case RAG script.
#
# Installs: NVIDIA driver (only if a GPU is present), Docker + compose,
# uv (Python), Ollama + the embedding model, project deps, and the Postgres
# + pgvector container. Idempotent — safe to re-run.
#
# Usage (run as the normal user, e.g. `ubuntu`, NOT root — it uses sudo itself):
#   chmod +x setup.sh
#   ./setup.sh
#
# Tested on Ubuntu 22.04 / 24.04. On a GPU instance the driver install needs a
# reboot; the script tells you when, then just re-run it to finish.
#
set -euo pipefail

# ----------------------------- config --------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"  # default: this repo
EMBED_MODEL="${EMBED_MODEL:-qwen3-embedding:0.6b}"
INSTALL_GPU_DRIVER="${INSTALL_GPU_DRIVER:-auto}"   # auto | yes | no
DB_URL_DEFAULT="postgresql://postgres:postgres@localhost:5432/legal_db"
# ---------------------------------------------------------------------------

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -ne 0 ] || die "Run as a normal user (e.g. ubuntu), not root. The script uses sudo where needed."

# --------------------------------------------------------------------------
log "1/7  Base system packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git gnupg lsb-release tmux pciutils

# --------------------------------------------------------------------------
log "2/7  GPU detection & NVIDIA driver"
HAS_GPU=no
if lspci 2>/dev/null | grep -qi 'nvidia'; then HAS_GPU=yes; fi

want_driver=no
case "$INSTALL_GPU_DRIVER" in
  yes)  want_driver=yes ;;
  no)   want_driver=no ;;
  auto) want_driver=$HAS_GPU ;;
esac

if [ "$want_driver" = yes ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "    NVIDIA driver already working:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  else
    warn "GPU found but no working driver. Installing — a REBOOT will be required."
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    warn "Driver installed. REBOOT now, then re-run this script to finish:"
    warn "    sudo reboot"
    exit 0
  fi
else
  if [ "$HAS_GPU" = yes ]; then
    warn "GPU present but INSTALL_GPU_DRIVER=no — Ollama will run the model on CPU."
  else
    log "    No GPU detected — Ollama will run on CPU (fine for the 0.6b model, just slower)."
  fi
fi

# --------------------------------------------------------------------------
log "3/7  Docker + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  warn "Added $USER to the 'docker' group. You may need to log out/in (or run 'newgrp docker') for non-sudo docker."
else
  log "    Docker already installed: $(docker --version)"
fi
# Use sudo for docker in this script so it works even before the group change takes effect.
DOCKER="sudo docker"

# --------------------------------------------------------------------------
log "4/7  uv (Python toolchain)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv not on PATH. Add \$HOME/.local/bin to PATH and re-run."
log "    $(uv --version)"

# --------------------------------------------------------------------------
log "5/7  Ollama + embedding model"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
# The installer registers a systemd service that auto-detects the GPU.
sudo systemctl enable --now ollama 2>/dev/null || true
sleep 3
log "    Pulling embedding model: $EMBED_MODEL"
ollama pull "$EMBED_MODEL"

# --------------------------------------------------------------------------
log "6/7  Project deps & .env"
cd "$PROJECT_DIR"
[ -f main.py ] || die "main.py not found in $PROJECT_DIR — run this from the repo (or set PROJECT_DIR)."
uv sync

if [ ! -f .env ]; then
  if [ -f .env.example ]; then cp .env.example .env; else printf 'DATABASE_URL=%s\nCOLLECTION_NAME=legal_cases\n' "$DB_URL_DEFAULT" > .env; fi
  log "    Created .env (edit if your DB creds differ)."
else
  log "    .env already exists — leaving it."
fi

# --------------------------------------------------------------------------
log "7/7  Postgres + pgvector container"
# Guard: compose mounts a *.sql seed into /docker-entrypoint-initdb.d. That seed
# is loaded ONLY when pgdata is empty (first boot). Make sure the referenced file
# exists, otherwise docker turns the missing bind-mount into an empty dir and init breaks.
SEED=$(grep -oE '\./[^:]+\.sql:/docker-entrypoint-initdb\.d' compose.yaml | head -1 | sed 's/:.*//' || true)
if [ -n "${SEED:-}" ] && [ ! -f "${SEED#./}" ]; then
  warn "compose.yaml seeds from '$SEED' but that file is missing here."
  warn "Options: (a) copy the seed file up (scp), or"
  warn "         (b) edit compose.yaml to point at legal_db_2005.sql (committed, 174MB), or"
  warn "         (c) comment out the seed line to start with an EMPTY DB and ingest fresh."
  die  "Fix the seed reference, then re-run."
fi

$DOCKER compose up -d db
log "    Waiting for Postgres to accept connections..."
for i in $(seq 1 60); do
  if $DOCKER exec legal-pgvector pg_isready -U postgres -d legal_db >/dev/null 2>&1; then
    log "    Postgres ready."
    break
  fi
  sleep 2
  [ "$i" -lt 60 ] || die "Postgres did not become ready in time. Check: $DOCKER compose logs db"
done

# --------------------------------------------------------------------------
cat <<EOF

Done. Environment is ready.

Next steps:
  1. (Chat model) The analysis step uses 'gemma4:31b-cloud', a CLOUD model via
     ollama.com. Sign the box in to your Ollama account so cloud calls work:
         ollama signin        # follow the link, then verify: ollama run gemma4:31b-cloud "hi"
     (Embedding/ingest works WITHOUT this — only analyze_case needs it.)

  2. Put your PDFs under ./documents/ (nested folders are fine), then ingest.
     Run it in tmux so an SSH drop doesn't kill it; it's resumable (skip_existing):
         tmux new -s ingest
         uv run python main.py
         # detach: Ctrl-b then d   |   reattach: tmux attach -t ingest

  3. Useful:
         $DOCKER compose logs -f db                 # DB logs
         $DOCKER exec -it legal-pgvector psql -U postgres -d legal_db
         ollama ps                                  # see if the model is on GPU/CPU

GPU: $( [ "$HAS_GPU" = yes ] && echo "present (Ollama will use it)" || echo "none (running on CPU)" )
EOF
