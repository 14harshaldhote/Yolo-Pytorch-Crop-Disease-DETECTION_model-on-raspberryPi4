#!/usr/bin/env bash
# Sets up the detection API on a Raspberry Pi 4 (64-bit Raspberry Pi OS, Bullseye or Bookworm).
# Usage: ./scripts/install_pi.sh            install into ./.venv
#        ./scripts/install_pi.sh --service  also install and start a systemd service
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This needs 64-bit Raspberry Pi OS (uname -m printed $(uname -m), expected aarch64)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-api.txt

if [[ ! -f .env ]]; then
  umask 077
  echo "API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" > .env
  echo "Created .env with a random API_KEY (readable only by you): $(cut -d= -f2 .env)"
fi

if [[ "${1:-}" == "--service" ]]; then
  sed -e "s#__REPO__#${REPO}#g" -e "s#__USER__#$(id -un)#g" scripts/crop-disease-api.service \
    | sudo tee /etc/systemd/system/crop-disease-api.service > /dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now crop-disease-api
  echo "Service running: curl http://localhost:8000/health"
else
  echo "Start the API with:"
  echo "  set -a; source .env; set +a; .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000"
fi
