#!/bin/bash
set -euo pipefail

echo "Fetching latest code..."
git -C "$(dirname "$0")/open-webui" pull
git -C "$(dirname "$0")/open-webui-pipelines" pull

echo "Rebuilding and starting containers..."
docker compose down --remove-orphans
docker compose build
docker compose up -d

echo "✔ All containers up to date and running."
