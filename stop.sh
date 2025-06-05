#!/bin/bash
set -euo pipefail

echo "⏹ Stopping and removing containers..."

docker compose down --remove-orphans

# Optional: remove unused volumes (if you're sure you don't need them)
docker volume prune -f

# Optional: clean dangling images (leftover from old builds)
docker image prune -f

echo "✔ Containers stopped and cleaned up."

