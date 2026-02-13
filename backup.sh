#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./backup.sh [OPTIONS]

Create a cold backup of Docker volumes and project config into backups/<timestamp>.

Options:
  --output-dir DIR       Backup parent directory (default: backups)
  --name NAME            Backup folder name (default: YYYYMMDD_HHMMSS)
  --no-restart           Do not restart services after backup
  --help                 Show this help text

Environment overrides:
  COMPOSE_FILE           Compose file path (default: docker-compose.yml)
  VOLUME_QDRANT          Volume name (default: mcp-research_qdrant_storage)
  VOLUME_REDIS           Volume name (default: mcp-research_redis_data)
  VOLUME_MINIO           Volume name (default: mcp-research_minio_data)
EOF
}

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
OUTPUT_DIR="backups"
BACKUP_NAME="$(date +%Y%m%d_%H%M%S)"
RESTART_AFTER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --name)
      BACKUP_NAME="${2:-}"
      shift 2
      ;;
    --no-restart)
      RESTART_AFTER=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH" >&2
  exit 1
fi

VOLUME_QDRANT="${VOLUME_QDRANT:-mcp-research_qdrant_storage}"
VOLUME_REDIS="${VOLUME_REDIS:-mcp-research_redis_data}"
VOLUME_MINIO="${VOLUME_MINIO:-mcp-research_minio_data}"
VOLUMES=("$VOLUME_QDRANT" "$VOLUME_REDIS" "$VOLUME_MINIO")

BACKUP_PATH="${OUTPUT_DIR%/}/${BACKUP_NAME}"
if [[ -e "$BACKUP_PATH" ]]; then
  echo "Backup path already exists: $BACKUP_PATH" >&2
  exit 1
fi
mkdir -p "$BACKUP_PATH"

echo "Backing up volumes to: $BACKUP_PATH"
echo "Stopping compose services..."
docker compose -f "$COMPOSE_FILE" down

for volume in "${VOLUMES[@]}"; do
  if ! docker volume inspect "$volume" >/dev/null 2>&1; then
    echo "Volume not found: $volume" >&2
    exit 1
  fi
  echo "Archiving $volume ..."
  docker run --rm \
    -v "$volume":/volume:ro \
    -v "$(pwd)/$BACKUP_PATH":/backup \
    alpine sh -c "cd /volume && tar czf /backup/${volume}.tgz ."
done

echo "Saving compose and environment metadata..."
cp "$COMPOSE_FILE" "$BACKUP_PATH/docker-compose.yml"
if [[ -f .env ]]; then
  cp .env "$BACKUP_PATH/.env"
fi
docker compose -f "$COMPOSE_FILE" config > "$BACKUP_PATH/compose.resolved.yml"
docker image ls --digests > "$BACKUP_PATH/image-digests.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$BACKUP_PATH/backup.created_utc.txt"

if [[ "$RESTART_AFTER" -eq 1 ]]; then
  echo "Restarting compose services..."
  docker compose -f "$COMPOSE_FILE" up -d
fi

echo "Backup complete: $BACKUP_PATH"
