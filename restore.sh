#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./restore.sh [OPTIONS]

Restore Docker volumes from a backup directory.

Options:
  --backup DIR           Backup directory to restore from (default: latest in backups/)
  --latest               Restore from latest backup in backups/ (same as default)
  --compose-file FILE    Compose file path (default: docker-compose.yml)
  --no-start             Do not start services after restore
  --restore-env          Restore .env from backup if present
  --help                 Show this help text

Environment overrides:
  VOLUME_QDRANT          Volume name (default: mcp-research_qdrant_storage)
  VOLUME_REDIS           Volume name (default: mcp-research_redis_data)
  VOLUME_MINIO           Volume name (default: mcp-research_minio_data)
EOF
}

latest_backup_dir() {
  if [[ ! -d backups ]]; then
    return 1
  fi
  local latest
  latest="$(ls -1dt backups/*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  echo "${latest%/}"
}

BACKUP_DIR=""
USE_LATEST=0
COMPOSE_FILE="docker-compose.yml"
START_AFTER=1
RESTORE_ENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup)
      BACKUP_DIR="${2:-}"
      shift 2
      ;;
    --latest)
      USE_LATEST=1
      shift
      ;;
    --compose-file)
      COMPOSE_FILE="${2:-}"
      shift 2
      ;;
    --no-start)
      START_AFTER=0
      shift
      ;;
    --restore-env)
      RESTORE_ENV=1
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

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH" >&2
  exit 1
fi

if [[ "$USE_LATEST" -eq 1 || -z "$BACKUP_DIR" ]]; then
  BACKUP_DIR="$(latest_backup_dir || true)"
fi

if [[ -z "$BACKUP_DIR" ]]; then
  echo "No backup directory selected and no backups found under backups/" >&2
  exit 1
fi

if [[ ! -d "$BACKUP_DIR" ]]; then
  echo "Backup directory not found: $BACKUP_DIR" >&2
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

VOLUME_QDRANT="${VOLUME_QDRANT:-mcp-research_qdrant_storage}"
VOLUME_REDIS="${VOLUME_REDIS:-mcp-research_redis_data}"
VOLUME_MINIO="${VOLUME_MINIO:-mcp-research_minio_data}"
VOLUMES=("$VOLUME_QDRANT" "$VOLUME_REDIS" "$VOLUME_MINIO")

echo "Restoring from: $BACKUP_DIR"
echo "Stopping compose services..."
docker compose -f "$COMPOSE_FILE" down

for volume in "${VOLUMES[@]}"; do
  archive="$BACKUP_DIR/${volume}.tgz"
  if [[ ! -f "$archive" ]]; then
    echo "Missing backup archive: $archive" >&2
    exit 1
  fi
  echo "Restoring $volume ..."
  docker volume create "$volume" >/dev/null
  docker run --rm \
    -v "$volume":/volume \
    -v "$(pwd)/$BACKUP_DIR":/backup:ro \
    alpine sh -c "rm -rf /volume/* /volume/.[!.]* /volume/..?* 2>/dev/null || true; tar xzf /backup/${volume}.tgz -C /volume"
done

if [[ "$RESTORE_ENV" -eq 1 && -f "$BACKUP_DIR/.env" ]]; then
  cp "$BACKUP_DIR/.env" .env
  echo "Restored .env from backup"
fi

if [[ "$START_AFTER" -eq 1 ]]; then
  echo "Starting compose services..."
  docker compose -f "$COMPOSE_FILE" up -d
fi

echo "Restore complete from: $BACKUP_DIR"
