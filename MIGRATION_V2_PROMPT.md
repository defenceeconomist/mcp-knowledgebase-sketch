# Migration Prompt: Redis/Qdrant/BibTeX v2 Schema Rollout

Use this prompt with a coding agent to execute the migration safely in phased gates.

## Prompt To Run
You are migrating this repository from v1 storage schema to v2 schema with zero data loss and rollback support.

### Objective
Implement and roll out v2 keys/payloads while preserving compatibility:
- Redis v2 doc/source/partition/chunk model
- Qdrant payload v2 fields (`doc_hash`, `partition_hash`, `chunk_hash`, `source_id`)
- BibTeX v2 keyed by document hash
- Deterministic Qdrant point IDs after parity validation

### Non-Negotiable Constraints
1. Never remove v1 reads/writes until v2 parity is proven.
2. Use flags for all behavior switches; default behavior must remain current behavior.
3. Every rollout gate must pass acceptance checks before advancing.
4. On any gate failure, stop and rollback to the prior gate.

### Canonical v2 Keys (exact names)
Assume:
- `{RP}` = `REDIS_PREFIX`
- `{BP}` = `BIBTEX_REDIS_PREFIX`
- `doc_hash` = file content sha256
- `source_id` = sha1(`s3://{bucket}/{key}?version_id={version_id or ''}`)
- `partition_hash`, `chunk_hash` = deterministic hashes

Redis v2:
- `{RP}:v2:doc_hashes`
- `{RP}:v2:doc:{doc_hash}:meta`
- `{RP}:v2:doc:{doc_hash}:sources`
- `{RP}:v2:doc:{doc_hash}:collections`
- `{RP}:v2:source:{source_id}:meta`
- `{RP}:v2:source:{source_id}:doc`
- `{RP}:v2:doc:{doc_hash}:partition_hashes`
- `{RP}:v2:partition:{partition_hash}`
- `{RP}:v2:partition:{partition_hash}:chunk_hashes`
- `{RP}:v2:doc:{doc_hash}:chunk_hashes`
- `{RP}:v2:chunk:{chunk_hash}`

BibTeX v2:
- `{BP}:v2:doc:{doc_hash}`
- `{BP}:v2:source:{source_id}:doc`

### Required Rollout Flags (exact names)
- `REDIS_SCHEMA_WRITE_MODE=v1|dual|v2`
- `REDIS_SCHEMA_READ_MODE=v1|prefer_v2|v2`
- `QDRANT_PAYLOAD_SCHEMA=v1|dual|v2`
- `QDRANT_POINT_ID_MODE=uuid|deterministic`
- `SEARCH_REDIS_ENRICHMENT=off|partition|document|both`
- `SEARCH_LINK_MODE=legacy|dual`
- `BIBTEX_SCHEMA_WRITE_MODE=v1|dual|v2`
- `BIBTEX_SCHEMA_READ_MODE=v1|prefer_v2|v2`

### Module Compatibility Requirements
- `src/mcp_research/ingest_unstructured.py`: dual write support, v2 skip-processed check
- `src/mcp_research/minio_ingest.py`: v2 source mapping read/write + collection re-upsert safety
- `src/mcp_research/upsert_chunks.py`: v2 Redis input + dual payload output + deterministic ID mode
- `src/mcp_research/mcp_app.py`: prefer_v2 reads with v1 fallback
- `src/mcp_research/bibtex_autofill.py`: dual write v1+v2
- `src/mcp_research/bibtex_ui_app.py`: prefer_v2 read with v1 fallback
- `src/mcp_research/link_resolver.py` / `src/mcp_research/resolver_app.py`: support dual link output from search layer

### Deliverables
1. Code changes for flags + compatibility logic
2. Backfill scripts for v2 data (`redis`, `qdrant`, `bibtex`)
3. Validator script that compares v1 vs v2 parity
4. Rollout runbook updates
5. Test updates/additions

## Acceptance Gates (exact checks)

### Gate 0: Backup + Baseline (must pass before code cutover)
Pass criteria:
1. Backup snapshot created with all 3 volume archives present
2. Baseline Redis collections validation runs without runtime errors
3. Baseline Qdrant duplicate scan runs without runtime errors

Command templates:
```bash
TS=$(date +%Y%m%d_%H%M%S)
./backup.sh --name pre_v2_${TS}
test -f backups/pre_v2_${TS}/mcp-research_qdrant_storage.tgz
test -f backups/pre_v2_${TS}/mcp-research_redis_data.tgz
test -f backups/pre_v2_${TS}/mcp-research_minio_data.tgz

docker compose exec -T mcp python validate_collections.py --redis-url "${REDIS_URL:-redis://redis:6379/0}"
docker compose exec -T mcp python dedupe_qdrant_chunks.py --all-collections
```

### Gate 1: Dual Write Enabled (reads still v1)
Flags:
- `REDIS_SCHEMA_WRITE_MODE=dual`
- `REDIS_SCHEMA_READ_MODE=v1`
- `QDRANT_PAYLOAD_SCHEMA=dual`
- `QDRANT_POINT_ID_MODE=uuid`
- `BIBTEX_SCHEMA_WRITE_MODE=dual`
- `BIBTEX_SCHEMA_READ_MODE=v1`

Pass criteria:
1. New ingests write v1 and v2 keys
2. Existing APIs unchanged for consumers

Command templates:
```bash
docker compose up -d --build mcp minio_ingest bibtex_ui resolver search_ui
# ingest one known PDF path/object through normal flow
# then verify both schemas exist for same document
docker compose exec -T redis redis-cli --raw KEYS "${REDIS_PREFIX:-unstructured}:pdf:*:meta" | head
docker compose exec -T redis redis-cli --raw KEYS "${REDIS_PREFIX:-unstructured}:v2:doc:*:meta" | head
```

### Gate 2: Backfill + Parity
Pass criteria:
1. v2 doc count equals v1 doc count
2. For sampled docs, chunk counts match between v1 and v2
3. Qdrant payload has required v2 fields for sampled points

Command templates:
```bash
docker compose exec -T mcp python backfill_source_index.py --redis-url "${REDIS_URL:-redis://redis:6379/0}"
docker compose exec -T mcp python backfill_collections.py --redis-url "${REDIS_URL:-redis://redis:6379/0}"
docker compose exec -T mcp python backfill_qdrant_metadata.py --url "${QDRANT_URL:-http://qdrant:6333}" --collection "${QDRANT_COLLECTION:-pdf_chunks}"

# run new parity validator (to be added in migration)
docker compose exec -T mcp python scripts/validate_schema_parity.py --sample-size 200 --fail-on-mismatch
```

### Gate 3: Read Cutover (prefer_v2)
Flags:
- `REDIS_SCHEMA_READ_MODE=prefer_v2`
- `BIBTEX_SCHEMA_READ_MODE=prefer_v2`

Pass criteria:
1. Search/fetch APIs return same functional results as Gate 2
2. `fetch_chunk_partition` uses v2 direct lookup when available
3. No increase in error rate

Command templates:
```bash
docker compose up -d mcp resolver search_ui bibtex_ui
docker compose exec -T mcp python -m pytest tests/test_mcp_app.py tests/test_link_resolver.py tests/test_bibtex_ui_app.py
```

### Gate 4: Deterministic IDs + Dedupe
Flags:
- `QDRANT_POINT_ID_MODE=deterministic`

Pass criteria:
1. Re-ingest does not increase point count for unchanged docs
2. Dedupe dry-run finds zero or expected historical duplicates

Command templates:
```bash
# ingest same test object twice, compare point count before/after
docker compose exec -T mcp python dedupe_qdrant_chunks.py --all-collections
docker compose exec -T mcp python dedupe_qdrant_chunks.py --all-collections --apply
```

### Gate 5: v2-only
Flags:
- `REDIS_SCHEMA_WRITE_MODE=v2`
- `REDIS_SCHEMA_READ_MODE=v2`
- `QDRANT_PAYLOAD_SCHEMA=v2`
- `BIBTEX_SCHEMA_WRITE_MODE=v2`
- `BIBTEX_SCHEMA_READ_MODE=v2`

Pass criteria:
1. All tests pass
2. Runtime smoke checks pass
3. Rollback tested with `restore.sh`

Command templates:
```bash
docker compose exec -T mcp python -m pytest
./restore.sh --latest --no-start
docker compose up -d
```

## Rollback Procedure (always available)
```bash
./restore.sh --latest --restore-env
docker compose up -d
```

## Required Reporting Format
For each gate, report:
1. Commands run
2. Raw outputs (or key excerpts)
3. Pass/fail per acceptance criterion
4. Go/no-go decision
