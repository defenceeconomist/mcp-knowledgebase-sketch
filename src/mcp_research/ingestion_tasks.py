from __future__ import annotations

from mcp_research.celery_app import celery_app


@celery_app.task(bind=True, name="mcp_research.ingest_minio_object")
def ingest_minio_object_task(
    self,
    bucket: str,
    object_name: str,
    version_id: str | None = None,
) -> dict:
    """Ingest a MinIO object into Qdrant/Redis as a Celery task."""
    from mcp_research.minio_ingest import process_object_from_env

    process_object_from_env(bucket=bucket, object_name=object_name, version_id=version_id)
    return {
        "bucket": bucket,
        "object_name": object_name,
        "version_id": version_id,
        "status": "completed",
    }


@celery_app.task(bind=True, name="mcp_research.delete_minio_object")
def delete_minio_object_task(
    self,
    bucket: str,
    object_name: str,
    version_id: str | None = None,
) -> dict:
    """Delete a MinIO object's vectors/metadata as a Celery task."""
    from mcp_research.minio_ingest import delete_object_from_env

    delete_object_from_env(bucket=bucket, object_name=object_name, version_id=version_id)
    return {
        "bucket": bucket,
        "object_name": object_name,
        "version_id": version_id,
        "status": "completed",
    }
