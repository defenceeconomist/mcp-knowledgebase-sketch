from __future__ import annotations

from typing import Any

from mcp_research.celery_app import celery_app
from mcp_research.ingest_unstructured import run_from_env


@celery_app.task(bind=True, name="mcp_research.ingest_unstructured")
def ingest_unstructured_task(self, pdf_path: str | None = None, data_dir: str | None = None) -> dict:
    """Run Unstructured ingestion as a Celery task with progress updates."""
    def _progress(payload: dict) -> None:
        """Proxy progress payloads to Celery task metadata."""
        self.update_state(state="PROGRESS", meta={"progress": payload})

    results = run_from_env(
        pdf_path_override=pdf_path,
        data_dir_override=data_dir,
        on_progress=_progress,
    )
    return {"count": len(results), "results": results}


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
