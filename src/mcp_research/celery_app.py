import os

from celery import Celery
from mcp_research.runtime_utils import load_env_int


def make_celery() -> Celery:
    """Configure and return the Celery application instance."""
    broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "redis://redis:6379/0")
    backend = os.getenv("CELERY_RESULT_BACKEND") or broker
    app = Celery(
        "mcp_research",
        broker=broker,
        backend=backend,
        include=["mcp_research.ingestion_tasks"],
    )
    app.conf.update(
        task_track_started=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        broker_transport_options={
            "visibility_timeout": load_env_int("CELERY_VISIBILITY_TIMEOUT", 3600),
        },
        result_extended=True,
    )
    return app


celery_app = make_celery()
