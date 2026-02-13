import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import ingestion_tasks


class IngestionTasksTests(unittest.TestCase):
    def test_ingest_minio_object_task_calls_processor(self):
        task = ingestion_tasks.ingest_minio_object_task
        with mock.patch("mcp_research.minio_ingest.process_object_from_env") as mocked:
            result = task.run(bucket="bucket", object_name="file.pdf", version_id="v1")

        mocked.assert_called_once_with(bucket="bucket", object_name="file.pdf", version_id="v1")
        self.assertEqual(result["status"], "completed")

    def test_delete_minio_object_task_calls_processor(self):
        task = ingestion_tasks.delete_minio_object_task
        with mock.patch("mcp_research.minio_ingest.delete_object_from_env") as mocked:
            result = task.run(bucket="bucket", object_name="file.pdf", version_id=None)

        mocked.assert_called_once_with(bucket="bucket", object_name="file.pdf", version_id=None)
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
