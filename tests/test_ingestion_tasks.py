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
    def test_ingest_unstructured_task_updates_progress(self):
        task = ingestion_tasks.ingest_unstructured_task
        calls = []

        def fake_update_state(state, meta):
            calls.append((state, meta))

        def fake_run_from_env(pdf_path_override, data_dir_override, on_progress):
            on_progress({"step": "started"})
            return ["result"]

        with mock.patch("mcp_research.ingestion_tasks.run_from_env", side_effect=fake_run_from_env):
            task.update_state = fake_update_state
            result = task.run(pdf_path="file.pdf", data_dir="/tmp")

        self.assertEqual(result["count"], 1)
        self.assertEqual(calls[0][0], "PROGRESS")
        self.assertEqual(calls[0][1]["progress"], {"step": "started"})

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
