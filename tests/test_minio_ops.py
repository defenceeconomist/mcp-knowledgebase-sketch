import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import minio_ops

TEST_DOC_TARGET = "mcp_research.minio_ops"
TEST_DOC_METHOD = (
    "Uses mocked MinIO clients plus temporary PDF fixtures to validate bucket/file commands and ingest cleanup flags through CLI entrypoints."
)


class MinioOpsTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()
        os.environ["MINIO_ACCESS_KEY"] = "test-access"
        os.environ["MINIO_SECRET_KEY"] = "test-secret"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_add_bucket_creates_when_missing(self):
        client = mock.Mock()
        client.bucket_exists.return_value = False

        minio_ops._add_bucket(client, "research")

        client.make_bucket.assert_called_once_with("research")

    def test_delete_bucket_force_removes_objects(self):
        client = mock.Mock()
        client.bucket_exists.return_value = True
        client.list_objects.return_value = [
            SimpleNamespace(object_name="a.pdf", version_id=None),
            SimpleNamespace(object_name="b.pdf", version_id="v2"),
        ]

        minio_ops._delete_bucket(client, "research", force=True)

        self.assertEqual(client.remove_object.call_count, 2)
        client.remove_bucket.assert_called_once_with("research")

    def test_upload_file_ingests_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "paper.pdf"
            file_path.write_bytes(b"%PDF test")

            client = mock.Mock()
            client.bucket_exists.return_value = True
            client.fput_object.return_value = SimpleNamespace(version_id="v1")

            with mock.patch("mcp_research.minio_ops._build_minio_client", return_value=client):
                with mock.patch("mcp_research.minio_ops.process_object_from_env") as ingest:
                    minio_ops.main(
                        [
                            "--endpoint",
                            "localhost:9000",
                            "--access-key",
                            "key",
                            "--secret-key",
                            "secret",
                            "--no-secure",
                            "upload-file",
                            "research",
                            str(file_path),
                        ]
                    )

        ingest.assert_called_once_with(bucket="research", object_name="paper.pdf", version_id="v1")

    def test_remove_file_can_skip_ingested_cleanup(self):
        client = mock.Mock()
        with mock.patch("mcp_research.minio_ops._build_minio_client", return_value=client):
            with mock.patch("mcp_research.minio_ops.delete_object_from_env") as cleanup:
                minio_ops.main(
                    [
                        "--endpoint",
                        "localhost:9000",
                        "--access-key",
                        "key",
                        "--secret-key",
                        "secret",
                        "--no-secure",
                        "remove-file",
                        "research",
                        "paper.pdf",
                        "--no-delete-ingested",
                    ]
                )

        client.remove_object.assert_called_once_with("research", "paper.pdf", version_id=None)
        cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
