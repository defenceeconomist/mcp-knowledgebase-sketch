import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import upload_data_to_redis


class UploadDataToRedisTests(unittest.TestCase):
    def test_upload_pair_calls_helper(self):
        redis_client = object()
        with mock.patch("mcp_research.upload_data_to_redis.upload_json_files_to_redis") as uploader:
            upload_data_to_redis._upload_pair(
                redis_client=redis_client,
                partitions_path=Path("/tmp/part.json"),
                chunks_path=Path("/tmp/chunk.json"),
                redis_prefix="unit",
                doc_id="doc",
                source="source.pdf",
            )

        uploader.assert_called_once_with(
            redis_client=redis_client,
            partitions_path=Path("/tmp/part.json"),
            chunks_path=Path("/tmp/chunk.json"),
            doc_id="doc",
            source="source.pdf",
            prefix="unit",
        )

    def test_upload_directory_requires_pairs_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            partitions_dir = Path(tmpdir) / "partitions"
            chunks_dir = Path(tmpdir) / "chunks"
            partitions_dir.mkdir()
            chunks_dir.mkdir()
            (chunks_dir / "a.json").write_text("[]", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                upload_data_to_redis._upload_directory(
                    redis_client=object(),
                    partitions_dir=partitions_dir,
                    chunks_dir=chunks_dir,
                    redis_prefix="unit",
                    strict=True,
                )

    def test_upload_directory_counts_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            partitions_dir = Path(tmpdir) / "partitions"
            chunks_dir = Path(tmpdir) / "chunks"
            partitions_dir.mkdir()
            chunks_dir.mkdir()
            (partitions_dir / "a.json").write_text("[]", encoding="utf-8")
            (chunks_dir / "a.json").write_text("[]", encoding="utf-8")

            with mock.patch("mcp_research.upload_data_to_redis.upload_json_files_to_redis"):
                total = upload_data_to_redis._upload_directory(
                    redis_client=object(),
                    partitions_dir=partitions_dir,
                    chunks_dir=chunks_dir,
                    redis_prefix="unit",
                    strict=False,
                )

        self.assertEqual(total, 1)


if __name__ == "__main__":
    unittest.main()
