import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import ingest_missing_minio

TEST_DOC_TARGET = "mcp_research.ingest_missing_minio"
TEST_DOC_METHOD = (
    "Mocks discovery/enqueue helpers and captures stdout/stderr to validate CLI control flow, dry-run output, and enqueue counts."
)


class IngestMissingMinioTests(unittest.TestCase):
    def test_print_errors_to_stderr(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            ingest_missing_minio._print_errors(["one", "two"])

        output = stderr.getvalue().strip().splitlines()
        self.assertEqual(output, ["error: one", "error: two"])

    def test_main_dry_run_outputs_missing(self):
        stdout = io.StringIO()
        with mock.patch("mcp_research.ingest_missing_minio._find_missing_minio_objects") as finder:
            finder.return_value = ([
                ("bucket", "file1.pdf"),
                ("bucket", "file2.pdf"),
            ], [])
            with redirect_stdout(stdout):
                with mock.patch.object(sys, "argv", ["ingest_missing_minio", "--dry-run"]):
                    ingest_missing_minio.main()

        output = stdout.getvalue()
        self.assertIn("bucket/file1.pdf", output)
        self.assertIn("missing_count=2", output)

    def test_main_enqueue_outputs_count(self):
        stdout = io.StringIO()
        with mock.patch("mcp_research.ingest_missing_minio._find_missing_minio_objects") as finder:
            with mock.patch("mcp_research.ingest_missing_minio._enqueue_missing_ingests") as enqueuer:
                finder.return_value = ([("bucket", "file1.pdf")], [])
                enqueuer.return_value = 1
                with redirect_stdout(stdout):
                    with mock.patch.object(sys, "argv", ["ingest_missing_minio"]):
                        ingest_missing_minio.main()

        self.assertIn("enqueued_count=1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
