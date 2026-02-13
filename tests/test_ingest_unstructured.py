import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import ingest_unstructured


class _FakeRedis:
    def __init__(self):
        self.data = {}
        self.sets = {}
        self.hashes = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key):
        return self.sets.get(key, set())

    def exists(self, key):
        return key in self.data or key in self.hashes


class IngestUnstructuredTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_load_dotenv_sets_new_values_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("NEW_KEY=new_value\nEXISTING=from_file\n", encoding="utf-8")
            os.environ["EXISTING"] = "from_env"

            ingest_unstructured.load_dotenv(env_path)

            self.assertEqual(os.environ["EXISTING"], "from_env")
            self.assertEqual(os.environ["NEW_KEY"], "new_value")

    def test_load_env_int_handles_invalid(self):
        os.environ["TEST_INT"] = "not-an-int"
        value = ingest_unstructured.load_env_int("TEST_INT", 42)
        self.assertEqual(value, 42)

    def test_load_env_bool_parses_true(self):
        os.environ["TEST_BOOL"] = "yes"
        value = ingest_unstructured.load_env_bool("TEST_BOOL", False)
        self.assertTrue(value)

    def test_collect_pdfs_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_text("data", encoding="utf-8")
            (Path(tmpdir) / "notes.txt").write_text("notes", encoding="utf-8")

            results = ingest_unstructured.collect_pdfs(Path(tmpdir))

            self.assertEqual(results, [pdf_path])

    def test_elements_to_chunks_extracts_page_ranges(self):
        elements = [
            {"text": "Hello", "metadata": {"page_number": 2}},
            {"text": "World", "metadata": {"page_range": "3-4"}},
            {"text": "", "metadata": {"page": 1}},
        ]

        chunks, pages = ingest_unstructured.elements_to_chunks(elements)

        self.assertEqual(chunks, ["Hello", "World"])
        self.assertEqual(pages, [[2], [3, 4]])

    def test_upload_to_redis_writes_keys(self):
        redis_client = _FakeRedis()

        result = ingest_unstructured.upload_to_redis(
            redis_client=redis_client,
            doc_id="doc123",
            source="file.pdf",
            partitions_payload=[{"id": 1}],
            chunks_payload=[{"text": "chunk"}],
            prefix="unit",
            collection="collection_a",
        )

        self.assertIn("unit:pdf:doc123:meta", result["meta_key"])
        self.assertIn("doc123", redis_client.sets.get("unit:pdf:hashes", set()))
        self.assertIn("collection_a", redis_client.sets.get("unit:pdf:doc123:collections", set()))

    def test_upload_json_files_to_redis_inferrs_doc_id(self):
        redis_client = _FakeRedis()
        with tempfile.TemporaryDirectory() as tmpdir:
            partitions_path = Path(tmpdir) / "partitions.json"
            chunks_path = Path(tmpdir) / "chunks.json"
            partitions_path.write_text("[{\"id\": 1}]", encoding="utf-8")
            chunks_path.write_text(
                "[{\"document_id\": \"docxyz\", \"source\": \"source.pdf\", \"text\": \"t\"}]",
                encoding="utf-8",
            )

            result = ingest_unstructured.upload_json_files_to_redis(
                redis_client=redis_client,
                partitions_path=partitions_path,
                chunks_path=chunks_path,
                doc_id=None,
                source=None,
                prefix="unit",
                collection=None,
            )

        self.assertIn("unit:pdf:docxyz:meta", result["meta_key"])

    def test_main_help_uses_argparse_and_does_not_ingest(self):
        output = io.StringIO()
        with mock.patch.object(ingest_unstructured, "run_from_env") as mocked_run:
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as exc:
                    ingest_unstructured.main(["--help"])
        self.assertEqual(exc.exception.code, 0)
        self.assertIn("usage:", output.getvalue().lower())
        mocked_run.assert_not_called()

    def test_main_forwards_pdf_path_and_data_dir_overrides(self):
        with mock.patch.object(ingest_unstructured, "run_from_env", return_value=[]) as mocked_run:
            ingest_unstructured.main(["--pdf-path", "sample.pdf", "--data-dir", "/tmp/data"])

        mocked_run.assert_called_once_with(
            pdf_path_override="sample.pdf",
            data_dir_override="/tmp/data",
        )


if __name__ == "__main__":
    unittest.main()
