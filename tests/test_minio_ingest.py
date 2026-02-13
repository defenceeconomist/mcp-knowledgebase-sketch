import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import minio_ingest


class _FakeRedis:
    def __init__(self):
        self.sets = {}
        self.store = {}
        self.deleted = []

    def smembers(self, key):
        return self.sets.get(key, set())

    def get(self, key):
        return self.store.get(key)

    def srem(self, key, value):
        if key in self.sets:
            self.sets[key].discard(value)

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def delete(self, key):
        self.deleted.append(key)
        self.sets.pop(key, None)
        self.store.pop(key, None)


class _DummyModel:
    def embed(self, docs):
        return [[0.1, 0.2] for _ in docs]


class MinioIngestTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_normalize_events_strips_blanks(self):
        events = minio_ingest._normalize_events([" s3:ObjectCreated:* ", "", "  "])
        self.assertEqual(events, ["s3:ObjectCreated:*"])

    def test_load_env_list_parses_values(self):
        os.environ["TEST_LIST"] = "a, b, ,c"
        self.assertEqual(minio_ingest._load_env_list("TEST_LIST"), ["a", "b", "c"])

    def test_source_doc_ids_prefers_set_members(self):
        redis_client = _FakeRedis()
        with mock.patch("mcp_research.minio_ingest.read_v2_source_doc_hash", return_value="doc1"):
            result = minio_ingest._source_doc_ids(redis_client, "unit", "bucket/file.pdf")
        self.assertEqual(result, ["doc1"])

    def test_remove_source_mapping_deletes_empty_set(self):
        redis_client = _FakeRedis()

        minio_ingest._remove_source_mapping(redis_client, "unit", "bucket/file.pdf", "doc")

        self.assertEqual(len(redis_client.deleted), 2)
        self.assertTrue(redis_client.deleted[0].startswith("unit:v2:source:"))

    def test_delete_from_qdrant_skips_missing_collection(self):
        client = mock.Mock()
        client.collection_exists.return_value = False

        deleted = minio_ingest._delete_from_qdrant(client, "bucket", "file.pdf")

        self.assertEqual(deleted, 0)
        client.delete.assert_not_called()

    def test_process_object_skips_non_pdf(self):
        minio_client = mock.Mock()
        minio_client.get_object.side_effect = AssertionError("should not fetch")
        qdrant_client = mock.Mock()

        ingestor = minio_ingest.MinioIngestor(
            minio_client=minio_client,
            qdrant_client=qdrant_client,
            dense_model=_DummyModel(),
            sparse_model=_DummyModel(),
            redis_client=None,
            redis_prefix="unit",
            unstructured_api_key="key",
            unstructured_api_url="url",
            unstructured_strategy="hi_res",
            unstructured_chunking=None,
            chunk_size=100,
            chunk_overlap=10,
            languages=None,
            skip_existing=True,
        )

        ingestor.process_object("bucket", "notes.txt")
        minio_client.get_object.assert_not_called()


if __name__ == "__main__":
    unittest.main()
