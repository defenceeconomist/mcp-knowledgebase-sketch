import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from qdrant_client import models

from mcp_research import upsert_chunks

TEST_DOC_TARGET = "mcp_research.upsert_chunks"
TEST_DOC_METHOD = (
    "Uses fake embedder/Qdrant/Redis fixtures and env overrides to verify chunk loading, payload schema, batching, and deterministic ID behavior."
)


class _Sparse:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


class _DenseModel:
    def __init__(self, vectors):
        self._vectors = vectors

    def embed(self, docs):
        return list(self._vectors[: len(docs)])


class _SparseModel:
    def __init__(self, sparse_vectors):
        self._sparse_vectors = sparse_vectors

    def embed(self, docs):
        return [
            _Sparse(indices=indices, values=values)
            for indices, values in self._sparse_vectors[: len(docs)]
        ]


class _Client:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, collection_name, points):
        self.upsert_calls.append((collection_name, points))


class _FakeRedis:
    def __init__(self, store):
        self.store = store

    def smembers(self, key):
        return self.store.get(key, set())

    def get(self, key):
        return self.store.get(key)


class UpsertChunksTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()
        os.environ["QDRANT_POINT_ID_MODE"] = "uuid"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_batched_splits_items(self):
        items = [{"id": i} for i in range(5)]
        batches = list(upsert_chunks.batched(items, batch_size=2))

        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[2]), 1)

    def test_load_chunk_items_filters_invalid_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_dir = Path(tmpdir)
            (chunks_dir / "a.json").write_text(
                "[{\"text\": \" alpha \"}, {\"text\": \"\"}, \"skip\"]",
                encoding="utf-8",
            )
            (chunks_dir / "b.json").write_text("{\"text\": \"nope\"}", encoding="utf-8")

            items = upsert_chunks.load_chunk_items(chunks_dir)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], " alpha ")

    def test_load_chunk_items_from_redis(self):
        redis_store = {"unit:v2:doc_hashes": {b"doc1"}}
        redis_client = _FakeRedis(redis_store)
        with unittest.mock.patch(
            "mcp_research.upsert_chunks.read_v2_doc_chunks",
            return_value=[{"text": "hello"}],
        ):
            items = upsert_chunks.load_chunk_items_from_redis(
                redis_client=redis_client,
                prefix="unit",
                doc_ids=None,
            )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "hello")

    def test_upsert_items_sends_points(self):
        client = _Client()
        dense_model = _DenseModel([[0.1, 0.2]])
        sparse_model = _SparseModel([([1, 2], [0.6, 0.4])])
        items = [
            {
                "text": "chunk",
                "document_id": "doc",
                "source": "file.pdf",
                "chunk_index": 0,
                "pages": [1],
            }
        ]

        total = upsert_chunks.upsert_items(
            client=client,
            collection="demo",
            items=items,
            dense_model=dense_model,
            sparse_model=sparse_model,
            batch_size=10,
        )

        self.assertEqual(total, 1)
        self.assertEqual(len(client.upsert_calls), 1)
        collection_name, points = client.upsert_calls[0]
        self.assertEqual(collection_name, "demo")
        self.assertIsInstance(points[0], models.PointStruct)
        self.assertEqual(points[0].payload["doc_hash"], "doc")

    def test_upsert_items_v2_payload_and_deterministic_id(self):
        client = _Client()
        dense_model = _DenseModel([[0.1, 0.2]])
        sparse_model = _SparseModel([([1, 2], [0.6, 0.4])])
        items = [
            {
                "text": "chunk",
                "document_id": "doc-v2",
                "source": "bucket/file.pdf",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "pages": [1],
            }
        ]

        with unittest.mock.patch.dict(
            os.environ,
            {"QDRANT_POINT_ID_MODE": "deterministic"},
            clear=False,
        ):
            first_total = upsert_chunks.upsert_items(
                client=client,
                collection="demo",
                items=items,
                dense_model=dense_model,
                sparse_model=sparse_model,
                batch_size=10,
            )
            second_total = upsert_chunks.upsert_items(
                client=client,
                collection="demo",
                items=items,
                dense_model=dense_model,
                sparse_model=sparse_model,
                batch_size=10,
            )

        self.assertEqual(first_total, 1)
        self.assertEqual(second_total, 1)
        self.assertEqual(len(client.upsert_calls), 2)
        first_point = client.upsert_calls[0][1][0]
        second_point = client.upsert_calls[1][1][0]
        self.assertEqual(first_point.id, second_point.id)
        self.assertIn("doc_hash", first_point.payload)
        self.assertIn("partition_hash", first_point.payload)
        self.assertIn("chunk_hash", first_point.payload)
        self.assertNotIn("document_id", first_point.payload)


if __name__ == "__main__":
    unittest.main()
