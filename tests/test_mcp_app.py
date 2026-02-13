import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

try:
    from mcp_research import mcp_app
except ModuleNotFoundError:  # pragma: no cover - optional local deps
    mcp_app = None  # type: ignore[assignment]


class _FakeRedis:
    def __init__(self, *, values=None, smembers=None, hashes=None):
        self._values = values or {}
        self._smembers = smembers or {}
        self._hashes = hashes or {}

    def get(self, key):
        return self._values.get(key)

    def smembers(self, key):
        return self._smembers.get(key, set())

    def hgetall(self, key):
        return self._hashes.get(key, {})


class _DummyPoint:
    def __init__(self, pid, payload, score=0.0):
        self.id = pid
        self.payload = payload
        self.score = score


class _DummySearchResponse:
    def __init__(self, points):
        self.points = points


class _FakeQdrant:
    def __init__(self, payload):
        self._payload = payload

    def retrieve(self, **_kwargs):
        return [_DummyPoint("42", self._payload)] if self._payload is not None else []


@unittest.skipIf(mcp_app is None, "mcp_app optional dependencies unavailable")
class McpAppTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_pages_to_range_handles_empty(self):
        self.assertEqual(mcp_app._pages_to_range([]), (None, None))
        self.assertEqual(mcp_app._pages_to_range([3, 1, 2]), (1, 3))

    def test_source_ref_from_payload_prefers_existing(self):
        payload = {"source_ref": "doc://bucket/file.pdf"}
        self.assertEqual(mcp_app._source_ref_from_payload(payload), "doc://bucket/file.pdf")

    def test_source_ref_from_payload_builds_ref(self):
        payload = {"bucket": "bucket", "key": "file.pdf", "pages": [2, 4]}
        with mock.patch("mcp_research.mcp_app.build_source_ref", return_value="doc://bucket/file.pdf"):
            self.assertEqual(mcp_app._source_ref_from_payload(payload), "doc://bucket/file.pdf")

    def test_coerce_qdrant_offset(self):
        self.assertEqual(mcp_app._coerce_qdrant_offset("12"), 12)
        self.assertEqual(mcp_app._coerce_qdrant_offset("abc"), "abc")

    def test_file_identity_prefers_document_id(self):
        identity = mcp_app._file_identity({"document_id": "doc", "source": "s"})
        self.assertEqual(identity[0], "doc:doc")

    def test_default_collection_key_uses_prefix(self):
        self.assertEqual(mcp_app._default_collection_key(), "unstructured:qdrant:default_collection")

    def test_normalize_retrieval_mode(self):
        self.assertEqual(mcp_app._normalize_retrieval_mode("HYBRID"), "hybrid")
        self.assertEqual(mcp_app._normalize_retrieval_mode(" cosine "), "cosine")
        with self.assertRaises(ValueError):
            mcp_app._normalize_retrieval_mode("bm25")

    def test_citation_key_from_payload_prefers_payload_value(self):
        payload = {"citation_key": "alpha2025"}
        self.assertEqual(mcp_app._citation_key_from_payload(payload), "alpha2025")

    def test_citation_key_from_payload_falls_back_to_bibtex_metadata(self):
        fake_redis = _FakeRedis(
            values={
                "bibtex:file:bucket-a/paper.pdf": b'{"citationKey":"doe2024paper"}',
            }
        )
        with mock.patch.object(mcp_app, "_get_redis_client", return_value=fake_redis):
            payload = {"bucket": "bucket-a", "key": "paper.pdf"}
            self.assertEqual(mcp_app._citation_key_from_payload(payload), "doe2024paper")

    def test_fetch_document_chunks_uses_meta_chunks_key_override(self):
        fake_redis = _FakeRedis()
        with mock.patch.object(mcp_app, "_get_redis_client", return_value=fake_redis):
            with mock.patch(
                "mcp_research.mcp_app.read_v2_source_doc_hash",
                return_value="doc-1",
            ):
                with mock.patch(
                    "mcp_research.mcp_app.read_v2_doc_chunks",
                    return_value=[{"text": "chunk-a"}],
                ):
                    with mock.patch(
                        "mcp_research.mcp_app.read_v2_doc_partitions",
                        return_value=[],
                    ):
                        result = mcp_app.fetch_document_chunks(bucket="bucket-a", key="paper.pdf")
        self.assertTrue(result["found"])
        self.assertEqual(result["count"], 1)
        self.assertIn("v2:doc:doc-1:chunk_hashes", str(result["chunks_key"]))
        self.assertEqual(result["chunks"][0]["text"], "chunk-a")

    def test_fetch_document_chunks_reads_v2(self):
        fake_redis = _FakeRedis()
        with mock.patch.object(mcp_app, "_get_redis_client", return_value=fake_redis):
            with mock.patch(
                "mcp_research.mcp_app.read_v2_source_doc_hash",
                return_value="doc-1",
            ):
                with mock.patch(
                    "mcp_research.mcp_app.read_v2_doc_chunks",
                    return_value=[{"text": "chunk-v2"}],
                ):
                    with mock.patch(
                        "mcp_research.mcp_app.read_v2_doc_partitions",
                        return_value=[{"text": "partition-v2"}],
                    ):
                        result = mcp_app.fetch_document_chunks(bucket="bucket-a", key="paper.pdf")
        self.assertTrue(result["found"])
        self.assertEqual(result["count"], 1)
        self.assertIn("v2:doc:doc-1:chunk_hashes", str(result.get("chunks_key")))
        self.assertEqual(result["chunks"][0]["text"], "chunk-v2")

    def test_fetch_chunk_bibtex_returns_metadata_for_chunk(self):
        fake_qdrant = _FakeQdrant({"bucket": "bucket-a", "key": "paper.pdf", "document_id": "doc-1"})
        fake_redis = _FakeRedis(
            values={
                "bibtex:file:bucket-a/paper.pdf": b'{"citationKey":"doe2024paper","title":"Paper"}',
            }
        )
        with mock.patch.object(mcp_app, "_get_qdrant_client", return_value=fake_qdrant):
            with mock.patch.object(mcp_app, "_get_redis_client", return_value=fake_redis):
                result = mcp_app.fetch_chunk_bibtex("42", collection="research_2026")
        self.assertTrue(result["found"])
        self.assertEqual(result["citation_key"], "doe2024paper")
        self.assertEqual(result["metadata"]["title"], "Paper")

    def test_fetch_chunk_partition_returns_partition_chunks_for_chunk_page(self):
        fake_qdrant = _FakeQdrant(
            {
                "doc_hash": "doc-1",
                "bucket": "bucket-a",
                "key": "paper.pdf",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
                "text": "seed chunk",
            }
        )
        fake_redis = _FakeRedis()
        with mock.patch.object(mcp_app, "_get_qdrant_client", return_value=fake_qdrant):
            with mock.patch.object(mcp_app, "_get_redis_client", return_value=fake_redis):
                with mock.patch(
                    "mcp_research.mcp_app.read_v2_doc_chunks",
                    return_value=[
                        {"page_start": 1, "page_end": 1, "chunk_index": 0, "text": "chunk page 1"},
                        {"page_start": 2, "page_end": 2, "chunk_index": 1, "text": "chunk page 2"},
                    ],
                ):
                    with mock.patch(
                        "mcp_research.mcp_app.read_v2_doc_partitions",
                        return_value=[
                            {"page_start": 1, "page_end": 1, "text": "partition 1"},
                            {"page_start": 2, "page_end": 2, "text": "partition 2"},
                        ],
                    ):
                        result = mcp_app.fetch_chunk_partition("42", collection="research_2026")
        self.assertTrue(result["found"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["partition"]["page_start"], 1)
        self.assertEqual(result["chunks"][0]["text"], "chunk page 1")

    def test_search_include_partition_true_returns_partition_summary(self):
        point = _DummyPoint(
            "p-1",
            {
                "document_id": "doc-1",
                "bucket": "bucket-a",
                "key": "paper.pdf",
                "page_start": 3,
                "page_end": 3,
                "chunk_index": 7,
                "text": "target chunk text",
            },
            score=0.87,
        )
        response = _DummySearchResponse([point])
        fake_redis = _FakeRedis()
        bundle = {
            "document_ids": ["doc-1"],
            "count": 2,
            "chunks": [
                {"page_start": 3, "page_end": 3, "chunk_index": 7, "text": "chunk page 3"},
                {"page_start": 4, "page_end": 4, "chunk_index": 8, "text": "chunk page 4"},
            ],
            "partitions": [
                {"page_start": 3, "page_end": 3, "text": "partition 3"},
            ],
        }

        with mock.patch.object(mcp_app, "_get_qdrant_client", return_value=object()):
            with mock.patch.object(mcp_app, "_get_dense_model", return_value=object()):
                with mock.patch.object(mcp_app, "_cosine_search", return_value=response):
                    with mock.patch.object(mcp_app, "_get_redis_client", return_value=fake_redis):
                        with mock.patch.object(mcp_app, "_fetch_document_bundle", return_value=bundle):
                            result = mcp_app.search(
                                query="target chunk",
                                retrieval_mode="cosine",
                                include_partition=True,
                                include_document=False,
                            )

        self.assertTrue(result["include_partition"])
        self.assertFalse(result["include_document"])
        self.assertEqual(len(result["results"]), 1)
        partition = result["results"][0]["partition"]
        self.assertIsNotNone(partition)
        self.assertEqual(partition["page_start"], 3)
        self.assertEqual(partition["page_end"], 3)
        self.assertEqual(partition["chunk_count"], 1)
        self.assertEqual(partition["partition_payload"]["text"], "partition 3")


if __name__ == "__main__":
    unittest.main()
