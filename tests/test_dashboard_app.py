import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import dashboard_app


class _FakePipeline:
    def __init__(self, client):
        self._client = client
        self._ops = []

    def hgetall(self, key):
        self._ops.append(("hgetall", key))
        return self

    def get(self, key):
        self._ops.append(("get", key))
        return self

    def zcard(self, key):
        self._ops.append(("zcard", key))
        return self

    def execute(self):
        out = []
        for op, key in self._ops:
            if op == "hgetall":
                out.append(self._client.hgetall(key))
            elif op == "get":
                out.append(self._client.get(key))
            elif op == "zcard":
                out.append(self._client.zcard(key))
            else:
                out.append(None)
        return out


class _FakeRedis:
    def __init__(self, *, smembers=None, values=None, hashes=None, zsets=None):
        self._smembers = smembers or {}
        self._values = values or {}
        self._hashes = hashes or {}
        self._zsets = zsets or {}

    def pipeline(self):
        return _FakePipeline(self)

    def smembers(self, key):
        return self._smembers.get(key, set())

    def get(self, key):
        return self._values.get(key)

    def hgetall(self, key):
        return self._hashes.get(key, {})

    def zcard(self, key):
        return self._zsets.get(key, 0)


class _DummyPoint:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class _FakeQdrant:
    def __init__(self, batches):
        self._batches = list(batches)
        self._idx = 0

    def scroll(self, **kwargs):
        if self._idx >= len(self._batches):
            return [], None
        batch = self._batches[self._idx]
        self._idx += 1
        next_offset = kwargs.get("offset")
        return batch, next_offset


class DashboardAppTests(unittest.TestCase):
    def test_file_identity_prefers_document_id(self):
        identity = dashboard_app._file_identity({"document_id": "doc", "source": "s", "bucket": "b", "key": "k"})
        self.assertEqual(identity[0], "doc:doc")

    def test_scan_collection_files_aggregates_chunks(self):
        client = _FakeQdrant(
            batches=[
                [
                    _DummyPoint("1", {"document_id": "a", "bucket": "b", "key": "k1"}),
                    _DummyPoint("2", {"document_id": "a", "bucket": "b", "key": "k1"}),
                    _DummyPoint("3", {"document_id": "c", "bucket": "b", "key": "k2"}),
                ],
                [],
            ]
        )
        result = dashboard_app.scan_collection_files(client, "bucket1", limit=200, batch_size=10, offset=None)
        files = sorted(result["files"], key=lambda f: f.get("document_id") or "")
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0]["document_id"], "a")
        self.assertEqual(files[0]["qdrant_chunks"], 2)
        self.assertEqual(files[0]["qdrant_metadata"]["document_id"], "a")
        self.assertEqual(files[0]["qdrant_metadata"]["bucket"], "b")
        self.assertEqual(files[0]["qdrant_metadata"]["key"], "k1")

    def test_scan_collection_files_uses_first_page_first_chunk_for_metadata(self):
        client = _FakeQdrant(
            batches=[
                [
                    _DummyPoint(
                        "1",
                        {"document_id": "a", "bucket": "b", "key": "k1", "page_start": 3, "chunk_index": 0, "source_ref": "doc://b/k1#page=3"},
                    ),
                    _DummyPoint(
                        "2",
                        {"document_id": "a", "bucket": "b", "key": "k1", "page_start": 1, "chunk_index": 2, "source_ref": "doc://b/k1#page=1"},
                    ),
                    _DummyPoint(
                        "3",
                        {"document_id": "a", "bucket": "b", "key": "k1", "page_start": 1, "chunk_index": 0, "source_ref": "doc://b/k1#page=1"},
                    ),
                ],
                [],
            ]
        )

        result = dashboard_app.scan_collection_files(client, "bucket1", limit=200, batch_size=10, offset=None)
        self.assertEqual(len(result["files"]), 1)
        file_row = result["files"][0]
        self.assertEqual(file_row["qdrant_chunks"], 3)
        self.assertEqual(file_row["page_start"], 1)
        self.assertEqual(file_row["qdrant_metadata"]["chunk_index"], 0)
        self.assertEqual(file_row["qdrant_metadata"]["source_ref"], "doc://b/k1#page=1")

    def test_scan_collection_files_dedupes_chunk_and_partition_counts(self):
        client = _FakeQdrant(
            batches=[
                [
                    _DummyPoint(
                        "1",
                        {"document_id": "a", "bucket": "b", "key": "k1", "page_start": 1, "page_end": 1, "chunk_index": 0, "text": "same"},
                    ),
                    _DummyPoint(
                        "2",
                        {"document_id": "a", "bucket": "b", "key": "k1", "page_start": 1, "page_end": 1, "chunk_index": 0, "text": "same"},
                    ),
                    _DummyPoint(
                        "3",
                        {"document_id": "a", "bucket": "b", "key": "k1", "page_start": 2, "page_end": 2, "chunk_index": 1, "text": "next"},
                    ),
                ],
                [],
            ]
        )

        result = dashboard_app.scan_collection_files(client, "bucket1", limit=200, batch_size=10, offset=None)
        self.assertEqual(len(result["files"]), 1)
        file_row = result["files"][0]
        self.assertEqual(file_row["qdrant_chunks"], 2)
        self.assertEqual(file_row["qdrant_partitions"], 2)

    def test_group_chunks_by_partition_in_sequential_order(self):
        raw_payloads = [
            {"chunk_index": 1, "page_start": 2, "page_end": 2, "text": "p2-c1"},
            {"chunk_index": 2, "page_start": 1, "page_end": 1, "text": "p1-c2"},
            {"chunk_index": 0, "page_start": 1, "page_end": 1, "text": "p1-c0"},
            {"chunk_index": 0, "page_start": 2, "page_end": 2, "text": "p2-c0"},
        ]
        chunks = [dashboard_app._chunk_detail_entry(str(idx), payload) for idx, payload in enumerate(raw_payloads)]
        chunks.sort(key=dashboard_app._chunk_sort_key)
        partitions = dashboard_app.group_chunks_by_partition(chunks)

        self.assertEqual([entry["label"] for entry in partitions], ["Page 1", "Page 2"])
        self.assertEqual([chunk["text"] for chunk in partitions[0]["chunks"]], ["p1-c0", "p1-c2"])
        self.assertEqual([chunk["text"] for chunk in partitions[1]["chunks"]], ["p2-c0", "p2-c1"])

    def test_dedupe_chunk_entries_removes_duplicate_payloads(self):
        chunks = [
            {
                "point_id": "p1",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "pages": [1],
                "source_ref": "doc://b/k#page=1",
                "text": "same chunk",
            },
            {
                "point_id": "p2",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "pages": [1],
                "source_ref": "doc://b/k#page=1",
                "text": "same chunk",
            },
            {
                "point_id": "p3",
                "chunk_index": 1,
                "page_start": 1,
                "page_end": 1,
                "pages": [1],
                "source_ref": "doc://b/k#page=1",
                "text": "different chunk",
            },
        ]
        deduped = dashboard_app.dedupe_chunk_entries(chunks)
        self.assertEqual(len(deduped), 2)
        self.assertEqual([chunk["point_id"] for chunk in deduped], ["p1", "p3"])

    def test_fetch_file_partition_summaries_omits_chunk_text_payload(self):
        with mock.patch.object(
            dashboard_app,
            "fetch_file_chunks",
            return_value={
                "collection": "bucket1",
                "count": 3,
                "raw_count": 3,
                "partition_count": 2,
                "scanned_points": 3,
                "truncated": False,
                "chunks": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                "partitions": [
                    {"partition_index": 1, "label": "Page 1", "page_start": 1, "page_end": 1, "chunk_count": 2, "chunks": [{"text": "a"}, {"text": "b"}]},
                    {"partition_index": 2, "label": "Page 2", "page_start": 2, "page_end": 2, "chunk_count": 1, "chunks": [{"text": "c"}]},
                ],
            },
        ):
            result = dashboard_app.fetch_file_partition_summaries(
                client=object(),
                collection_name="bucket1",
                document_id="doc",
                source=None,
                key=None,
                batch_size=50,
                limit=100,
            )

        self.assertEqual(result["partition_count"], 2)
        self.assertEqual(result["partitions"][0]["label"], "Page 1")
        self.assertNotIn("chunks", result["partitions"][0])

    def test_fetch_partition_chunks_returns_single_partition(self):
        with mock.patch.object(
            dashboard_app,
            "fetch_file_chunks",
            return_value={
                "collection": "bucket1",
                "count": 3,
                "raw_count": 3,
                "partition_count": 2,
                "scanned_points": 3,
                "truncated": False,
                "chunks": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                "partitions": [
                    {"partition_index": 1, "label": "Page 1", "page_start": 1, "page_end": 1, "chunk_count": 2, "chunks": [{"text": "a"}, {"text": "b"}]},
                    {"partition_index": 2, "label": "Page 2", "page_start": 2, "page_end": 2, "chunk_count": 1, "chunks": [{"text": "c"}]},
                ],
            },
        ):
            result = dashboard_app.fetch_partition_chunks(
                client=object(),
                collection_name="bucket1",
                document_id="doc",
                source=None,
                key=None,
                partition_index=2,
                batch_size=50,
                limit=100,
            )

        self.assertEqual(result["partition"]["partition_index"], 2)
        self.assertEqual(result["partition"]["chunk_count"], 1)
        self.assertEqual(result["partition"]["chunks"][0]["text"], "c")

    def test_enrich_files_with_redis_counts(self):
        prefix = "unstructured"
        source = "bucket1/file.pdf"
        doc_id = "doc123"
        redis_client = _FakeRedis(
            hashes={
                f"{prefix}:v2:doc:{doc_id}:meta": {
                    b"chunks_count": b"7",
                }
            },
            zsets={
                f"{prefix}:v2:doc:{doc_id}:chunk_hashes": 7,
                f"{prefix}:v2:doc:{doc_id}:partition_hashes": 3,
            },
        )

        files = [{"bucket": "bucket1", "key": "file.pdf", "document_id": None, "source": None, "qdrant_chunks": 4}]
        with mock.patch("mcp_research.dashboard_app.read_v2_source_doc_hash", return_value=doc_id):
            dashboard_app.enrich_files_with_redis(redis_client, prefix, "bucket1", files)

        self.assertEqual(files[0]["redis_doc_ids"], [doc_id])
        self.assertEqual(files[0]["redis_chunks"], 7)
        self.assertEqual(files[0]["redis_partitions"], 3)
        self.assertEqual(files[0]["redis_meta_key"], f"{prefix}:v2:doc:{doc_id}:meta")
        self.assertEqual(files[0]["redis_metadata"]["chunks_count"], "7")

    def test_build_original_file_url_from_bucket_key(self):
        entry = {"bucket": "bucket1", "key": "folder/file.pdf", "version_id": "v1"}
        with mock.patch.dict(
            os.environ,
            {"CITATION_BASE_URL": "http://resolver.local:8080", "CITATION_REF_PATH": "/r/doc"},
            clear=False,
        ):
            url = dashboard_app._build_original_file_url(entry)

        self.assertEqual(
            url,
            "http://resolver.local:8080/r/doc?ref=doc%3A%2F%2Fbucket1%2Ffolder%2Ffile.pdf%3Fversion_id%3Dv1",
        )

    def test_attach_original_file_links_uses_source_when_bucket_key_missing(self):
        files = [{"bucket": None, "key": None, "source": "bucket2/notes.pdf"}]
        with mock.patch.dict(
            os.environ,
            {"CITATION_BASE_URL": "http://resolver.local:8080", "CITATION_REF_PATH": "/r/doc"},
            clear=False,
        ):
            dashboard_app.attach_original_file_links(files)

        self.assertEqual(files[0]["original_file_url"], "http://resolver.local:8080/r/doc?ref=doc%3A%2F%2Fbucket2%2Fnotes.pdf")


if __name__ == "__main__":
    unittest.main()
