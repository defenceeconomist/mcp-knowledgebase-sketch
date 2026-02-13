import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import bibtex_ui_app
from mcp_research.schema_v2 import (
    redis_v2_chunk_key,
    redis_v2_doc_chunk_hashes_key,
    redis_v2_doc_partition_hashes_key,
    redis_v2_partition_key,
    redis_v2_source_doc_key,
    source_id,
)

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]


class _FakeRedisPipeline:
    def __init__(self, client):
        self._client = client
        self._keys = []

    def get(self, key):
        self._keys.append(key)
        return self

    def execute(self):
        return [self._client.get(key) for key in self._keys]


class _FakeRedis:
    def __init__(self):
        self.values = {}
        self.sets = {}
        self.hashes = {}
        self.sorted_sets = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def pipeline(self):
        return _FakeRedisPipeline(self)

    def smembers(self, key):
        return self.sets.get(key, set())

    def zadd(self, key, mapping):
        store = self.sorted_sets.setdefault(key, {})
        for member, score in (mapping or {}).items():
            store[member] = score

    def zrange(self, key, start, end):
        entries = self.sorted_sets.get(key, {})
        ordered = [member for member, _score in sorted(entries.items(), key=lambda item: item[1])]
        if end == -1:
            return ordered[start:]
        return ordered[start : end + 1]

    def mget(self, keys):
        return [self.get(key) for key in keys]


class _FakeBucket:
    def __init__(self, name):
        self.name = name


class _FakeObject:
    def __init__(self, object_name):
        self.object_name = object_name
        self.is_dir = False


class _FakeSearchTools:
    QDRANT_URL = "http://qdrant.local:6333"
    QDRANT_COLLECTION = "default_col"
    REDIS_URL = "redis://redis:6379/0"

    def __init__(self):
        self.default_collection = "default_col"
        self.last_search = None
        self.last_fetch = None
        self.last_fetch_chunk_document = None
        self.last_fetch_chunk_partition = None
        self.last_fetch_chunk_bibtex = None

    def ping(self):
        return "pong"

    def _get_default_collection(self):
        return self.default_collection

    def list_collections(self):
        return {"collections": ["default_col", "research_2026"]}

    def set_default_collection(self, name):
        self.default_collection = name
        return {"default_collection": name}

    def search(
        self,
        query,
        top_k=5,
        prefetch_k=40,
        collection=None,
        retrieval_mode="hybrid",
        include_partition=False,
        include_document=False,
    ):
        self.last_search = {
            "query": query,
            "top_k": top_k,
            "prefetch_k": prefetch_k,
            "collection": collection,
            "retrieval_mode": retrieval_mode,
            "include_partition": include_partition,
            "include_document": include_document,
        }
        return {
            "retrieval_mode": retrieval_mode,
            "include_partition": include_partition,
            "include_document": include_document,
            "results": [
                {
                    "id": "42",
                    "score": 0.9,
                    "text": "result text",
                    "bucket": "bucket-a",
                    "key": "paper.pdf",
                    "citation_key": "doe2026paper",
                }
            ],
        }

    def fetch(self, id, collection=None):
        self.last_fetch = {"id": id, "collection": collection}
        return {"id": id, "found": True, "text": "full chunk"}

    def fetch_chunk_document(self, id, collection=None):
        self.last_fetch_chunk_document = {"id": id, "collection": collection}
        return {"id": id, "found": True, "count": 2, "chunks": [{"text": "first"}, {"text": "second"}]}

    def fetch_chunk_partition(self, id, collection=None):
        self.last_fetch_chunk_partition = {"id": id, "collection": collection}
        return {
            "id": id,
            "found": True,
            "count": 1,
            "partition": {"label": "Page 1", "page_start": 1, "page_end": 1},
            "chunks": [{"text": "partition chunk"}],
        }

    def fetch_chunk_bibtex(self, id, collection=None):
        self.last_fetch_chunk_bibtex = {"id": id, "collection": collection}
        return {"id": id, "found": True, "citation_key": "doe2026paper", "metadata": {"title": "Paper"}}


class _FakeMinio:
    def __init__(self, buckets):
        self._buckets = {name: list(objects) for name, objects in buckets.items()}
        self.removed_buckets = []
        self.removed_objects = []
        self.uploaded_objects = []

    def list_buckets(self):
        return [_FakeBucket(name) for name in self._buckets.keys()]

    def list_objects(self, bucket, prefix="", recursive=True, **_kwargs):
        del recursive
        for object_name in self._buckets.get(bucket, []):
            if prefix and not object_name.startswith(prefix):
                continue
            yield _FakeObject(object_name)

    def bucket_exists(self, bucket):
        return bucket in self._buckets

    def make_bucket(self, bucket):
        self._buckets.setdefault(bucket, [])

    def remove_object(self, bucket, object_name, version_id=None):
        del version_id
        entries = self._buckets.get(bucket, [])
        if object_name in entries:
            entries.remove(object_name)
        self.removed_objects.append((bucket, object_name))

    def remove_bucket(self, bucket):
        entries = self._buckets.get(bucket, [])
        if entries:
            raise RuntimeError(f"BucketNotEmpty: {bucket}")
        self._buckets.pop(bucket, None)
        self.removed_buckets.append(bucket)

    def put_object(self, bucket_name, object_name, data, length, content_type=None):
        del data, length, content_type
        self._buckets.setdefault(bucket_name, [])
        if object_name not in self._buckets[bucket_name]:
            self._buckets[bucket_name].append(object_name)
        self.uploaded_objects.append((bucket_name, object_name))
        return SimpleNamespace(version_id="v1")


@unittest.skipIf(bibtex_ui_app.app is None or TestClient is None, "FastAPI test client unavailable")
class BibtexUiAppTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()
        os.environ["BIBTEX_SCHEMA_WRITE_MODE"] = "dual"
        os.environ["BIBTEX_SCHEMA_READ_MODE"] = "prefer_v2"
        self.client = TestClient(bibtex_ui_app.app)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_index_serves_bibtex_workspace_shell(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("BibTeX Metadata Workspace", response.text)
        self.assertIn("Buckets", response.text)
        self.assertIn("BibTeX Fields", response.text)
        self.assertIn("Qdrant Search", response.text)

    def test_api_search_status_reports_qdrant_default_collection(self):
        fake_search = _FakeSearchTools()
        with mock.patch.object(bibtex_ui_app, "mcp_search_tools", fake_search):
            response = self.client.get("/api/search/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["qdrant_url"], "http://qdrant.local:6333")
        self.assertEqual(payload["default_collection"], "default_col")

    def test_api_search_uses_requested_parameters(self):
        fake_search = _FakeSearchTools()
        with mock.patch.object(bibtex_ui_app, "mcp_search_tools", fake_search):
            response = self.client.post(
                "/api/search",
                json={
                    "query": "hybrid ranking",
                    "topK": 11,
                    "prefetchK": 120,
                    "collection": "research_2026",
                    "retrievalMode": "cosine",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(fake_search.last_search["query"], "hybrid ranking")
        self.assertEqual(fake_search.last_search["top_k"], 11)
        self.assertEqual(fake_search.last_search["prefetch_k"], 120)
        self.assertEqual(fake_search.last_search["collection"], "research_2026")
        self.assertEqual(fake_search.last_search["retrieval_mode"], "cosine")
        self.assertEqual(payload["collection"], "research_2026")
        self.assertEqual(payload["retrieval_mode"], "cosine")
        self.assertEqual(len(payload["results"]), 1)

    def test_api_search_fetch_uses_point_id_and_collection(self):
        fake_search = _FakeSearchTools()
        with mock.patch.object(bibtex_ui_app, "mcp_search_tools", fake_search):
            response = self.client.get("/api/search/fetch/42?collection=research_2026")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["found"])
        self.assertEqual(fake_search.last_fetch, {"id": "42", "collection": "research_2026"})

    def test_api_buckets_lists_minio_buckets(self):
        fake_minio = _FakeMinio({"bucket-a": [], "bucket-b": []})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(
                bibtex_ui_app,
                "_resolve_minio_buckets",
                return_value=(["bucket-a", "bucket-b"], None),
            ):
                response = self.client.get("/api/buckets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["buckets"], ["bucket-a", "bucket-b"])

    def test_api_bucket_files_loads_pdf_objects_and_bibtex_metadata(self):
        fake_minio = _FakeMinio({"bucket-a": ["folder/alpha.pdf", "beta.pdf", "ignore.txt"]})
        fake_redis = _FakeRedis()
        fake_redis.set(
            "bibtex:file:bucket-a/folder/alpha.pdf",
            json.dumps(
                {
                    "title": "Alpha Paper",
                    "year": "2024",
                    "citationKey": "alpha2024paper",
                    "entryType": "article",
                    "authors": [{"firstName": "Ada", "lastName": "Lovelace"}],
                }
            ),
        )

        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
                with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                    response = self.client.get("/api/buckets/bucket-a/files")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 2)
        files = payload["files"]
        self.assertEqual([entry["objectName"] for entry in files], ["beta.pdf", "folder/alpha.pdf"])
        alpha = next(entry for entry in files if entry["objectName"] == "folder/alpha.pdf")
        self.assertEqual(alpha["title"], "Alpha Paper")
        self.assertEqual(alpha["citationKey"], "alpha2024paper")
        self.assertEqual(alpha["authors"][0]["lastName"], "Lovelace")
        self.assertIn("/r/doc?ref=", alpha["originalFileUrl"])

    def test_api_add_bucket_creates_bucket(self):
        fake_minio = _FakeMinio({"bucket-a": []})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            response = self.client.post("/api/buckets", json={"bucket": "bucket-b"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["created"])
        self.assertIn("bucket-b", fake_minio._buckets)

    def test_api_delete_bucket_force_removes_objects_and_cleans_ingested_data(self):
        fake_minio = _FakeMinio({"bucket-a": ["alpha.pdf", "beta.pdf"]})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_delete_ingested_object_from_env") as cleanup:
                response = self.client.delete("/api/buckets/bucket-a?force=true&delete_ingested=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["removed_objects"], 2)
        self.assertIn("bucket-a", fake_minio.removed_buckets)
        self.assertEqual(cleanup.call_count, 2)
        self.assertTrue(payload["delete_ingested"])

    def test_api_delete_bucket_force_preserves_ingested_data_by_default(self):
        fake_minio = _FakeMinio({"bucket-a": ["alpha.pdf", "beta.pdf"]})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_delete_ingested_object_from_env") as cleanup:
                response = self.client.delete("/api/buckets/bucket-a?force=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["removed_objects"], 2)
        self.assertFalse(payload["delete_ingested"])
        self.assertIn("bucket-a", fake_minio.removed_buckets)
        cleanup.assert_not_called()

    def test_api_delete_file_removes_object_and_ingested_data(self):
        fake_minio = _FakeMinio({"bucket-a": ["folder/alpha.pdf"]})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_delete_ingested_object_from_env") as cleanup:
                response = self.client.delete(
                    "/api/buckets/bucket-a/files/folder/alpha.pdf?delete_ingested=true"
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], True)
        self.assertNotIn("folder/alpha.pdf", fake_minio._buckets["bucket-a"])
        cleanup.assert_called_once_with(bucket="bucket-a", object_name="folder/alpha.pdf", version_id=None)

    def test_api_delete_file_preserves_ingested_data_by_default(self):
        fake_minio = _FakeMinio({"bucket-a": ["folder/alpha.pdf"]})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_delete_ingested_object_from_env") as cleanup:
                response = self.client.delete("/api/buckets/bucket-a/files/folder/alpha.pdf")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["deleted"])
        self.assertFalse(payload["delete_ingested"])
        self.assertNotIn("folder/alpha.pdf", fake_minio._buckets["bucket-a"])
        cleanup.assert_not_called()

    def test_api_upload_file_forces_ingest_job_even_when_flag_is_false(self):
        fake_minio = _FakeMinio({"bucket-a": []})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_start_ingest_job", return_value="job-123"):
                response = self.client.post(
                    "/api/buckets/bucket-a/files/upload",
                    data={"ingest": "false", "create_bucket": "false"},
                    files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["bucket"], "bucket-a")
        self.assertEqual(payload["object_name"], "paper.pdf")
        self.assertEqual(payload["job_id"], "job-123")

    def test_api_upload_multiple_files_queues_celery_ingest_tasks(self):
        fake_minio = _FakeMinio({"bucket-a": []})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(
                bibtex_ui_app,
                "_queue_ingest_task_celery",
                side_effect=["task-1", "task-2"],
            ) as queue_task:
                with mock.patch.object(bibtex_ui_app, "_start_ingest_job") as start_job:
                    response = self.client.post(
                        "/api/buckets/bucket-a/files/upload",
                        data={"ingest": "true", "create_bucket": "false"},
                        files=[
                            ("files", ("paper-a.pdf", b"%PDF-1.4", "application/pdf")),
                            ("files", ("paper-b.pdf", b"%PDF-1.4", "application/pdf")),
                        ],
                    )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "batch")
        self.assertEqual(payload["uploaded_count"], 2)
        self.assertEqual(payload["queued_count"], 2)
        self.assertEqual(len(payload["queue_task_ids"]), 2)
        self.assertEqual(payload["queue_errors"], [])
        self.assertIsNone(payload["job_id"])
        self.assertEqual(queue_task.call_count, 2)
        start_job.assert_not_called()
        self.assertIn("paper-a.pdf", fake_minio._buckets["bucket-a"])
        self.assertIn("paper-b.pdf", fake_minio._buckets["bucket-a"])

    def test_api_upload_rejects_non_pdf_file(self):
        fake_minio = _FakeMinio({"bucket-a": []})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            response = self.client.post(
                "/api/buckets/bucket-a/files/upload",
                data={"ingest": "false", "create_bucket": "false"},
                files={"file": ("notes.md", b"# hello", "text/markdown")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF files are supported", response.json().get("detail", ""))

    def test_api_save_file_bibtex_persists_to_redis_prefix(self):
        fake_redis = _FakeRedis()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                response = self.client.put(
                    "/api/buckets/bucket-a/files/folder%2Falpha.pdf/bibtex",
                    json={
                        "title": "Updated Alpha",
                        "year": "2025",
                        "citationKey": "alpha2025updated",
                        "entryType": "incollection",
                        "editors": "John Editor; Jane Editor",
                        "publisher": "ACM Press",
                        "authors": [
                            {"firstName": "Ada", "lastName": "Lovelace"},
                            {"firstName": "Grace", "lastName": "Hopper"},
                        ],
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["redis_key"], "bibtex:file:bucket-a/folder/alpha.pdf")
        self.assertEqual(body["file"]["title"], "Updated Alpha")
        self.assertEqual(body["file"]["entryType"], "incollection")
        self.assertEqual(body["file"]["editors"], "John Editor; Jane Editor")
        self.assertEqual(body["file"]["publisher"], "ACM Press")
        self.assertIn("bucket-a/folder/alpha.pdf", fake_redis.sets.get("bibtex:files", set()))

        stored = json.loads(fake_redis.get("bibtex:file:bucket-a/folder/alpha.pdf"))
        self.assertEqual(stored["year"], "2025")
        self.assertEqual(stored["editors"], "John Editor; Jane Editor")
        self.assertEqual(stored["authors"][1]["lastName"], "Hopper")

    def test_api_file_redis_summary_counts_partitions_and_chunks(self):
        fake_redis = _FakeRedis()
        fake_redis.sets["unstructured:pdf:source:bucket-a/folder/alpha.pdf"] = {"doc1"}
        fake_redis.values["unstructured:pdf:doc1:partitions"] = json.dumps([{"text": "p1"}, {"text": "p2"}])
        fake_redis.values["unstructured:pdf:doc1:chunks"] = json.dumps([{"text": "c1"}, {"text": "c2"}, {"text": "c3"}])

        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"REDIS_PREFIX": "unstructured"}, clear=False):
                response = self.client.get("/api/buckets/bucket-a/files/folder%2Falpha.pdf/redis-summary")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["partition_count"], 2)
        self.assertEqual(payload["chunk_count"], 3)
        self.assertEqual(payload["doc_ids"], ["doc1"])

    def test_api_save_file_bibtex_preserves_trailing_spaces_in_title(self):
        fake_redis = _FakeRedis()
        trailing_title = "Updated Alpha "

        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                response = self.client.put(
                    "/api/buckets/bucket-a/files/folder%2Falpha.pdf/bibtex",
                    json={"title": trailing_title, "year": "2025"},
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["file"]["title"], trailing_title)

        stored = json.loads(fake_redis.get("bibtex:file:bucket-a/folder/alpha.pdf"))
        self.assertEqual(stored["title"], trailing_title)

    def test_api_file_redis_data_lazy_limits_items(self):
        fake_redis = _FakeRedis()
        fake_redis.sets["unstructured:pdf:source:bucket-a/folder/alpha.pdf"] = {"doc1"}
        fake_redis.values["unstructured:pdf:doc1:chunks"] = json.dumps(
            [{"text": "chunk one"}, {"text": "chunk two"}, {"text": "chunk three"}]
        )

        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"REDIS_PREFIX": "unstructured"}, clear=False):
                response = self.client.get("/api/buckets/bucket-a/files/folder%2Falpha.pdf/redis-data?kind=chunks&limit=2")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["kind"], "chunks")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["total_available"], 3)
        self.assertTrue(payload["truncated"])
        self.assertIn("chunk one", payload["items"][0]["text"])

    def test_api_file_redis_summary_reads_v2_schema(self):
        fake_redis = _FakeRedis()
        sid = source_id("bucket-a", "folder/alpha.pdf", None)
        fake_redis.set(redis_v2_source_doc_key("unstructured", sid), "doc-v2")
        fake_redis.zadd(redis_v2_doc_partition_hashes_key("unstructured", "doc-v2"), {"p1": 1, "p2": 2})
        fake_redis.zadd(redis_v2_doc_chunk_hashes_key("unstructured", "doc-v2"), {"c1": 1, "c2": 2, "c3": 3})
        fake_redis.set(redis_v2_partition_key("unstructured", "p1"), json.dumps({"text": "part one"}))
        fake_redis.set(redis_v2_partition_key("unstructured", "p2"), json.dumps({"text": "part two"}))
        fake_redis.set(redis_v2_chunk_key("unstructured", "c1"), json.dumps({"text": "chunk one"}))
        fake_redis.set(redis_v2_chunk_key("unstructured", "c2"), json.dumps({"text": "chunk two"}))
        fake_redis.set(redis_v2_chunk_key("unstructured", "c3"), json.dumps({"text": "chunk three"}))

        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"REDIS_PREFIX": "unstructured"}, clear=False):
                response = self.client.get("/api/buckets/bucket-a/files/folder%2Falpha.pdf/redis-summary")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["doc_ids"], ["doc-v2"])
        self.assertEqual(payload["partition_count"], 2)
        self.assertEqual(payload["chunk_count"], 3)

    def test_api_bucket_autofill_missing_processes_in_batches(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")
        fake_minio = _FakeMinio({"bucket-a": ["alpha.pdf", "beta.pdf", "gamma.pdf"]})
        fake_redis = _FakeRedis()
        calls = []

        def fake_enrich_file_metadata(**kwargs):
            calls.append(kwargs)
            object_name = kwargs["object_name"]
            if object_name == "alpha.pdf":
                return {"status": "updated"}
            if object_name == "beta.pdf":
                return {"status": "no_match"}
            return {"status": "skipped_existing"}

        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
                with mock.patch.object(bibtex_ui_app.bibtex_autofill, "CrossrefClient", return_value=object()):
                    with mock.patch.object(
                        bibtex_ui_app.bibtex_autofill,
                        "enrich_file_metadata",
                        side_effect=fake_enrich_file_metadata,
                    ):
                        first = self.client.post(
                            "/api/buckets/bucket-a/autofill-missing",
                            json={
                                "objectNames": ["alpha.pdf", "beta.pdf", "gamma.pdf"],
                                "offset": 0,
                                "batchSize": 2,
                            },
                        )
                        second = self.client.post(
                            "/api/buckets/bucket-a/autofill-missing",
                            json={
                                "objectNames": ["alpha.pdf", "beta.pdf", "gamma.pdf"],
                                "offset": 2,
                                "batchSize": 2,
                            },
                        )

        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertEqual(first_payload["processed_total"], 2)
        self.assertEqual(first_payload["processed_in_batch"], 2)
        self.assertFalse(first_payload["done"])
        self.assertEqual(first_payload["next_offset"], 2)
        self.assertEqual(first_payload["counts"]["updated"], 1)
        self.assertEqual(first_payload["counts"]["no_match"], 1)

        self.assertEqual(second.status_code, 200)
        second_payload = second.json()
        self.assertEqual(second_payload["processed_total"], 3)
        self.assertEqual(second_payload["processed_in_batch"], 1)
        self.assertTrue(second_payload["done"])
        self.assertIsNone(second_payload["next_offset"])
        self.assertEqual(second_payload["counts"]["skipped_existing"], 1)

        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertFalse(call["overwrite"])
            self.assertTrue(call["skip_complete"])
            self.assertFalse(call["dry_run"])

    def test_api_file_lookup_by_doi_overwrites_metadata_from_crossref(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")

        fake_redis = _FakeRedis()
        fake_redis.set(
            "bibtex:file:bucket-a/folder/alpha.pdf",
            json.dumps(
                {
                    "title": "Old Title",
                    "doi": "10.1111/old",
                    "citationKey": "oldkey",
                    "entryType": "article",
                    "authors": [{"firstName": "Old", "lastName": "Author"}],
                }
            ),
        )

        class _FakeCrossrefClient:
            def lookup_by_doi(self, doi):
                self.last_doi = doi
                return {
                    "DOI": "10.1000/testdoi",
                    "title": ["CrossRef Title"],
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "issued": {"date-parts": [[2024]]},
                    "type": "journal-article",
                    "container-title": ["Journal of Testing"],
                    "URL": "https://doi.org/10.1000/testdoi",
                }

        fake_crossref = _FakeCrossrefClient()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.object(bibtex_ui_app, "_crossref_client_from_env", return_value=fake_crossref):
                with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                    response = self.client.post(
                        "/api/buckets/bucket-a/files/folder%2Falpha.pdf/lookup-by-doi",
                        json={"doi": "10.1000/testdoi", "overwrite": True},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["doi"], "10.1000/testdoi")
        self.assertEqual(body["file"]["title"], "CrossRef Title")
        self.assertEqual(body["file"]["year"], "2024")
        self.assertEqual(body["file"]["doi"], "10.1000/testdoi")
        self.assertEqual(body["file"]["authors"][0]["lastName"], "Lovelace")

        stored = json.loads(fake_redis.get("bibtex:file:bucket-a/folder/alpha.pdf"))
        self.assertEqual(stored["title"], "CrossRef Title")
        self.assertEqual(stored["doi"], "10.1000/testdoi")

    def test_api_file_lookup_by_doi_keeps_existing_authors_when_crossref_has_no_author(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")

        fake_redis = _FakeRedis()
        fake_redis.set(
            "bibtex:file:bucket-a/folder/alpha.pdf",
            json.dumps(
                {
                    "title": "Existing",
                    "doi": "10.1111/old",
                    "citationKey": "oldkey",
                    "entryType": "article",
                    "authors": [{"firstName": "Existing", "lastName": "Author"}],
                }
            ),
        )

        class _FakeCrossrefClient:
            def lookup_by_doi(self, doi):
                self.last_doi = doi
                return {
                    "DOI": "10.1000/noauthors",
                    "title": ["CrossRef Without Author"],
                    "issued": {"date-parts": [[2024]]},
                    "type": "journal-article",
                    "container-title": ["Journal of Testing"],
                    "URL": "https://doi.org/10.1000/noauthors",
                }

        fake_crossref = _FakeCrossrefClient()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.object(bibtex_ui_app, "_crossref_client_from_env", return_value=fake_crossref):
                with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                    response = self.client.post(
                        "/api/buckets/bucket-a/files/folder%2Falpha.pdf/lookup-by-doi",
                        json={"doi": "10.1000/noauthors", "overwrite": True},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["file"]["authors"][0]["firstName"], "Existing")
        self.assertEqual(body["file"]["authors"][0]["lastName"], "Author")

    def test_api_file_lookup_by_doi_uses_parent_doi_authors_for_chapter_doi(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")

        fake_redis = _FakeRedis()

        class _FakeCrossrefClient:
            def lookup_by_doi(self, doi):
                if doi == "10.4324/9780203392300-11":
                    return {
                        "DOI": "10.4324/9780203392300-11",
                        "title": ["Using procurement offsets as an economic development strategy"],
                        "type": "book-chapter",
                        "container-title": ["Arms Trade and Economic Development"],
                        "issued": {"date-parts": [[2004]]},
                        "URL": "https://doi.org/10.4324/9780203392300-11",
                    }
                if doi == "10.4324/9780203392300":
                    return {
                        "DOI": "10.4324/9780203392300",
                        "title": ["Arms Trade and Economic Development"],
                        "type": "monograph",
                        "author": [
                            {"given": "Jurgen", "family": "Brauer"},
                            {"given": "Paul", "family": "Dunne"},
                        ],
                    }
                return None

        fake_crossref = _FakeCrossrefClient()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.object(bibtex_ui_app, "_crossref_client_from_env", return_value=fake_crossref):
                with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                    response = self.client.post(
                        "/api/buckets/bucket-a/files/folder%2Falpha.pdf/lookup-by-doi",
                        json={"doi": "10.4324/9780203392300-11", "overwrite": True},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["file"]["authors"][0]["lastName"], "Brauer")
        self.assertEqual(body["file"]["authors"][1]["lastName"], "Dunne")

    def test_api_file_lookup_by_doi_requires_valid_doi(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")

        fake_redis = _FakeRedis()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            response = self.client.post(
                "/api/buckets/bucket-a/files/folder%2Falpha.pdf/lookup-by-doi",
                json={"doi": ""},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid DOI", response.json().get("detail", ""))

    def test_api_save_file_bibtex_accepts_extended_entry_types_and_fields(self):
        fake_redis = _FakeRedis()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                response = self.client.put(
                    "/api/buckets/bucket-a/files/folder%2Fthesis.pdf/bibtex",
                    json={
                        "citationKey": "examplethesis2026",
                        "entryType": "phdthesis",
                        "title": "An Example Dissertation",
                        "year": "2026",
                        "authors": [{"firstName": "Ada", "lastName": "Lovelace"}],
                        "school": "Example University",
                        "address": "Boston",
                        "month": "jan",
                        "type": "PhD dissertation",
                        "annote": "Extended notes",
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["file"]["entryType"], "phdthesis")
        self.assertEqual(body["file"]["school"], "Example University")
        self.assertEqual(body["file"]["address"], "Boston")
        self.assertEqual(body["file"]["type"], "PhD dissertation")
        self.assertEqual(body["file"]["annote"], "Extended notes")

    def test_api_file_lookup_by_doi_handles_literal_author_payload(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")

        fake_redis = _FakeRedis()

        class _FakeCrossrefClient:
            def lookup_by_doi(self, doi):
                self.last_doi = doi
                return {
                    "DOI": "10.1000/literal",
                    "title": ["CrossRef Literal Author"],
                    "author": [{"literal": "OpenAI Research"}],
                    "issued": {"date-parts": [[2026]]},
                    "type": "journal-article",
                    "container-title": ["Journal of Testing"],
                    "URL": "https://doi.org/10.1000/literal",
                }

        fake_crossref = _FakeCrossrefClient()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.object(bibtex_ui_app, "_crossref_client_from_env", return_value=fake_crossref):
                with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                    response = self.client.post(
                        "/api/buckets/bucket-a/files/folder%2Falpha.pdf/lookup-by-doi",
                        json={"doi": "10.1000/literal", "overwrite": True},
                    )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["file"]["authors"][0]["firstName"], "OpenAI")
        self.assertEqual(payload["file"]["authors"][0]["lastName"], "Research")

    def test_api_save_file_bibtex_writes_v2_keys_in_v2_mode(self):
        fake_redis = _FakeRedis()
        fake_redis.sets["unstructured:pdf:source:bucket-a/folder/alpha.pdf"] = {"doc-alpha"}
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(
                os.environ,
                {
                    "BIBTEX_REDIS_PREFIX": "bibtex",
                    "BIBTEX_SCHEMA_WRITE_MODE": "v2",
                    "BIBTEX_SOURCE_REDIS_PREFIX": "unstructured",
                },
                clear=False,
            ):
                response = self.client.put(
                    "/api/buckets/bucket-a/files/folder%2Falpha.pdf/bibtex",
                    json={
                        "title": "Updated Alpha",
                        "year": "2025",
                        "citationKey": "alpha2025updated",
                        "entryType": "article",
                        "authors": [{"firstName": "Ada", "lastName": "Lovelace"}],
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("bibtex:file:bucket-a/folder/alpha.pdf", fake_redis.values)
        self.assertIn("bibtex:v2:doc:doc-alpha", fake_redis.values)
        sid = source_id("bucket-a", "folder/alpha.pdf", None)
        self.assertEqual(fake_redis.values.get(f"bibtex:v2:source:{sid}:doc"), "doc-alpha")


if __name__ == "__main__":
    unittest.main()
