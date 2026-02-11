import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import search_ui_app

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]


class _FakeTools:
    QDRANT_URL = "http://qdrant.local:6333"
    QDRANT_COLLECTION = "default_col"
    REDIS_URL = "redis://redis:6379/0"

    def __init__(self):
        self.last_search = None
        self.last_fetch = None
        self.last_fetch_chunk_document = None
        self.last_fetch_chunk_partition = None
        self.last_fetch_chunk_bibtex = None
        self.default_collection = "default_col"

    def ping(self):
        return "pong"

    def _get_default_collection(self):
        return self.default_collection

    def list_collections(self):
        return {"collections": ["default_col", "research_2026"]}

    def set_default_collection(self, name):
        if name == "missing":
            raise ValueError("Collection not found: missing")
        self.default_collection = name
        return {"default_collection": name}

    def search(self, query, top_k=5, prefetch_k=40, collection=None, retrieval_mode="hybrid"):
        self.last_search = {
            "query": query,
            "top_k": top_k,
            "prefetch_k": prefetch_k,
            "collection": collection,
            "retrieval_mode": retrieval_mode,
        }
        return {
            "retrieval_mode": retrieval_mode,
            "results": [
                {
                    "id": "42",
                    "score": 0.9,
                    "text": "result text",
                    "bucket": "default_col",
                    "key": "paper.pdf",
                    "citation_key": "doe2024paper",
                }
            ]
        }

    def fetch(self, id, collection=None):
        self.last_fetch = {"id": id, "collection": collection}
        return {"id": id, "found": True, "text": "full chunk"}

    def fetch_chunk_document(self, id, collection=None):
        self.last_fetch_chunk_document = {"id": id, "collection": collection}
        return {
            "id": id,
            "found": True,
            "count": 2,
            "chunks": [{"text": "first"}, {"text": "second"}],
        }

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
        return {
            "id": id,
            "found": True,
            "citation_key": "doe2024paper",
            "metadata": {"citationKey": "doe2024paper", "title": "Paper"},
        }


class _ToolWrapper:
    def __init__(self, fn):
        self.fn = fn


@unittest.skipIf(search_ui_app.app is None or TestClient is None, "FastAPI test client unavailable")
class SearchUiAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(search_ui_app.app)
        self.fake_tools = _FakeTools()
        self.tools_patch = mock.patch.object(search_ui_app, "mcp_tools", self.fake_tools)
        self.tools_patch.start()

    def tearDown(self):
        self.tools_patch.stop()

    def test_index_serves_search_workspace_shell(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Qdrant Search", response.text)
        self.assertIn("Search Workspace", response.text)
        self.assertIn("top_k (default 8)", response.text)
        self.assertIn("prefetch_k (default 60)", response.text)
        self.assertIn("retrieval mode", response.text)
        self.assertIn("Hybrid search", response.text)
        self.assertIn("Cosine search", response.text)

    def test_favicon_returns_no_content(self):
        response = self.client.get("/favicon.ico")
        self.assertEqual(response.status_code, 204)

    def test_api_status_reports_qdrant_and_default_collection(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["qdrant_url"], "http://qdrant.local:6333")
        self.assertEqual(payload["default_collection"], "default_col")

    def test_api_collections_lists_collections(self):
        response = self.client.get("/api/collections")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["collections"], ["default_col", "research_2026"])
        self.assertEqual(payload["default_collection"], "default_col")

    def test_api_set_default_collection_updates_selection(self):
        response = self.client.post("/api/default-collection", json={"collection": "research_2026"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["default_collection"], "research_2026")

    def test_api_set_default_collection_returns_404_for_missing_collection(self):
        response = self.client.post("/api/default-collection", json={"collection": "missing"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("Collection not found", response.json()["detail"])

    def test_api_search_requires_query(self):
        response = self.client.post("/api/search", json={"query": ""})
        self.assertEqual(response.status_code, 400)
        self.assertIn("query is required", response.json()["detail"])

    def test_api_search_uses_requested_parameters(self):
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
        self.assertEqual(self.fake_tools.last_search["query"], "hybrid ranking")
        self.assertEqual(self.fake_tools.last_search["top_k"], 11)
        self.assertEqual(self.fake_tools.last_search["prefetch_k"], 120)
        self.assertEqual(self.fake_tools.last_search["collection"], "research_2026")
        self.assertEqual(self.fake_tools.last_search["retrieval_mode"], "cosine")
        payload = response.json()
        self.assertEqual(payload["collection"], "research_2026")
        self.assertEqual(payload["retrieval_mode"], "cosine")
        self.assertEqual(len(payload["results"]), 1)

    def test_api_fetch_uses_point_id_and_collection(self):
        response = self.client.get("/api/fetch/42?collection=research_2026")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["found"])
        self.assertEqual(self.fake_tools.last_fetch, {"id": "42", "collection": "research_2026"})

    def test_api_chunk_document_uses_point_id_and_collection(self):
        response = self.client.get("/api/chunk/42/document?collection=research_2026")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["found"])
        self.assertEqual(self.fake_tools.last_fetch_chunk_document, {"id": "42", "collection": "research_2026"})

    def test_api_chunk_partition_uses_point_id_and_collection(self):
        response = self.client.get("/api/chunk/42/partition?collection=research_2026")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["found"])
        self.assertEqual(self.fake_tools.last_fetch_chunk_partition, {"id": "42", "collection": "research_2026"})

    def test_api_chunk_bibtex_uses_point_id_and_collection(self):
        response = self.client.get("/api/chunk/42/bibtex?collection=research_2026")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["found"])
        self.assertEqual(self.fake_tools.last_fetch_chunk_bibtex, {"id": "42", "collection": "research_2026"})

    def test_api_supports_fn_wrapped_tools(self):
        self.fake_tools.ping = _ToolWrapper(self.fake_tools.ping)
        self.fake_tools.list_collections = _ToolWrapper(self.fake_tools.list_collections)

        status_response = self.client.get("/api/status")
        self.assertEqual(status_response.status_code, 200)
        self.assertTrue(status_response.json()["ok"])

        collections_response = self.client.get("/api/collections")
        self.assertEqual(collections_response.status_code, 200)
        self.assertEqual(collections_response.json()["collections"], ["default_col", "research_2026"])


if __name__ == "__main__":
    unittest.main()
