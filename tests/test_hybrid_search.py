import os
import sys
import unittest

import httpx
from dotenv import load_dotenv

load_dotenv(".env")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from qdrant_client import QdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from mcp_research import hybrid_search

TEST_DOC_TARGET = "mcp_research.hybrid_search"
TEST_DOC_METHOD = (
    "Combines fake embedding models and fake/live Qdrant clients to verify vector construction, upsert payloads, and hybrid query orchestration."
)


class _Sparse:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


class _DenseModel:
    def __init__(self, vectors):
        self._vectors = vectors

    def embed(self, docs):
        # Return one vector per doc.
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
        self.query_calls = []
        self.query_result = object()

    def upsert(self, collection_name, points):
        self.upsert_calls.append((collection_name, points))

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return self.query_result


class HybridSearchTests(unittest.TestCase):
    def test_embed_corpus_builds_dense_and_sparse_vectors(self):
        dense_model = _DenseModel([[1.0, 0.0], [0.5, 0.5]])
        sparse_model = _SparseModel([
            ([1, 2], [0.2, 0.8]),
            ([3], [1.0]),
        ])
        docs = ["alpha", "beta"]

        dense_vectors, sparse_vectors = hybrid_search.embed_corpus(
            dense_model=dense_model,
            sparse_model=sparse_model,
            docs=docs,
        )

        self.assertEqual(dense_vectors, [[1.0, 0.0], [0.5, 0.5]])
        self.assertEqual(len(sparse_vectors), 2)
        self.assertIsInstance(sparse_vectors[0], models.SparseVector)
        self.assertEqual(sparse_vectors[0].indices, [1, 2])
        self.assertEqual(sparse_vectors[0].values, [0.2, 0.8])

    def test_upsert_docs_sends_payloads_and_vectors(self):
        client = _Client()
        docs = ["doc one", "doc two"]
        dense_vectors = [[0.1, 0.2], [0.3, 0.4]]
        sparse_vectors = [
            models.SparseVector(indices=[1], values=[0.9]),
            models.SparseVector(indices=[2], values=[0.8]),
        ]

        hybrid_search.upsert_docs(
            client=client,
            collection_name="demo",
            docs=docs,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )

        self.assertEqual(len(client.upsert_calls), 1)
        collection_name, points = client.upsert_calls[0]
        self.assertEqual(collection_name, "demo")
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].payload, {"text": "doc one"})
        self.assertIn("dense", points[0].vector)
        self.assertIn("sparse", points[0].vector)

    def test_hybrid_search_calls_query_points(self):
        client = _Client()
        dense_model = _DenseModel([[0.2, 0.8]])
        sparse_model = _SparseModel([
            ([4, 5], [0.6, 0.4]),
        ])

        result = hybrid_search.hybrid_search(
            client=client,
            collection_name="demo",
            dense_model=dense_model,
            sparse_model=sparse_model,
            query_text="hello",
            top_k=3,
            prefetch_k=7,
        )

        self.assertIs(result, client.query_result)
        self.assertEqual(len(client.query_calls), 1)
        call = client.query_calls[0]
        self.assertEqual(call["collection_name"], "demo")
        self.assertEqual(call["limit"], 3)
        self.assertEqual(len(call["prefetch"]), 2)
        self.assertEqual(call["prefetch"][0].using, "sparse")
        self.assertEqual(call["prefetch"][0].limit, 7)
        self.assertEqual(call["prefetch"][1].using, "dense")
        self.assertEqual(call["prefetch"][1].limit, 7)
        self.assertTrue(call["with_payload"])


@unittest.skipUnless(
    os.getenv("RUN_QDRANT_LIVE_TESTS") == "1",
    "Set RUN_QDRANT_LIVE_TESTS=1 to run live Qdrant tests.",
)
class HybridSearchLiveTests(unittest.TestCase):
    def test_hybrid_search_hits_live_qdrant(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = "hybrid_search_live_test"
        client = QdrantClient(url=qdrant_url)

        dense_dim = 3
        dense_vectors = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]]
        sparse_vectors = [
            models.SparseVector(indices=[1, 2], values=[0.7, 0.3]),
            models.SparseVector(indices=[2, 3], values=[0.4, 0.6]),
        ]
        docs = ["first doc", "second doc"]

        try:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )
            hybrid_search.upsert_docs(
                client=client,
                collection_name=collection_name,
                docs=docs,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
            )

            dense_model = _DenseModel([[0.05, 0.1, 0.2]])
            sparse_model = _SparseModel([([1], [1.0])])
            result = hybrid_search.hybrid_search(
                client=client,
                collection_name=collection_name,
                dense_model=dense_model,
                sparse_model=sparse_model,
                query_text="live query",
                top_k=2,
                prefetch_k=5,
            )

            self.assertGreaterEqual(len(result.points), 1)
        except (httpx.ConnectError, qdrant_exceptions.ResponseHandlingException) as exc:
            self.skipTest(f"Qdrant not reachable at {qdrant_url}: {exc}")
        finally:
            try:
                if client.collection_exists(collection_name):
                    client.delete_collection(collection_name)
            except (httpx.ConnectError, qdrant_exceptions.ResponseHandlingException):
                pass


if __name__ == "__main__":
    unittest.main()
