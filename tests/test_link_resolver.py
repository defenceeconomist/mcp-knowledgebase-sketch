import os
import sys
import unittest
from datetime import timedelta
from unittest import mock

from dotenv import load_dotenv

load_dotenv(".env")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from fastapi.testclient import TestClient

from mcp_research import link_resolver, resolver_app


class LinkResolverTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_build_source_ref_encodes_key_and_pages(self):
        ref = link_resolver.build_source_ref(
            bucket="docs",
            key="/folder/file name.pdf",
            page_start=2,
            page_end=5,
            version_id="abc123",
        )

        self.assertEqual(
            ref,
            "doc://docs/folder/file%20name.pdf?version_id=abc123#page=2-5",
        )

    def test_parse_source_ref_from_portal_url(self):
        source_ref = "doc://bucket/path/file.pdf#page=3"
        portal_url = (
            "https://example.test/r/doc?ref=doc%3A%2F%2Fbucket%2Fpath%2Ffile.pdf%23page%3D3"
        )

        parsed = link_resolver.parse_source_ref(portal_url)

        self.assertEqual(parsed["bucket"], "bucket")
        self.assertEqual(parsed["key"], "path/file.pdf")
        self.assertEqual(parsed["page_start"], 3)
        self.assertEqual(parsed["page_end"], 3)
        self.assertEqual(parsed["version_id"], None)
        self.assertEqual(link_resolver._normalize_source_ref(portal_url), source_ref)

    def test_build_citation_url_uses_env_and_path(self):
        os.environ["CITATION_BASE_URL"] = "https://docs.example"
        os.environ["CITATION_REF_PATH"] = "refs/doc"

        url = link_resolver.build_citation_url("doc://bucket/file.txt")

        self.assertEqual(url, "https://docs.example/refs/doc?ref=doc%3A%2F%2Fbucket%2Ffile.txt")

    def test_resolve_link_portal_requires_base(self):
        os.environ.pop("CITATION_BASE_URL", None)
        os.environ.pop("DOCS_BASE_URL", None)

        with self.assertRaises(RuntimeError):
            link_resolver.resolve_link(bucket="bucket", key="file.txt")

    def test_resolve_link_cdn_builds_url(self):
        os.environ["CDN_BASE_URL"] = "https://cdn.example"

        result = link_resolver.resolve_link(
            bucket="bucket",
            key="/nested/file.txt",
            mode="cdn",
        )

        self.assertEqual(result["mode"], "cdn")
        self.assertEqual(result["url"], "https://cdn.example/nested/file.txt")

    def test_resolve_link_presign_uses_minio_client(self):
        os.environ["MINIO_PRESIGN_EXPIRY_SECONDS"] = "120"
        minio_client = mock.Mock()
        minio_client.presigned_get_object.return_value = "https://signed.example/object"

        with mock.patch(
            "mcp_research.link_resolver._get_minio_client",
            return_value=minio_client,
        ):
            result = link_resolver.resolve_link(
                bucket="bucket",
                key="file.txt",
                version_id="v1",
                mode="presign",
            )

        self.assertEqual(result["mode"], "presign")
        self.assertEqual(result["url"], "https://signed.example/object")
        minio_client.presigned_get_object.assert_called_once_with(
            bucket_name="bucket",
            object_name="file.txt",
            expires=timedelta(seconds=120),
            version_id="v1",
        )


class ResolverAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(resolver_app.app)

    def test_resolve_doc_redirects(self):
        with mock.patch(
            "mcp_research.resolver_app.resolve_link",
            return_value={"url": "https://example.test/target"},
        ):
            response = self.client.get(
                "/r/doc",
                params={"ref": "doc://bucket/file.txt"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers.get("location"), "https://example.test/target")

    def test_resolve_doc_json_returns_payload(self):
        payload = {"source_ref": "doc://bucket/file.txt", "url": "https://example.test/target"}
        with mock.patch("mcp_research.resolver_app.resolve_link", return_value=payload):
            response = self.client.get("/r/doc.json", params={"ref": "doc://bucket/file.txt"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)


if __name__ == "__main__":
    unittest.main()
