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

    def test_resolve_link_portal_defaults_to_local_base(self):
        os.environ.pop("CITATION_BASE_URL", None)
        os.environ.pop("DOCS_BASE_URL", None)

        result = link_resolver.resolve_link(bucket="bucket", key="file.txt")
        self.assertEqual(
            result["url"],
            "http://localhost:8080/r/doc?ref=doc%3A%2F%2Fbucket%2Ffile.txt",
        )
        self.assertEqual(result["mode"], "portal")

    def test_resolve_link_portal_appends_page_and_highlight_query(self):
        os.environ["CITATION_BASE_URL"] = "https://docs.example"
        os.environ["CITATION_REF_PATH"] = "/r/doc"

        result = link_resolver.resolve_link(
            source_ref="doc://bucket/file.pdf",
            page_start=7,
            page_end=7,
            highlight="transfer pricing",
            mode="portal",
        )

        self.assertEqual(
            result["url"],
            "https://docs.example/r/doc?ref=doc%3A%2F%2Fbucket%2Ffile.pdf&page=7&page_start=7&page_end=7&highlight=transfer+pricing",
        )
        self.assertEqual(result["mode"], "portal")

    def test_resolve_link_cdn_builds_url(self):
        os.environ["CDN_BASE_URL"] = "https://cdn.example"

        result = link_resolver.resolve_link(
            bucket="bucket",
            key="/nested/file.txt",
            mode="cdn",
        )

        self.assertEqual(result["mode"], "cdn")
        self.assertEqual(result["url"], "https://cdn.example/nested/file.txt")

    def test_resolve_link_proxy_builds_reverse_proxy_url(self):
        os.environ["CITATION_BASE_URL"] = "https://resolver.example"
        os.environ["CITATION_PDF_PROXY_PATH"] = "/r/pdf-proxy"

        result = link_resolver.resolve_link(
            source_ref="doc://bucket/path/file.pdf",
            page=4,
            highlight="transfer pricing",
            mode="proxy",
        )

        self.assertEqual(result["mode"], "proxy")
        self.assertEqual(
            result["url"],
            "https://resolver.example/r/pdf-proxy?ref=doc%3A%2F%2Fbucket%2Fpath%2Ffile.pdf#page=4&search=transfer%20pricing",
        )

    def test_resolve_link_proxy_uses_relative_path_for_localhost_base(self):
        os.environ["CITATION_BASE_URL"] = "http://localhost:8080"
        os.environ["CITATION_PDF_PROXY_PATH"] = "/r/pdf-proxy"

        result = link_resolver.resolve_link(
            source_ref="doc://bucket/path/file.pdf",
            mode="proxy",
        )

        self.assertEqual(result["mode"], "proxy")
        self.assertEqual(
            result["url"],
            "/r/pdf-proxy?ref=doc%3A%2F%2Fbucket%2Fpath%2Ffile.pdf",
        )

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

    def test_resolve_link_presign_appends_page_fragment_from_source_ref(self):
        minio_client = mock.Mock()
        minio_client.presigned_get_object.return_value = "https://signed.example/object?sig=abc"

        with mock.patch(
            "mcp_research.link_resolver._get_minio_client",
            return_value=minio_client,
        ):
            result = link_resolver.resolve_link(
                source_ref="doc://bucket/path/file.pdf#page=10",
                mode="presign",
            )

        self.assertEqual(result["url"], "https://signed.example/object?sig=abc#page=10")

    def test_resolve_link_presign_appends_highlight_fragment(self):
        minio_client = mock.Mock()
        minio_client.presigned_get_object.return_value = "https://signed.example/object?sig=abc"

        with mock.patch(
            "mcp_research.link_resolver._get_minio_client",
            return_value=minio_client,
        ):
            result = link_resolver.resolve_link(
                bucket="bucket",
                key="path/file.pdf",
                page=7,
                highlight="transfer pricing",
                mode="presign",
            )

        self.assertEqual(
            result["url"],
            "https://signed.example/object?sig=abc#page=7&search=transfer%20pricing",
        )


class ResolverAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(resolver_app.app)

    def test_resolve_doc_redirects(self):
        with mock.patch(
            "mcp_research.resolver_app.resolve_link",
            return_value={"url": "https://example.test/target"},
        ) as resolve_link_mock:
            response = self.client.get(
                "/r/doc",
                params={"ref": "doc://bucket/file.txt", "highlight": "transfer pricing"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers.get("location"), "https://example.test/target")
        resolve_link_mock.assert_called_once_with(
            source_ref="doc://bucket/file.txt",
            bucket=None,
            key=None,
            version_id=None,
            page=None,
            page_start=None,
            page_end=None,
            highlight="transfer pricing",
            mode=None,
        )

    def test_resolve_doc_json_returns_payload(self):
        payload = {"source_ref": "doc://bucket/file.txt", "url": "https://example.test/target"}
        with mock.patch("mcp_research.resolver_app.resolve_link", return_value=payload):
            response = self.client.get("/r/doc.json", params={"ref": "doc://bucket/file.txt"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)

    def test_resolve_doc_focus_params_render_embed_page_for_presign(self):
        payload = {
            "source_ref": "doc://bucket/file.pdf#page=7",
            "url": "https://example.test/file.pdf#page=7&search=transfer%20pricing",
            "mode": "presign",
        }
        with mock.patch("mcp_research.resolver_app.resolve_link", return_value=payload):
            response = self.client.get(
                "/r/doc",
                params={
                    "ref": "doc://bucket/file.pdf#page=7",
                    "page_start": 7,
                    "page_end": 7,
                    "highlight": "transfer pricing",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("<iframe", response.text)
        self.assertIn("Open direct PDF", response.text)
        self.assertIn("file.pdf#page=7&amp;search=transfer%20pricing", response.text)

    def test_resolve_doc_proxy_mode_redirects_to_pdf_proxy(self):
        payload = {
            "source_ref": "doc://bucket/path/file.pdf",
            "url": "https://resolver.example/r/pdf-proxy?ref=doc%3A%2F%2Fbucket%2Fpath%2Ffile.pdf",
            "mode": "proxy",
        }
        with mock.patch("mcp_research.resolver_app.resolve_link", return_value=payload):
            response = self.client.get(
                "/r/doc",
                params={"ref": "doc://bucket/path/file.pdf", "mode": "proxy"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers.get("location"),
            "https://resolver.example/r/pdf-proxy?ref=doc%3A%2F%2Fbucket%2Fpath%2Ffile.pdf",
        )

    def test_resolve_pdf_proxy_returns_pdf_payload(self):
        class _FakeUpstream:
            def __init__(self):
                self.headers = {"Content-Type": "application/pdf"}

            def read(self):
                return b"%PDF-1.7\\n"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        with mock.patch("mcp_research.resolver_app.urllib.request.urlopen", return_value=_FakeUpstream()):
            response = self.client.get("/r/pdf-proxy", params={"url": "http://example.test/file.pdf"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/pdf")
        self.assertEqual(response.content, b"%PDF-1.7\\n")

    def test_resolve_pdf_proxy_rejects_non_http_url(self):
        response = self.client.get("/r/pdf-proxy", params={"url": "file:///tmp/demo.pdf"})
        self.assertEqual(response.status_code, 400)

    def test_resolve_pdf_proxy_with_ref_reads_from_minio(self):
        class _FakeObject:
            def read(self):
                return b"%PDF-ref\\n"

            def close(self):
                return None

            def release_conn(self):
                return None

        class _FakeMinio:
            def get_object(self, bucket, key, version_id=None):
                self.last_call = (bucket, key, version_id)
                return _FakeObject()

        fake_minio = _FakeMinio()
        with mock.patch("mcp_research.resolver_app._get_minio_client", return_value=(fake_minio, None)):
            response = self.client.get("/r/pdf-proxy", params={"ref": "doc://bucket/path/file.pdf?version_id=v1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"%PDF-ref\\n")
        self.assertEqual(fake_minio.last_call, ("bucket", "path/file.pdf", "v1"))


if __name__ == "__main__":
    unittest.main()
