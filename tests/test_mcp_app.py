import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import mcp_app


class McpAppTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
