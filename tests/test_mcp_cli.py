import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mcp_cli


class McpCliTests(unittest.TestCase):
    def test_main_help_exits_zero(self):
        with self.assertRaises(SystemExit) as exc:
            mcp_cli.main(["--help"])
        self.assertEqual(exc.exception.code, 0)

    def test_dispatch_forwards_arguments_to_module_main(self):
        seen_argv = {}

        def fake_main():
            seen_argv["value"] = list(sys.argv)

        fake_module = type("FakeModule", (), {"main": staticmethod(fake_main)})
        with mock.patch.object(mcp_cli.importlib, "import_module", return_value=fake_module):
            code = mcp_cli.main(["upsert-chunks", "--batch-size", "123"])

        self.assertEqual(code, 0)
        self.assertEqual(
            seen_argv["value"],
            ["mcp_research.upsert_chunks", "--batch-size", "123"],
        )


if __name__ == "__main__":
    unittest.main()
