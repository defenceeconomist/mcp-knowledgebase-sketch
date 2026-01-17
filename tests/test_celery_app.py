import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import celery_app


class CeleryAppTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_env_int_invalid_returns_default(self):
        os.environ["CELERY_VISIBILITY_TIMEOUT"] = "nope"
        value = celery_app._env_int("CELERY_VISIBILITY_TIMEOUT", 3600)
        self.assertEqual(value, 3600)

    def test_make_celery_uses_env_urls(self):
        os.environ["CELERY_BROKER_URL"] = "redis://example/1"
        os.environ["CELERY_RESULT_BACKEND"] = "redis://example/2"

        app = celery_app.make_celery()

        self.assertEqual(app.conf.broker_url, "redis://example/1")
        self.assertEqual(app.conf.result_backend, "redis://example/2")


if __name__ == "__main__":
    unittest.main()
