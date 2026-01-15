import os
import unittest

import httpx
from dotenv import load_dotenv


load_dotenv(".env")


@unittest.skipUnless(
    os.getenv("RUN_E2E_TESTS") == "1",
    "Set RUN_E2E_TESTS=1 to run end-to-end integration tests.",
)
class ResolverE2ETests(unittest.TestCase):
    def setUp(self):
        self.base_url = os.getenv("RESOLVER_BASE_URL", "http://localhost:8080").rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def test_resolve_doc_json_presign(self):
        params = {"bucket": "e2e-bucket", "key": "sample.txt", "mode": "presign"}
        try:
            response = httpx.get(self._url("/r/doc.json"), params=params, timeout=5.0)
        except httpx.RequestError as exc:
            self.skipTest(f"Resolver not reachable at {self.base_url}: {exc}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("mode"), "presign")
        self.assertIn("url", payload)
        self.assertTrue(payload["url"])
        self.assertIn("source_ref", payload)

    def test_resolve_doc_redirects(self):
        params = {"bucket": "e2e-bucket", "key": "sample.txt", "mode": "presign"}
        try:
            response = httpx.get(
                self._url("/r/doc"),
                params=params,
                timeout=5.0,
                follow_redirects=False,
            )
        except httpx.RequestError as exc:
            self.skipTest(f"Resolver not reachable at {self.base_url}: {exc}")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers.get("location"))


if __name__ == "__main__":
    unittest.main()
