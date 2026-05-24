from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.ha_client import HomeAssistantClient, HomeAssistantError


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


class FakeClock:
    def __init__(self, times: list[float]) -> None:
        self._times = iter(times)
        self.sleep = Mock()

    def time(self) -> float:
        return next(self._times)


class HomeAssistantClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = HomeAssistantClient(
            base_url="http://example.test:8123",
            token="test-token",
            camera_entity="camera.placeholder",
        )

    def test_retries_after_transient_http_500_and_uses_cache_buster(self) -> None:
        requested_urls: list[str] = []
        authorization_headers: list[str | None] = []
        clock = FakeClock([1000.001, 1000.002])

        def fake_urlopen(request: object, timeout: float) -> FakeResponse:
            del timeout
            requested_urls.append(request.full_url)  # type: ignore[attr-defined]
            authorization_headers.append(request.get_header("Authorization"))  # type: ignore[attr-defined]
            if len(requested_urls) == 1:
                raise HTTPError(request.full_url, 500, "Internal Server Error", None, None)  # type: ignore[attr-defined]
            return FakeResponse(b"image-bytes")

        with (
            patch("yorkie_watch.ha_client.urlopen", side_effect=fake_urlopen),
            patch("yorkie_watch.ha_client.time", clock),
            self.assertLogs("yorkie_watch.ha_client", level="WARNING") as captured_logs,
        ):
            image_bytes = self.client.fetch_snapshot(delay_seconds=0.01)

        self.assertEqual(image_bytes, b"image-bytes")
        self.assertEqual(len(requested_urls), 2)
        self.assertEqual(authorization_headers, ["Bearer test-token", "Bearer test-token"])
        self.assertIn("?ts=1000001", requested_urls[0])
        self.assertIn("?ts=1000002", requested_urls[1])
        self.assertNotEqual(requested_urls[0], requested_urls[1])
        clock.sleep.assert_called_once_with(0.01)
        log_output = "\n".join(captured_logs.output)
        self.assertIn("attempt 1/3 failed", log_output)
        self.assertNotIn("test-token", log_output)

    def test_raises_after_final_failed_attempt_with_http_500_message(self) -> None:
        clock = FakeClock([1000.001, 1000.002, 1000.003])

        def fake_urlopen(request: object, timeout: float) -> FakeResponse:
            del timeout
            raise HTTPError(request.full_url, 500, "Internal Server Error", None, None)  # type: ignore[attr-defined]

        with (
            patch("yorkie_watch.ha_client.urlopen", side_effect=fake_urlopen) as urlopen_mock,
            patch("yorkie_watch.ha_client.time", clock),
            self.assertLogs("yorkie_watch.ha_client", level="WARNING") as captured_logs,
        ):
            with self.assertRaises(HomeAssistantError) as error:
                self.client.fetch_snapshot()

        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(clock.sleep.call_count, 2)
        self.assertIn("HTTP 500", str(error.exception))
        self.assertIn("camera backend may be temporarily unavailable", str(error.exception))
        log_output = "\n".join(captured_logs.output)
        self.assertIn("attempt 1/3 failed", log_output)
        self.assertIn("attempt 2/3 failed", log_output)
        self.assertIn("attempt 3/3 failed", log_output)
        self.assertNotIn("test-token", log_output)


if __name__ == "__main__":
    unittest.main()
