from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.openclaw_client import OpenClawClient


class OpenClawClientTests(unittest.TestCase):
    def make_client(self) -> OpenClawClient:
        return OpenClawClient(
            notify_mode="ssh",
            ssh_host="openclaw-host-placeholder",
            ssh_user="openclaw-user-placeholder",
            whatsapp_target="+1 234-567-8900",
            ssh_media_remote_dir="/tmp/yorkie-watch",
            ssh_media_command_template=(
                "{binary} message send --channel {channel} --account {account} "
                "--target {target} --message {message} --media {media_path}"
            ),
        )

    def test_text_ssh_command_has_no_shell_wrapper(self) -> None:
        argv = self.make_client()._build_ssh_argv("Test message")

        self.assertEqual(argv[:5], ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"])
        self.assertIn("openclaw message send", argv[-1])
        self.assertNotIn("sh", argv)
        self.assertNotIn("-lc", argv)

    def test_media_command_template_quotes_values(self) -> None:
        client = self.make_client()

        command = client._build_remote_media_command(
            message="Dog detected near gate",
            remote_media_path="/tmp/yorkie-watch/snapshot 1.jpg",
        )

        self.assertIn("--target '+1 234-567-8900'", command)
        self.assertIn("--message 'Dog detected near gate'", command)
        self.assertIn("--media '/tmp/yorkie-watch/snapshot 1.jpg'", command)

    def test_ssh_output_redacts_target_and_host(self) -> None:
        client = self.make_client()

        output = client._redact_ssh_output(
            "failed for +12345678900 on openclaw-host-placeholder as openclaw-user-placeholder"
        )

        self.assertNotIn("+12345678900", output)
        self.assertNotIn("openclaw-host-placeholder", output)
        self.assertNotIn("openclaw-user-placeholder", output)


if __name__ == "__main__":
    unittest.main()
