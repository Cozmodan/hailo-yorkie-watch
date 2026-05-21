# Hailo Yorkie Watch

Hailo Yorkie Watch is a Raspberry Pi 5 + Raspberry Pi AI HAT+ 2 / Hailo-10H vision project designed to pull camera snapshots from Home Assistant and send future detection events to OpenClaw for WhatsApp alerts.

This first milestone implements plumbing plus an optional dog-detection stage:

- Fetch one snapshot from a Home Assistant camera proxy endpoint.
- Save local test snapshots under `data/snapshots/`.
- Optionally run a Hailo Apps object detector against the saved snapshot.
- Send a WhatsApp notification through OpenClaw only when the configured detection condition matches.
- Provide a small command-line entry point.

VLM support and Yorkie breed recognition are intentionally not implemented yet. For now, a matching `dog` object is treated as the alert condition.

## Repository safety

This repository is public. Do not commit secrets, tokens, or real camera images.

- Keep local credentials in `.env` only.
- Use `.env.example` as a template.
- Snapshot outputs in `data/snapshots/` are ignored by Git.
- Common camera image extensions such as `.jpg`, `.jpeg`, `.png`, and `.webp` are ignored by Git.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Create a local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` with your local values:

```dotenv
HOME_ASSISTANT_URL=<home-assistant-url>
HOME_ASSISTANT_TOKEN=<home-assistant-token>
HOME_ASSISTANT_CAMERA_ENTITY=<home-assistant-camera-entity>

OPENCLAW_NOTIFY_MODE=ssh
OPENCLAW_WHATSAPP_TARGET=<whatsapp-target>

OPENCLAW_URL=<openclaw-http-url>
OPENCLAW_TOKEN=<openclaw-http-token>
OPENCLAW_EVENT_ENDPOINT=/api/events/yorkie-watch

OPENCLAW_SSH_HOST=<openclaw-ssh-host>
OPENCLAW_SSH_USER=<openclaw-ssh-user>
OPENCLAW_SSH_PORT=22
OPENCLAW_BINARY=openclaw
OPENCLAW_WHATSAPP_ACCOUNT=business
OPENCLAW_SSH_MEDIA_REMOTE_DIR=/tmp/yorkie-watch
OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE=

YORKIE_DETECTOR_ENABLED=0
YORKIE_DETECTOR_BACKEND=hailo_apps
YORKIE_HAILO_HEF=/usr/share/hailo-models/yolov8m_h10.hef
YORKIE_HAILO_APPS_ROOT=<hailo-apps-root>
YORKIE_HAILO_PYTHON=python3
YORKIE_DOG_CONFIDENCE=0.35
YORKIE_TARGET_CLASSES=dog
YORKIE_DETECTOR_TIMEOUT=60
YORKIE_HAILO_DETECT_COMMAND=
```

Do not put real values in committed files. Keep real URLs, hostnames, tokens, camera entity names, and WhatsApp targets in your local `.env` only.

`OPENCLAW_NOTIFY_MODE` supports:

- `http`: send the existing JSON event to `OPENCLAW_URL` using `OPENCLAW_TOKEN`.
- `ssh`: run OpenClaw on the remote machine over SSH.
- `disabled`: skip notification sends without contacting OpenClaw.

`OPENCLAW_NOTIFY_MODE` defaults to `http` if unset. `OPENCLAW_EVENT_ENDPOINT` is optional and defaults to `/api/events/yorkie-watch`. `OPENCLAW_SSH_PORT`, `OPENCLAW_BINARY`, and `OPENCLAW_WHATSAPP_ACCOUNT` default to `22`, `openclaw`, and `business`.

Snapshot attachments over SSH are opt-in because OpenClaw media CLI syntax must be verified on the Nano first. When `OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE` is set, the Pi copies the snapshot to `OPENCLAW_SSH_MEDIA_REMOTE_DIR` with `scp`, then runs the configured OpenClaw media command on the Nano. Available template placeholders are `{binary}`, `{channel}`, `{account}`, `{target}`, `{message}`, and `{media_path}`. Leave the template empty until the media command syntax is confirmed.

`YORKIE_DETECTOR_ENABLED` defaults to `0`, so `--once` still only saves a Home Assistant snapshot unless you opt in to detection. The default detector backend is `hailo_apps`, using `/usr/share/hailo-models/yolov8m_h10.hef`, `dog` as the target class, and `0.35` as the minimum confidence.

`YORKIE_HAILO_APPS_ROOT` should point to the installed `hailo-apps` checkout on the Pi. `YORKIE_HAILO_PYTHON` defaults to `python3` so the detector subprocess can use system Hailo packages outside this project virtual environment. `YORKIE_HAILO_DETECT_COMMAND` is optional; leave it empty to use the repo wrapper, or set it to a JSON-emitting command template with `{image}`, `{hef}`, `{hailo_apps_root}`, `{threshold}`, and `{classes}` placeholders after verifying a custom Hailo command.

## Home Assistant snapshot test

Fetch one image from Home Assistant and save it to `data/snapshots/test_snapshot.jpg`:

```powershell
python scripts/test_home_assistant_snapshot.py
```

The script prints the saved path and file size when successful.

## OpenClaw notification test

Send a test WhatsApp notification through the configured OpenClaw path:

```powershell
python scripts/test_openclaw_notify.py
```

For HTTP mode, the test payload is:

```json
{
  "event_type": "yorkie_watch_test",
  "message": "Test alert from Hailo Yorkie Watch",
  "confidence": 0.0
}
```

The configured `OPENCLAW_WHATSAPP_TARGET` is added to the outgoing JSON as `whatsapp_target`.

For SSH mode, the script invokes OpenClaw with `subprocess.run` using argv and no local shell:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 -p <port> <ssh-user>@<ssh-host> <openclaw-binary> message send --channel whatsapp --account <account> --target <whatsapp-target> --message <message>
```

## OpenClaw media discovery

Probe OpenClaw help output over the configured SSH connection:

```powershell
python scripts/probe_openclaw_media.py
```

Equivalent manual commands using placeholders:

```bash
ssh <ssh-user>@<ssh-host> '<openclaw-binary> --help || true'
ssh <ssh-user>@<ssh-host> '<openclaw-binary> message --help || true'
ssh <ssh-user>@<ssh-host> '<openclaw-binary> message send --help || true'
ssh <ssh-user>@<ssh-host> '<openclaw-binary> messages --help || true'
ssh <ssh-user>@<ssh-host> '<openclaw-binary> media --help || true'
```

After confirming the media send syntax, set `OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE` in local `.env`. Example shape only:

```dotenv
OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE={binary} message send --channel {channel} --account {account} --target {target} --message {message} --media {media_path}
```

## Hailo dog detection test

Run the detector against an existing snapshot:

```powershell
python scripts/test_hailo_detect.py data/snapshots/test_snapshot.jpg
```

The script prints JSON with `detections`, `matched`, and `matched_reason`. If `YORKIE_DETECTOR_ENABLED=0`, it prints a disabled detector result. If the Hailo Apps path or runtime is not available, it prints a clear JSON error instead of importing Hailo modules into the main app.

## Command-line usage

Fetch one Home Assistant snapshot and save it under `data/snapshots/`. If `YORKIE_DETECTOR_ENABLED=1`, this also runs detection and sends OpenClaw WhatsApp only when the dog condition matches:

```powershell
python -m yorkie_watch.main --once
```

Send one OpenClaw test notification:

```powershell
python -m yorkie_watch.main --test-openclaw
```

Run one detector test from the module entry point:

```powershell
python -m yorkie_watch.main --test-detect data/snapshots/test_snapshot.jpg
```

Ask for a snapshot plus detector summary:

```powershell
python -m yorkie_watch.main --what-see
```

When SSH media is configured, `--what-see` sends the saved snapshot as an attachment plus a detection summary. If media is not configured yet, it sends the text summary and logs that the snapshot attachment was skipped.

You can also use the installed console script:

```powershell
yorkie-watch --once
yorkie-watch --test-openclaw
```

## Current hardware architecture

- Home Assistant Pi receives the external security camera feed.
- Raspberry Pi 5 with AI HAT+ 2 pulls snapshots/frames from Home Assistant.
- Hailo object detection is wired as an optional subprocess stage.
- VLM detection will be added later.
- Jetson Nano running OpenClaw will later send WhatsApp alerts.
- GitHub remains the source of truth for non-secret project code.

