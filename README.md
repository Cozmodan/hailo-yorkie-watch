# Hailo Yorkie Watch

Hailo Yorkie Watch is a Raspberry Pi 5 + Raspberry Pi AI HAT+ 2 / Hailo-10H vision project designed to pull camera snapshots from Home Assistant and send future detection events to OpenClaw for WhatsApp alerts.

This first milestone only implements plumbing:

- Fetch one snapshot from a Home Assistant camera proxy endpoint.
- Save local test snapshots under `data/snapshots/`.
- Send a JSON test event to OpenClaw.
- Provide a small command-line entry point.

Hailo inference, VLM support, and Yorkie breed detection are intentionally not implemented yet.

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
```

Do not put real values in committed files. Keep real URLs, hostnames, tokens, camera entity names, and WhatsApp targets in your local `.env` only.

`OPENCLAW_NOTIFY_MODE` supports:

- `http`: send the existing JSON event to `OPENCLAW_URL` using `OPENCLAW_TOKEN`.
- `ssh`: run OpenClaw on the remote machine over SSH.
- `disabled`: skip notification sends without contacting OpenClaw.

`OPENCLAW_NOTIFY_MODE` defaults to `http` if unset. `OPENCLAW_EVENT_ENDPOINT` is optional and defaults to `/api/events/yorkie-watch`. `OPENCLAW_SSH_PORT`, `OPENCLAW_BINARY`, and `OPENCLAW_WHATSAPP_ACCOUNT` default to `22`, `openclaw`, and `business`.

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
ssh -p <port> <ssh-user>@<ssh-host> <openclaw-binary> message send --channel whatsapp --account <account> --target <whatsapp-target> --message <message>
```

## Command-line usage

Fetch one Home Assistant snapshot and save it under `data/snapshots/`:

```powershell
python -m yorkie_watch.main --once
```

Send one OpenClaw test notification:

```powershell
python -m yorkie_watch.main --test-openclaw
```

You can also use the installed console script:

```powershell
yorkie-watch --once
yorkie-watch --test-openclaw
```

## Current hardware architecture

- Home Assistant Pi receives the external security camera feed.
- Raspberry Pi 5 with AI HAT+ 2 pulls snapshots/frames from Home Assistant.
- Hailo/VLM detection will be added later.
- Jetson Nano running OpenClaw will later send WhatsApp alerts.
- GitHub remains the source of truth for non-secret project code.

