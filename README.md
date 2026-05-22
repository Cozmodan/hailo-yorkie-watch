# Hailo Yorkie Watch

Hailo Yorkie Watch is a Raspberry Pi 5 + Raspberry Pi AI HAT+ 2 / Hailo-10H vision project designed to pull camera snapshots from Home Assistant and send future detection events to OpenClaw for WhatsApp alerts.

This first milestone implements plumbing plus an optional dog-detection stage:

- Fetch one snapshot from a Home Assistant camera proxy endpoint.
- Save local test snapshots under `data/snapshots/`.
- Optionally sample frames from a configured live camera stream.
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
YORKIE_FULL_FRAME_DOG_CONFIDENCE=0.35
YORKIE_CROP_DOG_CONFIDENCE=0.20
YORKIE_PERSON_CONFIDENCE=0.35
YORKIE_TARGET_CLASSES=dog,person
YORKIE_DETECTOR_TIMEOUT=60
YORKIE_HAILO_DETECT_COMMAND=

YORKIE_NIGHT_MODE=auto
YORKIE_SCAN_TILES=2x2
YORKIE_ENABLE_CROP_SCAN=1
YORKIE_ENABLE_PERSON_ROI_SCAN=1
YORKIE_CONFIRM_FRAMES=2
YORKIE_CONFIRM_INTERVAL_SECONDS=1.0
YORKIE_MAX_CROPS_PER_IMAGE=8
YORKIE_SAVE_DEBUG_CROPS=1

YORKIE_WATCH_INTERVAL_SECONDS=5
YORKIE_WATCH_COOLDOWN_SECONDS=300
YORKIE_WATCH_MAX_ITERATIONS=
YORKIE_WATCH_SEND_NO_MATCH_LOG=1
YORKIE_WATCH_HEARTBEAT_EVERY=0
YORKIE_WATCH_REUSE_LAST_SNAPSHOT_ON_HA_FAIL=0
YORKIE_WATCH_STOP_ON_ERROR=0

YORKIE_STREAM_ENABLED=0
YORKIE_STREAM_URL=
YORKIE_STREAM_BACKEND=home_assistant
YORKIE_STREAM_USE_HOME_ASSISTANT=1
YORKIE_HA_BASE_URL=http://<home-assistant-host>:8123
YORKIE_HA_STREAM_ENTITY=camera.<placeholder>
YORKIE_HA_STREAM_URL=
YORKIE_HA_LONG_LIVED_TOKEN=
YORKIE_HA_STREAM_AUTH_MODE=bearer
YORKIE_STREAM_FRAME_INTERVAL=5
YORKIE_STREAM_RECONNECT_SECONDS=5
YORKIE_STREAM_MAX_FAILURES=0
YORKIE_STREAM_SAVE_DEBUG_FRAMES=1
YORKIE_STREAM_DEBUG_DIR=data/stream_frames
YORKIE_STREAM_ALERT_COOLDOWN_SECONDS=300
YORKIE_STREAM_PYTHON=python3
```

Do not put real values in committed files. Keep real URLs, hostnames, tokens, camera entity names, and WhatsApp targets in your local `.env` only.

`OPENCLAW_NOTIFY_MODE` supports:

- `http`: send the existing JSON event to `OPENCLAW_URL` using `OPENCLAW_TOKEN`.
- `ssh`: run OpenClaw on the remote machine over SSH.
- `disabled`: skip notification sends without contacting OpenClaw.

`OPENCLAW_NOTIFY_MODE` defaults to `http` if unset. `OPENCLAW_EVENT_ENDPOINT` is optional and defaults to `/api/events/yorkie-watch`. `OPENCLAW_SSH_PORT`, `OPENCLAW_BINARY`, and `OPENCLAW_WHATSAPP_ACCOUNT` default to `22`, `openclaw`, and `business`.

Snapshot attachments over SSH are opt-in because OpenClaw media CLI syntax must be verified on the Nano first. When `OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE` is set, the Pi copies the snapshot to `OPENCLAW_SSH_MEDIA_REMOTE_DIR` with `scp`, then runs the configured OpenClaw media command on the Nano. Available template placeholders are `{binary}`, `{channel}`, `{account}`, `{target}`, `{message}`, and `{media_path}`. Leave the template empty until the media command syntax is confirmed.

`YORKIE_DETECTOR_ENABLED` defaults to `0`, so `--once` still only saves a Home Assistant snapshot unless you opt in to detection. The default detector backend is `hailo_apps`, using `/usr/share/hailo-models/yolov8m_h10.hef`, `dog,person` as requested detector classes, `0.35` as the full-frame dog confidence, and `0.20` as the crop/zoom dog confidence.

`YORKIE_HAILO_APPS_ROOT` should point to the installed `hailo-apps` checkout on the Pi. `YORKIE_HAILO_PYTHON` defaults to `python3` so the detector subprocess can use system Hailo packages outside this project virtual environment. `YORKIE_HAILO_DETECT_COMMAND` is optional; leave it empty to use the repo wrapper, or set it to a JSON-emitting command template with `{image}`, `{hef}`, `{hailo_apps_root}`, `{threshold}`, and `{classes}` placeholders after verifying a custom Hailo command.

`YORKIE_STREAM_ENABLED` defaults to `0`. Home Assistant stream mode builds an authenticated camera proxy stream URL from local runtime settings:

```text
http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.<placeholder>
```

Set `YORKIE_STREAM_BACKEND=home_assistant` or `YORKIE_STREAM_BACKEND=ha_hls`, enable `YORKIE_STREAM_USE_HOME_ASSISTANT=1`, and keep the Home Assistant base URL, camera entity, and long-lived token in local `.env`:

```dotenv
YORKIE_STREAM_BACKEND=home_assistant
YORKIE_STREAM_USE_HOME_ASSISTANT=1
YORKIE_HA_BASE_URL=http://<home-assistant-host>:8123
YORKIE_HA_STREAM_ENTITY=camera.<placeholder>
YORKIE_HA_LONG_LIVED_TOKEN=<home-assistant-long-lived-token>
YORKIE_HA_STREAM_AUTH_MODE=bearer
```

The Home Assistant stream helper uses ffmpeg for the bearer-authenticated stream and redacts the authorization header, long-lived token, and stream URL query values from stream helper output handling. `YORKIE_HA_STREAM_URL` remains an optional manual override for a ready Home Assistant stream URL such as:

```dotenv
YORKIE_HA_STREAM_URL=http://<home-assistant-host>:8123/api/hls/<placeholder>/master_playlist.m3u8
```

Direct live stream mode still uses the OpenCV helper. Set `YORKIE_STREAM_BACKEND=opencv`, `YORKIE_STREAM_USE_HOME_ASSISTANT=0`, and put a direct stream URL such as `rtsp://<camera-stream-host>/<placeholder>` in local `YORKIE_STREAM_URL`. `YORKIE_STREAM_PYTHON` defaults to `python3` so ffmpeg/OpenCV helper processes can run outside this project virtual environment when needed.

## Multi-stage night scanner

The scanner keeps the existing full-frame dog alert behavior, then runs digital crop scans only when the full-frame dog threshold is not met. It does not move the camera or call Home Assistant PTZ services.

Current scanner passes are:

- Full-frame detection for `dog` and generic `person` classes.
- Center zoom crop.
- Lower-half crop for road/ground areas.
- `2x2` tiles by default, with `3x3` available by setting `YORKIE_SCAN_TILES=3x3`.
- Person-expanded ROI crops when generic `person` detections meet `YORKIE_PERSON_CONFIDENCE`.

Person detections are only used as region-of-interest cues. The project does not do face recognition or identify people.

Crop detections are mapped back to original snapshot coordinates and include a `source` value in JSON such as `full_frame`, `tile`, `lower_half`, `person_roi`, or `center_zoom`. When `YORKIE_SAVE_DEBUG_CROPS=1`, crop images are saved under `data/debug_crops/`; repository image ignore rules keep these real camera crops out of Git.

`YORKIE_CONFIRM_FRAMES` controls optional multi-frame confirmation for `--once` and `--watch` alerts. The default `2` means the app can take two snapshots separated by `YORKIE_CONFIRM_INTERVAL_SECONDS` and alert only when dog detection appears in both frames. Set `YORKIE_CONFIRM_FRAMES=1` for faster manual single-frame testing.

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

Run continuously until Ctrl+C or a service manager stops the process:

```powershell
python -m yorkie_watch.main --watch
```

Watch mode fetches a fresh Home Assistant snapshot every `YORKIE_WATCH_INTERVAL_SECONDS`, runs the same multi-stage scanner used by alert scans, and only sends WhatsApp when the alert condition matches. It keeps scanning during `YORKIE_WATCH_COOLDOWN_SECONDS`; matched alerts inside the cooldown are logged without sending a repeated message. Home Assistant and detector failures are logged and retried on the next watch iteration unless `YORKIE_WATCH_STOP_ON_ERROR=1`.

Use `YORKIE_WATCH_MAX_ITERATIONS` or a CLI override for bounded checks:

```powershell
python -m yorkie_watch.main --watch --watch-iterations 2
```

An empty or zero `YORKIE_WATCH_MAX_ITERATIONS` runs forever. `YORKIE_WATCH_SEND_NO_MATCH_LOG=1` logs no-alert scans. `YORKIE_WATCH_HEARTBEAT_EVERY=0` disables heartbeat WhatsApp messages; set a positive iteration count to opt in. `YORKIE_WATCH_REUSE_LAST_SNAPSHOT_ON_HA_FAIL=0` keeps failed Home Assistant fetches from scanning stale images by default.

Watch a live camera stream instead of requesting Home Assistant snapshots:

```powershell
python -m yorkie_watch.main --watch-stream
```

Stream watch mode is opt-in: set `YORKIE_STREAM_ENABLED=1` and keep real Home Assistant or RTSP stream credentials in local `.env` only. The stream helper reads frames continuously, saves one sampled frame every `YORKIE_STREAM_FRAME_INTERVAL` seconds under `YORKIE_STREAM_DEBUG_DIR`, runs the existing multi-stage scanner on that JPEG, and sends the sampled frame as the alert attachment when the detector condition matches. It reconnects after stream failures using `YORKIE_STREAM_RECONNECT_SECONDS`; `YORKIE_STREAM_MAX_FAILURES=0` leaves reconnect attempts unlimited.

Run a bounded stream test and keep its sampled frames:

```powershell
python -m yorkie_watch.main --watch-stream --stream-frames 3 --stream-save-debug-frame
```

`YORKIE_STREAM_ALERT_COOLDOWN_SECONDS` applies the live-stream alert cooldown. `YORKIE_STREAM_SAVE_DEBUG_FRAMES=1` keeps sampled stream JPGs by default for inspection; set it to `0` to remove sampled frames after each scan. Generated stream frames stay under ignored image paths and must not be committed.

You can also use the installed console script:

```powershell
yorkie-watch --once
yorkie-watch --test-openclaw
```

## Systemd watch service

Example service file path:

```text
/etc/systemd/system/yorkie-watch.service
```

Example contents using placeholders:

```ini
[Unit]
Description=Hailo Yorkie Watch
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<user>
WorkingDirectory=/home/<user>/hailo-yorkie-watch
ExecStart=/home/<user>/hailo-yorkie-watch/.venv/bin/python -m yorkie_watch.main --watch
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Keep the local `.env` file in the working directory so Python loads runtime credentials on the Pi. Do not put Home Assistant, OpenClaw, SSH, or WhatsApp secrets into the public service example.

## Current hardware architecture

- Home Assistant Pi receives the external security camera feed.
- Raspberry Pi 5 with AI HAT+ 2 pulls snapshots/frames from Home Assistant.
- Hailo object detection is wired as an optional subprocess stage.
- VLM detection will be added later.
- Jetson Nano running OpenClaw will later send WhatsApp alerts.
- GitHub remains the source of truth for non-secret project code.
