# Hailo Yorkie Watch

Hailo Yorkie Watch is a Raspberry Pi 5 + Raspberry Pi AI HAT+ 2 / Hailo-10H vision project designed to pull camera snapshots from Home Assistant and send future detection events to OpenClaw for WhatsApp alerts.

This first milestone implements plumbing plus an optional dog-detection stage:

- Fetch one snapshot from a Home Assistant camera proxy endpoint.
- Save local test snapshots under `data/snapshots/`.
- Optionally sample frames from a configured live camera stream.
- Optionally run a Hailo Apps object detector against the saved snapshot.
- Optionally ask a local Ollama-compatible VLM for a short explanation of alert evidence.
- Send a WhatsApp notification through OpenClaw only when the configured detection condition matches.
- Provide a small command-line entry point.

Yorkie breed recognition is intentionally not implemented yet. For now, a matching `dog` object is treated as the alert condition, and optional VLM text is only a secondary explanation.

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
DOG_MIN_CONFIDENCE=0.45
DOG_ALERT_COOLDOWN_SECONDS=180
DOG_CONFIRMATION_FRAMES=2
DOG_MIN_BOX_AREA_RATIO=0.01
SAVE_DEBUG_FRAMES=false
IMAGE_RETENTION_SECONDS=3600
MAX_EVIDENCE_IMAGES=100
DOG_EVIDENCE_DIR=data/evidence
YORKIE_DOG_CONFIDENCE=0.45
YORKIE_FULL_FRAME_DOG_CONFIDENCE=0.45
YORKIE_CROP_DOG_CONFIDENCE=0.45
YORKIE_PERSON_CONFIDENCE=0.35
YORKIE_TARGET_CLASSES=dog,person
YORKIE_DETECTOR_TIMEOUT=60
YORKIE_HAILO_DETECT_COMMAND=

YORKIE_VLM_ENABLED=0
YORKIE_VLM_BASE_URL=http://127.0.0.1:8010
YORKIE_VLM_MODEL=<vlm-model-name>
YORKIE_VLM_TIMEOUT_SECONDS=60
YORKIE_VLM_MAX_IMAGE_WIDTH=1280
YORKIE_VLM_PROMPT="Look at this image. Is there a dog or Yorkie? Briefly describe what you see and mention uncertainty."

HAILO_VLM_HEF=/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef
HAILO_VLM_HOST=127.0.0.1
HAILO_VLM_PORT=8010
HAILO_VLM_MAX_TOKENS=80
HAILO_VLM_OPTIMIZE_MEMORY=1
HAILO_VLM_CLEAR_CONTEXT=1

YORKIE_NIGHT_MODE=auto
YORKIE_SCAN_TILES=2x2
YORKIE_ENABLE_CROP_SCAN=1
YORKIE_ENABLE_PERSON_ROI_SCAN=1
YORKIE_CONFIRM_FRAMES=2
YORKIE_CONFIRM_INTERVAL_SECONDS=1.0
YORKIE_MAX_CROPS_PER_IMAGE=8
YORKIE_SAVE_DEBUG_CROPS=0

YORKIE_WATCH_INTERVAL_SECONDS=5
YORKIE_WATCH_COOLDOWN_SECONDS=180
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
YORKIE_STREAM_KEEP_FRAMES=0
YORKIE_STREAM_SAVE_DEBUG_FRAMES=0
YORKIE_STREAM_DEBUG_DIR=data/stream_frames
YORKIE_STREAM_RETENTION_MINUTES=60
YORKIE_STREAM_MAX_FRAME_FILES=500
YORKIE_DEBUG_CROP_RETENTION_MINUTES=60
YORKIE_DEBUG_CROP_MAX_FILES=500
YORKIE_STREAM_ALERT_COOLDOWN_SECONDS=180
YORKIE_STREAM_PYTHON=python3
```

Do not put real values in committed files. Keep real URLs, hostnames, tokens, camera entity names, and WhatsApp targets in your local `.env` only.

`OPENCLAW_NOTIFY_MODE` supports:

- `http`: send the existing JSON event to `OPENCLAW_URL` using `OPENCLAW_TOKEN`.
- `ssh`: run OpenClaw on the remote machine over SSH.
- `disabled`: skip notification sends without contacting OpenClaw.

`OPENCLAW_NOTIFY_MODE` defaults to `http` if unset. `OPENCLAW_EVENT_ENDPOINT` is optional and defaults to `/api/events/yorkie-watch`. `OPENCLAW_SSH_PORT`, `OPENCLAW_BINARY`, and `OPENCLAW_WHATSAPP_ACCOUNT` default to `22`, `openclaw`, and `business`.

Snapshot attachments over SSH are opt-in because OpenClaw media CLI syntax must be verified on the Nano first. When `OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE` is set, the Pi copies the snapshot to `OPENCLAW_SSH_MEDIA_REMOTE_DIR` with `scp`, then runs the configured OpenClaw media command on the Nano. Available template placeholders are `{binary}`, `{channel}`, `{account}`, `{target}`, `{message}`, and `{media_path}`. Leave the template empty until the media command syntax is confirmed.

`YORKIE_DETECTOR_ENABLED` defaults to `0`, so `--once` still only saves a Home Assistant snapshot unless you opt in to detection. The default detector backend is `hailo_apps`, using `/usr/share/hailo-models/yolov8m_h10.hef`, `dog,person` as requested detector classes, and `0.45` as the starting dog confidence threshold.

Dog alerts use a stricter policy layer after scanner output:

- `DOG_MIN_CONFIDENCE=0.45` is the minimum dog confidence that can send an alert.
- `DOG_ALERT_COOLDOWN_SECONDS=180` suppresses repeated dog alerts after one sends.
- `DOG_CONFIRMATION_FRAMES=2` requires two consecutive valid dog detections before alerting.
- `DOG_MIN_BOX_AREA_RATIO=0.01` ignores tiny dog boxes smaller than 1% of the full image.
- `SAVE_DEBUG_FRAMES=false` keeps non-alert frames and debug crops from building up by default.
- `IMAGE_RETENTION_SECONDS=3600` and `MAX_EVIDENCE_IMAGES=100` limit annotated alert evidence images under `DOG_EVIDENCE_DIR`.

When an alert is sent, Yorkie Watch draws a dog bounding box, confidence label, timestamp, and scanner region onto an evidence image and sends that annotated image to OpenClaw instead of the raw frame.

If false alerts continue, raise `DOG_MIN_CONFIDENCE` to `0.50` or `0.55`. If alerts are still too frequent, increase `DOG_ALERT_COOLDOWN_SECONDS` to `300`. If real dogs are missed, reduce `DOG_MIN_CONFIDENCE` slightly or reduce `DOG_CONFIRMATION_FRAMES`.

## Local VLM reasoning

VLM reasoning is optional and disabled by default. It is intended for a local Hailo VLM, Hailo-Ollama style bridge, or Ollama-compatible service that accepts image prompts. Keep the real service URL and model name in local `.env` only.

Starting placeholder settings:

```dotenv
YORKIE_VLM_ENABLED=0
YORKIE_VLM_BASE_URL=http://127.0.0.1:8010
YORKIE_VLM_MODEL=<vlm-model-name>
YORKIE_VLM_TIMEOUT_SECONDS=60
YORKIE_VLM_MAX_IMAGE_WIDTH=1280
YORKIE_VLM_PROMPT="Look at this image. Is there a dog or Yorkie? Briefly describe what you see and mention uncertainty."
```

The repo includes a local Hailo VLM wrapper that exposes the Ollama-style endpoints Yorkie Watch already calls. It must run with system Python because `hailo_platform`, `cv2`, and `numpy` are installed outside the project virtual environment:

```bash
/usr/bin/python3 scripts/hailo_vlm_server.py
```

Wrapper settings:

```dotenv
HAILO_VLM_HEF=/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef
HAILO_VLM_HOST=127.0.0.1
HAILO_VLM_PORT=8010
HAILO_VLM_MAX_TOKENS=80
HAILO_VLM_OPTIMIZE_MEMORY=1
HAILO_VLM_CLEAR_CONTEXT=1
```

The wrapper serves:

- `GET /health`
- `POST /api/chat`
- `POST /api/generate`

It accepts Ollama-style base64 image requests, decodes JPEG/PNG images with OpenCV, converts BGR to RGB, resizes to the Hailo VLM input frame shape `336x336x3`, and serializes generation with a process-local lock. By default it clears VLM context before each request.

If a separate `hailo-ollama.service` or other process owns the Hailo device, stop it before starting this wrapper:

```bash
sudo systemctl stop hailo-ollama.service
```

Check the wrapper locally:

```bash
curl http://127.0.0.1:8010/health
```

When `YORKIE_VLM_ENABLED=1`, Yorkie Watch sends the annotated alert evidence image to the local VLM after the dog detector confirms an alert. The VLM summary is appended to the WhatsApp message:

```text
Dog detected by Hailo Yorkie Watch: lower_half
Detector: dog confidence 0.52 >= 0.45
VLM: A small dog-like animal appears near the doorway. Confidence moderate.
```

The original evidence image is not overwritten. A resized temporary image copy is created under `data/vlm_tmp/`, sent to the VLM, and removed after the request. If the VLM is unavailable or times out, the normal detector alert still sends without VLM text.

Yorkie Watch stores non-secret metadata for the latest alert in `data/latest_event.json`, including the annotated evidence image path, detector class, confidence, region, and VLM summary if one was available. Runtime state files and temporary VLM images are ignored by Git.

Ask a local VLM question about the latest alert image:

```powershell
python -m yorkie_watch.main --chat "Was that actually my dog or just a shadow?"
```

By default, chat mode prints the answer only. To send the generated answer through OpenClaw, explicitly opt in:

```powershell
python -m yorkie_watch.main --chat "Was that actually my dog or just a shadow?" --send-chat-reply
```

This is a safe CLI bridge for testing. It does not add OpenClaw inbound webhook handling yet.

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

Crop detections are mapped back to original snapshot coordinates and include a `source` value in JSON such as `full_frame`, `tile`, `lower_half`, `person_roi`, or `center_zoom`. The alert policy requires the dog bounding-box centre to remain inside the active scanner region for simple regions such as `lower_half`, `center_zoom`, and tiles. When `YORKIE_SAVE_DEBUG_CROPS=1` or `SAVE_DEBUG_FRAMES=true`, crop images are saved under `data/debug_crops/`; repository image ignore rules keep these real camera crops out of Git.

`DOG_CONFIRMATION_FRAMES` controls alert confirmation. The default `2` means watch modes require two consecutive valid dog detections before sending. `YORKIE_CONFIRM_FRAMES` remains available for snapshot confirmation captures in `--once` and `--watch`; keep it aligned with `DOG_CONFIRMATION_FRAMES` unless you are deliberately testing.

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

Ask the local VLM about the latest alert image:

```powershell
python -m yorkie_watch.main --chat "What does the latest alert image show?"
```

Send that VLM answer through OpenClaw only when explicitly requested:

```powershell
python -m yorkie_watch.main --chat "What does the latest alert image show?" --send-chat-reply
```

Run continuously until Ctrl+C or a service manager stops the process:

```powershell
python -m yorkie_watch.main --watch
```

Watch mode fetches a fresh Home Assistant snapshot every `YORKIE_WATCH_INTERVAL_SECONDS`, runs the same multi-stage scanner used by alert scans, and only sends WhatsApp when the dog alert policy matches. It keeps scanning during `DOG_ALERT_COOLDOWN_SECONDS`; matched alerts inside the cooldown are logged without sending a repeated message. Home Assistant and detector failures are logged and retried on the next watch iteration unless `YORKIE_WATCH_STOP_ON_ERROR=1`.

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

`DOG_ALERT_COOLDOWN_SECONDS` applies the live-stream alert cooldown. By default, `YORKIE_STREAM_KEEP_FRAMES=0`, `YORKIE_STREAM_SAVE_DEBUG_FRAMES=0`, and `SAVE_DEBUG_FRAMES=false` delete each sampled stream JPG after it has been scanned, including frames that produced an alert once the annotated evidence image has been created. Set debug values to `1`/`true` only for local debugging. Startup, periodic stream-loop, and bounded-test cleanup remove old `data/stream_frames/`, `data/debug_crops/`, and `data/evidence/` images by the configured retention/count limits. Cleanup is limited to project `data/` image directories, and generated camera frames/crops/evidence must not be committed.

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
- Optional local VLM reasoning can explain annotated alert evidence and answer latest-alert chat questions.
- Jetson Nano running OpenClaw sends WhatsApp alerts.
- GitHub remains the source of truth for non-secret project code.
