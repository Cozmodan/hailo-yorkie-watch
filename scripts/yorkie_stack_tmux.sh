#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
LOG_DIR="${REPO_ROOT}/logs/tmux"
DEFAULT_SESSION="yorkie-watch"

DEFAULT_VLM_CMD="HAILO_VLM_UNLOAD_AFTER_REQUEST=1 HAILO_DEVICE_LOCK_TIMEOUT_SECONDS=120 /usr/bin/python3 scripts/hailo_vlm_server.py"
DEFAULT_VISION_CMD="source .venv/bin/activate && PYTHONPATH=src python scripts/openclaw_vision_tool_server.py"
DEFAULT_INBOUND_CMD="source .venv/bin/activate && PYTHONPATH=src python scripts/openclaw_inbound_bridge.py"
DEFAULT_WATCH_CMD="source .venv/bin/activate && PYTHONPATH=src python -m yorkie_watch.main --watch"
DEFAULT_STREAM_WATCH_CMD="source .venv/bin/activate && PYTHONPATH=src python -m yorkie_watch.main --watch-stream"

usage() {
    cat <<'EOF'
Usage:
  scripts/yorkie_stack_tmux.sh start [stream]
  scripts/yorkie_stack_tmux.sh stop
  scripts/yorkie_stack_tmux.sh restart [stream]
  scripts/yorkie_stack_tmux.sh attach
  scripts/yorkie_stack_tmux.sh status
  scripts/yorkie_stack_tmux.sh logs
  scripts/yorkie_stack_tmux.sh smoke-test

Environment overrides:
  YORKIE_TMUX_SESSION
  YORKIE_STACK_VLM_CMD
  YORKIE_STACK_VISION_CMD
  YORKIE_STACK_INBOUND_CMD
  YORKIE_STACK_WATCH_CMD
EOF
}

dotenv_get() {
    local key="$1"
    local default_value="${2:-}"
    local env_file="${REPO_ROOT}/.env"
    local line value="" found=0

    if [[ -n "${!key+x}" ]]; then
        printf '%s' "${!key}"
        return
    fi

    if [[ -f "${env_file}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="${line%$'\r'}"
            [[ "${line}" =~ ^[[:space:]]*$ ]] && continue
            [[ "${line}" =~ ^[[:space:]]*# ]] && continue
            if [[ "${line}" == "${key}="* ]]; then
                value="${line#*=}"
                found=1
            fi
        done < "${env_file}"
    fi

    if [[ "${found}" -eq 1 ]]; then
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        printf '%s' "${value}"
        return
    fi

    printf '%s' "${default_value}"
}

sh_quote() {
    local value="${1:-}"
    printf "'%s'" "$(printf '%s' "${value}" | sed "s/'/'\\\\''/g")"
}

session_name() {
    local name
    name="$(dotenv_get YORKIE_TMUX_SESSION "${DEFAULT_SESSION}")"
    printf '%s' "${name:-${DEFAULT_SESSION}}"
}

require_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is required but was not found. Install tmux, then retry." >&2
        exit 1
    fi
}

require_curl() {
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required for this action but was not found." >&2
        exit 1
    fi
}

require_files() {
    local missing=0
    local required=(
        "scripts/hailo_vlm_server.py"
        "scripts/openclaw_vision_tool_server.py"
        "scripts/openclaw_inbound_bridge.py"
        "src/yorkie_watch/main.py"
    )

    for path in "${required[@]}"; do
        if [[ ! -f "${REPO_ROOT}/${path}" ]]; then
            echo "Missing required file: ${path}" >&2
            missing=1
        fi
    done

    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
}

session_exists() {
    local session="$1"
    tmux has-session -t "${session}" 2>/dev/null
}

runner_script() {
    local label="$1"
    local command="$2"
    local log_file="$3"

    cat <<EOF
set -uo pipefail
cd $(sh_quote "${REPO_ROOT}")
mkdir -p $(sh_quote "${LOG_DIR}")
exec > >(tee -a $(sh_quote "${log_file}")) 2>&1
echo "[$label] starting at \$(date -Is)"
bash -lc $(sh_quote "set -Eeuo pipefail; ${command}")
rc=\$?
echo
echo "[$label] exited with status \$rc at \$(date -Is). Pane left open for inspection."
exec "\${SHELL:-/bin/bash}"
EOF
}

new_window() {
    local session="$1"
    local name="$2"
    local command="$3"
    local log_file="${LOG_DIR}/${name}.log"
    local script
    script="$(runner_script "${name}" "${command}" "${log_file}")"

    if [[ "${name}" == "hailo-vlm" ]]; then
        tmux new-session -d -s "${session}" -n "${name}" "bash -lc $(sh_quote "${script}")"
    else
        tmux new-window -d -t "${session}" -n "${name}" "bash -lc $(sh_quote "${script}")"
    fi
}

watch_command_for_mode() {
    local mode="$1"
    local override
    override="$(dotenv_get YORKIE_STACK_WATCH_CMD "")"
    if [[ -n "${override}" ]]; then
        printf '%s' "${override}"
        return
    fi

    if [[ "${mode}" == "stream" ]]; then
        printf '%s' "${DEFAULT_STREAM_WATCH_CMD}"
    else
        printf '%s' "${DEFAULT_WATCH_CMD}"
    fi
}

normalize_mode() {
    local mode="${1:-watch}"
    case "${mode}" in
        watch|snapshot|"")
            printf 'watch'
            ;;
        stream)
            printf 'stream'
            ;;
        *)
            echo "Unknown watcher mode: ${mode}" >&2
            echo "Use no mode for --watch, or use: stream" >&2
            exit 1
            ;;
    esac
}

start_stack() {
    local mode="$1"
    local session="$2"
    local vlm_cmd vision_cmd inbound_cmd watch_cmd

    require_tmux
    require_files

    if session_exists "${session}"; then
        echo "tmux session '${session}' already exists. Run stop, restart, or attach." >&2
        exit 1
    fi

    mkdir -p "${LOG_DIR}"
    vlm_cmd="$(dotenv_get YORKIE_STACK_VLM_CMD "${DEFAULT_VLM_CMD}")"
    vision_cmd="$(dotenv_get YORKIE_STACK_VISION_CMD "${DEFAULT_VISION_CMD}")"
    inbound_cmd="$(dotenv_get YORKIE_STACK_INBOUND_CMD "${DEFAULT_INBOUND_CMD}")"
    watch_cmd="$(watch_command_for_mode "${mode}")"

    new_window "${session}" "hailo-vlm" "${vlm_cmd:-${DEFAULT_VLM_CMD}}"
    new_window "${session}" "vision-tool" "${vision_cmd:-${DEFAULT_VISION_CMD}}"
    new_window "${session}" "inbound-bridge" "${inbound_cmd:-${DEFAULT_INBOUND_CMD}}"
    new_window "${session}" "yorkie-watch" "${watch_cmd}"
    tmux select-window -t "${session}:hailo-vlm" >/dev/null

    echo "Started tmux session '${session}' in ${mode} mode."
    echo "Attach with: scripts/yorkie_stack_tmux.sh attach"
    echo "Logs are under: logs/tmux/"
}

stop_stack() {
    local session="$1"
    require_tmux
    if session_exists "${session}"; then
        tmux kill-session -t "${session}"
        echo "Stopped tmux session '${session}'."
    else
        echo "tmux session '${session}' is not running."
    fi
}

attach_stack() {
    local session="$1"
    require_tmux
    if ! session_exists "${session}"; then
        echo "tmux session '${session}' is not running. Start it first." >&2
        exit 1
    fi
    exec tmux attach-session -t "${session}"
}

health_check() {
    local name="$1"
    local url="$2"
    local response

    printf '%s health: ' "${name}"
    if response="$(curl -fsS --connect-timeout 2 --max-time 5 "${url}" 2>&1)"; then
        echo "ok ${response}"
        return 0
    fi

    echo "failed ${response}"
    return 1
}

status_stack() {
    local session="$1"
    local hailo_port vision_port inbound_port rc=0
    require_tmux
    require_curl

    if session_exists "${session}"; then
        echo "tmux session '${session}' is running."
        tmux list-windows -t "${session}"
        tmux list-panes -a -F '#S:#W.#P #{pane_current_command} #{pane_active} #{pane_dead_status}'
    else
        echo "tmux session '${session}' is not running."
        rc=1
    fi

    hailo_port="$(dotenv_get HAILO_VLM_PORT "8010")"
    vision_port="$(dotenv_get OPENCLAW_VISION_TOOL_PORT "8021")"
    inbound_port="$(dotenv_get OPENCLAW_INBOUND_PORT "8020")"
    health_check "hailo-vlm" "http://127.0.0.1:${hailo_port}/health" || rc=1
    health_check "vision-tool" "http://127.0.0.1:${vision_port}/health" || rc=1
    health_check "inbound-bridge" "http://127.0.0.1:${inbound_port}/health" || rc=1
    return "${rc}"
}

show_logs() {
    local name log_file
    for name in hailo-vlm vision-tool inbound-bridge yorkie-watch; do
        log_file="${LOG_DIR}/${name}.log"
        echo
        echo "==> ${log_file}"
        if [[ -f "${log_file}" ]]; then
            tail -n 80 "${log_file}"
        else
            echo "Log file not found."
        fi
    done
}

smoke_test() {
    local hailo_port vision_port inbound_port vision_url inbound_url vision_secret inbound_secret smoke_send
    local body='{"prompt":"Describe the camera view briefly. Is a dog or Yorkie visible?"}'
    local inbound_body='{"sender":"test","message":"status"}'
    require_curl

    hailo_port="$(dotenv_get HAILO_VLM_PORT "8010")"
    vision_port="$(dotenv_get OPENCLAW_VISION_TOOL_PORT "8021")"
    inbound_port="$(dotenv_get OPENCLAW_INBOUND_PORT "8020")"
    vision_url="http://127.0.0.1:${vision_port}"
    inbound_url="http://127.0.0.1:${inbound_port}"
    vision_secret="$(dotenv_get OPENCLAW_VISION_TOOL_SHARED_SECRET "")"
    inbound_secret="$(dotenv_get OPENCLAW_INBOUND_SHARED_SECRET "")"
    smoke_send="$(dotenv_get OPENCLAW_INBOUND_SMOKE_SEND "0")"

    echo "Checking Hailo VLM health..."
    curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1:${hailo_port}/health"
    echo
    echo "Checking OpenClaw vision tool health..."
    curl -fsS --connect-timeout 3 --max-time 10 "${vision_url}/health"
    echo
    echo "Checking OpenClaw inbound bridge health..."
    curl -fsS --connect-timeout 3 --max-time 10 "${inbound_url}/health"
    echo
    echo "Testing inbound bridge status route..."
    local inbound_curl_args=(-fsS --connect-timeout 5 --max-time 30 -H "Content-Type: application/json")
    if [[ -n "${inbound_secret}" ]]; then
        inbound_curl_args+=(-H "X-OpenClaw-Secret: ${inbound_secret}")
    fi
    if [[ "${smoke_send}" != "1" ]]; then
        inbound_curl_args+=(-H "X-OpenClaw-Smoke-Test: 1")
    fi
    curl "${inbound_curl_args[@]}" -d "${inbound_body}" "${inbound_url}/openclaw/inbound"
    echo
    echo "Requesting camera snapshot description..."

    local curl_args=(-fsS --connect-timeout 5 --max-time 180 -H "Content-Type: application/json")
    if [[ -n "${vision_secret}" ]]; then
        curl_args+=(-H "X-OpenClaw-Secret: ${vision_secret}")
    fi

    curl "${curl_args[@]}" -d "${body}" "${vision_url}/vision/camera-snapshot"
    echo
}

main() {
    local action="${1:-}"
    local mode_arg="${2:-}"
    local session
    session="$(session_name)"

    case "${action}" in
        start)
            start_stack "$(normalize_mode "${mode_arg:-watch}")" "${session}"
            ;;
        stop)
            stop_stack "${session}"
            ;;
        restart)
            require_tmux
            if session_exists "${session}"; then
                tmux kill-session -t "${session}"
                echo "Stopped existing tmux session '${session}'."
            fi
            start_stack "$(normalize_mode "${mode_arg:-watch}")" "${session}"
            ;;
        attach)
            attach_stack "${session}"
            ;;
        status)
            status_stack "${session}"
            ;;
        logs)
            show_logs
            ;;
        smoke-test)
            smoke_test
            ;;
        -h|--help|help|"")
            usage
            ;;
        *)
            echo "Unknown action: ${action}" >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
