from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hailo Apps object detection and emit Yorkie Watch JSON.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--hef", required=True, help="Hailo HEF path or model name.")
    parser.add_argument("--hailo-apps-root", required=True, help="Path to the installed hailo-apps repository.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum confidence to include in output.")
    parser.add_argument("--classes", default="dog", help="Comma-separated target classes.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = Path(args.image)
    result: dict[str, Any] = {
        "ok": False,
        "backend": "hailo_apps",
        "image": str(image_path),
        "detections": [],
        "matched": False,
        "matched_reason": "detector failed",
    }

    try:
        detections = run_hailo_apps_detection(args)
    except Exception as exc:  # Hailo runtime/import errors must come back as JSON.
        result["error"] = str(exc)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    result["ok"] = True
    result["detections"] = detections
    result["matched_reason"] = "raw detections emitted; yorkie_watch.detector applies alert criteria"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_hailo_apps_detection(args: argparse.Namespace) -> list[dict[str, Any]]:
    image_path = Path(args.image)
    if not image_path.exists():
        raise RuntimeError(f"Input image does not exist: {image_path}")

    hailo_apps_root = Path(args.hailo_apps_root)
    app_path = (
        hailo_apps_root
        / "hailo_apps"
        / "python"
        / "standalone_apps"
        / "object_detection"
        / "object_detection.py"
    )
    post_process_path = app_path.with_name("object_detection_post_process.py")
    if not app_path.exists():
        raise RuntimeError(f"Hailo Apps object_detection.py was not found under: {hailo_apps_root}")
    if not post_process_path.exists():
        raise RuntimeError(f"Hailo Apps object_detection_post_process.py was not found under: {hailo_apps_root}")

    # The Hailo Apps standalone detector does not expose a JSON CLI. This hook keeps
    # Hailo imports isolated in the subprocess and captures the first output frame.
    sys.path.insert(0, str(hailo_apps_root))
    app_module = _load_module("hailo_yorkie_watch_hailo_object_detection", app_path)
    post_process_module = _load_module("hailo_yorkie_watch_hailo_post_process", post_process_path)

    captured: list[dict[str, Any]] = []

    def capture_visualize(
        input_context: Any,
        visualization_settings: Any,
        output_queue: Any,
        post_process_callback_fn: Any,
        fps_tracker: Any,
        stop_event: Any,
    ) -> None:
        del input_context, visualization_settings, fps_tracker
        while not stop_event.is_set():
            item = output_queue.get()
            if item is None:
                break
            original_frame, infer_results = item
            config_data = dict(post_process_callback_fn.keywords.get("config_data", {}))
            config_data.setdefault("visualization_params", {})
            config_data["visualization_params"]["score_thres"] = 0.0
            labels = post_process_callback_fn.keywords.get("labels", [])
            extracted = post_process_module.extract_detections(original_frame, infer_results, config_data)
            captured.extend(_serialize_detections(extracted, labels, args.threshold))
            stop_event.set()
            break

    original_argv = sys.argv[:]
    original_visualize = app_module.visualize
    try:
        app_module.visualize = capture_visualize
        sys.argv = [
            str(app_path),
            "--hef-path",
            args.hef,
            "--input",
            str(image_path),
            "--no-display",
        ]
        app_module.main()
    finally:
        app_module.visualize = original_visualize
        sys.argv = original_argv

    return captured


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serialize_detections(extracted: dict[str, Any], labels: list[str], threshold: float) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    boxes = extracted.get("detection_boxes", [])
    classes = extracted.get("detection_classes", [])
    scores = extracted.get("detection_scores", [])
    for box, class_id, score in zip(boxes, classes, scores):
        confidence = float(score)
        if confidence < threshold:
            continue
        class_index = int(class_id)
        class_name = labels[class_index] if 0 <= class_index < len(labels) else str(class_index)
        detections.append(
            {
                "class_name": class_name.lower(),
                "class_id": class_index,
                "confidence": confidence,
                "bbox": [float(value) for value in box[:4]],
            }
        )
    return detections


if __name__ == "__main__":
    raise SystemExit(main())
