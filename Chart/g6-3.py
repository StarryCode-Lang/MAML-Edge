import json
import re

from _paper_plot import CHART_DIR, repo_path


SOURCE_FILES = [
    "system_layer/frontend/webui/index.html",
    "system_layer/frontend/webui/app.js",
    "system_layer/frontend/webui/styles.css",
    "system_layer/backend/main.py",
    "system_layer/backend/websocket_manager.py",
    "system_layer/backend/predictor.py",
    "system_layer/backend/service_stats.py",
    "system_layer/storage/history_store.py",
    "system_layer/storage/alert_store.py",
    "logs/thesis_tables/paper_balanced/system_benchmark.json",
]

OUTPUT_JSON = CHART_DIR / "fig6_3_capture_manifest.json"
OUTPUT_MD = CHART_DIR / "fig6_3_capture_notes.md"


def extract_section_titles(index_html_text):
    return re.findall(r"<h2>(.*?)</h2>", index_html_text)


def load_benchmark_snapshot():
    payload = json.loads(repo_path("logs/thesis_tables/paper_balanced/system_benchmark.json").read_text(encoding="utf-8"))
    direct_channel = next(channel for channel in payload["channels"] if channel["channel"] == "direct")
    return {
        "request_count": direct_channel["request_count"],
        "avg_preprocess_latency_ms": direct_channel["avg_preprocess_latency_ms"],
        "avg_inference_latency_ms": direct_channel["avg_inference_latency_ms"],
        "avg_end_to_end_latency_ms": direct_channel["avg_end_to_end_latency_ms"],
    }


def build_manifest():
    index_html = repo_path("system_layer/frontend/webui/index.html").read_text(encoding="utf-8")
    app_js = repo_path("system_layer/frontend/webui/app.js").read_text(encoding="utf-8")
    benchmark_snapshot = load_benchmark_snapshot()

    required_sections = [
        "Model Switcher",
        "Direct Predict",
        "Simulation Controls",
        "Latency + Confidence Monitor",
        "Runtime Stats",
        "Benchmark Snapshot",
        "Alerts",
        "History",
    ]
    discovered_titles = extract_section_titles(index_html)
    section_status = {
        title: (title in discovered_titles)
        for title in required_sections
    }

    manifest = {
        "figure": "Fig. 6-3",
        "type": "real_screenshot_only",
        "capture_target": "/webui",
        "required_sections": required_sections,
        "section_presence": section_status,
        "expected_numbers": {
            "direct_request_count": benchmark_snapshot["request_count"],
            "avg_preprocess_latency_ms": benchmark_snapshot["avg_preprocess_latency_ms"],
            "avg_inference_latency_ms": benchmark_snapshot["avg_inference_latency_ms"],
            "avg_end_to_end_latency_ms": benchmark_snapshot["avg_end_to_end_latency_ms"],
        },
        "ws_signal_present_in_code": "connectWebSocket" in app_js and "/ws/realtime" in app_js,
        "history_alert_storage_present": True,
        "source_files": SOURCE_FILES,
    }
    return manifest


def build_notes(manifest):
    lines = [
        "# Fig. 6-3 Capture Notes",
        "",
        "This figure must be a real screenshot, not a rendered mockup.",
        "",
        "## Required sections",
    ]
    for title in manifest["required_sections"]:
        status = "OK" if manifest["section_presence"][title] else "MISSING"
        lines.append(f"- {title}: {status}")
    lines.extend([
        "",
        "## Numeric targets for the screenshot caption/check",
        f"- direct request count: {manifest['expected_numbers']['direct_request_count']}",
        f"- avg preprocess latency: {manifest['expected_numbers']['avg_preprocess_latency_ms']:.4f} ms",
        f"- avg inference latency: {manifest['expected_numbers']['avg_inference_latency_ms']:.4f} ms",
        f"- avg end-to-end latency: {manifest['expected_numbers']['avg_end_to_end_latency_ms']:.4f} ms",
        "",
        "## Capture recommendation",
        "- Open /webui after the backend is running with the paper-balanced model loaded.",
        "- Keep the latency monitor, runtime stats, alerts, and history visible in one frame.",
        "- Trigger at least one direct simulation batch before capturing the screen.",
    ])
    return "\n".join(lines) + "\n"


def main():
    manifest = build_manifest()
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_MD.write_text(build_notes(manifest), encoding="utf-8")

    print("Generated:")
    print(OUTPUT_JSON)
    print(OUTPUT_MD)
    print("Source files:")
    for source in SOURCE_FILES:
        print(source)


if __name__ == "__main__":
    main()
