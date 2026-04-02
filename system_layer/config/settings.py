import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_latest_summary(root_dir):
    summary_files = sorted(
        root_dir.glob('deploy_artifacts/*/compression_summary.json'),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return str(summary_files[0]) if summary_files else None


@dataclass
class SystemSettings:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    mqtt_host: str = '127.0.0.1'
    mqtt_port: int = 1883
    mqtt_topic: str = 'maml-edge/devices/+/signal'
    mqtt_client_id: str = 'maml_edge_backend'
    enable_mqtt_consumer: bool = True
    runtime_backend: str = 'onnxruntime'
    prefer_int8: bool = True
    model_summary_path: str = None
    dataset_name: str = 'CWRU'
    preprocess: str = 'STFT'
    time_steps: int = 1024
    image_size: int = 84
    stft_window_size: int = 64
    stft_overlap: float = 0.5
    normal_label: int = 0
    alert_confidence_threshold: float = 0.8
    storage_dir: str = None
    history_path: str = None
    alert_path: str = None
    websocket_path: str = '/ws/realtime'
    api_host: str = '0.0.0.0'
    api_port: int = 8000

    def __post_init__(self):
        if self.model_summary_path is None:
            self.model_summary_path = _find_latest_summary(self.root_dir)
        if self.storage_dir is None:
            self.storage_dir = str(self.root_dir / 'system_layer' / 'storage' / 'runtime')
        os.makedirs(self.storage_dir, exist_ok=True)
        if self.history_path is None:
            self.history_path = os.path.join(self.storage_dir, 'history.json')
        if self.alert_path is None:
            self.alert_path = os.path.join(self.storage_dir, 'alerts.json')

    @classmethod
    def from_env(cls):
        root_dir = Path(__file__).resolve().parents[2]
        return cls(
            root_dir=root_dir,
            mqtt_host=os.getenv('MAML_EDGE_MQTT_HOST', '127.0.0.1'),
            mqtt_port=int(os.getenv('MAML_EDGE_MQTT_PORT', '1883')),
            mqtt_topic=os.getenv('MAML_EDGE_MQTT_TOPIC', 'maml-edge/devices/+/signal'),
            mqtt_client_id=os.getenv('MAML_EDGE_MQTT_CLIENT_ID', 'maml_edge_backend'),
            enable_mqtt_consumer=os.getenv('MAML_EDGE_ENABLE_MQTT', '1') != '0',
            runtime_backend=os.getenv('MAML_EDGE_RUNTIME_BACKEND', 'onnxruntime'),
            prefer_int8=os.getenv('MAML_EDGE_PREFER_INT8', '1') != '0',
            model_summary_path=os.getenv('MAML_EDGE_MODEL_SUMMARY_PATH'),
            dataset_name=os.getenv('MAML_EDGE_DATASET', 'CWRU'),
            preprocess=os.getenv('MAML_EDGE_PREPROCESS', 'STFT'),
            time_steps=int(os.getenv('MAML_EDGE_TIME_STEPS', '1024')),
            image_size=int(os.getenv('MAML_EDGE_IMAGE_SIZE', '84')),
            stft_window_size=int(os.getenv('MAML_EDGE_STFT_WINDOW_SIZE', '64')),
            stft_overlap=float(os.getenv('MAML_EDGE_STFT_OVERLAP', '0.5')),
            normal_label=int(os.getenv('MAML_EDGE_NORMAL_LABEL', '0')),
            alert_confidence_threshold=float(os.getenv('MAML_EDGE_ALERT_CONFIDENCE', '0.8')),
            storage_dir=os.getenv('MAML_EDGE_STORAGE_DIR'),
            history_path=os.getenv('MAML_EDGE_HISTORY_PATH'),
            alert_path=os.getenv('MAML_EDGE_ALERT_PATH'),
            websocket_path=os.getenv('MAML_EDGE_WEBSOCKET_PATH', '/ws/realtime'),
            api_host=os.getenv('MAML_EDGE_API_HOST', '0.0.0.0'),
            api_port=int(os.getenv('MAML_EDGE_API_PORT', '8000')),
        )
