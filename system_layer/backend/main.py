import asyncio
import importlib.util
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from edge_layer.mqtt_client.publisher import MQTTPublisher
from edge_layer.simulator.publish_signal import _build_synthetic_signal, _load_cwru_signal
from edge_layer.simulator.sample_payloads import build_signal_payload
from system_layer.backend.mqtt_worker import MQTTWorker
from system_layer.backend.predictor import RealTimePredictor
from system_layer.backend.websocket_manager import WebSocketManager
from system_layer.config.settings import SystemSettings
from system_layer.storage.alert_store import AlertStore
from system_layer.storage.history_store import HistoryStore
from test_layer.benchmark import check_thresholds, load_summary


settings = SystemSettings.from_env()
history_store = HistoryStore(settings.history_path)
alert_store = AlertStore(settings.alert_path)
websocket_manager = WebSocketManager()
predictor = RealTimePredictor(settings)
mqtt_worker = MQTTWorker(settings, predictor, history_store, alert_store, websocket_manager)
WEBUI_DIR = Path(ROOT_DIR) / 'system_layer' / 'frontend' / 'webui'


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_mqtt_consumer:
        try:
            mqtt_worker.start(loop=asyncio.get_running_loop())
        except Exception as exc:
            app.state.mqtt_error = str(exc)
    yield
    mqtt_worker.stop()


app = FastAPI(title='MAML-Edge System Service', lifespan=lifespan)
if WEBUI_DIR.exists():
    app.mount('/webui', StaticFiles(directory=str(WEBUI_DIR)), name='webui')


def _should_raise_alert(result):
    return (
        result['predicted_label'] != settings.normal_label and
        result['confidence'] >= settings.alert_confidence_threshold
    )


async def _process_payload(payload):
    result = predictor.predict_payload(payload)
    history_store.append(result)
    if _should_raise_alert(result):
        alert_store.append(result)
    await websocket_manager.broadcast_json({'type': 'diagnosis', 'data': result})
    return result


def _list_summary_catalog():
    catalog = []
    artifact_root = Path(ROOT_DIR) / 'deploy_artifacts'
    for summary_path in sorted(artifact_root.glob('*/compression_summary.json')):
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        relative_path = summary_path.relative_to(Path(ROOT_DIR)).as_posix()
        deploy_metrics = summary.get('deployment_int8_metrics') or summary.get('deployment_float_metrics') or {}
        catalog.append({
            'summary_path': relative_path,
            'experiment_title': summary.get('experiment_title'),
            'algorithm': summary.get('algorithm'),
            'deployment_backend': summary.get('deployment_backend', 'onnxruntime'),
            'accuracy': deploy_metrics.get('accuracy'),
            'avg_latency_ms': deploy_metrics.get('avg_latency_ms'),
        })
    return catalog


def _detect_capabilities():
    torch_available = importlib.util.find_spec('torch') is not None
    return {
        'supports_cwru_source': torch_available,
        'supports_mqtt_publish': True,
        'supports_direct_simulation': True,
        'webui_path': '/',
    }


def _reload_predictor(summary_path=None, runtime_backend=None, prefer_int8=None):
    global predictor
    if summary_path is not None:
        if not os.path.exists(summary_path):
            raise FileNotFoundError('Compression summary not found: {}'.format(summary_path))
        settings.model_summary_path = summary_path
    if runtime_backend is not None:
        settings.runtime_backend = runtime_backend
    if prefer_int8 is not None:
        settings.prefer_int8 = bool(prefer_int8)
    predictor = RealTimePredictor(settings)
    mqtt_worker.predictor = predictor
    return predictor.model_info()


@app.get('/', include_in_schema=False)
def index():
    index_path = WEBUI_DIR / 'index.html'
    if not index_path.exists():
        return {'status': 'missing-ui', 'message': 'Frontend assets are not available.'}
    return FileResponse(index_path)


@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'runtime_backend': settings.runtime_backend,
        'mqtt_enabled': settings.enable_mqtt_consumer,
        'mqtt_error': getattr(app.state, 'mqtt_error', None),
        'model_summary_path': settings.model_summary_path,
        'experiment_title': predictor.service.summary.get('experiment_title'),
        'capabilities': _detect_capabilities(),
    }


@app.get('/model/info')
def model_info():
    return predictor.model_info()


@app.get('/artifacts/summaries')
def list_artifact_summaries():
    return _list_summary_catalog()


@app.post('/model/select')
def select_model(payload: dict):
    summary_path = payload.get('summary_path')
    if not summary_path:
        raise HTTPException(status_code=400, detail='summary_path is required.')
    try:
        return _reload_predictor(
            summary_path=summary_path,
            runtime_backend=payload.get('runtime_backend'),
            prefer_int8=payload.get('prefer_int8'),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/benchmark/current')
def benchmark_current():
    return check_thresholds(load_summary(settings.model_summary_path))


@app.get('/history')
def get_history(limit: int = 50):
    return history_store.load_recent(limit)


@app.get('/alerts')
def get_alerts(limit: int = 50):
    return alert_store.load_recent(limit)


@app.post('/storage/reset')
def reset_storage():
    history_store.clear()
    alert_store.clear()
    return {'status': 'ok'}


@app.post('/predict')
async def predict(payload: dict):
    return await _process_payload(payload)


@app.post('/simulate/publish')
async def simulate_publish(payload: dict):
    source = payload.get('source', 'synthetic')
    mode = payload.get('mode', 'direct')
    if source not in {'synthetic', 'cwru'}:
        raise HTTPException(status_code=400, detail='source must be "synthetic" or "cwru".')
    if mode not in {'direct', 'mqtt'}:
        raise HTTPException(status_code=400, detail='mode must be "direct" or "mqtt".')
    device_id = payload.get('device_id', 'esp32-sim-01')
    count = int(payload.get('count', 1))
    interval = float(payload.get('interval', 1.0))
    domain = int(payload.get('domain', 3))
    label = int(payload.get('label', 0))
    time_steps = int(payload.get('time_steps', settings.time_steps))
    temperature = float(payload.get('temperature', 36.5))
    seed = int(payload.get('seed', 42))

    publisher = None
    results = []
    if mode == 'mqtt':
        publisher = MQTTPublisher(
            host=payload.get('host', settings.mqtt_host),
            port=int(payload.get('port', settings.mqtt_port)),
            client_id='{}_publisher'.format(device_id),
        )
        publisher.connect()

    try:
        for publish_index in range(count):
            if source == 'cwru':
                try:
                    signal = _load_cwru_signal(
                        data_dir_path=payload.get('data_dir_path', './data'),
                        domain=domain,
                        label=label,
                        seed=seed + publish_index,
                    )
                except ModuleNotFoundError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail='CWRU simulation requires the full training environment with torch installed.',
                    ) from exc
            else:
                signal = _build_synthetic_signal(time_steps, seed + publish_index)

            signal_payload = build_signal_payload(
                device_id=device_id,
                raw_signal=signal,
                temperature=temperature,
                metadata={
                    'source': source,
                    'domain': domain,
                    'label': label,
                    'publish_index': publish_index,
                },
            )

            if mode == 'mqtt':
                publisher.publish_payload(
                    payload.get('topic', 'maml-edge/devices/{}/signal'.format(device_id)),
                    signal_payload,
                )
            else:
                results.append(await _process_payload(signal_payload))

            if publish_index < count - 1 and interval > 0:
                await asyncio.sleep(interval)
    finally:
        if publisher is not None:
            publisher.disconnect()

    return {
        'status': 'ok',
        'mode': mode,
        'count': count,
        'results': results,
    }


@app.post('/adapt')
def adapt():
    return {
        'status': 'pending',
        'message': 'Online few-shot adaptation interface is reserved for the next stage.',
    }


@app.websocket(settings.websocket_path)
async def realtime_websocket(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
