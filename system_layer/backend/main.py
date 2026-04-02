import asyncio
import os
import sys
from contextlib import asynccontextmanager


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from system_layer.backend.mqtt_worker import MQTTWorker
from system_layer.backend.predictor import RealTimePredictor
from system_layer.backend.websocket_manager import WebSocketManager
from system_layer.config.settings import SystemSettings
from system_layer.storage.alert_store import AlertStore
from system_layer.storage.history_store import HistoryStore


settings = SystemSettings.from_env()
history_store = HistoryStore(settings.history_path)
alert_store = AlertStore(settings.alert_path)
websocket_manager = WebSocketManager()
predictor = RealTimePredictor(settings)
mqtt_worker = MQTTWorker(settings, predictor, history_store, alert_store, websocket_manager)


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


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'runtime_backend': settings.runtime_backend,
        'mqtt_enabled': settings.enable_mqtt_consumer,
        'mqtt_error': getattr(app.state, 'mqtt_error', None),
    }


@app.get('/model/info')
def model_info():
    return predictor.model_info()


@app.get('/history')
def get_history(limit: int = 50):
    return history_store.load_recent(limit)


@app.get('/alerts')
def get_alerts(limit: int = 50):
    return alert_store.load_recent(limit)


@app.post('/predict')
async def predict(payload: dict):
    result = predictor.predict_payload(payload)
    history_store.append(result)
    if result['predicted_label'] != settings.normal_label and result['confidence'] >= settings.alert_confidence_threshold:
        alert_store.append(result)
    await websocket_manager.broadcast_json({'type': 'diagnosis', 'data': result})
    return result


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
