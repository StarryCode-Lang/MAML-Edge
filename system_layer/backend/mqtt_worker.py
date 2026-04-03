import json
import logging
import asyncio


class MQTTWorker:
    def __init__(self, settings, predictor, history_store, alert_store, websocket_manager, result_callback=None):
        self.settings = settings
        self.predictor = predictor
        self.history_store = history_store
        self.alert_store = alert_store
        self.websocket_manager = websocket_manager
        self.result_callback = result_callback
        self._client = None
        self._loop = None

    def _should_raise_alert(self, result):
        return (
            result['predicted_label'] != self.settings.normal_label and
            result['confidence'] >= self.settings.alert_confidence_threshold
        )

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self.settings.mqtt_topic)

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            result = self.predictor.predict_payload(payload)
            self.history_store.append(result)
            alert_raised = self._should_raise_alert(result)
            if alert_raised:
                self.alert_store.append(result)
            if self.result_callback is not None:
                self.result_callback(result=result, source='mqtt', alert_raised=alert_raised)
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self.websocket_manager.broadcast_json({'type': 'diagnosis', 'data': result}),
                    self._loop,
                )
        except Exception as exc:
            logging.exception('Failed to process MQTT payload: %s', exc)

    def start(self, loop=None):
        import paho.mqtt.client as mqtt

        self._loop = loop
        self._client = mqtt.Client(client_id=self.settings.mqtt_client_id)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60)
        self._client.loop_start()

    def stop(self):
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
