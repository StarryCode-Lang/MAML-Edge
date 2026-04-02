import json


class MQTTPublisher:
    def __init__(self, host='127.0.0.1', port=1883, client_id='maml_edge_publisher'):
        self.host = host
        self.port = int(port)
        self.client_id = client_id
        self._client = None

    def connect(self):
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(client_id=self.client_id)
        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()
        return self._client

    def publish_payload(self, topic, payload, qos=0):
        if self._client is None:
            self.connect()
        self._client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=qos)

    def disconnect(self):
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
