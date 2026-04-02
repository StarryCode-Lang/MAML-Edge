from fastapi import WebSocket


class WebSocketManager:
    def __init__(self):
        self._connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast_json(self, payload):
        stale_connections = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(payload)
            except Exception:
                stale_connections.append(websocket)
        for websocket in stale_connections:
            self.disconnect(websocket)
