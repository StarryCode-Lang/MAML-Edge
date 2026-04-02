# Web UI

This directory hosts a zero-build Vue 3 + ECharts browser console served directly by FastAPI.

Routing:

- `GET /` serves `index.html`
- static assets are mounted at `/webui`

Current capabilities:

- switch deployment artifacts from the browser
- inspect system health and current model info
- run direct prediction requests
- trigger synthetic or CWRU simulation in `direct` or `mqtt` mode
- clear runtime history and alerts
- view history and alerts
- watch realtime WebSocket diagnosis updates
- inspect the current deployment benchmark snapshot

Implementation notes:

- Vue 3 is loaded from CDN and drives the page state without a separate frontend build step
- ECharts renders the realtime latency/confidence visualization
- FastAPI remains the only service that needs to be started for local demos
