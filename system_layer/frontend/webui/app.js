const { createApp } = Vue;

function pretty(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function basename(path) {
  if (!path) {
    return "-";
  }
  const normalized = String(path).replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function formatNumber(value, digits = 3) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatTimestamp(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(Number(value) * 1000);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

async function requestJson(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const config = { ...options, headers };
  if (config.body && typeof config.body !== "string") {
    config.body = JSON.stringify(config.body);
  }

  const response = await fetch(path, config);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "string"
      ? payload
      : payload.detail || payload.message || pretty(payload);
    throw new Error(detail);
  }
  return payload;
}

createApp({
  data() {
    return {
      health: {},
      modelInfo: null,
      benchmark: null,
      history: [],
      alerts: [],
      summaries: [],
      capabilities: {},
      selectedSummaryPath: "",
      signalInput: "",
      deviceIdInput: "web-manual-01",
      temperatureInput: 36.5,
      simulateMode: "direct",
      simulateSource: "synthetic",
      simulateCount: 5,
      simulateInterval: 0.5,
      predictResult: null,
      simulationResult: null,
      notice: { message: "", type: "info" },
      noticeTimer: null,
      busy: {
        refresh: false,
        modelApply: false,
        modelInfo: false,
        predict: false,
        simulate: false,
        reset: false,
      },
      wsStatus: "disconnected",
      ws: null,
      chart: null,
      chartObserver: null,
      liveFeedRecords: [],
      resizeHandler: null,
      sceneContext: null,
      sceneAnimationFrame: null,
      sceneNodes: [],
      sceneStars: [],
      sceneLastTimestamp: 0,
      tiltTarget: null,
      densityMode: "balanced",
      focusMode: "all",
      pointerEnabled: false,
      pointerVisible: false,
      pointerX: 0,
      pointerY: 0,
      pointerTrail: Array.from({ length: 6 }, () => ({ x: 0, y: 0 })),
      hoverHud: {
        visible: false,
        section: "",
        title: "",
        detail: "",
      },
    };
  },
  computed: {
    shellClasses() {
      return {
        "mode-compact": this.densityMode === "compact",
        "mode-relaxed": this.densityMode === "relaxed",
        "focus-controls": this.focusMode === "controls",
        "focus-monitor": this.focusMode === "monitor",
      };
    },
    sectionShortcuts() {
      return [
        { key: "modelSection", label: "Model" },
        { key: "predictSection", label: "Predict" },
        { key: "simulateSection", label: "Simulate" },
        { key: "realtimeSection", label: "Realtime" },
        { key: "historySection", label: "History" },
        { key: "alertsSection", label: "Alerts" },
      ];
    },
    densityOptions() {
      return [
        { value: "compact", label: "Compact" },
        { value: "balanced", label: "Balanced" },
        { value: "relaxed", label: "Relaxed" },
      ];
    },
    focusOptions() {
      return [
        { value: "all", label: "All" },
        { value: "controls", label: "Controls" },
        { value: "monitor", label: "Monitor" },
      ];
    },
    serviceTone() {
      return this.health.status === "ok" ? "ok" : this.health.status ? "error" : "warn";
    },
    mqttTone() {
      if (this.health.mqtt_enabled && !this.health.mqtt_error) {
        return "ok";
      }
      return this.health.mqtt_error ? "warn" : "warn";
    },
    mqttSummary() {
      if (this.health.mqtt_enabled && !this.health.mqtt_error) {
        return "consumer ready";
      }
      if (this.health.mqtt_error) {
        return `consumer error: ${this.health.mqtt_error}`;
      }
      return "consumer disabled";
    },
    cwruTone() {
      return this.capabilities.supports_cwru_source ? "ok" : "warn";
    },
    cwruSummary() {
      return this.capabilities.supports_cwru_source ? "full env ready" : "needs torch + data layer";
    },
    wsTone() {
      if (this.wsStatus === "connected") {
        return "ok";
      }
      if (this.wsStatus === "reconnecting") {
        return "warn";
      }
      return "warn";
    },
    wsSummary() {
      return this.wsStatus;
    },
    pipelinePills() {
      const modelMode = this.modelInfo?.algorithm
        ? `${this.modelInfo.algorithm}/${this.modelInfo.deployment_type || "deploy"}`
        : "-";
      const edgeMode = this.capabilities.supports_cwru_source ? "synthetic + CWRU" : "synthetic only";
      const realtimeMode = this.health.mqtt_enabled
        ? (this.wsStatus === "connected" ? "MQTT + WS live" : "MQTT ready")
        : "HTTP only";
      return [
        { label: "Deploy", value: this.health.runtime_backend || "-" },
        { label: "Model", value: modelMode },
        { label: "Edge", value: edgeMode },
        { label: "Realtime", value: realtimeMode },
      ];
    },
    wsBadgeClass() {
      if (this.wsStatus === "connected") {
        return "online";
      }
      if (this.wsStatus === "reconnecting") {
        return "warn";
      }
      return "offline";
    },
    liveBadgeText() {
      if (this.wsStatus === "connected") {
        return "Live updates enabled";
      }
      if (this.wsStatus === "reconnecting") {
        return "Realtime reconnecting";
      }
      return "Realtime unavailable";
    },
    summaryPathLabel() {
      return basename(this.health.model_summary_path) || "Not configured";
    },
    modelMetaCards() {
      const info = this.modelInfo || {};
      return [
        { label: "Algorithm", value: info.algorithm || "-" },
        { label: "Deployment", value: info.deployment_type || "-" },
        { label: "Backend", value: info.deployment_backend || info.runtime_backend || "-" },
        { label: "Providers", value: (info.providers || []).join(", ") || "-" },
        { label: "Model Artifact", value: basename(info.model_path) },
        { label: "Summary File", value: basename(info.summary_path) },
      ];
    },
    predictSummaryCards() {
      const result = this.predictResult || {};
      return [
        { label: "Predicted Label", value: result.predicted_label ?? "-" },
        { label: "Confidence", value: formatNumber(result.confidence) },
        { label: "Preprocess ms", value: formatNumber(result.preprocess_latency_ms) },
        { label: "Inference ms", value: formatNumber(result.inference_latency_ms) },
        { label: "End-to-End ms", value: formatNumber(result.end_to_end_latency_ms ?? result.latency_ms) },
        { label: "Source", value: result.metadata?.source || "manual" },
      ];
    },
    simulationSummaryCards() {
      const result = this.simulationResult || {};
      return [
        { label: "Mode", value: result.mode || "-" },
        { label: "Source", value: this.simulateSource || "-" },
        { label: "Count", value: result.count ?? "-" },
        { label: "Direct Results", value: Array.isArray(result.results) ? result.results.length : 0 },
        {
          label: "MQTT Path",
          value: result.mode === "mqtt" ? "published to broker" : result.mode ? "processed in service" : "-",
        },
        { label: "Device", value: "esp32-sim-01" },
      ];
    },
    benchmarkCards() {
      const benchmark = this.benchmark || {};
      return [
        { label: "Accuracy Pass", value: benchmark.accuracy_pass ? "pass" : benchmark.accuracy_pass === false ? "fail" : "-" },
        { label: "Latency Pass", value: benchmark.latency_pass ? "pass" : benchmark.latency_pass === false ? "fail" : "-" },
        { label: "Accuracy", value: formatNumber(benchmark.accuracy) },
        { label: "Avg Inference ms", value: formatNumber(benchmark.avg_latency_ms) },
      ];
    },
    historyRows() {
      return this.history
        .slice()
        .reverse()
        .map((record, index) => ({
          key: `${record.device_id || "device"}-${record.timestamp || index}-${index}`,
          time: formatTimestamp(record.timestamp),
          device: record.device_id ?? "-",
          label: record.predicted_label ?? "-",
          confidence: formatNumber(record.confidence),
          endToEnd: formatNumber(record.end_to_end_latency_ms ?? record.latency_ms),
          inference: formatNumber(record.inference_latency_ms),
          source: record.metadata?.source ?? "-",
        }));
    },
    alertRows() {
      return this.alerts
        .slice()
        .reverse()
        .map((record, index) => ({
          key: `${record.device_id || "device"}-${record.timestamp || index}-${index}`,
          time: formatTimestamp(record.timestamp),
          device: record.device_id ?? "-",
          label: record.predicted_label ?? "-",
          confidence: formatNumber(record.confidence),
          source: record.metadata?.source ?? "-",
        }));
    },
    liveFeed() {
      return this.liveFeedRecords.map((record, index) => ({
        key: `${record.device_id || "device"}-${record.timestamp || index}-${index}`,
        title: `${record.device_id || "unknown-device"} -> label ${record.predicted_label ?? "-"}`,
        detail: `conf ${formatNumber(record.confidence)}, preprocess ${formatNumber(record.preprocess_latency_ms)} ms, inference ${formatNumber(record.inference_latency_ms)} ms, end-to-end ${formatNumber(record.end_to_end_latency_ms ?? record.latency_ms)} ms`,
      }));
    },
    cursorOrbStyle() {
      return {
        transform: `translate3d(${this.pointerX - 22}px, ${this.pointerY - 22}px, 0)`,
      };
    },
    cursorHudStyle() {
      const offsetX = 18;
      const offsetY = 18;
      return {
        transform: `translate3d(${this.pointerX + offsetX}px, ${this.pointerY + offsetY}px, 0)`,
      };
    },
    simulateHint() {
      return this.capabilities.supports_cwru_source
        ? "Synthetic is fastest for demos. CWRU is available because the full training environment is present."
        : "Synthetic works in the minimal edge-system environment. CWRU requires torch and the full training stack.";
    },
  },
  watch: {
    history: {
      handler() {
        this.renderChart();
      },
      deep: true,
    },
  },
  mounted() {
    this.pointerEnabled = window.matchMedia("(pointer: fine)").matches;
    this.loadExampleSignal();
    this.connectWebSocket();
    this.initScene();
    this.resizeHandler = () => {
      if (this.chart) {
        this.chart.resize();
      }
      this.resizeScene();
    };
    window.addEventListener("resize", this.resizeHandler);
    if (window.ResizeObserver) {
      this.chartObserver = new ResizeObserver(() => {
        if (this.chart) {
          this.chart.resize();
        }
      });
      this.chartObserver.observe(this.$refs.latencyChart);
    }
    this.refreshAll();
  },
  beforeUnmount() {
    if (this.noticeTimer) {
      window.clearTimeout(this.noticeTimer);
    }
    if (this.ws) {
      this.ws.close();
    }
    if (this.chartObserver) {
      this.chartObserver.disconnect();
    }
    if (this.resizeHandler) {
      window.removeEventListener("resize", this.resizeHandler);
    }
    if (this.sceneAnimationFrame) {
      window.cancelAnimationFrame(this.sceneAnimationFrame);
    }
    this.resetTiltTarget();
  },
  methods: {
    pretty,
    formatAccuracy(value) {
      return formatNumber(value);
    },
    statusClass(tone) {
      if (tone === "ok") {
        return "status-ok";
      }
      if (tone === "error") {
        return "status-error";
      }
      return "status-warn";
    },
    showNotice(message, type = "info", timeoutMs = 4800) {
      this.notice = { message, type };
      if (this.noticeTimer) {
        window.clearTimeout(this.noticeTimer);
      }
      if (timeoutMs > 0 && type !== "error") {
        this.noticeTimer = window.setTimeout(() => {
          this.notice = { message: "", type: "info" };
        }, timeoutMs);
      }
    },
    async refreshAll(options = {}) {
      const { showSuccess = false, setBusy = true } = options;
      try {
        if (setBusy) {
          this.busy.refresh = true;
        }
        const [health, info, catalog, benchmark, history, alerts] = await Promise.all([
          requestJson("/health"),
          requestJson("/model/info"),
          requestJson("/artifacts/summaries"),
          requestJson("/benchmark/current"),
          requestJson("/history"),
          requestJson("/alerts"),
        ]);

        this.health = health;
        this.capabilities = health.capabilities || {};
        this.modelInfo = info;
        this.summaries = catalog;
        this.benchmark = benchmark;
        this.history = history;
        this.alerts = alerts;
        this.selectedSummaryPath = health.model_summary_path
          ? String(health.model_summary_path).replaceAll("\\", "/")
          : "";

        if (!this.capabilities.supports_cwru_source && this.simulateSource === "cwru") {
          this.simulateSource = "synthetic";
        }
        if (showSuccess) {
          this.showNotice("Console data refreshed.", "success", 1800);
        }
      } catch (error) {
        this.showNotice(`Refresh failed: ${error.message}`, "error", 0);
      } finally {
        if (setBusy) {
          this.busy.refresh = false;
        }
      }
    },
    scrollToSection(shortcut) {
      const target = this.$refs[shortcut.key];
      if (!target) {
        return;
      }
      target.scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" });
      this.showNotice(`Jumped to ${shortcut.label}.`, "info", 1200);
    },
    setDensityMode(mode) {
      this.densityMode = mode;
      this.showNotice(`Density set to ${mode}.`, "info", 1200);
    },
    setFocusMode(mode) {
      this.focusMode = mode;
      this.showNotice(`Focus mode set to ${mode}.`, "info", 1200);
    },
    initScene() {
      const canvas = this.$refs.fxCanvas;
      if (!canvas) {
        return;
      }
      const context = canvas.getContext("2d");
      if (!context) {
        return;
      }
      this.sceneContext = context;
      this.resizeScene();
      this.seedSceneEntities();
      this.sceneAnimationFrame = window.requestAnimationFrame((timestamp) => this.animateScene(timestamp));
    },
    resizeScene() {
      const canvas = this.$refs.fxCanvas;
      if (!canvas) {
        return;
      }
      const ratio = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      if (this.sceneContext) {
        this.sceneContext.setTransform(ratio, 0, 0, ratio, 0, 0);
      }
      this.seedSceneEntities();
    },
    seedSceneEntities() {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const nodeCount = clamp(Math.round(width / 160), 8, 18);
      const starCount = clamp(Math.round(width / 95), 14, 32);
      this.sceneNodes = Array.from({ length: nodeCount }, (_, index) => ({
        x: ((index + 1) / (nodeCount + 1)) * width,
        y: height * (0.18 + ((index % 5) * 0.09)),
        radius: 1.8 + (index % 3) * 0.8,
        drift: 0.12 + (index % 4) * 0.03,
        phase: index * 0.7,
      }));
      this.sceneStars = Array.from({ length: starCount }, (_, index) => ({
        x: Math.random() * width,
        y: Math.random() * height * 0.82,
        radius: 0.8 + Math.random() * 1.8,
        speed: 0.12 + Math.random() * 0.18,
        alpha: 0.2 + Math.random() * 0.35,
        offset: index * 0.43,
      }));
    },
    animateScene(timestamp) {
      if (!this.sceneContext) {
        return;
      }
      this.updatePointerTrail();
      const context = this.sceneContext;
      const width = window.innerWidth;
      const height = window.innerHeight;
      const delta = this.sceneLastTimestamp ? Math.min(timestamp - this.sceneLastTimestamp, 32) : 16;
      this.sceneLastTimestamp = timestamp;

      context.clearRect(0, 0, width, height);

      const pointerX = this.pointerVisible ? this.pointerX : width * 0.55;
      const pointerY = this.pointerVisible ? this.pointerY : height * 0.28;
      const horizon = height * 0.2;
      const centerX = width * 0.55 + (pointerX - width * 0.5) * 0.02;

      const glow = context.createRadialGradient(pointerX, pointerY, 12, pointerX, pointerY, width * 0.35);
      glow.addColorStop(0, "rgba(255,255,255,0.18)");
      glow.addColorStop(0.18, "rgba(77, 152, 255, 0.13)");
      glow.addColorStop(0.48, "rgba(207, 95, 52, 0.10)");
      glow.addColorStop(1, "rgba(255,255,255,0)");
      context.fillStyle = glow;
      context.fillRect(0, 0, width, height);

      context.save();
      context.globalAlpha = 0.5;
      context.strokeStyle = "rgba(35, 75, 96, 0.15)";
      context.lineWidth = 1;
      for (let i = -8; i <= 8; i += 1) {
        const x = centerX + i * width * 0.08;
        context.beginPath();
        context.moveTo(x, height);
        context.lineTo(centerX + i * width * 0.015, horizon);
        context.stroke();
      }
      for (let row = 0; row < 10; row += 1) {
        const t = row / 9;
        const y = horizon + Math.pow(t, 2.1) * (height - horizon);
        context.beginPath();
        context.moveTo(width * 0.1, y);
        context.lineTo(width * 0.98, y);
        context.stroke();
      }
      context.restore();

      context.save();
      context.strokeStyle = "rgba(255,255,255,0.08)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(width * 0.06, horizon);
      context.lineTo(width * 0.98, horizon);
      context.stroke();
      context.restore();

      this.sceneStars.forEach((star, index) => {
        const y = (star.y + timestamp * star.speed * 0.01 + index * 0.02) % (height * 0.84);
        const twinkle = 0.55 + Math.sin(timestamp * 0.001 + star.offset) * 0.25;
        context.beginPath();
        context.fillStyle = `rgba(255,248,243,${star.alpha * twinkle})`;
        context.arc(star.x, y, star.radius, 0, Math.PI * 2);
        context.fill();
      });

      this.sceneNodes.forEach((node, index) => {
        const driftY = Math.sin(timestamp * 0.0012 * node.drift + node.phase) * 14;
        const driftX = Math.cos(timestamp * 0.001 + node.phase) * 12;
        const x = node.x + driftX + (pointerX - width * 0.5) * 0.01;
        const y = node.y + driftY;
        const nextNode = this.sceneNodes[index + 1];
        if (nextNode) {
          context.beginPath();
          context.strokeStyle = "rgba(35, 75, 96, 0.1)";
          context.lineWidth = 1;
          context.moveTo(x, y);
          context.lineTo(
            nextNode.x + Math.cos(timestamp * 0.001 + nextNode.phase) * 12,
            nextNode.y + Math.sin(timestamp * 0.0012 * nextNode.drift + nextNode.phase) * 14,
          );
          context.stroke();
        }
        const nodeGlow = context.createRadialGradient(x, y, 0, x, y, 18);
        nodeGlow.addColorStop(0, "rgba(255,255,255,0.72)");
        nodeGlow.addColorStop(0.38, "rgba(64,103,215,0.24)");
        nodeGlow.addColorStop(1, "rgba(64,103,215,0)");
        context.fillStyle = nodeGlow;
        context.beginPath();
        context.arc(x, y, 18, 0, Math.PI * 2);
        context.fill();
        context.beginPath();
        context.fillStyle = "rgba(255, 252, 247, 0.95)";
        context.arc(x, y, node.radius, 0, Math.PI * 2);
        context.fill();
      });

      context.save();
      context.globalCompositeOperation = "screen";
      const beam = context.createLinearGradient(0, pointerY - 40, width, pointerY + 120);
      beam.addColorStop(0, "rgba(255,255,255,0)");
      beam.addColorStop(0.5, "rgba(255,255,255,0.045)");
      beam.addColorStop(1, "rgba(255,255,255,0)");
      context.fillStyle = beam;
      context.fillRect(0, pointerY - 80, width, 180);
      context.restore();

      this.sceneAnimationFrame = window.requestAnimationFrame((nextTimestamp) => this.animateScene(nextTimestamp));
    },
    updatePointerTrail() {
      if (!this.pointerTrail.length) {
        return;
      }
      const headX = this.pointerVisible ? this.pointerX : this.pointerTrail[0].x;
      const headY = this.pointerVisible ? this.pointerY : this.pointerTrail[0].y;
      this.pointerTrail[0].x += (headX - this.pointerTrail[0].x) * 0.28;
      this.pointerTrail[0].y += (headY - this.pointerTrail[0].y) * 0.28;
      for (let index = 1; index < this.pointerTrail.length; index += 1) {
        const previous = this.pointerTrail[index - 1];
        const current = this.pointerTrail[index];
        current.x += (previous.x - current.x) * (0.22 - index * 0.015);
        current.y += (previous.y - current.y) * (0.22 - index * 0.015);
      }
    },
    trailStyle(trail, index) {
      const size = 28 - index * 3.2;
      const opacity = 0.22 - index * 0.028;
      return {
        width: `${size}px`,
        height: `${size}px`,
        opacity: this.pointerVisible ? Math.max(opacity, 0.03) : 0,
        transform: `translate3d(${trail.x - size / 2}px, ${trail.y - size / 2}px, 0) scale(${1 - index * 0.04})`,
      };
    },
    updateTiltTarget(target, event) {
      const tiltTarget = target?.closest(".hud-target");
      if (!tiltTarget) {
        this.resetTiltTarget();
        return;
      }
      if (this.tiltTarget && this.tiltTarget !== tiltTarget) {
        this.resetTiltTarget(this.tiltTarget);
      }
      const rect = tiltTarget.getBoundingClientRect();
      const localX = clamp(event.clientX - rect.left, 0, rect.width);
      const localY = clamp(event.clientY - rect.top, 0, rect.height);
      const rotateY = ((localX / rect.width) - 0.5) * 10;
      const rotateX = (0.5 - (localY / rect.height)) * 10;
      tiltTarget.classList.add("tilt-active");
      tiltTarget.style.setProperty("--tilt-x", `${rotateX.toFixed(2)}deg`);
      tiltTarget.style.setProperty("--tilt-y", `${rotateY.toFixed(2)}deg`);
      tiltTarget.style.setProperty("--glow-x", `${((localX / rect.width) * 100).toFixed(1)}%`);
      tiltTarget.style.setProperty("--glow-y", `${((localY / rect.height) * 100).toFixed(1)}%`);
      tiltTarget.style.setProperty("--lift", "-4px");
      this.tiltTarget = tiltTarget;
    },
    resetTiltTarget(target = this.tiltTarget) {
      if (!target) {
        return;
      }
      target.classList.remove("tilt-active");
      target.style.removeProperty("--tilt-x");
      target.style.removeProperty("--tilt-y");
      target.style.removeProperty("--glow-x");
      target.style.removeProperty("--glow-y");
      target.style.removeProperty("--lift");
      if (this.tiltTarget === target) {
        this.tiltTarget = null;
      }
    },
    handlePointerMove(event) {
      if (!this.pointerEnabled) {
        return;
      }
      this.pointerVisible = true;
      this.pointerX = event.clientX;
      this.pointerY = event.clientY;
      if (!this.pointerTrail[0].x && !this.pointerTrail[0].y) {
        this.pointerTrail.forEach((trail) => {
          trail.x = event.clientX;
          trail.y = event.clientY;
        });
      }
      this.updateTiltTarget(event.target, event);
      const target = event.target.closest("[data-hud-title]");
      if (!target) {
        this.hoverHud = {
          visible: false,
          section: "",
          title: "",
          detail: "",
        };
        return;
      }
      this.hoverHud = {
        visible: true,
        section: target.dataset.hudSection || "Console",
        title: target.dataset.hudTitle || "Interactive element",
        detail: target.dataset.hudDetail || "Move through the console to inspect live context.",
      };
    },
    clearPointerState() {
      this.pointerVisible = false;
      this.resetTiltTarget();
      this.hoverHud = {
        visible: false,
        section: "",
        title: "",
        detail: "",
      };
    },
    async refreshModelInfo() {
      try {
        this.busy.modelInfo = true;
        this.modelInfo = await requestJson("/model/info");
        this.showNotice("Model info refreshed.", "info");
      } catch (error) {
        this.showNotice(`Failed to load model info: ${error.message}`, "error", 0);
      } finally {
        this.busy.modelInfo = false;
      }
    },
    loadExampleSignal() {
      this.signalInput = "0.01,0.03,0.02,0.15,0.22,0.18,0.03,0.02";
      this.showNotice("Example signal loaded into the editor.", "info", 1200);
    },
    parseSignalInput() {
      return this.signalInput
        .split(",")
        .map((item) => Number(item.trim()))
        .filter((item) => !Number.isNaN(item));
    },
    async applySelectedModel() {
      try {
        if (!this.selectedSummaryPath) {
          this.showNotice("Select a deployment bundle before applying a model.", "warn");
          return;
        }
        this.busy.modelApply = true;
        const result = await requestJson("/model/select", {
          method: "POST",
          body: { summary_path: this.selectedSummaryPath },
        });
        this.modelInfo = result;
        await this.refreshAll({ setBusy: false });
        this.showNotice(`Loaded ${result.experiment_title || "selected model"}.`, "success");
      } catch (error) {
        this.showNotice(`Model switch failed: ${error.message}`, "error", 0);
      } finally {
        this.busy.modelApply = false;
      }
    },
    handleSummarySelection() {
      if (!this.selectedSummaryPath) {
        return;
      }
      this.showNotice(`Staged ${basename(this.selectedSummaryPath)} for model apply.`, "info", 1400);
    },
    async runPredict() {
      try {
        const signal = this.parseSignalInput();
        if (!signal.length) {
          this.showNotice("Provide at least one numeric raw signal value before running prediction.", "warn");
          return;
        }
        this.busy.predict = true;
        const result = await requestJson("/predict", {
          method: "POST",
          body: {
            device_id: this.deviceIdInput || "web-manual-01",
            timestamp: Math.floor(Date.now() / 1000),
            temperature: Number(this.temperatureInput || 36.5),
            raw_signal: signal,
            event_triggered: true,
            feature_summary: {},
            metadata: { source: "web-manual" },
          },
        });

        this.predictResult = result;
        const [history, alerts] = await Promise.all([
          requestJson("/history"),
          requestJson("/alerts"),
        ]);
        this.history = history;
        this.alerts = alerts;
        this.showNotice("Direct prediction finished.", "success");
      } catch (error) {
        this.showNotice(`Prediction failed: ${error.message}`, "error", 0);
      } finally {
        this.busy.predict = false;
      }
    },
    handleSimulationSourceChange() {
      if (this.simulateSource === "cwru" && !this.capabilities.supports_cwru_source) {
        this.showNotice("CWRU simulation needs the full training environment. Synthetic is the recommended demo path here.", "warn");
        this.simulateSource = "synthetic";
        return;
      }
      this.showNotice(`Simulation source set to ${this.simulateSource}.`, "info", 1200);
    },
    handleSimulationModeChange() {
      const message = this.simulateMode === "mqtt"
        ? "Simulation mode set to MQTT publish."
        : "Simulation mode set to direct process.";
      this.showNotice(message, "info", 1200);
    },
    announceSimulationConfig() {
      this.showNotice(
        `Simulation configured for ${this.simulateCount} runs at ${this.simulateInterval}s interval.`,
        "info",
        1200,
      );
    },
    async runSimulation() {
      try {
        if (this.simulateSource === "cwru" && !this.capabilities.supports_cwru_source) {
          this.showNotice("CWRU simulation is unavailable in the minimal environment. Switch to Synthetic or use the full training stack.", "warn");
          return;
        }
        if (this.simulateMode === "mqtt" && this.health && !this.health.mqtt_enabled) {
          this.showNotice("MQTT consumer is disabled. Use Direct process for the demo or enable MQTT in system settings.", "warn");
          return;
        }
        this.busy.simulate = true;
        const result = await requestJson("/simulate/publish", {
          method: "POST",
          body: {
            mode: this.simulateMode,
            source: this.simulateSource,
            count: Number(this.simulateCount || 1),
            interval: Number(this.simulateInterval || 0),
            device_id: "esp32-sim-01",
          },
        });

        this.simulationResult = result;
        const [history, alerts] = await Promise.all([
          requestJson("/history"),
          requestJson("/alerts"),
        ]);
        this.history = history;
        this.alerts = alerts;
        this.showNotice(`Simulation finished in ${this.simulateMode} mode.`, "success");
      } catch (error) {
        this.showNotice(`Simulation failed: ${error.message}`, "error", 0);
      } finally {
        this.busy.simulate = false;
      }
    },
    async resetStorage() {
      try {
        this.busy.reset = true;
        await requestJson("/storage/reset", { method: "POST", body: {} });
        this.history = [];
        this.alerts = [];
        this.predictResult = null;
        this.simulationResult = "Runtime storage cleared.";
        this.liveFeedRecords = [];
        this.showNotice("History and alert storage cleared.", "success");
      } catch (error) {
        this.showNotice(`Failed to clear storage: ${error.message}`, "error", 0);
      } finally {
        this.busy.reset = false;
      }
    },
    connectWebSocket() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws/realtime`);
      this.ws = socket;

      socket.onopen = () => {
        this.wsStatus = "connected";
      };
      socket.onclose = () => {
        this.wsStatus = "reconnecting";
        window.setTimeout(() => this.connectWebSocket(), 1600);
      };
      socket.onerror = () => {
        this.wsStatus = "unavailable";
      };
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "diagnosis" && payload.data) {
          this.upsertRealtimeRecord(payload.data);
        }
      };
    },
    upsertRealtimeRecord(record) {
      this.history.push(record);
      this.history = this.history.slice(-50);
      if (record.predicted_label !== 0) {
        this.alerts.push(record);
        this.alerts = this.alerts.slice(-50);
      }
      this.liveFeedRecords.unshift(record);
      this.liveFeedRecords = this.liveFeedRecords.slice(0, 8);
    },
    renderChart() {
      if (!window.echarts || !this.$refs.latencyChart) {
        return;
      }
      if (!this.chart) {
        this.chart = window.echarts.init(this.$refs.latencyChart);
      }

      const latest = this.history.slice(-12);
      const labels = latest.map((item) => formatTimestamp(item.timestamp).slice(-8));
      const endToEnd = latest.map((item) => item.end_to_end_latency_ms ?? item.latency_ms ?? 0);
      const inference = latest.map((item) => item.inference_latency_ms ?? 0);
      const confidence = latest.map((item) => item.confidence ?? 0);

      this.chart.setOption({
        animationDuration: 350,
        color: ["#4067d7", "#71b968", "#f0a428"],
        grid: { left: 64, right: 52, top: 46, bottom: 36 },
        tooltip: { trigger: "axis" },
        legend: {
          top: 0,
          textStyle: { color: "#6e6d68" },
          data: ["End-to-End ms", "Inference ms", "Confidence"],
        },
        xAxis: {
          type: "category",
          boundaryGap: false,
          data: labels,
          axisLine: { lineStyle: { color: "rgba(23, 25, 29, 0.14)" } },
          axisLabel: { color: "#6e6d68" },
        },
        yAxis: [
          {
            type: "value",
            name: "Latency ms",
            nameTextStyle: { color: "#6e6d68" },
            axisLabel: { color: "#6e6d68" },
            splitLine: { lineStyle: { color: "rgba(23, 25, 29, 0.08)" } },
          },
          {
            type: "value",
            name: "Confidence",
            min: 0,
            max: 1,
            nameTextStyle: { color: "#6e6d68" },
            axisLabel: { color: "#6e6d68" },
            splitLine: { show: false },
          },
        ],
        series: [
          {
            name: "End-to-End ms",
            type: "line",
            smooth: true,
            symbolSize: 7,
            data: endToEnd,
            areaStyle: { opacity: 0.22 },
            lineStyle: { width: 3 },
          },
          {
            name: "Inference ms",
            type: "line",
            smooth: true,
            symbolSize: 6,
            data: inference,
            lineStyle: { width: 2.5 },
          },
          {
            name: "Confidence",
            type: "line",
            yAxisIndex: 1,
            smooth: true,
            symbolSize: 6,
            data: confidence,
            lineStyle: { width: 2 },
          },
        ],
      });
    },
  },
}).mount("#app");
