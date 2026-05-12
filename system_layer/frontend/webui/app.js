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

function downloadJsonFile(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

createApp({
  data() {
    return {
      health: {},
      modelInfo: null,
      benchmark: null,
      systemStats: null,
      history: [],
      alerts: [],
      summaries: [],
      capabilities: {},
      selectedSummaryPath: "",
      signalInput: "",
      deviceIdInput: "web-manual-01",
      temperatureInput: 36.5,
      adaptSignalInput: "",
      adaptLabelInput: 0,
      adaptCopiesInput: 3,
      adaptBlendFactor: 0.5,
      adaptDevicePrefix: "support-web",
      simulateMode: "direct",
      simulateSource: "synthetic",
      simulateCount: 5,
      simulateInterval: 0.5,
      predictResult: null,
      adaptResult: null,
      simulationResult: null,
      notice: { message: "", type: "info" },
      noticeTimer: null,
      busy: {
        refresh: false,
        modelApply: false,
        modelInfo: false,
        predict: false,
        adapt: false,
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
        { key: "modelSection", label: "模型" },
        { key: "predictSection", label: "预测" },
        { key: "adaptSection", label: "适配" },
        { key: "simulateSection", label: "仿真" },
        { key: "realtimeSection", label: "实时" },
        { key: "statsSection", label: "统计" },
        { key: "historySection", label: "历史" },
        { key: "alertsSection", label: "告警" },
      ];
    },
    densityOptions() {
      return [
        { value: "compact", label: "紧凑" },
        { value: "balanced", label: "均衡" },
        { value: "relaxed", label: "宽松" },
      ];
    },
    focusOptions() {
      return [
        { value: "all", label: "全部" },
        { value: "controls", label: "控制" },
        { value: "monitor", label: "监控" },
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
        return "消费者就绪";
      }
      if (this.health.mqtt_error) {
        return `消费者错误: ${this.health.mqtt_error}`;
      }
      return "消费者已禁用";
    },
    cwruTone() {
      return this.capabilities.supports_cwru_source ? "ok" : "warn";
    },
    cwruSummary() {
      return this.capabilities.supports_cwru_source ? "完整环境就绪" : "需要torch+data layer";
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
      const edgeMode = this.capabilities.supports_cwru_source ? "合成+CWRU" : "仅合成";
      const realtimeMode = this.health.mqtt_enabled
        ? (this.wsStatus === "connected" ? "MQTT+WS实时" : "MQTT就绪")
        : "仅HTTP";
      return [
        { label: "部署", value: this.health.runtime_backend || "-" },
        { label: "模型", value: modelMode },
        { label: "边缘", value: edgeMode },
        { label: "实时", value: realtimeMode },
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
        return "实时更新已启用";
      }
      if (this.wsStatus === "reconnecting") {
        return "实时连接重试中";
      }
      return "实时连接不可用";
    },
    summaryPathLabel() {
      return basename(this.health.model_summary_path) || "未配置";
    },
    adaptationSupported() {
      return Boolean(this.modelInfo?.adaptation_supported);
    },
    modelMetaCards() {
      const info = this.modelInfo || {};
      return [
        { label: "算法", value: info.algorithm || "-" },
        { label: "部署类型", value: info.deployment_type || "-" },
        { label: "后端", value: info.deployment_backend || info.runtime_backend || "-" },
        { label: "提供者", value: (info.providers || []).join(", ") || "-" },
        { label: "模型文件", value: basename(info.model_path) },
        { label: "摘要文件", value: basename(info.summary_path) },
        { label: "适配能力", value: info.adaptation_supported ? "原型更新就绪" : "不支持" },
        { label: "原型标签", value: (info.prototype_labels || []).join(", ") || "-" },
      ];
    },
    predictSummaryCards() {
      const result = this.predictResult || {};
      return [
        { label: "预测标签", value: result.predicted_label ?? "-" },
        { label: "置信度", value: formatNumber(result.confidence) },
        { label: "预处理 ms", value: formatNumber(result.preprocess_latency_ms) },
        { label: "推理 ms", value: formatNumber(result.inference_latency_ms) },
        { label: "端到端 ms", value: formatNumber(result.end_to_end_latency_ms ?? result.latency_ms) },
        { label: "来源", value: result.metadata?.source || "手动" },
      ];
    },
    simulationSummaryCards() {
      const result = this.simulationResult || {};
      return [
        { label: "模式", value: result.mode || "-" },
        { label: "数据源", value: this.simulateSource || "-" },
        { label: "数量", value: result.count ?? "-" },
        { label: "直接结果", value: Array.isArray(result.results) ? result.results.length : 0 },
        {
          label: "MQTT路径",
          value: result.mode === "mqtt" ? "已发布到代理" : result.mode ? "服务内处理" : "-",
        },
        { label: "设备", value: "esp32-sim-01" },
      ];
    },
    adaptSummaryCards() {
      const result = this.adaptResult?.adaptation || {};
      return [
        { label: "状态", value: this.adaptResult?.status || "-" },
        { label: "已更新标签", value: (result.updated_labels || []).join(", ") || "-" },
        { label: "新增标签", value: (result.new_labels_added || []).join(", ") || "-" },
        { label: "样本数", value: result.sample_count ?? "-" },
        { label: "原型数量", value: result.prototype_count ?? "-" },
        { label: "混合因子", value: result.blend_factor ?? this.adaptBlendFactor },
      ];
    },
    benchmarkCards() {
      const benchmark = this.benchmark || {};
      return [
        { label: "精度通过", value: benchmark.accuracy_pass ? "通过" : benchmark.accuracy_pass === false ? "失败" : "-" },
        { label: "时延通过", value: benchmark.latency_pass ? "通过" : benchmark.latency_pass === false ? "失败" : "-" },
        { label: "精度", value: formatNumber(benchmark.accuracy) },
        { label: "平均推理 ms", value: formatNumber(benchmark.avg_latency_ms) },
      ];
    },
    systemStatCards() {
      const direct = this.systemStats?.channels?.direct || {};
      const mqtt = this.systemStats?.channels?.mqtt || {};
      const adaptation = this.systemStats?.adaptation || {};
      return [
        { label: "直接请求", value: direct.request_count ?? 0 },
        { label: "直接平均端到端 ms", value: formatNumber(direct.avg_end_to_end_latency_ms) },
        { label: "MQTT请求", value: mqtt.request_count ?? 0 },
        { label: "MQTT平均端到端 ms", value: formatNumber(mqtt.avg_end_to_end_latency_ms) },
        { label: "适配操作", value: adaptation.request_count ?? 0 },
        { label: "适配样本", value: adaptation.sample_count ?? 0 },
        { label: "触发的告警", value: this.systemStats?.alerts_triggered ?? 0 },
        { label: "运行时间 秒", value: formatNumber(this.systemStats?.uptime_seconds) },
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
    consoleSnapshot() {
      return {
        exported_at: new Date().toISOString(),
        health: this.health,
        model_info: this.modelInfo,
        benchmark: this.benchmark,
        system_stats: this.systemStats,
        latest_predict: this.predictResult,
        latest_adapt: this.adaptResult,
      };
    },
    snapshotFilename() {
      const stamp = new Date().toISOString().replaceAll(":", "-");
      return `maml-edge-console-snapshot-${stamp}.json`;
    },
    adaptHint() {
      if (this.adaptationSupported) {
        return "此部署支持运行时原型更新。提供一个小支持信号和标签以刷新原型质心。";
      }
      return "当前加载的部署是分类器路径。切换到ProtoNet编码器包以启用运行时原型更新。";
    },
    simulateHint() {
      return this.capabilities.supports_cwru_source
        ? "合成数据最适合演示。CWRU可用，因为完整训练环境已就绪。"
        : "合成数据在最小edge-system环境中工作。CWRU需要torch和完整训练堆栈。";
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
    this.loadAdaptExample();
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
        const [health, info, catalog, benchmark, systemStats, history, alerts] = await Promise.all([
          requestJson("/health"),
          requestJson("/model/info"),
          requestJson("/artifacts/summaries"),
          requestJson("/benchmark/current"),
          requestJson("/system/stats"),
          requestJson("/history"),
          requestJson("/alerts"),
        ]);

        this.health = health;
        this.capabilities = health.capabilities || {};
        this.modelInfo = info;
        this.summaries = catalog;
        this.benchmark = benchmark;
        this.systemStats = systemStats;
        this.history = history;
        this.alerts = alerts;
        this.selectedSummaryPath = health.model_summary_path
          ? String(health.model_summary_path).replaceAll("\\", "/")
          : "";

        if (!this.capabilities.supports_cwru_source && this.simulateSource === "cwru") {
          this.simulateSource = "synthetic";
        }
        if (showSuccess) {
          this.showNotice("控制台数据已刷新。", "success", 1800);
        }
      } catch (error) {
        this.showNotice(`刷新失败: ${error.message}`, "error", 0);
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
      this.showNotice(`已跳转至 ${shortcut.label}。`, "info", 1200);
    },
    setDensityMode(mode) {
      this.densityMode = mode;
      this.showNotice(`密度已设置为 ${mode}。`, "info", 1200);
    },
    setFocusMode(mode) {
      this.focusMode = mode;
      this.showNotice(`焦点模式已设置为 ${mode}。`, "info", 1200);
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
        section: target.dataset.hudSection || "控制台",
        title: target.dataset.hudTitle || "交互元素",
        detail: target.dataset.hudDetail || "在控制台中移动以检查实时上下文。",
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
        this.showNotice("模型信息已刷新。", "info");
      } catch (error) {
        this.showNotice(`加载模型信息失败: ${error.message}`, "error", 0);
      } finally {
        this.busy.modelInfo = false;
      }
    },
    loadExampleSignal() {
      this.signalInput = "0.01,0.03,0.02,0.15,0.22,0.18,0.03,0.02";
      this.showNotice("示例信号已加载到编辑器。", "info", 1200);
    },
    loadAdaptExample() {
      this.adaptSignalInput = this.signalInput || "0.01,0.03,0.02,0.15,0.22,0.18,0.03,0.02";
      this.adaptLabelInput = 0;
      this.adaptCopiesInput = 3;
      this.showNotice("原型适配示例支持集已加载。", "info", 1200);
    },
    parseSignalInput() {
      return this.signalInput
        .split(",")
        .map((item) => Number(item.trim()))
        .filter((item) => !Number.isNaN(item));
    },
    parseSupportSignalInput() {
      return this.adaptSignalInput
        .split(",")
        .map((item) => Number(item.trim()))
        .filter((item) => !Number.isNaN(item));
    },
    async applySelectedModel() {
      try {
        if (!this.selectedSummaryPath) {
          this.showNotice("在应用模型前请选择部署包。", "warn");
          return;
        }
        this.busy.modelApply = true;
        const result = await requestJson("/model/select", {
          method: "POST",
          body: { summary_path: this.selectedSummaryPath },
        });
        this.modelInfo = result;
        await this.refreshAll({ setBusy: false });
        this.showNotice(`已加载 ${result.experiment_title || "所选模型"}。`, "success");
      } catch (error) {
        this.showNotice(`模型切换失败: ${error.message}`, "error", 0);
      } finally {
        this.busy.modelApply = false;
      }
    },
    handleSummarySelection() {
      if (!this.selectedSummaryPath) {
        return;
      }
      this.showNotice(`已暂存 ${basename(this.selectedSummaryPath)} 用于模型应用。`, "info", 1400);
    },
    async runPredict() {
      try {
        const signal = this.parseSignalInput();
        if (!signal.length) {
          this.showNotice("运行预测前请提供至少一个数值原始信号值。", "warn");
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
        await this.refreshAll({ setBusy: false });
        this.showNotice("直接预测完成。", "success");
      } catch (error) {
        this.showNotice(`预测失败: ${error.message}`, "error", 0);
      } finally {
        this.busy.predict = false;
      }
    },
    async runAdapt() {
      try {
        if (!this.adaptationSupported) {
          this.showNotice("当前部署不支持运行时原型更新。", "warn");
          return;
        }
        const signal = this.parseSupportSignalInput();
        if (!signal.length) {
          this.showNotice("运行运行时适配前请提供数值支持信号。", "warn");
          return;
        }
        const sampleCount = Math.max(1, Number(this.adaptCopiesInput || 1));
        const supportSamples = Array.from({ length: sampleCount }, (_, index) => ({
          device_id: `${this.adaptDevicePrefix || "support-web"}-${index + 1}`,
          label: Number(this.adaptLabelInput || 0),
          raw_signal: signal,
        }));
        this.busy.adapt = true;
        const result = await requestJson("/adapt", {
          method: "POST",
          body: {
            blend_factor: Number(this.adaptBlendFactor),
            support_samples: supportSamples,
          },
        });
        this.adaptResult = result;
        this.modelInfo = result.model_info || this.modelInfo;
        await this.refreshAll({ setBusy: false });
        this.showNotice("运行时原型更新完成。", "success");
      } catch (error) {
        this.showNotice(`适配失败: ${error.message}`, "error", 0);
      } finally {
        this.busy.adapt = false;
      }
    },
    handleSimulationSourceChange() {
      if (this.simulateSource === "cwru" && !this.capabilities.supports_cwru_source) {
        this.showNotice("CWRU仿真需要完整训练环境。合成数据是此处推荐的演示路径。", "warn");
        this.simulateSource = "synthetic";
        return;
      }
      this.showNotice(`仿真数据源已设置为 ${this.simulateSource}。`, "info", 1200);
    },
    handleSimulationModeChange() {
      const message = this.simulateMode === "mqtt"
        ? "仿真模式已设置为MQTT发布。"
        : "仿真模式已设置为直接处理。";
      this.showNotice(message, "info", 1200);
    },
    announceSimulationConfig() {
      this.showNotice(
        `仿真已配置为 ${this.simulateCount} 次运行，间隔 ${this.simulateInterval}秒。`,
        "info",
        1200,
      );
    },
    async runSimulation() {
      try {
        if (this.simulateSource === "cwru" && !this.capabilities.supports_cwru_source) {
          this.showNotice("CWRU仿真在最小环境中不可用。切换到合成数据或使用完整训练堆栈。", "warn");
          return;
        }
        if (this.simulateMode === "mqtt" && this.health && !this.health.mqtt_enabled) {
          this.showNotice("MQTT消费者已禁用。使用直接处理进行演示或启用系统设置中的MQTT。", "warn");
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
        if (this.simulateMode === "mqtt") {
          await new Promise((resolve) => window.setTimeout(resolve, 900));
        }
        await this.refreshAll({ setBusy: false });
        this.showNotice(`仿真在${this.simulateMode}模式下完成。`, "success");
      } catch (error) {
        this.showNotice(`仿真失败: ${error.message}`, "error", 0);
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
        this.adaptResult = null;
        this.simulationResult = "运行时存储已清除。";
        this.liveFeedRecords = [];
        await this.refreshAll({ setBusy: false });
        this.showNotice("历史和告警存储已清除。", "success");
      } catch (error) {
        this.showNotice(`清除存储失败: ${error.message}`, "error", 0);
      } finally {
        this.busy.reset = false;
      }
    },
    async copyConsoleSnapshot() {
      try {
        const payload = JSON.stringify(this.consoleSnapshot, null, 2);
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(payload);
        } else {
          const textarea = document.createElement("textarea");
          textarea.value = payload;
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          document.body.removeChild(textarea);
        }
        this.showNotice("控制台快照已复制到剪贴板。", "success", 1600);
      } catch (error) {
        this.showNotice(`复制失败: ${error.message}`, "error", 0);
      }
    },
    downloadConsoleSnapshot() {
      downloadJsonFile(this.snapshotFilename, this.consoleSnapshot);
      this.showNotice("控制台快照下载已开始。", "success", 1600);
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
          data: ["端到端 ms", "推理 ms", "置信度"],
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
            name: "时延 ms",
            nameTextStyle: { color: "#6e6d68" },
            axisLabel: { color: "#6e6d68" },
            splitLine: { lineStyle: { color: "rgba(23, 25, 29, 0.08)" } },
          },
          {
            type: "value",
            name: "置信度",
            min: 0,
            max: 1,
            nameTextStyle: { color: "#6e6d68" },
            axisLabel: { color: "#6e6d68" },
            splitLine: { show: false },
          },
        ],
        series: [
          {
            name: "端到端 ms",
            type: "line",
            smooth: true,
            symbolSize: 7,
            data: endToEnd,
            areaStyle: { opacity: 0.22 },
            lineStyle: { width: 3 },
          },
          {
            name: "推理 ms",
            type: "line",
            smooth: true,
            symbolSize: 6,
            data: inference,
            lineStyle: { width: 2.5 },
          },
          {
            name: "置信度",
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
