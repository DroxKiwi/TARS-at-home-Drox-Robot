(() => {
  const SAMPLE_RATE = 16000;

  const el = {
    conn: document.getElementById("conn"),
    sessionPill: document.getElementById("sessionPill"),
    state: document.getElementById("state"),
    configPanel: document.getElementById("configPanel"),
    chatPanel: document.getElementById("chatPanel"),
    systemPrompt: document.getElementById("systemPrompt"),
    saveSettings: document.getElementById("saveSettings"),
    settingsHint: document.getElementById("settingsHint"),
    ollamaUrl: document.getElementById("ollamaUrl"),
    testOllama: document.getElementById("testOllama"),
    ollamaHint: document.getElementById("ollamaHint"),
    llmField: document.getElementById("llmField"),
    ollamaModel: document.getElementById("ollamaModel"),
    enableThinking: document.getElementById("enableThinking"),
    enableThinkingLive: document.getElementById("enableThinkingLive"),
    sttModel: document.getElementById("sttModel"),
    ttsBackend: document.getElementById("ttsBackend"),
    ttsVoice: document.getElementById("ttsVoice"),
    kokoroOpts: document.getElementById("kokoroOpts"),
    cosyOpts: document.getElementById("cosyOpts"),
    cosyPromptFile: document.getElementById("cosyPromptFile"),
    cosyPromptHint: document.getElementById("cosyPromptHint"),
    cosyPromptText: document.getElementById("cosyPromptText"),
    cosyUseDefault: document.getElementById("cosyUseDefault"),
    testCosy: document.getElementById("testCosy"),
    cosyHealthHint: document.getElementById("cosyHealthHint"),
    loadSession: document.getElementById("loadSession"),
    loadSteps: document.getElementById("loadSteps"),
    loadError: document.getElementById("loadError"),
    reconfig: document.getElementById("reconfig"),
    activeModels: document.getElementById("activeModels"),
    chatToggle: document.getElementById("chatToggle"),
    userText: document.getElementById("userText"),
    agentText: document.getElementById("agentText"),
    thinkingBlock: document.getElementById("thinkingBlock"),
    thinkingText: document.getElementById("thinkingText"),
    error: document.getElementById("error"),
    metricsLive: document.getElementById("metricsLive"),
    metricsDetail: document.getElementById("metricsDetail"),
    metricsHistory: document.getElementById("metricsHistory"),
    shapeCanvas: document.getElementById("shapeCanvas"),
    toolHint: document.getElementById("toolHint"),
  };

  let promptUploaded = false;

  let ws = null;
  let wsReady = false;
  let ollamaOk = false;
  let sessionReady = false;
  let chatActive = false;
  /** true pendant STT/LLM (pas pendant speaking — barge-in autorisé) */
  let processingBusy = false;
  let mediaStream = null;
  let audioCtx = null;
  let processor = null;
  let metricsHistory = [];
  let livePartial = {};
  let utteranceSpeechStartedAt = 0;

  let vad = { silenceMs: 700, rmsThreshold: 0.015 };
  let bargeIn = { enabled: true, minSpeechMs: 180 };
  let speechSeen = false;
  let silenceStartedAt = 0;
  let bargeSpeechStartedAt = 0;

  /** Playback syncé texte ↔ audio */
  const player = {
    queue: [],
    playing: false,
    stopped: false,
    currentSource: null,
    currentStartAt: 0,
    currentDurationMs: 0,
    currentText: "",
    completedTexts: [],
    revealRaf: 0,
    turnActive: false,
    barged: false,
  };

  function fmtMs(v) {
    if (v == null || Number.isNaN(v)) return "—";
    return `${Number(v).toFixed(0)} ms`;
  }

  function renderMetricsSummary(m) {
    if (!m) return "—";
    const lines = [
      `<b>Tour ${m.turn_id}</b>` + (m.interrupted ? " · <b>interrompu</b>" : ""),
      `Audio in: ${fmtMs(m.input_duration_ms)} (${m.input_chunks} chunks)`,
      `STT: ${fmtMs(m.stt_ms)} · RTF ${m.stt_realtime_factor ?? "—"}`,
      `LLM TTFT: ${fmtMs(m.llm_ttft_ms)} · think TTFT: ${fmtMs(m.llm_ttft_thinking_ms)}`,
      `LLM stream: ${fmtMs(m.llm_stream_ms)} · chars ${m.llm_content_chars} (+think ${m.llm_thinking_chars})`,
      `Spoken: ${m.spoken_chars ?? "—"} chars · 1er audio: ${fmtMs(m.e2e_to_first_audio_ms)} · E2E: ${fmtMs(m.e2e_total_ms)}`,
      `TTS synth total: ${fmtMs(m.tts_total_synthesize_ms)} · audio gen: ${fmtMs(m.tts_total_audio_ms)}`,
    ];
    if (m.client_vad_ms != null) lines.splice(2, 0, `VAD client→fin: ${fmtMs(m.client_vad_ms)}`);
    if (m.error) lines.push(`Erreur: ${m.error}`);
    return lines.join("<br>");
  }

  function showMetrics(m) {
    el.metricsLive.innerHTML = renderMetricsSummary(m);
    el.metricsDetail.textContent = JSON.stringify(m, null, 2);
    metricsHistory.unshift(m);
    metricsHistory = metricsHistory.slice(0, 12);
    el.metricsHistory.innerHTML = "";
    for (const item of metricsHistory) {
      const btn = document.createElement("button");
      btn.type = "button";
      const tag = item.interrupted ? "✂ " : "";
      btn.textContent = `${tag}${item.turn_id} · e2e ${fmtMs(item.e2e_total_ms)} · first_audio ${fmtMs(item.e2e_to_first_audio_ms)} · stt ${fmtMs(item.stt_ms)}`;
      btn.addEventListener("click", () => {
        el.metricsLive.innerHTML = renderMetricsSummary(item);
        el.metricsDetail.textContent = JSON.stringify(item, null, 2);
      });
      el.metricsHistory.appendChild(btn);
    }
  }

  function updateLivePartial() {
    const bits = [];
    if (livePartial.stt_ms != null) bits.push(`STT ${fmtMs(livePartial.stt_ms)}`);
    if (livePartial.ttft_ms != null) bits.push(`TTFT ${fmtMs(livePartial.ttft_ms)}`);
    if (livePartial.first_audio_ms != null) bits.push(`1er audio ${fmtMs(livePartial.first_audio_ms)}`);
    if (livePartial.last_tts_ms != null) bits.push(`TTS phrase ${fmtMs(livePartial.last_tts_ms)}`);
    if (bits.length) el.metricsLive.textContent = `En cours · ${bits.join(" · ")}`;
  }

  function setPill(node, ok, label, muted) {
    node.textContent = label;
    node.classList.toggle("ok", !!ok);
    node.classList.toggle("bad", ok === false && !muted);
    node.classList.toggle("muted", !!muted);
  }

  function setError(msg) {
    el.error.hidden = !msg;
    el.error.textContent = msg || "";
  }

  function setLoadError(msg) {
    el.loadError.hidden = !msg;
    el.loadError.textContent = msg || "";
  }

  function syncTtsBackendUi() {
    const backend = el.ttsBackend.value || "kokoro";
    el.kokoroOpts.hidden = backend !== "kokoro";
    el.cosyOpts.hidden = backend !== "cosyvoice";
  }

  function syncLoadButton() {
    const backend = el.ttsBackend.value || "kokoro";
    let ttsOk = false;
    if (backend === "kokoro") {
      ttsOk = !!el.ttsVoice.value;
    } else {
      ttsOk = !!(promptUploaded || el.cosyUseDefault.checked);
    }
    el.loadSession.disabled = !(
      ollamaOk &&
      el.ollamaModel.value &&
      el.sttModel.value &&
      ttsOk
    );
  }

  function syncChatToggle() {
    el.chatToggle.disabled = !(wsReady && sessionReady);
    el.chatToggle.classList.toggle("active", chatActive);
    el.chatToggle.textContent = chatActive ? "Arrêter le chat" : "Démarrer le chat";
  }

  function showChat(active) {
    el.configPanel.hidden = active;
    el.chatPanel.hidden = !active;
  }

  function fillSelect(select, items, valueKey, labelKey, preferred) {
    select.innerHTML = "";
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item[valueKey];
      opt.textContent = item[labelKey];
      select.appendChild(opt);
    }
    if (preferred && [...select.options].some((o) => o.value === preferred)) {
      select.value = preferred;
    }
  }

  function fillLlmModels(models, preferred) {
    el.ollamaModel.innerHTML = "";
    if (!models.length) {
      el.ollamaModel.innerHTML = '<option value="">Aucun modèle</option>';
      el.ollamaModel.disabled = true;
      el.llmField.hidden = false;
      return;
    }
    for (const name of models) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      el.ollamaModel.appendChild(opt);
    }
    if (preferred && models.includes(preferred)) el.ollamaModel.value = preferred;
    el.ollamaModel.disabled = false;
    el.llmField.hidden = false;
  }

  function renderSteps(steps, ttsHint) {
    const labels = {
      ollama: "Ollama",
      stt: "STT (Whisper)",
      tts: "TTS",
      llm_warm: "Préchauffage LLM",
    };
    const statusFr = {
      ok: "ok",
      error: "erreur",
      loading: "chargement",
      testing: "test",
      pending: "en attente",
    };
    el.loadSteps.hidden = false;
    el.loadSteps.innerHTML = "";
    for (const [key, status] of Object.entries(steps || {})) {
      const li = document.createElement("li");
      li.className = status;
      const mark =
        status === "ok" ? "✓" :
        status === "error" ? "✗" :
        status === "loading" || status === "testing" ? "…" : "·";
      const statusLabel = statusFr[status] || status;
      let label = `${mark} ${labels[key] || key} — ${statusLabel}`;
      if (key === "tts" && ttsHint && (status === "loading" || status === "pending")) {
        label += ` · ${ttsHint}`;
      }
      li.textContent = label;
      el.loadSteps.appendChild(li);
    }
  }

  function formatCosyHealth(data) {
    if (!data || data.ok === false) {
      return data?.error || "service Cosy injoignable";
    }
    if (data.model_ready) {
      return data.loaded
        ? "CosyVoice prêt (modèle + prompt)"
        : "modèle CosyVoice téléchargé — prêt à charger";
    }
    const pct = data.progress_pct != null ? `${data.progress_pct} %` : "? %";
    const got = data.bytes_on_disk != null
      ? (data.bytes_on_disk / 1e9).toFixed(1).replace(".", ",")
      : "?";
    const exp = data.bytes_expected != null
      ? (data.bytes_expected / 1e9).toFixed(1).replace(".", ",")
      : "3,2";
    return `téléchargement du modèle ${pct} (${got} / ~${exp} Go) — ne redémarre pas Docker`;
  }

  async function pollCosyHint() {
    try {
      const data = await fetch("/api/tts/cosy/health").then((r) => r.json());
      return formatCosyHealth(data);
    } catch (err) {
      return err.message || "Cosy injoignable";
    }
  }

  function clearShapeCanvas() {
    const c = el.shapeCanvas;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
  }

  function drawShape(shape, color) {
    const c = el.shapeCanvas;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width;
    const h = c.height;
    const cx = w / 2;
    const cy = h / 2;
    const size = Math.min(w, h) * 0.32;
    ctx.fillStyle = color || "blue";
    ctx.strokeStyle = color || "blue";
    ctx.lineWidth = 3;
    ctx.beginPath();
    if (shape === "square") {
      ctx.rect(cx - size, cy - size, size * 2, size * 2);
      ctx.fill();
    } else if (shape === "triangle") {
      ctx.moveTo(cx, cy - size);
      ctx.lineTo(cx + size, cy + size);
      ctx.lineTo(cx - size, cy + size);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.arc(cx, cy, size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function applyToolUi(ui) {
    if (!ui || !ui.action) return;
    if (ui.action === "clear") {
      clearShapeCanvas();
      if (el.toolHint) el.toolHint.textContent = "Panneau effacé";
      return;
    }
    if (ui.action === "show_shape") {
      if (ui.clear) clearShapeCanvas();
      drawShape(ui.shape, ui.color);
      if (el.toolHint) {
        el.toolHint.textContent = `${ui.shape} · ${ui.color}`;
        el.toolHint.className = "hint ok";
      }
      return;
    }
    if (ui.action === "web_search") {
      if (el.toolHint) {
        const n = ui.count != null ? ui.count : (ui.results || []).length;
        el.toolHint.textContent = `Recherche · ${n} résultat${n === 1 ? "" : "s"}`;
        el.toolHint.className = "hint ok";
      }
    }
  }

  function rms(float32) {
    let s = 0;
    for (let i = 0; i < float32.length; i++) s += float32[i] * float32[i];
    return Math.sqrt(s / Math.max(1, float32.length));
  }

  function truncateByRatio(text, ratio) {
    if (!text) return "";
    const r = Math.max(0, Math.min(1, ratio));
    if (r <= 0) return "";
    if (r >= 0.98) return text;
    const cut = Math.max(1, Math.floor(text.length * r));
    const slice = text.slice(0, cut);
    const sp = slice.lastIndexOf(" ");
    if (sp > 8) return slice.slice(0, sp).trim();
    return slice.trim();
  }

  function joinSpoken(parts) {
    return parts.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
  }

  function currentSpokenText() {
    const done = [...player.completedTexts];
    if (player.playing && player.currentText) {
      const elapsed = performance.now() - player.currentStartAt;
      const ratio = player.currentDurationMs > 0
        ? elapsed / player.currentDurationMs
        : 0;
      const partial = truncateByRatio(player.currentText, ratio);
      if (partial) done.push(partial);
    }
    return joinSpoken(done);
  }

  function revealUiFromPlayback() {
    el.agentText.textContent = currentSpokenText() || "—";
  }

  function stopRevealLoop() {
    if (player.revealRaf) {
      cancelAnimationFrame(player.revealRaf);
      player.revealRaf = 0;
    }
  }

  function startRevealLoop() {
    stopRevealLoop();
    const tick = () => {
      if (!player.playing) return;
      revealUiFromPlayback();
      player.revealRaf = requestAnimationFrame(tick);
    };
    player.revealRaf = requestAnimationFrame(tick);
  }

  function resetPlayer() {
    stopRevealLoop();
    if (player.currentSource) {
      try { player.currentSource.stop(); } catch { /* ok */ }
      player.currentSource = null;
    }
    player.queue = [];
    player.playing = false;
    player.stopped = false;
    player.currentDurationMs = 0;
    player.currentText = "";
    player.completedTexts = [];
    player.turnActive = false;
    player.barged = false;
  }

  function enqueueAudio(int16, sampleRate, text, durationMs) {
    if (player.stopped || player.barged) return;
    player.queue.push({ int16, sampleRate, text: text || "", durationMs: durationMs || 0 });
    if (!player.playing) playNext();
  }

  async function playNext() {
    if (player.stopped || player.barged) {
      player.playing = false;
      return;
    }
    const item = player.queue.shift();
    if (!item) {
      player.playing = false;
      stopRevealLoop();
      revealUiFromPlayback();
      if (player.turnActive && !player.barged) {
        sendSpokenCommit(currentSpokenText());
        player.turnActive = false;
      }
      return;
    }

    player.playing = true;
    player.currentText = item.text;
    audioCtx = audioCtx || new AudioContext();
    if (audioCtx.state === "suspended") await audioCtx.resume();

    const f32 = new Float32Array(item.int16.length);
    for (let i = 0; i < item.int16.length; i++) f32[i] = item.int16[i] / 32768;
    const buffer = audioCtx.createBuffer(1, f32.length, item.sampleRate);
    buffer.copyToChannel(f32, 0);
    const durMs = item.durationMs || (buffer.duration * 1000);
    player.currentDurationMs = durMs;
    player.currentStartAt = performance.now();

    const src = audioCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtx.destination);
    player.currentSource = src;
    startRevealLoop();

    await new Promise((resolve) => {
      src.onended = resolve;
      try {
        src.start();
      } catch {
        resolve();
      }
    });

    player.currentSource = null;
    if (!player.stopped && !player.barged) {
      player.completedTexts.push(item.text);
      player.currentText = "";
      revealUiFromPlayback();
      playNext();
    } else {
      player.playing = false;
      stopRevealLoop();
    }
  }

  function stopPlaybackForBargeIn() {
    const spoken = currentSpokenText();
    player.barged = true;
    player.stopped = true;
    player.queue = [];
    stopRevealLoop();
    if (player.currentSource) {
      try { player.currentSource.stop(); } catch { /* ok */ }
      player.currentSource = null;
    }
    player.playing = false;
    player.turnActive = false;
    el.agentText.textContent = spoken || "—";
    return spoken;
  }

  function sendBargeIn(spoken) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      type: "barge_in",
      spoken_text: spoken || "",
    }));
  }

  function sendSpokenCommit(spoken) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      type: "spoken_commit",
      spoken_text: spoken || "",
    }));
  }

  async function testOllama() {
    const url = el.ollamaUrl.value.trim();
    setLoadError("");
    el.ollamaHint.textContent = "Test…";
    el.ollamaHint.className = "hint";
    el.testOllama.disabled = true;
    ollamaOk = false;
    syncLoadButton();

    try {
      const res = await fetch("/api/ollama/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_url: url }),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Échec");
      ollamaOk = true;
      fillLlmModels(data.models || [], el.ollamaModel.value || "");
      el.ollamaHint.textContent = `${data.models.length} modèle(s)`;
      el.ollamaHint.className = "hint ok";
      syncLoadButton();
    } catch (err) {
      ollamaOk = false;
      el.llmField.hidden = true;
      el.ollamaHint.textContent = err.message || "Échec";
      el.ollamaHint.className = "hint bad";
      syncLoadButton();
    } finally {
      el.testOllama.disabled = false;
    }
  }

  async function testCosy() {
    el.cosyHealthHint.textContent = "Test…";
    el.cosyHealthHint.className = "hint";
    try {
      const data = await fetch("/api/tts/cosy/health").then((r) => r.json());
      if (data.ok === false) throw new Error(data.error || "injoignable");
      el.cosyHealthHint.textContent = formatCosyHealth(data);
      el.cosyHealthHint.className =
        data.model_ready || data.loaded ? "hint ok" : "hint";
    } catch (err) {
      el.cosyHealthHint.textContent = err.message || "Échec";
      el.cosyHealthHint.className = "hint bad";
    }
  }

  async function uploadCosyPrompt(file) {
    if (!file) return;
    el.cosyPromptHint.textContent = "Upload…";
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/tts/prompt", { method: "POST", body: fd });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Upload échoué");
    promptUploaded = true;
    el.cosyPromptHint.textContent = `Sample OK (${data.bytes} octets)`;
    el.cosyPromptHint.className = "hint ok";
    syncLoadButton();
  }

  function collectSettingsPayload() {
    const backend = el.ttsBackend.value || "kokoro";
    return {
      system_prompt: el.systemPrompt.value,
      ollama_base_url: el.ollamaUrl.value.trim(),
      ollama_model: el.ollamaModel.value || undefined,
      stt_model: el.sttModel.value || undefined,
      tts_backend: backend,
      tts_voice: backend === "kokoro" ? el.ttsVoice.value : "cosyvoice-clone",
      enable_thinking: el.enableThinking.checked,
    };
  }

  async function saveSettingsToDb(showHint = true) {
    const payload = collectSettingsPayload();
    // Ne pas envoyer ollama_model vide (select pas encore rempli)
    if (!payload.ollama_model) delete payload.ollama_model;
    if (!payload.stt_model) delete payload.stt_model;
    if (!payload.tts_voice) delete payload.tts_voice;
    const res = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }
    const data = await res.json();
    if (showHint && el.settingsHint) {
      el.settingsHint.textContent = "Enregistré";
      el.settingsHint.className = "hint ok";
    }
    return data;
  }

  async function loadSession() {
    setLoadError("");
    setError("");
    el.loadSession.disabled = true;
    el.loadSession.textContent = "Chargement…";
    setPill(el.sessionPill, null, "chargement…", true);
    renderSteps({ ollama: "testing", stt: "pending", tts: "pending", llm_warm: "pending" });

    const backend = el.ttsBackend.value || "kokoro";
    if (backend === "cosyvoice" && el.cosyPromptFile.files[0] && !promptUploaded) {
      try {
        await uploadCosyPrompt(el.cosyPromptFile.files[0]);
      } catch (err) {
        setLoadError(err.message || "Upload sample échoué");
        el.loadSession.textContent = "Charger en VRAM";
        syncLoadButton();
        return;
      }
    }

    try {
      await saveSettingsToDb(false);
      if (el.settingsHint) {
        el.settingsHint.textContent = "Réglages sauvés";
        el.settingsHint.className = "hint ok";
      }
    } catch (err) {
      setLoadError(`Sauvegarde DB: ${err.message || err}`);
      el.loadSession.textContent = "Charger en VRAM";
      syncLoadButton();
      return;
    }

    const payload = {
      ollama_base_url: el.ollamaUrl.value.trim(),
      ollama_model: el.ollamaModel.value,
      stt_model: el.sttModel.value,
      tts_voice: backend === "kokoro" ? el.ttsVoice.value : "cosyvoice-clone",
      tts_backend: backend,
      enable_thinking: el.enableThinking.checked,
      cosy_prompt_text: el.cosyPromptText.value.trim(),
      cosy_use_default_prompt: !!el.cosyUseDefault.checked && !promptUploaded,
    };

    try {
      const poll = setInterval(async () => {
        try {
          const s = await fetch("/api/session").then((r) => r.json());
          let ttsHint = "";
          if (backend === "cosyvoice" && s.steps?.tts === "loading") {
            ttsHint = await pollCosyHint();
            el.cosyHealthHint.textContent = ttsHint;
            el.cosyHealthHint.className = "hint";
          }
          if (s.steps) renderSteps(s.steps, ttsHint);
        } catch { /* ignore */ }
      }, 1500);

      const res = await fetch("/api/session/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      clearInterval(poll);
      const data = await res.json();
      renderSteps(data.steps || {});
      if (!data.ok) throw new Error(data.error || "Échec du chargement");

      sessionReady = true;
      el.enableThinkingLive.checked = payload.enable_thinking;
      el.thinkingBlock.hidden = !payload.enable_thinking;

      setPill(el.sessionPill, true, "session prête");
      el.activeModels.textContent =
        `STT ${payload.stt_model} · LLM ${payload.ollama_model} · TTS ${payload.tts_backend}/${payload.tts_voice}` +
        (payload.enable_thinking ? " · thinking ON" : " · thinking OFF");
      showChat(true);
      syncChatToggle();
    } catch (err) {
      sessionReady = false;
      setPill(el.sessionPill, false, "échec chargement");
      setLoadError(err.message || "Erreur");
      syncChatToggle();
    } finally {
      el.loadSession.textContent = "Charger en VRAM";
      syncLoadButton();
    }
  }

  function floatTo16BitPCM(float32) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function downsampleTo16k(input, inRate) {
    if (inRate === SAMPLE_RATE) return input;
    const ratio = inRate / SAMPLE_RATE;
    const newLen = Math.floor(input.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) result[i] = input[Math.floor(i * ratio)];
    return result;
  }

  function pcm16ToBase64(int16) {
    const bytes = new Uint8Array(int16.buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  function base64ToInt16(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Int16Array(bytes.buffer);
  }

  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => {
      wsReady = true;
      setPill(el.conn, true, "connecté");
      syncChatToggle();
    };
    ws.onclose = () => {
      wsReady = false;
      setPill(el.conn, false, "déconnecté");
      if (chatActive) stopChat(false);
      syncChatToggle();
      setTimeout(connectWs, 1500);
    };
    ws.onerror = () => setPill(el.conn, false, "erreur WS");

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") {
        if (msg.vad) {
          vad.silenceMs = msg.vad.silence_ms || 700;
          vad.rmsThreshold = msg.vad.rms_threshold || 0.015;
        }
        if (msg.barge_in) {
          bargeIn.enabled = msg.barge_in.enabled !== false;
          bargeIn.minSpeechMs = msg.barge_in.min_speech_ms || 180;
        }
      }
      if (msg.type === "status") {
        el.state.textContent = msg.state;
        processingBusy = ["stt", "streaming"].includes(msg.state);
        if (msg.state === "listening") {
          processingBusy = false;
          speechSeen = false;
          silenceStartedAt = 0;
          utteranceSpeechStartedAt = 0;
          bargeSpeechStartedAt = 0;
        }
        if (msg.state === "speaking") {
          processingBusy = false;
        }
      }
      if (msg.type === "user_transcript") {
        el.userText.textContent = msg.text || "—";
        if (msg.metrics) {
          livePartial.stt_ms = msg.metrics.stt_ms;
          updateLivePartial();
        }
      }
      if (msg.type === "assistant_reset") {
        resetPlayer();
        player.turnActive = true;
        el.agentText.textContent = "";
        el.thinkingText.textContent = "";
        livePartial = {};
        el.metricsLive.textContent = "Tour en cours…";
      }
      if (msg.type === "thinking_token") {
        el.thinkingBlock.hidden = false;
        el.thinkingText.textContent += msg.text;
      }
      if (msg.type === "tool_call") {
        if (el.toolHint) {
          el.toolHint.textContent = `Outil… ${msg.name || "?"}`;
          el.toolHint.className = "hint";
        }
      }
      if (msg.type === "tool_result") {
        applyToolUi(msg.ui);
        if (el.toolHint && !msg.ui) {
          el.toolHint.textContent = msg.content || (msg.ok ? "OK" : "Échec outil");
          el.toolHint.className = msg.ok ? "hint ok" : "hint bad";
        }
      }
      if (msg.type === "assistant_token") {
        // display:false — sync UI uniquement via playback
        if (msg.ttft_ms != null) {
          livePartial.ttft_ms = msg.ttft_ms;
          updateLivePartial();
        }
      }
      if (msg.type === "assistant_audio") {
        if (msg.metrics) {
          livePartial.last_tts_ms = msg.metrics.synthesize_ms;
          if (msg.metrics.e2e_to_first_audio_ms != null) {
            livePartial.first_audio_ms = msg.metrics.e2e_to_first_audio_ms;
          }
          updateLivePartial();
        }
        const pcm = base64ToInt16(msg.audio);
        enqueueAudio(
          pcm,
          msg.sample_rate || SAMPLE_RATE,
          msg.text || "",
          msg.duration_ms || (msg.metrics && msg.metrics.duration_ms) || 0
        );
      }
      if (msg.type === "assistant_done" || msg.type === "assistant_interrupted") {
        if (msg.text != null) el.agentText.textContent = msg.text || "—";
        player.turnActive = false;
      }
      if (msg.type === "turn_metrics") {
        showMetrics(msg.metrics);
      }
      if (msg.type === "error") setError(msg.message || "Erreur");
    };
  }

  async function startMic() {
    if (processor) return;
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
      video: false,
    });
    audioCtx = audioCtx || new AudioContext({ sampleRate: 48000 });
    if (audioCtx.state === "suspended") await audioCtx.resume();
    const source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (!chatActive || !ws || ws.readyState !== WebSocket.OPEN) return;

      const input = e.inputBuffer.getChannelData(0);
      const down = downsampleTo16k(input, audioCtx.sampleRate);
      const level = rms(down);
      const now = performance.now();
      const playbackActive = player.playing || player.queue.length > 0;

      // Barge-in pendant lecture TTS
      if (
        bargeIn.enabled &&
        playbackActive &&
        !player.barged &&
        !processingBusy
      ) {
        if (level >= vad.rmsThreshold) {
          if (!bargeSpeechStartedAt) bargeSpeechStartedAt = now;
          else if (now - bargeSpeechStartedAt >= bargeIn.minSpeechMs) {
            const spoken = stopPlaybackForBargeIn();
            sendBargeIn(spoken);
            // Enchaîne sur capture d'utterance
            speechSeen = true;
            utteranceSpeechStartedAt = bargeSpeechStartedAt;
            silenceStartedAt = 0;
            bargeSpeechStartedAt = 0;
            processingBusy = false;
            el.state.textContent = "listening";
            const pcm = floatTo16BitPCM(down);
            ws.send(JSON.stringify({
              type: "audio_chunk",
              audio: pcm16ToBase64(pcm),
            }));
            return;
          }
        } else {
          bargeSpeechStartedAt = 0;
        }
        return; // ne pas envoyer audio pendant TTS (sauf barge déclenché)
      }

      if (processingBusy) return;

      if (level >= vad.rmsThreshold) {
        if (!speechSeen) utteranceSpeechStartedAt = now;
        speechSeen = true;
        silenceStartedAt = 0;
      } else if (speechSeen) {
        if (!silenceStartedAt) silenceStartedAt = now;
        else if (now - silenceStartedAt >= vad.silenceMs) {
          const clientVadMs = utteranceSpeechStartedAt
            ? Math.round(now - utteranceSpeechStartedAt)
            : null;
          speechSeen = false;
          silenceStartedAt = 0;
          utteranceSpeechStartedAt = 0;
          processingBusy = true;
          ws.send(JSON.stringify({
            type: "end_utterance",
            client_vad_ms: clientVadMs,
          }));
          el.state.textContent = "stt";
          return;
        }
      }

      if (speechSeen || silenceStartedAt) {
        const pcm = floatTo16BitPCM(down);
        ws.send(JSON.stringify({
          type: "audio_chunk",
          audio: pcm16ToBase64(pcm),
        }));
      }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);
  }

  function stopMic() {
    if (processor) {
      processor.disconnect();
      processor = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
  }

  async function startChat() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !sessionReady || chatActive) return;
    setError("");
    el.userText.textContent = "—";
    el.agentText.textContent = "—";
    el.thinkingText.textContent = "—";
    speechSeen = false;
    silenceStartedAt = 0;
    processingBusy = false;
    resetPlayer();

    ws.send(JSON.stringify({
      type: "set_thinking",
      enable_thinking: el.enableThinkingLive.checked,
    }));
    ws.send(JSON.stringify({ type: "chat_start" }));
    chatActive = true;
    syncChatToggle();
    await startMic();
    el.state.textContent = "listening";
  }

  function stopChat(notifyServer = true) {
    chatActive = false;
    processingBusy = false;
    speechSeen = false;
    silenceStartedAt = 0;
    resetPlayer();
    stopMic();
    if (notifyServer && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "chat_stop" }));
    }
    syncChatToggle();
    el.state.textContent = "idle";
  }

  el.chatToggle.addEventListener("click", () => {
    if (chatActive) stopChat(true);
    else startChat();
  });

  el.testOllama.addEventListener("click", () => testOllama());
  el.testCosy.addEventListener("click", () => testCosy());
  el.loadSession.addEventListener("click", () => loadSession());
  el.ollamaModel.addEventListener("change", syncLoadButton);
  el.sttModel.addEventListener("change", syncLoadButton);
  el.ttsVoice.addEventListener("change", syncLoadButton);
  el.ttsBackend.addEventListener("change", () => {
    syncTtsBackendUi();
    syncLoadButton();
  });
  el.cosyPromptText.addEventListener("input", syncLoadButton);
  el.cosyUseDefault.addEventListener("change", syncLoadButton);
  el.cosyPromptFile.addEventListener("change", async () => {
    const f = el.cosyPromptFile.files[0];
    if (!f) return;
    try {
      await uploadCosyPrompt(f);
    } catch (err) {
      promptUploaded = false;
      el.cosyPromptHint.textContent = err.message || "Échec upload";
      el.cosyPromptHint.className = "hint bad";
      syncLoadButton();
    }
  });
  el.ollamaUrl.addEventListener("change", () => {
    ollamaOk = false;
    el.llmField.hidden = true;
    el.ollamaHint.textContent = "Reteste Ollama";
    el.ollamaHint.className = "hint";
    syncLoadButton();
  });

  el.enableThinkingLive.addEventListener("change", () => {
    el.thinkingBlock.hidden = !el.enableThinkingLive.checked;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "set_thinking",
        enable_thinking: el.enableThinkingLive.checked,
      }));
    }
  });

  el.reconfig.addEventListener("click", () => {
    stopChat(true);
    sessionReady = false;
    setPill(el.sessionPill, null, "session non chargée", true);
    showChat(false);
    syncChatToggle();
  });

  if (el.saveSettings) {
    el.saveSettings.addEventListener("click", async () => {
      el.saveSettings.disabled = true;
      el.settingsHint.textContent = "Enregistrement…";
      el.settingsHint.className = "hint";
      try {
        await saveSettingsToDb(true);
      } catch (err) {
        el.settingsHint.textContent = err.message || "Échec";
        el.settingsHint.className = "hint bad";
      } finally {
        el.saveSettings.disabled = false;
      }
    });
  }

  async function init() {
    const cat = await fetch("/api/catalog").then((r) => r.json());
    let saved = null;
    try {
      saved = await fetch("/api/settings").then((r) => {
        if (!r.ok) throw new Error(`settings ${r.status}`);
        return r.json();
      });
    } catch (err) {
      console.warn("GET /api/settings:", err);
      if (el.settingsHint) {
        el.settingsHint.textContent = "DB injoignable — defaults";
        el.settingsHint.className = "hint bad";
      }
    }
    const d = { ...(cat.defaults || {}), ...(saved || {}) };

    el.systemPrompt.value = d.system_prompt || "";
    el.ollamaUrl.value =
      d.ollama_base_url || "http://host.docker.internal:11434";
    el.enableThinking.checked = !!d.enable_thinking;
    fillSelect(
      el.sttModel,
      cat.stt || [],
      "id",
      "label",
      d.stt_model
    );
    fillSelect(
      el.ttsBackend,
      cat.tts_backends || [
        { id: "kokoro", label: "Kokoro" },
        { id: "cosyvoice", label: "CosyVoice 3" },
      ],
      "id",
      "label",
      d.tts_backend || "kokoro"
    );
    fillSelect(
      el.ttsVoice,
      cat.tts || [],
      "id",
      "label",
      d.tts_voice || "ff_siwis"
    );
    el.cosyPromptText.value = "";
    syncTtsBackendUi();

    // Préremplir le modèle LLM si Ollama répond
    try {
      const probe = await fetch("/api/ollama/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_url: el.ollamaUrl.value.trim() }),
      }).then((r) => r.json());
      if (probe.ok) {
        ollamaOk = true;
        fillLlmModels(probe.models || [], d.ollama_model || "");
        el.ollamaHint.textContent = `${(probe.models || []).length} modèle(s)`;
        el.ollamaHint.className = "hint ok";
      }
    } catch { /* ignore */ }

    const session = await fetch("/api/session").then((r) => r.json()).catch(() => null);
    if (session && session.ready && session.active) {
      sessionReady = true;
      setPill(el.sessionPill, true, "session prête");
      const think = !!session.active.enable_thinking;
      el.enableThinkingLive.checked = think;
      el.thinkingBlock.hidden = !think;
      el.activeModels.textContent =
        `STT ${session.active.stt} · LLM ${session.active.llm} · TTS ${session.active.tts}`;
      showChat(true);
    }

    connectWs();
    syncLoadButton();
    syncChatToggle();
  }

  init();
})();
