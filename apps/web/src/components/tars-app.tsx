"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  cosyHealth,
  createRole,
  deleteRole,
  getCatalog,
  getSession,
  getSettings,
  listRoles,
  listFunctions,
  loadSession,
  putSettings,
  testOllama,
  unloadSession,
  updateRole,
  uploadCosyPrompt,
  type FunctionDefInfo,
  type LlmRole,
} from "@/lib/api";
import {
  SAMPLE_RATE,
  base64ToInt16,
  downsampleTo16k,
  floatTo16BitPCM,
  joinSpoken,
  pcm16ToBase64,
  rms,
  truncateByRatio,
} from "@/lib/audio";
import { voiceWsUrl } from "@/lib/voice-url";
import { ShapeCanvas } from "@/components/shape-canvas";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

type StepMap = Record<string, string>;

type PlayerState = {
  queue: { int16: Int16Array; sampleRate: number; text: string; durationMs: number }[];
  playing: boolean;
  stopped: boolean;
  barged: boolean;
  turnActive: boolean;
  currentSource: AudioBufferSourceNode | null;
  currentStartAt: number;
  currentDurationMs: number;
  currentText: string;
  completedTexts: string[];
  revealRaf: number;
};

function fmtMs(v: unknown) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${Number(v).toFixed(0)} ms`;
}

function formatCosyHealth(data: Record<string, unknown>) {
  if (!data || data.ok === false) return String(data?.error || "Cosy injoignable");
  if (data.model_ready) {
    return data.loaded
      ? "CosyVoice prêt (modèle + prompt)"
      : "modèle CosyVoice téléchargé — prêt à charger";
  }
  const pct = data.progress_pct != null ? `${data.progress_pct} %` : "? %";
  return `téléchargement modèle ${pct}`;
}

export function TarsApp() {
  const [view, setView] = useState<"config" | "chat">("config");
  const [conn, setConn] = useState<"…"| "connecté" | "déconnecté" | "erreur WS">("…");
  const [sessionLabel, setSessionLabel] = useState("session non chargée");
  const [sessionTone, setSessionTone] = useState<"muted" | "ok" | "bad">("muted");
  const [stateLabel, setStateLabel] = useState("idle");

  const [systemPrompt, setSystemPrompt] = useState("");
  const [ollamaUrl, setOllamaUrl] = useState("http://host.docker.internal:11434");
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaModel, setOllamaModel] = useState("");
  const [ollamaOk, setOllamaOk] = useState(false);
  const [ollamaHint, setOllamaHint] = useState("");
  const [enableThinking, setEnableThinking] = useState(false);
  const [enableThinkingLive, setEnableThinkingLive] = useState(false);

  const [sttOptions, setSttOptions] = useState<{ id: string; label: string }[]>([]);
  const [ttsOptions, setTtsOptions] = useState<{ id: string; label: string }[]>([]);
  const [sttModel, setSttModel] = useState("");
  const [ttsBackend, setTtsBackend] = useState("kokoro");
  const [ttsVoice, setTtsVoice] = useState("");
  const [cosyPromptText, setCosyPromptText] = useState("");
  const [cosyUseDefault, setCosyUseDefault] = useState(false);
  const [promptUploaded, setPromptUploaded] = useState(false);
  const [cosyHint, setCosyHint] = useState("");
  const [settingsHint, setSettingsHint] = useState("");
  const [loadError, setLoadError] = useState("");
  const [loadSteps, setLoadSteps] = useState<StepMap | null>(null);
  const [loading, setLoading] = useState(false);
  const [unloading, setUnloading] = useState(false);
  const [sessionReady, setSessionReady] = useState(false);
  const [activeModels, setActiveModels] = useState("");
  const [chatActive, setChatActive] = useState(false);
  const [wsReady, setWsReady] = useState(false);

  const [userText, setUserText] = useState("—");
  const [agentText, setAgentText] = useState("—");
  const [thinkingText, setThinkingText] = useState("");
  const [showThinking, setShowThinking] = useState(false);
  const [error, setError] = useState("");
  const [toolHint, setToolHint] = useState("En attente d’un outil…");
  const [specialistReply, setSpecialistReply] = useState("");
  const [specialistMeta, setSpecialistMeta] = useState("");
  const [shape, setShape] = useState<string | null>(null);
  const [shapeColor, setShapeColor] = useState("blue");
  const [clearToken, setClearToken] = useState(0);
  const [metricsLive, setMetricsLive] = useState("En attente d’un tour…");
  const [metricsDetail, setMetricsDetail] = useState("—");
  const [metricsHistory, setMetricsHistory] = useState<Record<string, unknown>[]>([]);
  const [actor, setActor] = useState("idle");
  const [resource, setResource] = useState<Record<string, unknown> | null>(null);
  const [stageLog, setStageLog] = useState<
    { stage: string; ms?: number; ok?: boolean; detail?: string; at: number }[]
  >([]);
  const [toolLog, setToolLog] = useState<
    { name: string; ok?: boolean; at: number; detail?: string }[]
  >([]);
  const [roles, setRoles] = useState<LlmRole[]>([]);
  const [functionCatalog, setFunctionCatalog] = useState<FunctionDefInfo[]>([]);
  const [roleHint, setRoleHint] = useState("");
  const [roleHistoryN, setRoleHistoryN] = useState(8);
  const [roleForm, setRoleForm] = useState({
    id: null as number | null,
    key: "",
    name: "",
    description: "",
    system_prompt: "",
    ollama_model: "",
    function_keys: [] as string[],
    enabled: true,
  });

  const cosyFileRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const chatActiveRef = useRef(false);
  const sessionReadyRef = useRef(false);
  const processingBusyRef = useRef(false);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const vadRef = useRef({ silenceMs: 700, rmsThreshold: 0.015 });
  const bargeRef = useRef({ enabled: true, minSpeechMs: 180 });
  const speechSeenRef = useRef(false);
  const silenceStartedAtRef = useRef(0);
  const utteranceStartRef = useRef(0);
  const bargeSpeechStartRef = useRef(0);
  const livePartialRef = useRef<Record<string, number>>({});
  const playerRef = useRef<PlayerState>({
    queue: [],
    playing: false,
    stopped: false,
    barged: false,
    turnActive: false,
    currentSource: null,
    currentStartAt: 0,
    currentDurationMs: 0,
    currentText: "",
    completedTexts: [],
    revealRaf: 0,
  });

  const canLoad = useMemo(() => {
    const ttsOk =
      ttsBackend === "kokoro"
        ? !!ttsVoice
        : promptUploaded || cosyUseDefault;
    return ollamaOk && !!ollamaModel && !!sttModel && ttsOk && !loading;
  }, [ollamaOk, ollamaModel, sttModel, ttsBackend, ttsVoice, promptUploaded, cosyUseDefault, loading]);

  const updateLivePartial = useCallback(() => {
    const p = livePartialRef.current;
    const bits: string[] = [];
    if (p.stt_ms != null) bits.push(`STT ${fmtMs(p.stt_ms)}`);
    if (p.ttft_ms != null) bits.push(`TTFT ${fmtMs(p.ttft_ms)}`);
    if (p.first_audio_ms != null) bits.push(`1er audio ${fmtMs(p.first_audio_ms)}`);
    if (p.last_tts_ms != null) bits.push(`TTS phrase ${fmtMs(p.last_tts_ms)}`);
    if (bits.length) setMetricsLive(`En cours · ${bits.join(" · ")}`);
  }, []);

  const currentSpokenText = useCallback(() => {
    const player = playerRef.current;
    const done = [...player.completedTexts];
    if (player.playing && player.currentText) {
      const elapsed = performance.now() - player.currentStartAt;
      const ratio =
        player.currentDurationMs > 0 ? elapsed / player.currentDurationMs : 0;
      const partial = truncateByRatio(player.currentText, ratio);
      if (partial) done.push(partial);
    }
    return joinSpoken(done);
  }, []);

  const revealUi = useCallback(() => {
    setAgentText(currentSpokenText() || "—");
  }, [currentSpokenText]);

  const stopRevealLoop = useCallback(() => {
    const p = playerRef.current;
    if (p.revealRaf) {
      cancelAnimationFrame(p.revealRaf);
      p.revealRaf = 0;
    }
  }, []);

  const startRevealLoop = useCallback(() => {
    stopRevealLoop();
    const tick = () => {
      if (!playerRef.current.playing) return;
      revealUi();
      playerRef.current.revealRaf = requestAnimationFrame(tick);
    };
    playerRef.current.revealRaf = requestAnimationFrame(tick);
  }, [revealUi, stopRevealLoop]);

  const sendSpokenCommit = useCallback((spoken: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "spoken_commit", spoken_text: spoken || "" }));
  }, []);

  const playNext = useCallback(async () => {
    const player = playerRef.current;
    if (player.stopped || player.barged) {
      player.playing = false;
      return;
    }
    const item = player.queue.shift();
    if (!item) {
      player.playing = false;
      stopRevealLoop();
      revealUi();
      if (player.turnActive && !player.barged) {
        sendSpokenCommit(currentSpokenText());
        player.turnActive = false;
      }
      return;
    }
    player.playing = true;
    player.currentText = item.text;
    audioCtxRef.current =
      audioCtxRef.current || new AudioContext({ sampleRate: 48000 });
    const ctx = audioCtxRef.current;
    if (ctx.state === "suspended") await ctx.resume();
    const f32 = new Float32Array(item.int16.length);
    for (let i = 0; i < item.int16.length; i++) f32[i] = item.int16[i] / 32768;
    const buffer = ctx.createBuffer(1, f32.length, item.sampleRate);
    buffer.copyToChannel(f32, 0);
    player.currentDurationMs = item.durationMs || buffer.duration * 1000;
    player.currentStartAt = performance.now();
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    player.currentSource = src;
    startRevealLoop();
    await new Promise<void>((resolve) => {
      src.onended = () => resolve();
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
      revealUi();
      void playNext();
    } else {
      player.playing = false;
      stopRevealLoop();
    }
  }, [currentSpokenText, revealUi, sendSpokenCommit, startRevealLoop, stopRevealLoop]);

  const resetPlayer = useCallback(() => {
    const player = playerRef.current;
    stopRevealLoop();
    if (player.currentSource) {
      try {
        player.currentSource.stop();
      } catch {
        /* ok */
      }
      player.currentSource = null;
    }
    player.queue = [];
    player.playing = false;
    player.stopped = false;
    player.barged = false;
    player.turnActive = false;
    player.currentDurationMs = 0;
    player.currentText = "";
    player.completedTexts = [];
  }, [stopRevealLoop]);

  const enqueueAudio = useCallback(
    (int16: Int16Array, sampleRate: number, text: string, durationMs: number) => {
      const player = playerRef.current;
      if (player.stopped || player.barged) return;
      player.queue.push({ int16, sampleRate, text, durationMs });
      if (!player.playing) void playNext();
    },
    [playNext]
  );

  const stopPlaybackForBargeIn = useCallback(() => {
    const player = playerRef.current;
    const spoken = currentSpokenText();
    player.barged = true;
    player.stopped = true;
    player.queue = [];
    stopRevealLoop();
    if (player.currentSource) {
      try {
        player.currentSource.stop();
      } catch {
        /* ok */
      }
      player.currentSource = null;
    }
    player.playing = false;
    player.turnActive = false;
    setAgentText(spoken || "—");
    return spoken;
  }, [currentSpokenText, stopRevealLoop]);

  const applyResource = useCallback((res: Record<string, unknown> | null | undefined) => {
    if (!res) return;
    setResource(res);
    if (typeof res.actor === "string") setActor(res.actor);
  }, []);

  const applyToolUi = useCallback((ui: Record<string, unknown> | null | undefined) => {
    if (!ui || !ui.action) return;
    if (ui.action === "clear") {
      setShape(null);
      setClearToken((n) => n + 1);
      setToolHint("Panneau effacé");
      return;
    }
    if (ui.action === "show_shape") {
      if (ui.clear) {
        setShape(null);
        setClearToken((n) => n + 1);
      }
      setShape(String(ui.shape || "circle"));
      setShapeColor(String(ui.color || "blue"));
      setToolHint(`${ui.shape} · ${ui.color}`);
      return;
    }
    if (ui.action === "web_search") {
      const n =
        ui.count != null
          ? Number(ui.count)
          : Array.isArray(ui.results)
            ? ui.results.length
            : 0;
      setToolHint(`Recherche · ${n} résultat${n === 1 ? "" : "s"}`);
      return;
    }
    if (ui.action === "heavy_delegate" || ui.action === "role_delegate") {
      const ms = ui.total_ms != null ? `${ui.total_ms} ms` : "?";
      const label = ui.role_name || ui.role_key || ui.heavy_model || "?";
      setToolHint(`${label} · ${ms}`);
      const reply = String(ui.reply || "").trim();
      setSpecialistReply(reply);
      const actions = Array.isArray(ui.actions)
        ? (ui.actions as unknown[]).map((a) => String(a)).filter(Boolean)
        : [];
      setSpecialistMeta(
        actions.length
          ? `Actions : ${actions.join(" · ")}`
          : "Réponse dispo — TARS peut la lire à voix haute si tu confirmes"
      );
      if (Array.isArray(ui.stages)) {
        setStageLog(
          (ui.stages as Record<string, unknown>[]).map((s) => ({
            stage: String(s.stage || ""),
            ms: s.ms != null ? Number(s.ms) : undefined,
            ok: s.ok != null ? Boolean(s.ok) : undefined,
            detail: s.detail != null ? String(s.detail) : undefined,
            at: Date.now(),
          }))
        );
      }
      return;
    }
    if (ui.action === "read_specialist") {
      setToolHint(`Lecture complète · ${ui.role_name || "spécialiste"}`);
      if (ui.reply) setSpecialistReply(String(ui.reply));
      setSpecialistMeta("Lecture à voix haute en cours…");
    }
  }, []);

  const showMetrics = useCallback((m: Record<string, unknown>) => {
    const lines = [
      `Tour ${m.turn_id}` + (m.interrupted ? " · interrompu" : ""),
      `STT: ${fmtMs(m.stt_ms)} · LLM TTFT: ${fmtMs(m.llm_ttft_ms)}`,
      `1er audio: ${fmtMs(m.e2e_to_first_audio_ms)} · E2E: ${fmtMs(m.e2e_total_ms)}`,
    ];
    setMetricsLive(lines.join(" · "));
    setMetricsDetail(JSON.stringify(m, null, 2));
    setMetricsHistory((prev) => [m, ...prev].slice(0, 12));
  }, []);

  const stopMic = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  const startMic = useCallback(async () => {
    if (processorRef.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
      video: false,
    });
    mediaStreamRef.current = stream;
    audioCtxRef.current =
      audioCtxRef.current || new AudioContext({ sampleRate: 48000 });
    const ctx = audioCtxRef.current;
    if (ctx.state === "suspended") await ctx.resume();
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      if (!chatActiveRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN)
        return;
      const input = e.inputBuffer.getChannelData(0);
      const down = downsampleTo16k(input, ctx.sampleRate);
      const level = rms(down);
      const now = performance.now();
      const player = playerRef.current;
      const playbackActive = player.playing || player.queue.length > 0;
      const vad = vadRef.current;
      const barge = bargeRef.current;

      if (barge.enabled && playbackActive && !player.barged && !processingBusyRef.current) {
        if (level >= vad.rmsThreshold) {
          if (!bargeSpeechStartRef.current) bargeSpeechStartRef.current = now;
          else if (now - bargeSpeechStartRef.current >= barge.minSpeechMs) {
            const spoken = stopPlaybackForBargeIn();
            wsRef.current.send(
              JSON.stringify({ type: "barge_in", spoken_text: spoken || "" })
            );
            speechSeenRef.current = true;
            utteranceStartRef.current = bargeSpeechStartRef.current;
            silenceStartedAtRef.current = 0;
            bargeSpeechStartRef.current = 0;
            processingBusyRef.current = false;
            setStateLabel("listening");
            const pcm = floatTo16BitPCM(down);
            wsRef.current.send(
              JSON.stringify({ type: "audio_chunk", audio: pcm16ToBase64(pcm) })
            );
            return;
          }
        } else {
          bargeSpeechStartRef.current = 0;
        }
        return;
      }

      if (processingBusyRef.current) return;

      if (level >= vad.rmsThreshold) {
        if (!speechSeenRef.current) utteranceStartRef.current = now;
        speechSeenRef.current = true;
        silenceStartedAtRef.current = 0;
      } else if (speechSeenRef.current) {
        if (!silenceStartedAtRef.current) silenceStartedAtRef.current = now;
        else if (now - silenceStartedAtRef.current >= vad.silenceMs) {
          const clientVadMs = utteranceStartRef.current
            ? Math.round(now - utteranceStartRef.current)
            : null;
          speechSeenRef.current = false;
          silenceStartedAtRef.current = 0;
          utteranceStartRef.current = 0;
          processingBusyRef.current = true;
          wsRef.current.send(
            JSON.stringify({ type: "end_utterance", client_vad_ms: clientVadMs })
          );
          setStateLabel("stt");
          return;
        }
      }

      if (speechSeenRef.current || silenceStartedAtRef.current) {
        const pcm = floatTo16BitPCM(down);
        wsRef.current.send(
          JSON.stringify({ type: "audio_chunk", audio: pcm16ToBase64(pcm) })
        );
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);
  }, [stopPlaybackForBargeIn]);

  const stopChat = useCallback(
    (notifyServer = true) => {
      chatActiveRef.current = false;
      setChatActive(false);
      processingBusyRef.current = false;
      speechSeenRef.current = false;
      silenceStartedAtRef.current = 0;
      resetPlayer();
      stopMic();
      if (notifyServer && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "chat_stop" }));
      }
      setStateLabel("idle");
    },
    [resetPlayer, stopMic]
  );

  const startChat = useCallback(async () => {
    if (
      !wsRef.current ||
      wsRef.current.readyState !== WebSocket.OPEN ||
      !sessionReadyRef.current ||
      chatActiveRef.current
    )
      return;
    setError("");
    setUserText("—");
    setAgentText("—");
    setThinkingText("");
    speechSeenRef.current = false;
    silenceStartedAtRef.current = 0;
    processingBusyRef.current = false;
    resetPlayer();
    wsRef.current.send(
      JSON.stringify({ type: "set_thinking", enable_thinking: enableThinkingLive })
    );
    wsRef.current.send(JSON.stringify({ type: "chat_start" }));
    chatActiveRef.current = true;
    setChatActive(true);
    await startMic();
    setStateLabel("listening");
  }, [enableThinkingLive, resetPlayer, startMic]);

  // WebSocket
  useEffect(() => {
    let cancelled = false;
    let retry: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      const ws = new WebSocket(voiceWsUrl());
      wsRef.current = ws;
      ws.onopen = () => {
        setWsReady(true);
        setConn("connecté");
      };
      ws.onclose = () => {
        setWsReady(false);
        setConn("déconnecté");
        if (chatActiveRef.current) stopChat(false);
        retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => setConn("erreur WS");
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "ready") {
          if (msg.vad) {
            vadRef.current.silenceMs = msg.vad.silence_ms || 700;
            vadRef.current.rmsThreshold = msg.vad.rms_threshold || 0.015;
          }
          if (msg.barge_in) {
            bargeRef.current.enabled = msg.barge_in.enabled !== false;
            bargeRef.current.minSpeechMs = msg.barge_in.min_speech_ms || 180;
          }
        }
        if (msg.type === "ready") {
          if (msg.resource) applyResource(msg.resource);
        }
        if (msg.type === "status") {
          setStateLabel(msg.state);
          processingBusyRef.current = ["stt", "streaming"].includes(msg.state);
          if (msg.state === "listening") {
            processingBusyRef.current = false;
            speechSeenRef.current = false;
            silenceStartedAtRef.current = 0;
            utteranceStartRef.current = 0;
            bargeSpeechStartRef.current = 0;
          }
          if (msg.state === "speaking") processingBusyRef.current = false;
        }
        if (msg.type === "resource") {
          applyResource(msg.resource);
        }
        if (msg.type === "stage") {
          if (msg.resource) applyResource(msg.resource);
          else if (msg.actor) setActor(String(msg.actor));
          if (msg.kind === "stage_start") {
            setToolHint(`Étape… ${msg.stage || "?"}`);
          } else if (msg.stage) {
            setStageLog((prev) =>
              [
                {
                  stage: String(msg.stage),
                  ms: msg.ms != null ? Number(msg.ms) : undefined,
                  ok: msg.ok != null ? Boolean(msg.ok) : undefined,
                  detail: msg.detail != null ? String(msg.detail) : undefined,
                  at: Date.now(),
                },
                ...prev,
              ].slice(0, 40)
            );
          }
        }
        if (msg.type === "user_transcript") {
          setUserText(msg.text || "—");
          if (msg.metrics?.stt_ms != null) {
            livePartialRef.current.stt_ms = msg.metrics.stt_ms;
            updateLivePartial();
          }
        }
        if (msg.type === "assistant_reset") {
          resetPlayer();
          playerRef.current.turnActive = true;
          setAgentText("");
          setThinkingText("");
          setSpecialistReply("");
          setSpecialistMeta("");
          livePartialRef.current = {};
          setMetricsLive("Tour en cours…");
          setStageLog([]);
        }
        if (msg.type === "thinking_token") {
          setShowThinking(true);
          setThinkingText((t) => t + msg.text);
        }
        if (msg.type === "tool_call") {
          setToolHint(`Outil… ${msg.name || "?"}`);
          setToolLog((prev) =>
            [
              {
                name: String(msg.name || "?"),
                at: Date.now(),
                detail: msg.arguments
                  ? JSON.stringify(msg.arguments).slice(0, 120)
                  : undefined,
              },
              ...prev,
            ].slice(0, 24)
          );
        }
        if (msg.type === "tool_result") {
          applyToolUi(msg.ui);
          setToolLog((prev) => {
            if (!prev.length) return prev;
            const [head, ...rest] = prev;
            if (head.name !== msg.name) return prev;
            return [{ ...head, ok: !!msg.ok }, ...rest];
          });
          if (!msg.ui) {
            setToolHint(msg.content || (msg.ok ? "OK" : "Échec outil"));
          }
        }
        if (msg.type === "assistant_token" && msg.ttft_ms != null) {
          livePartialRef.current.ttft_ms = msg.ttft_ms;
          updateLivePartial();
        }
        if (msg.type === "assistant_audio") {
          if (msg.metrics) {
            livePartialRef.current.last_tts_ms = msg.metrics.synthesize_ms;
            if (msg.metrics.e2e_to_first_audio_ms != null) {
              livePartialRef.current.first_audio_ms = msg.metrics.e2e_to_first_audio_ms;
            }
            updateLivePartial();
          }
          const pcm = base64ToInt16(msg.audio);
          enqueueAudio(
            pcm,
            msg.sample_rate || SAMPLE_RATE,
            msg.text || "",
            msg.duration_ms || msg.metrics?.duration_ms || 0
          );
        }
        if (msg.type === "assistant_done" || msg.type === "assistant_interrupted") {
          if (msg.text != null) setAgentText(msg.text || "—");
          playerRef.current.turnActive = false;
        }
        if (msg.type === "turn_metrics") showMetrics(msg.metrics);
        if (msg.type === "error") setError(msg.message || "Erreur");
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (retry) clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [
    applyResource,
    applyToolUi,
    enqueueAudio,
    resetPlayer,
    showMetrics,
    stopChat,
    updateLivePartial,
  ]);

  // Init catalog + settings
  useEffect(() => {
    (async () => {
      try {
        const cat = await getCatalog();
        let saved: Awaited<ReturnType<typeof getSettings>> | null = null;
        try {
          saved = await getSettings();
        } catch {
          setSettingsHint("DB injoignable — defaults");
        }
        const d = { ...(cat.defaults || {}), ...(saved || {}) } as Record<string, unknown>;
        setSystemPrompt(String(d.system_prompt || ""));
        setOllamaUrl(String(d.ollama_base_url || "http://host.docker.internal:11434"));
        setRoleHistoryN(Number(d.role_history_messages ?? 8) || 8);
        setEnableThinking(!!d.enable_thinking);
        setSttOptions(cat.stt || []);
        setTtsOptions(cat.tts || []);
        setSttModel(String(d.stt_model || cat.stt?.[0]?.id || ""));
        setTtsBackend(String(d.tts_backend || "kokoro"));
        setTtsVoice(String(d.tts_voice || cat.tts?.[0]?.id || ""));

        try {
          const probe = await testOllama(
            String(d.ollama_base_url || "http://host.docker.internal:11434")
          );
          if (probe.ok) {
            setOllamaOk(true);
            setOllamaModels(probe.models || []);
            const pref = String(d.ollama_model || "");
            setOllamaModel(
              pref && (probe.models || []).includes(pref)
                ? pref
                : probe.models?.[0] || ""
            );
            setOllamaHint(`${(probe.models || []).length} modèle(s)`);
          }
        } catch {
          /* ignore */
        }

        try {
          const rs = await listRoles();
          setRoles(rs);
        } catch {
          /* ignore */
        }
        try {
          const fns = await listFunctions();
          setFunctionCatalog(fns.specialist || fns.functions || []);
        } catch {
          /* ignore */
        }

        const session = await getSession().catch(() => null);
        if (session?.ready && session.active) {
          setSessionReady(true);
          sessionReadyRef.current = true;
          setSessionLabel("session prête");
          setSessionTone("ok");
          setEnableThinkingLive(!!session.active.enable_thinking);
          setShowThinking(!!session.active.enable_thinking);
          const roleBits = Array.isArray(session.roles)
            ? session.roles.map((r: { tool_name?: string }) => r.tool_name).filter(Boolean)
            : [];
          setActiveModels(
            `STT ${session.active.stt} · LLM ${session.active.llm}` +
              (roleBits.length ? ` · ${roleBits.join(", ")}` : "") +
              ` · TTS ${session.active.tts}`
          );
          if (session.resource) applyResource(session.resource as Record<string, unknown>);
          setView("chat");
        }
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : "Init échouée");
      }
    })();
  }, [applyResource]);

  async function onTestOllama() {
    setOllamaHint("Test…");
    setOllamaOk(false);
    try {
      const data = await testOllama(ollamaUrl.trim());
      if (!data.ok) throw new Error(data.error || "Échec");
      setOllamaOk(true);
      setOllamaModels(data.models || []);
      setOllamaModel((prev) =>
        prev && (data.models || []).includes(prev) ? prev : data.models?.[0] || ""
      );
      setOllamaHint(`${(data.models || []).length} modèle(s)`);
    } catch (err) {
      setOllamaOk(false);
      setOllamaHint(err instanceof Error ? err.message : "Échec");
    }
  }

  async function onSaveSettings() {
    setSettingsHint("Enregistrement…");
    try {
      const patch: Record<string, unknown> = {
        system_prompt: systemPrompt,
        ollama_base_url: ollamaUrl.trim(),
        tts_backend: ttsBackend,
        enable_thinking: enableThinking,
      };
      if (ollamaModel) patch.ollama_model = ollamaModel;
      patch.role_history_messages = roleHistoryN;
      if (sttModel) patch.stt_model = sttModel;
      if (ttsVoice)
        patch.tts_voice =
          ttsBackend === "kokoro" ? ttsVoice : "cosyvoice-clone";
      await putSettings(patch);
      setSettingsHint("Enregistré");
    } catch (err) {
      setSettingsHint(err instanceof Error ? err.message : "Échec");
    }
  }

  async function onLoadSession() {
    setLoadError("");
    setLoading(true);
    setSessionLabel("chargement…");
    setSessionTone("muted");
    setLoadSteps({
      ollama: "testing",
      stt: "pending",
      tts: "pending",
      llm_warm: "pending",
    });

    try {
      if (ttsBackend === "cosyvoice" && cosyFileRef.current?.files?.[0] && !promptUploaded) {
        const up = await uploadCosyPrompt(cosyFileRef.current.files[0]);
        if (!up.ok) throw new Error(up.error || "Upload sample échoué");
        setPromptUploaded(true);
        setCosyHint(`Sample OK (${up.bytes} octets)`);
      }

      await putSettings({
        system_prompt: systemPrompt,
        ollama_base_url: ollamaUrl.trim(),
        ollama_model: ollamaModel,
        role_history_messages: roleHistoryN,
        stt_model: sttModel,
        tts_backend: ttsBackend,
        tts_voice: ttsBackend === "kokoro" ? ttsVoice : "cosyvoice-clone",
        enable_thinking: enableThinking,
      });
      setSettingsHint("Réglages sauvés");

      const poll = setInterval(async () => {
        try {
          const s = await getSession();
          if (s.steps) setLoadSteps(s.steps);
          if (ttsBackend === "cosyvoice" && s.steps?.tts === "loading") {
            const h = await cosyHealth();
            setCosyHint(formatCosyHealth(h));
          }
        } catch {
          /* ignore */
        }
      }, 1500);

      const payload = {
        ollama_base_url: ollamaUrl.trim(),
        ollama_model: ollamaModel,
        stt_model: sttModel,
        tts_voice: ttsBackend === "kokoro" ? ttsVoice : "cosyvoice-clone",
        tts_backend: ttsBackend,
        enable_thinking: enableThinking,
        cosy_prompt_text: cosyPromptText.trim(),
        cosy_use_default_prompt: cosyUseDefault && !promptUploaded,
      };
      const data = await loadSession(payload);
      clearInterval(poll);
      setLoadSteps(data.steps || {});
      if (!data.ok) throw new Error(data.error || "Échec du chargement");

      setSessionReady(true);
      sessionReadyRef.current = true;
      setEnableThinkingLive(enableThinking);
      setShowThinking(enableThinking);
      setSessionLabel("session prête");
      setSessionTone("ok");
      setActiveModels(
        `STT ${payload.stt_model} · LLM ${payload.ollama_model}` +
          (roles.filter((r) => r.enabled).length
            ? ` · ${roles
                .filter((r) => r.enabled)
                .map((r) => r.tool_name)
                .join(", ")}`
            : "") +
          ` · TTS ${payload.tts_backend}/${payload.tts_voice}` +
          (payload.enable_thinking ? " · thinking ON" : " · thinking OFF")
      );
      if (data.resource) applyResource(data.resource as Record<string, unknown>);
      setView("chat");
    } catch (err) {
      setSessionReady(false);
      sessionReadyRef.current = false;
      setSessionLabel("échec chargement");
      setSessionTone("bad");
      setLoadError(err instanceof Error ? err.message : "Erreur");
    } finally {
      setLoading(false);
    }
  }

  async function onUnloadSession() {
    setLoadError("");
    setUnloading(true);
    try {
      if (chatActiveRef.current) stopChat(true);
      const data = await unloadSession();
      setSessionReady(false);
      sessionReadyRef.current = false;
      setSessionLabel("VRAM libre");
      setSessionTone("muted");
      setActiveModels("");
      setLoadSteps(
        Object.fromEntries(
          (data.unloaded || []).map((u) => [u, "unloaded"])
        ) as StepMap
      );
      if (data.resource) applyResource(data.resource as Record<string, unknown>);
      else applyResource({ actor: "idle", stt_loaded: false, tts_loaded: false });
      setView("config");
      if (!data.ok && data.errors?.length) {
        setLoadError(data.errors.join(" · "));
      }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Déchargement échoué");
    } finally {
      setUnloading(false);
    }
  }

  return (
    <main className="mx-auto max-w-xl px-4 pb-10 pt-8 sm:max-w-2xl sm:px-6">
      <header className="mb-6">
        <p className="text-[clamp(2.6rem,10vw,4rem)] font-bold tracking-[0.08em] text-[var(--accent)] leading-none">
          TARS
        </p>
        <p className="mt-2 text-sm text-[var(--muted)]">
          STT → Ollama → TTS · mobile LAN
        </p>
      </header>

      <div className="mb-5 flex flex-wrap gap-2">
        <Badge tone={conn === "connecté" ? "ok" : conn === "…" ? "muted" : "bad"}>
          {conn}
        </Badge>
        <Badge tone={sessionTone}>{sessionLabel}</Badge>
        <Badge tone="muted">{stateLabel}</Badge>
      </div>

      {view === "config" ? (
        <section className="space-y-5 rounded-lg border border-[var(--line)] bg-[var(--bg1)]/70 p-4 sm:p-5">
          <h1 className="text-lg font-semibold tracking-wide">1 · Configuration</h1>

          <div className="space-y-2">
            <Label>System prompt</Label>
            <Textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={5}
            />
            <div className="flex items-center gap-3">
              <Button type="button" variant="ghost" size="sm" onClick={onSaveSettings}>
                Enregistrer
              </Button>
              <span className="text-xs text-[var(--muted)]">{settingsHint}</span>
            </div>
          </div>

          <div className="space-y-2">
            <Label>URL Ollama</Label>
            <Input
              value={ollamaUrl}
              onChange={(e) => {
                setOllamaUrl(e.target.value);
                setOllamaOk(false);
                setOllamaHint("Reteste Ollama");
              }}
            />
            <div className="flex flex-wrap items-center gap-3">
              <Button type="button" onClick={onTestOllama}>
                Tester Ollama
              </Button>
              <span className="text-xs text-[var(--muted)]">{ollamaHint}</span>
            </div>
            {ollamaModels.length > 0 && (
              <>
                <Label>Modèle LLM (chat)</Label>
                <select
                  className="h-11 w-full rounded-md border border-[var(--line)] bg-[var(--bg0)] px-3 text-sm"
                  value={ollamaModel}
                  onChange={(e) => setOllamaModel(e.target.value)}
                >
                  {ollamaModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </>
            )}
            <label className="flex items-center gap-2 text-sm text-[var(--muted)]">
              <input
                type="checkbox"
                checked={enableThinking}
                onChange={(e) => setEnableThinking(e.target.checked)}
              />
              Activer le thinking
            </label>
          </div>

          <div className="space-y-3 rounded-md border border-[var(--line)] p-3">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold tracking-wide">Spécialistes (rôles)</h2>
              <span className="text-xs text-[var(--muted)]">{roleHint}</span>
            </div>
            <p className="text-xs text-[var(--muted)]">
              Chaque rôle crée un outil ask_&lt;clé&gt; pour le chat. L&apos;historique
              récent est transmis automatiquement au spécialiste.
            </p>
            <div className="flex items-center gap-2">
              <Label className="shrink-0">Historique (msgs)</Label>
              <Input
                type="number"
                min={0}
                max={40}
                className="h-9 w-20"
                value={roleHistoryN}
                onChange={(e) => setRoleHistoryN(Number(e.target.value) || 0)}
              />
            </div>
            {roles.length > 0 && (
              <ul className="space-y-2 text-sm">
                {roles.map((r) => (
                  <li
                    key={r.id}
                    className="flex flex-wrap items-center justify-between gap-2 rounded border border-[var(--line)] bg-[var(--bg0)] px-3 py-2"
                  >
                    <div>
                      <p className="font-medium">
                        {r.name}{" "}
                        <span className="text-xs text-[var(--muted)]">{r.tool_name}</span>
                        {!r.enabled && (
                          <span className="ml-2 text-xs text-[var(--danger)]">off</span>
                        )}
                      </p>
                      <p className="text-xs text-[var(--muted)]">{r.ollama_model}</p>
                      <p className="text-[10px] text-[var(--muted)]">
                        Fonctions ·{" "}
                        {(r.function_keys || []).length
                          ? (r.function_keys || []).join(", ")
                          : "aucune"}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setRoleForm({
                            id: r.id,
                            key: r.key,
                            name: r.name,
                            description: r.description || "",
                            system_prompt: r.system_prompt || "",
                            ollama_model: r.ollama_model,
                            function_keys: [...(r.function_keys || [])],
                            enabled: r.enabled,
                          })
                        }
                      >
                        Éditer
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={async () => {
                          await deleteRole(r.id);
                          setRoles(await listRoles());
                          setRoleHint(`Supprimé ${r.key}`);
                          if (roleForm.id === r.id) {
                            setRoleForm({
                              id: null,
                              key: "",
                              name: "",
                              description: "",
                              system_prompt: "",
                              ollama_model: ollamaModels[0] || "",
                              function_keys: [],
                              enabled: true,
                            });
                          }
                        }}
                      >
                        Suppr.
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
            <div className="space-y-2 border-t border-[var(--line)] pt-3">
              <Label>{roleForm.id ? "Éditer le rôle" : "Nouveau rôle"}</Label>
              <Input
                placeholder="Nom (ex. Développeur)"
                value={roleForm.name}
                onChange={(e) => setRoleForm((f) => ({ ...f, name: e.target.value }))}
              />
              <Input
                placeholder="Clé outil (ex. developer → ask_developer)"
                value={roleForm.key}
                onChange={(e) => setRoleForm((f) => ({ ...f, key: e.target.value }))}
              />
              <Input
                placeholder="Description courte (pour le chat léger)"
                value={roleForm.description}
                onChange={(e) =>
                  setRoleForm((f) => ({ ...f, description: e.target.value }))
                }
              />
              <Textarea
                rows={4}
                placeholder="System prompt du spécialiste…"
                value={roleForm.system_prompt}
                onChange={(e) =>
                  setRoleForm((f) => ({ ...f, system_prompt: e.target.value }))
                }
              />
              <select
                className="h-11 w-full rounded-md border border-[var(--line)] bg-[var(--bg0)] px-3 text-sm"
                value={roleForm.ollama_model}
                onChange={(e) =>
                  setRoleForm((f) => ({ ...f, ollama_model: e.target.value }))
                }
              >
                <option value="">— modèle Ollama —</option>
                {ollamaModels.map((m) => (
                  <option key={`role-${m}`} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <div className="space-y-2 rounded border border-[var(--line)] p-2">
                <Label>Fonctions assignées à l&apos;expert</Label>
                <p className="text-[10px] text-[var(--muted)]">
                  Le spécialiste ne pourra appeler que ces fonctions pendant sa
                  délégation.
                </p>
                {functionCatalog.length === 0 ? (
                  <p className="text-xs text-[var(--muted)]">Catalogue vide</p>
                ) : (
                  <ul className="space-y-1.5">
                    {functionCatalog.map((fn) => {
                      const checked = roleForm.function_keys.includes(fn.key);
                      return (
                        <li key={fn.key}>
                          <label className="flex items-start gap-2 text-sm">
                            <input
                              type="checkbox"
                              className="mt-1"
                              checked={checked}
                              onChange={(e) => {
                                const on = e.target.checked;
                                setRoleForm((f) => ({
                                  ...f,
                                  function_keys: on
                                    ? [...f.function_keys, fn.key]
                                    : f.function_keys.filter((k) => k !== fn.key),
                                }));
                              }}
                            />
                            <span>
                              <span className="font-medium">{fn.name}</span>{" "}
                              <span className="text-xs text-[var(--muted)]">
                                {fn.key}
                              </span>
                              <br />
                              <span className="text-xs text-[var(--muted)]">
                                {fn.description}
                              </span>
                            </span>
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
              <label className="flex items-center gap-2 text-sm text-[var(--muted)]">
                <input
                  type="checkbox"
                  checked={roleForm.enabled}
                  onChange={(e) =>
                    setRoleForm((f) => ({ ...f, enabled: e.target.checked }))
                  }
                />
                Activé (exposé au chat)
              </label>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  onClick={async () => {
                    setRoleHint("Enregistrement…");
                    const body = {
                      key: roleForm.key || undefined,
                      name: roleForm.name,
                      description: roleForm.description,
                      system_prompt: roleForm.system_prompt,
                      ollama_model: roleForm.ollama_model,
                      function_keys: roleForm.function_keys,
                      enabled: roleForm.enabled,
                    };
                    const res = roleForm.id
                      ? await updateRole(roleForm.id, body)
                      : await createRole(body);
                    if (!res.ok) {
                      setRoleHint(res.error || "Échec");
                      return;
                    }
                    setRoles(await listRoles());
                    setRoleHint(roleForm.id ? "Rôle mis à jour" : "Rôle créé");
                    setRoleForm({
                      id: null,
                      key: "",
                      name: "",
                      description: "",
                      system_prompt: "",
                      ollama_model: ollamaModels[0] || "",
                      function_keys: [],
                      enabled: true,
                    });
                  }}
                  disabled={
                    !roleForm.name.trim() ||
                    !roleForm.system_prompt.trim() ||
                    !roleForm.ollama_model
                  }
                >
                  {roleForm.id ? "Mettre à jour" : "Créer le rôle"}
                </Button>
                {roleForm.id && (
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() =>
                      setRoleForm({
                        id: null,
                        key: "",
                        name: "",
                        description: "",
                        system_prompt: "",
                        ollama_model: ollamaModels[0] || "",
                        function_keys: [],
                        enabled: true,
                      })
                    }
                  >
                    Annuler
                  </Button>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label>STT</Label>
            <select
              className="h-11 w-full rounded-md border border-[var(--line)] bg-[var(--bg0)] px-3 text-sm"
              value={sttModel}
              onChange={(e) => setSttModel(e.target.value)}
            >
              {sttOptions.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <Label>TTS</Label>
            <select
              className="h-11 w-full rounded-md border border-[var(--line)] bg-[var(--bg0)] px-3 text-sm"
              value={ttsBackend}
              onChange={(e) => setTtsBackend(e.target.value)}
            >
              <option value="kokoro">Kokoro</option>
              <option value="cosyvoice">CosyVoice 3</option>
            </select>
            {ttsBackend === "kokoro" ? (
              <select
                className="h-11 w-full rounded-md border border-[var(--line)] bg-[var(--bg0)] px-3 text-sm"
                value={ttsVoice}
                onChange={(e) => setTtsVoice(e.target.value)}
              >
                {ttsOptions.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
            ) : (
              <div className="space-y-2 text-sm text-[var(--muted)]">
                <Input
                  ref={cosyFileRef}
                  type="file"
                  accept="audio/wav,audio/*"
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    try {
                      const up = await uploadCosyPrompt(f);
                      if (!up.ok) throw new Error(up.error || "Upload échoué");
                      setPromptUploaded(true);
                      setCosyHint(`Sample OK (${up.bytes} octets)`);
                    } catch (err) {
                      setPromptUploaded(false);
                      setCosyHint(err instanceof Error ? err.message : "Échec");
                    }
                  }}
                />
                <Textarea
                  rows={2}
                  placeholder="Transcription optionnelle…"
                  value={cosyPromptText}
                  onChange={(e) => setCosyPromptText(e.target.value)}
                />
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={cosyUseDefault}
                    onChange={(e) => setCosyUseDefault(e.target.checked)}
                  />
                  Sample démo CosyVoice
                </label>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={async () => {
                    try {
                      const h = await cosyHealth();
                      setCosyHint(formatCosyHealth(h));
                    } catch (err) {
                      setCosyHint(err instanceof Error ? err.message : "Échec");
                    }
                  }}
                >
                  Tester CosyVoice
                </Button>
                <p className="text-xs">{cosyHint}</p>
              </div>
            )}
          </div>

          <Button
            type="button"
            variant="accent"
            className="w-full"
            disabled={!canLoad}
            onClick={onLoadSession}
          >
            {loading ? "Chargement…" : "Charger en VRAM"}
          </Button>
          <Button
            type="button"
            variant="ghost"
            className="w-full"
            disabled={!sessionReady || unloading || loading}
            onClick={() => void onUnloadSession()}
          >
            {unloading ? "Déchargement…" : "Décharger la VRAM"}
          </Button>
          {loadSteps && (
            <ul className="space-y-1 text-sm text-[var(--muted)]">
              {Object.entries(loadSteps).map(([k, v]) => (
                <li key={k}>
                  {k} — {v}
                </li>
              ))}
            </ul>
          )}
          {loadError && <p className="text-sm text-[var(--danger)]">{loadError}</p>}
        </section>
      ) : (
        <section className="space-y-4 rounded-lg border border-[var(--line)] bg-[var(--bg1)]/70 p-4 sm:p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h1 className="text-lg font-semibold tracking-wide">2 · Discussion</h1>
              <p className="mt-1 text-xs text-[var(--muted)]">{activeModels}</p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <Badge
                className={
                  actor === "heavy"
                    ? "border-[var(--accent)] text-[var(--accent)]"
                    : actor === "swapping"
                      ? "border-amber-500 text-amber-400"
                      : actor === "speech"
                        ? "border-sky-500 text-sky-400"
                        : actor === "chat"
                          ? "border-emerald-500 text-emerald-400"
                          : ""
                }
              >
                Acteur · {actor}
              </Badge>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={unloading}
                  onClick={() => void onUnloadSession()}
                >
                  {unloading ? "…" : "Décharger VRAM"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    stopChat(true);
                    setView("config");
                  }}
                >
                  Config
                </Button>
              </div>
            </div>
          </div>

          <details
            open
            className="rounded-md border border-[var(--line)] p-3 text-sm"
          >
            <summary className="cursor-pointer text-[var(--muted)]">
              Ressources GPU / modèles
            </summary>
            <ul className="mt-2 space-y-1 text-xs text-[var(--muted)]">
              <li>
                Chat LLM · {String(resource?.chat_model || "—")}
                {resource?.actor === "chat" ? " · actif" : ""}
              </li>
              <li>
                Spécialiste ·{" "}
                {resource?.active_role_name || resource?.active_role
                  ? `${resource?.active_role_name || resource?.active_role} (${resource?.heavy_model || "?"})`
                  : resource?.heavy_model
                    ? String(resource.heavy_model)
                    : "—"}
                {resource?.actor === "heavy" ? " · actif" : ""}
              </li>
              <li>
                Rôles ·{" "}
                {roles.filter((r) => r.enabled).length
                  ? roles
                      .filter((r) => r.enabled)
                      .map((r) => r.tool_name)
                      .join(", ")
                  : "aucun"}
              </li>
              <li>
                STT ·{" "}
                {resource?.stt_loaded
                  ? String(resource?.stt_id || "chargé")
                  : "déchargé"}
              </li>
              <li>
                TTS ·{" "}
                {resource?.tts_loaded
                  ? `${resource?.tts_backend || "?"} / ${resource?.tts_id || "?"}`
                  : "déchargé"}
              </li>
              <li>Étape · {String(resource?.stage || "—")}</li>
            </ul>
            {stageLog.length > 0 && (
              <div className="mt-3">
                <p className="text-[10px] uppercase tracking-wide text-[var(--muted)]">
                  Timeline délégation
                </p>
                <ul className="mt-1 max-h-36 space-y-0.5 overflow-auto font-mono text-[10px]">
                  {stageLog.map((s, i) => (
                    <li key={`${s.stage}-${s.at}-${i}`}>
                      {s.ok === false ? "✗" : s.ok ? "✓" : "·"} {s.stage}
                      {s.ms != null ? ` ${Math.round(s.ms)}ms` : ""}
                      {s.detail ? ` — ${s.detail}` : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {toolLog.length > 0 && (
              <div className="mt-3">
                <p className="text-[10px] uppercase tracking-wide text-[var(--muted)]">
                  Appels d&apos;outils
                </p>
                <ul className="mt-1 max-h-28 space-y-0.5 overflow-auto font-mono text-[10px]">
                  {toolLog.map((t, i) => (
                    <li key={`${t.name}-${t.at}-${i}`}>
                      {t.ok === false ? "✗" : t.ok ? "✓" : "…"} {t.name}
                      {t.detail ? ` ${t.detail}` : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </details>

          <label className="flex items-center gap-2 text-sm text-[var(--muted)]">
            <input
              type="checkbox"
              checked={enableThinkingLive}
              onChange={(e) => {
                setEnableThinkingLive(e.target.checked);
                setShowThinking(e.target.checked);
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                  wsRef.current.send(
                    JSON.stringify({
                      type: "set_thinking",
                      enable_thinking: e.target.checked,
                    })
                  );
                }
              }}
            />
            Thinking pendant le chat
          </label>

          <Button
            type="button"
            variant={chatActive ? "danger" : "accent"}
            className="w-full"
            disabled={!(wsReady && sessionReady)}
            onClick={() => (chatActive ? stopChat(true) : void startChat())}
          >
            {chatActive ? "Arrêter le chat" : "Démarrer le chat"}
          </Button>

          <div className="space-y-3">
            <div>
              <p className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">Vous</p>
              <p className="rounded-md border border-[var(--line)] bg-[var(--bg0)] p-3 text-sm min-h-12">
                {userText}
              </p>
            </div>
            {showThinking && (
              <div>
                <p className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">
                  Thinking
                </p>
                <p className="rounded-md border border-[var(--line)] bg-[var(--bg0)] p-3 text-sm text-[var(--muted)] min-h-12 whitespace-pre-wrap">
                  {thinkingText || "—"}
                </p>
              </div>
            )}
            {specialistReply && (
              <div>
                <p className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">
                  Spécialiste (texte complet)
                </p>
                <p className="rounded-md border border-[var(--line)] bg-[var(--bg0)] p-3 text-sm min-h-12 whitespace-pre-wrap">
                  {specialistReply}
                </p>
                {specialistMeta && (
                  <p className="mt-1 text-xs text-[var(--muted)]">{specialistMeta}</p>
                )}
              </div>
            )}
            <div>
              <p className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">TARS</p>
              <p className="rounded-md border border-[var(--line)] bg-[var(--bg0)] p-3 text-sm min-h-12">
                {agentText}
              </p>
            </div>
          </div>

          <aside className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs uppercase tracking-wide text-[var(--muted)]">Panneau</p>
              <span className="text-xs text-[var(--muted)]">{toolHint}</span>
            </div>
            <ShapeCanvas shape={shape} color={shapeColor} clearToken={clearToken} />
          </aside>

          <details className="rounded-md border border-[var(--line)] p-3 text-sm">
            <summary className="cursor-pointer text-[var(--muted)]">Métriques</summary>
            <p className="mt-2 text-xs text-[var(--muted)]">{metricsLive}</p>
            <pre className="mt-2 max-h-40 overflow-auto text-[10px] text-[var(--muted)]">
              {metricsDetail}
            </pre>
            <div className="mt-2 flex flex-wrap gap-1">
              {metricsHistory.map((m, i) => (
                <button
                  key={i}
                  type="button"
                  className="rounded border border-[var(--line)] px-2 py-1 text-[10px]"
                  onClick={() => showMetrics(m)}
                >
                  {String(m.turn_id)} · {fmtMs(m.e2e_total_ms)}
                </button>
              ))}
            </div>
          </details>

          {error && <p className="text-sm text-[var(--danger)]">{error}</p>}
        </section>
      )}
    </main>
  );
}
