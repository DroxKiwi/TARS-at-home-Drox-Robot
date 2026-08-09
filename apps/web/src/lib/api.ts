import { apiFetch } from "./voice-url";

export type AppSettings = {
  system_prompt: string;
  ollama_base_url: string;
  ollama_model: string;
  stt_model: string;
  tts_backend: string;
  tts_voice: string;
  enable_thinking: boolean;
  kokoro_speed?: number;
  silence_ms?: number;
  vad_rms_threshold?: number;
  barge_in_enabled?: boolean;
  barge_in_min_speech_ms?: number;
  role_history_messages?: number;
  updated_at?: string;
};

export type LlmRole = {
  id: number;
  key: string;
  name: string;
  description: string;
  system_prompt: string;
  ollama_model: string;
  function_keys?: string[];
  enabled: boolean;
  sort_order: number;
  tool_name: string;
  created_at?: string;
  updated_at?: string;
};

export type FunctionDefInfo = {
  key: string;
  name: string;
  description: string;
  scopes: string[];
};

export type Catalog = {
  stt: { id: string; label: string }[];
  tts: { id: string; label: string }[];
  tts_backends?: { id: string; label: string }[];
  defaults: Record<string, unknown>;
};

export async function getCatalog(): Promise<Catalog> {
  const r = await apiFetch("/api/catalog");
  if (!r.ok) throw new Error(`catalog ${r.status}`);
  return r.json();
}

export async function getSettings(): Promise<AppSettings> {
  const r = await apiFetch("/api/settings");
  if (!r.ok) throw new Error(`settings ${r.status}`);
  return r.json();
}

export async function putSettings(
  patch: Partial<AppSettings> | Record<string, unknown>
): Promise<AppSettings> {
  const r = await apiFetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function listRoles(): Promise<LlmRole[]> {
  const r = await apiFetch("/api/roles");
  if (!r.ok) throw new Error(`roles ${r.status}`);
  const data = await r.json();
  return (data.roles || []) as LlmRole[];
}

export async function listFunctions(): Promise<{
  functions: FunctionDefInfo[];
  specialist: FunctionDefInfo[];
  chat: FunctionDefInfo[];
}> {
  const r = await apiFetch("/api/functions");
  if (!r.ok) throw new Error(`functions ${r.status}`);
  return r.json();
}

export async function createRole(body: Record<string, unknown>) {
  const r = await apiFetch("/api/roles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json() as Promise<{ ok: boolean; role?: LlmRole; error?: string }>;
}

export async function updateRole(id: number, body: Record<string, unknown>) {
  const r = await apiFetch(`/api/roles/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json() as Promise<{ ok: boolean; role?: LlmRole; error?: string }>;
}

export async function deleteRole(id: number) {
  const r = await apiFetch(`/api/roles/${id}`, { method: "DELETE" });
  return r.json() as Promise<{ ok: boolean }>;
}

export async function testOllama(baseUrl: string) {
  const r = await apiFetch("/api/ollama/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_url: baseUrl }),
  });
  return r.json() as Promise<{ ok: boolean; models?: string[]; error?: string }>;
}

export async function getSession() {
  const r = await apiFetch("/api/session");
  return r.json();
}

export async function loadSession(body: Record<string, unknown>) {
  const r = await apiFetch("/api/session/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

export async function unloadSession() {
  const r = await apiFetch("/api/session/unload", { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{
    ok: boolean;
    unloaded?: string[];
    errors?: string[];
    ready?: boolean;
    error?: string;
  }>;
}

export async function cosyHealth() {
  const r = await apiFetch("/api/tts/cosy/health");
  return r.json();
}

export async function uploadCosyPrompt(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await apiFetch("/api/tts/prompt", { method: "POST", body: fd });
  return r.json() as Promise<{ ok: boolean; bytes?: number; error?: string }>;
}
