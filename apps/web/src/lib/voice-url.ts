/** Même origin que la page (HTTPS) — le serveur Next proxy vers tars-voice. */

export function voiceHttpBase(): string {
  return "";
}

export function voiceWsUrl(): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}

export async function apiFetch(path: string, init?: RequestInit) {
  const res = await fetch(`${voiceHttpBase()}${path}`, init);
  return res;
}
