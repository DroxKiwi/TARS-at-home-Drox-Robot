export const SAMPLE_RATE = 16000;

export function rms(float32: Float32Array): number {
  let s = 0;
  for (let i = 0; i < float32.length; i++) s += float32[i] * float32[i];
  return Math.sqrt(s / Math.max(1, float32.length));
}

export function floatTo16BitPCM(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

export function downsampleTo16k(input: Float32Array, inRate: number): Float32Array {
  if (inRate === SAMPLE_RATE) return input;
  const ratio = inRate / SAMPLE_RATE;
  const newLen = Math.floor(input.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) result[i] = input[Math.floor(i * ratio)];
  return result;
}

export function pcm16ToBase64(int16: Int16Array): string {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export function base64ToInt16(b64: string): Int16Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

export function truncateByRatio(text: string, ratio: number): string {
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

export function joinSpoken(parts: string[]): string {
  return parts.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}
