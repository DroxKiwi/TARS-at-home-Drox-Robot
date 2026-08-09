"use client";

import { useEffect, useRef } from "react";

type Props = {
  shape: string | null;
  color: string;
  clearToken: number;
};

export function ShapeCanvas({ shape, color, clearToken }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!shape) return;
    const w = c.width;
    const h = c.height;
    const cx = w / 2;
    const cy = h / 2;
    const size = Math.min(w, h) * 0.32;
    ctx.fillStyle = color || "blue";
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
  }, [shape, color, clearToken]);

  return (
    <canvas
      ref={ref}
      width={320}
      height={180}
      className="w-full max-w-sm rounded-md border border-[var(--line)] bg-[var(--bg0)]"
    />
  );
}
