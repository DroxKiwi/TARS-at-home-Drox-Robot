import * as React from "react";
import { cn } from "@/lib/utils";

export const Badge = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement> & {
    tone?: "default" | "ok" | "bad" | "muted";
  }
>(({ className, tone = "default", ...props }, ref) => (
  <span
    ref={ref}
    className={cn(
      "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs",
      tone === "ok" && "border-[var(--ok)] text-[var(--ok)]",
      tone === "bad" && "border-[var(--danger)] text-[var(--danger)]",
      tone === "muted" && "border-[var(--line)] text-[var(--muted)]",
      tone === "default" && "border-[var(--line)] text-[var(--ink)]",
      className
    )}
    {...props}
  />
));
Badge.displayName = "Badge";
