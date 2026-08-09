import * as React from "react";
import { cn } from "@/lib/utils";

export const Label = React.forwardRef<
  HTMLLabelElement,
  React.LabelHTMLAttributes<HTMLLabelElement>
>(({ className, ...props }, ref) => (
  <label
    ref={ref}
    className={cn(
      "text-xs tracking-wide text-[var(--muted)] uppercase",
      className
    )}
    {...props}
  />
));
Label.displayName = "Label";
