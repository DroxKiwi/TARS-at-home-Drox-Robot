"use client";

import dynamic from "next/dynamic";

const TarsApp = dynamic(
  () => import("@/components/tars-app").then((m) => m.TarsApp),
  {
    ssr: false,
    loading: () => (
      <main className="mx-auto max-w-xl px-4 pt-8">
        <p className="text-[clamp(2.6rem,10vw,4rem)] font-bold tracking-[0.08em] text-[var(--accent)]">
          TARS
        </p>
        <p className="mt-3 text-sm text-[var(--muted)]">Chargement…</p>
      </main>
    ),
  }
);

export default function HomePage() {
  return <TarsApp />;
}
