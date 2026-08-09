"""Métriques de latence fines pour debug / optimisation."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("tars.metrics")


def _ms(t0: float, t1: float | None = None) -> float:
    end = time.perf_counter() if t1 is None else t1
    return round((end - t0) * 1000.0, 2)


@dataclass
class TtsSentenceMetrics:
    index: int
    chars: int
    synthesize_ms: float
    audio_samples: int
    audio_duration_ms: float
    sample_rate: int
    resample_encode_ms: float = 0.0
    ws_send_ms: float = 0.0
    # Depuis début du tour jusqu'à fin synth de cette phrase
    since_turn_start_ms: float = 0.0
    # Depuis 1er token LLM content
    since_ttft_ms: float = 0.0


@dataclass
class TurnMetrics:
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    # Audio entrée
    input_samples: int = 0
    input_duration_ms: float = 0.0
    input_chunks: int = 0
    sample_rate: int = 16000

    # Horodatages relatifs (ms depuis t0)
    t0: float = field(default_factory=time.perf_counter, repr=False)

    # STT
    stt_ms: float = 0.0
    stt_realtime_factor: float = 0.0  # input_duration / stt_ms
    transcript_chars: int = 0
    transcript_empty: bool = False

    # LLM
    llm_ttft_ms: float | None = None  # time to first content token
    llm_ttft_thinking_ms: float | None = None  # time to first thinking token
    llm_stream_ms: float = 0.0  # jusqu'à fin stream (avant TTS tail éventuel)
    llm_content_chars: int = 0
    llm_thinking_chars: int = 0
    llm_content_events: int = 0
    llm_thinking_events: int = 0
    llm_sentences: int = 0

    # TTS agrégé
    tts_sentences: list[TtsSentenceMetrics] = field(default_factory=list)
    tts_total_synthesize_ms: float = 0.0
    tts_total_audio_ms: float = 0.0
    tts_first_audio_ms: float | None = None  # depuis t0 jusqu'à 1ère phrase audio prête

    # E2E
    e2e_to_first_audio_ms: float | None = None
    e2e_total_ms: float = 0.0
    # Temps « utilisateur perçoit une réponse » ≈ e2e_to_first_audio
    # Pipeline hors réseau client playback
    pipeline_compute_ms: float = 0.0

    interrupted: bool = False
    spoken_chars: int = 0
    error: str | None = None

    # Internes
    _mark_stt0: float | None = field(default=None, repr=False)
    _mark_llm0: float | None = field(default=None, repr=False)
    _mark_llm_end: float | None = field(default=None, repr=False)

    def mark(self) -> float:
        return _ms(self.t0)

    def start_stt(self) -> None:
        self._mark_stt0 = time.perf_counter()

    def end_stt(self, transcript: str) -> None:
        assert self._mark_stt0 is not None
        self.stt_ms = _ms(self._mark_stt0)
        self.transcript_chars = len(transcript or "")
        self.transcript_empty = not bool((transcript or "").strip())
        if self.stt_ms > 0 and self.input_duration_ms > 0:
            self.stt_realtime_factor = round(self.input_duration_ms / self.stt_ms, 3)

    def start_llm(self) -> None:
        self._mark_llm0 = time.perf_counter()

    def on_thinking_delta(self, text: str) -> None:
        if self.llm_ttft_thinking_ms is None:
            self.llm_ttft_thinking_ms = self.mark()
        self.llm_thinking_chars += len(text or "")
        self.llm_thinking_events += 1

    def on_content_delta(self, text: str) -> None:
        if self.llm_ttft_ms is None:
            self.llm_ttft_ms = self.mark()
        self.llm_content_chars += len(text or "")
        self.llm_content_events += 1

    def end_llm_stream(self) -> None:
        if self._mark_llm0 is not None:
            self.llm_stream_ms = _ms(self._mark_llm0)
        self._mark_llm_end = time.perf_counter()

    def add_tts_sentence(
        self,
        *,
        index: int,
        text: str,
        synthesize_ms: float,
        audio: Any,
        sample_rate: int,
        resample_encode_ms: float = 0.0,
        ws_send_ms: float = 0.0,
    ) -> TtsSentenceMetrics:
        n = int(getattr(audio, "size", len(audio)))
        dur_ms = round((n / max(1, sample_rate)) * 1000.0, 2)
        since_ttft = (
            round(self.mark() - (self.llm_ttft_ms or 0.0), 2)
            if self.llm_ttft_ms is not None
            else 0.0
        )
        item = TtsSentenceMetrics(
            index=index,
            chars=len(text or ""),
            synthesize_ms=round(synthesize_ms, 2),
            audio_samples=n,
            audio_duration_ms=dur_ms,
            sample_rate=sample_rate,
            resample_encode_ms=round(resample_encode_ms, 2),
            ws_send_ms=round(ws_send_ms, 2),
            since_turn_start_ms=self.mark(),
            since_ttft_ms=since_ttft,
        )
        self.tts_sentences.append(item)
        self.tts_total_synthesize_ms = round(
            self.tts_total_synthesize_ms + item.synthesize_ms, 2
        )
        self.tts_total_audio_ms = round(self.tts_total_audio_ms + dur_ms, 2)
        self.llm_sentences = len(self.tts_sentences)
        if self.tts_first_audio_ms is None:
            self.tts_first_audio_ms = item.since_turn_start_ms
            self.e2e_to_first_audio_ms = item.since_turn_start_ms
        return item

    def finalize(self, error: str | None = None) -> dict[str, Any]:
        self.error = error
        self.e2e_total_ms = self.mark()
        self.pipeline_compute_ms = round(
            self.stt_ms
            + self.llm_stream_ms
            + self.tts_total_synthesize_ms
            + sum(s.resample_encode_ms + s.ws_send_ms for s in self.tts_sentences),
            2,
        )
        payload = self.to_dict()
        logger.info("TURN_METRICS %s", _compact_log(payload))
        return payload

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # drop private
        d.pop("_mark_stt0", None)
        d.pop("_mark_llm0", None)
        d.pop("_mark_llm_end", None)
        d.pop("t0", None)
        return d

    def set_input_audio(self, samples: int, sample_rate: int, chunks: int) -> None:
        self.input_samples = samples
        self.sample_rate = sample_rate
        self.input_chunks = chunks
        self.input_duration_ms = round((samples / max(1, sample_rate)) * 1000.0, 2)


def _compact_log(payload: dict[str, Any]) -> str:
    """Une ligne lisible pour docker logs."""
    tts = payload.get("tts_sentences") or []
    tts_brief = ",".join(
        f"s{s['index']}:{s['synthesize_ms']}ms/{s['audio_duration_ms']}ms"
        for s in tts
    )
    return (
        f"id={payload.get('turn_id')} "
        f"in={payload.get('input_duration_ms')}ms "
        f"stt={payload.get('stt_ms')}ms(x{payload.get('stt_realtime_factor')}) "
        f"ttft={payload.get('llm_ttft_ms')}ms "
        f"ttft_think={payload.get('llm_ttft_thinking_ms')}ms "
        f"llm_stream={payload.get('llm_stream_ms')}ms "
        f"chars={payload.get('llm_content_chars')}+think{payload.get('llm_thinking_chars')} "
        f"first_audio={payload.get('e2e_to_first_audio_ms')}ms "
        f"e2e={payload.get('e2e_total_ms')}ms "
        f"interrupted={payload.get('interrupted')} "
        f"spoken={payload.get('spoken_chars')} "
        f"tts=[{tts_brief}] "
        f"err={payload.get('error')}"
    )
