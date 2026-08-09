"""Nettoyage texte pour TTS — retire le markdown / bruit de code."""

from __future__ import annotations

import re


_CODE_FENCE = re.compile(r"```[^\n`]*\n?([\s\S]*?)```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BOLD_ITALIC = re.compile(r"(\*\*\*|___|\*\*|__|\*|_)(.+?)\1")
_STRIKE = re.compile(r"~~(.+?)~~")
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_HR = re.compile(r"^(-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_LIST_MARK = re.compile(r"^(\s*)([-*+]|\d+\.)\s+", re.MULTILINE)
_TABLE_SEP = re.compile(r"^\|?[\s:-]+\|[\s|:-]*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\|(.+)\|$", re.MULTILINE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def markdown_to_speech(text: str) -> str:
    """Convertit markdown / artefacts LLM en prose lisible à voix haute."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    def _fence(m: re.Match[str]) -> str:
        body = (m.group(1) or "").strip()
        if not body:
            return " "
        lines = [ln for ln in body.splitlines() if ln.strip()]
        # Court extrait littéral ; sinon on évite de lire du code
        if len(lines) == 1 and len(body) < 80:
            return f" {lines[0]} "
        return " (extrait de code omis) "

    s = _CODE_FENCE.sub(_fence, s)
    # Fences non fermés / restes
    s = re.sub(r"```[^\n`]*", " ", s)
    s = _IMAGE.sub(lambda m: f" {m.group(1)} " if m.group(1) else " ", s)
    s = _LINK.sub(lambda m: m.group(1), s)
    s = _INLINE_CODE.sub(lambda m: m.group(1), s)
    s = _HEADING.sub("", s)
    s = _BLOCKQUOTE.sub("", s)
    s = _HR.sub("", s)
    s = _STRIKE.sub(lambda m: m.group(1), s)
    # Gras / italique : garder le texte
    for _ in range(3):
        s2 = _BOLD_ITALIC.sub(lambda m: m.group(2), s)
        if s2 == s:
            break
        s = s2
    s = _LIST_MARK.sub(r"\1", s)
    s = _TABLE_SEP.sub("", s)

    def _table_row(m: re.Match[str]) -> str:
        cells = [c.strip() for c in m.group(1).split("|") if c.strip()]
        return ", ".join(cells) + ". "

    s = _TABLE_ROW.sub(_table_row, s)
    s = _HTML_TAG.sub("", s)
    s = s.replace("|", " ")
    s = s.replace("\\n", " ")
    s = s.replace("*", "")
    s = s.replace("#", "")
    s = s.replace("`", "")
    s = _MULTI_NL.sub("\n\n", s)
    s = _MULTI_SPACE.sub(" ", s)
    # Phrases plus naturelles à l'oral
    lines = [ln.strip() for ln in s.splitlines()]
    s = " ".join(ln for ln in lines if ln)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s
