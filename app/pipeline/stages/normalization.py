from __future__ import annotations

import re
from typing import Iterable, List

from app.pipeline.stages.sections import SectionDetector

from app.pipeline.stages.text_extraction import TextBlock


_LIGATURE_REPLACEMENTS = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

_CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\u200b\u200c\u200d\u2060\ufeff]+")
_WHITESPACE_REGEX = re.compile(r"[ \t]+")


class TextNormalizer:
    _WHITESPACE_REGEX = _WHITESPACE_REGEX
    @staticmethod
    def normalize_blocks(blocks: Iterable[TextBlock]) -> list[TextBlock]:
        """
        Normalize and coalesce adjacent TextBlocks that are fragments of the
        same visual line. This runs after reading-order so blocks are in visual
        sequence. The merge is conservative and adaptive (uses
        SectionDetector.LINE_Y_TOLERANCE for vertical alignment and font-size
        based horizontal tolerance).
        """
        b_list: List[TextBlock] = list(blocks)
        if not b_list:
            return []

        merged: list[TextBlock] = []

        def _is_bullet_marker(s: str) -> bool:
            if not s:
                return False
            t = s.strip()
            if t in ("•", "\u2022", "\u2023", "\u25E6", "-", "*", "●", "\u00B7"):
                return True
            if len(t) <= 3 and not any(ch.isalnum() for ch in t):
                return True
            return False

        i = 0
        n = len(b_list)
        tol_y = getattr(SectionDetector, "LINE_Y_TOLERANCE", 1.0)

        while i < n:
            base = b_list[i]
            # start a candidate run
            run = [base]
            run_x1 = getattr(base, "x1", 0) or 0
            run_y0 = getattr(base, "y0", 0) or 0
            page = getattr(base, "page_number", 1)

            j = i + 1
            while j < n:
                nxt = b_list[j]
                # rule 1: same page
                if getattr(nxt, "page_number", 1) != page:
                    break
                # rule 2/3: same visual baseline (conservative)
                ny = getattr(nxt, "y0", 0) or 0
                if abs(ny - run_y0) > tol_y:
                    break
                # rule 4: x0 must progress left-to-right
                nx0 = getattr(nxt, "x0", 0) or 0
                if nx0 < (getattr(run[-1], "x0", 0) or 0):
                    break
                # rule 5: do not merge bullets
                if _is_bullet_marker(getattr(nxt, "text", "") or "") or _is_bullet_marker(getattr(run[-1], "text", "") or ""):
                    break
                # adaptive horizontal tolerance based on font size
                # prefer the later block's font_size, fallback to base, then conservative 0
                fsize = getattr(nxt, "font_size", None) or getattr(run[-1], "font_size", None) or getattr(base, "font_size", None) or 0
                hor_tol = (fsize * 1.5) if fsize and fsize > 0 else 0
                gap = nx0 - (run_x1 or 0)
                if gap > hor_tol:
                    break

                # all checks passed: extend run
                run.append(nxt)
                run_x1 = max(run_x1, getattr(nxt, "x1", 0) or 0)
                j += 1

            if len(run) == 1:
                merged.append(TextNormalizer._normalize_block(base))
                i += 1
            else:
                # coalesce run into single TextBlock
                parts = [TextNormalizer.normalize_text(getattr(x, "text", "") or "") for x in run]
                parts = [p for p in parts if p]
                merged_text = " ".join(parts)

                x0 = min((getattr(x, "x0", 0) or 0) for x in run)
                x1 = max((getattr(x, "x1", 0) or 0) for x in run)
                y0 = min((getattr(x, "y0", 0) or 0) for x in run)
                y1 = max((getattr(x, "y1", 0) or 0) for x in run)

                merged.append(
                    TextBlock(
                        text=merged_text,
                        page_number=page,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        font_size=getattr(run[0], "font_size", None),
                        bold=getattr(run[0], "bold", None),
                    )
                )
                i = j

        # finally ensure each block's text is normalized properly (already done above for merged blocks)
        return [TextNormalizer._normalize_block(b) if not isinstance(b, TextBlock) or b is None else b for b in merged]

    @staticmethod
    def _normalize_block(block: TextBlock) -> TextBlock:
        normalized_text = TextNormalizer.normalize_text(block.text)
        return TextBlock(
            text=normalized_text,
            page_number=block.page_number,
            x0=block.x0,
            y0=block.y0,
            x1=block.x1,
            y1=block.y1,
            font_size=block.font_size,
            bold=block.bold,
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        if text is None:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = TextNormalizer._replace_control_chars(text)
        text = TextNormalizer._replace_ligatures(text)

        lines = text.split("\n")
        normalized_lines = []
        for line in lines:
            clean = TextNormalizer._WHITESPACE_REGEX.sub(" ", line).strip()
            normalized_lines.append(clean)

        return "\n".join(normalized_lines).strip()

    @staticmethod
    def _replace_control_chars(text: str) -> str:
        return _CONTROL_CHARS_REGEX.sub(" ", text)

    @staticmethod
    def _replace_ligatures(text: str) -> str:
        for old, new in _LIGATURE_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text
