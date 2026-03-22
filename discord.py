import requests
import logging
import traceback
import numpy as np
from typing import Iterable, Optional, List
import threading
import time
import sys
import unicodedata


logger = logging.getLogger(__name__)


def send_panic_notification(
    content: str,
    webhook_urls: Optional[Iterable[str]] = None,
    mention_everyone: bool = True,
) -> None:
    """Send an @everyone panic message to Discord.

    This is meant for crash/error alerts. It will never raise.
    """

    try:
        if webhook_urls is None:
            from config import PANIC_WEBHOOK_URL
            webhook_urls = PANIC_WEBHOOK_URL

        webhook_urls = list(webhook_urls)
        if not webhook_urls:
            return

        allowed_mentions = {"parse": ["everyone"]} if mention_everyone else {"parse": []}
        payload = {"content": content, "allowed_mentions": allowed_mentions}

        for url in webhook_urls:
            try:
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
            except Exception:
                # IMPORTANT: do not call logging here (prevents recursion when this is
                # used from a logging.Handler)
                try:
                    print(traceback.format_exc(), file=sys.stderr)
                except Exception:
                    pass
    except Exception:
        try:
            print(traceback.format_exc(), file=sys.stderr)
        except Exception:
            pass


def _split_discord_content(text: str, limit: int = 1900) -> List[str]:
    """Split text into chunks suitable for Discord messages."""
    if not text:
        return [""]
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + limit])
        i += limit
    return chunks


def _truncate_discord_text(text: object, limit: int) -> str:
    """Truncate text to Discord-safe length with an ellipsis."""
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s
    if limit <= 1:
        return s[:limit]
    return s[: limit - 1] + "…"


def _add_embed_field(embed: dict, name: object, value: object, inline: bool = False) -> None:
    """Append a field while respecting Discord embed constraints."""
    fields = embed.setdefault("fields", [])
    if len(fields) >= 25:
        return

    safe_name = _truncate_discord_text(name, 256) or "-"
    safe_value = _truncate_discord_text(value, 1024) or "-"
    fields.append({"name": safe_name, "value": safe_value, "inline": inline})


def _chunk_codeblock_lines(lines: List[str], max_chars: int = 1000) -> List[str]:
    """Chunk lines into multiple code blocks that fit in embed field values."""
    if not lines:
        return ["```\n\n```"]

    chunks: List[str] = []
    current: List[str] = []

    for line in lines:
        candidate = "```\n" + "\n".join(current + [line]) + "\n```"
        if len(candidate) > max_chars and current:
            chunks.append("```\n" + "\n".join(current) + "\n```")
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("```\n" + "\n".join(current) + "\n```")

    return chunks


class DiscordWebhookLogHandler(logging.Handler):
    """Logging handler that posts logs to PANIC_WEBHOOK_URL.

    - Sends INFO/WARNING/ERROR logs to the panic webhook (batching to reduce spam).
    - Only ERROR+ logs will mention @everyone.
    """

    def __init__(
        self,
        webhook_urls: Iterable[str],
        level: int = logging.INFO,
        flush_interval_seconds: float = 5.0,
        max_buffer_records: int = 40,
    ) -> None:
        super().__init__(level=level)
        self._webhook_urls = list(webhook_urls)
        self._flush_interval = flush_interval_seconds
        self._max_buffer = max_buffer_records
        self._lock = threading.Lock()
        self._buffer: List[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="discord-log-flush", daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        if not self._webhook_urls:
            return

        try:
            msg = self.format(record)
            is_error = record.levelno >= logging.ERROR

            # Include exception traceback if present
            if record.exc_info:
                msg = msg + "\n" + "".join(traceback.format_exception(*record.exc_info))[-1800:]

            if is_error:
                # Flush buffered logs first, then send this error immediately with @everyone
                self.flush()
                self._post_lines([msg], mention_everyone=True)
                return

            with self._lock:
                self._buffer.append(msg)
                if len(self._buffer) >= self._max_buffer:
                    buffered = self._buffer
                    self._buffer = []
                else:
                    buffered = []

            if buffered:
                self._post_lines(buffered, mention_everyone=False)
        except Exception:
            # Never raise from logging
            return

    def flush(self) -> None:
        try:
            with self._lock:
                buffered = self._buffer
                self._buffer = []
            if buffered:
                self._post_lines(buffered, mention_everyone=False)
        except Exception:
            return

    def close(self) -> None:
        try:
            self._stop.set()
            if self._thread.is_alive():
                self._thread.join(timeout=2)
            self.flush()
        finally:
            super().close()

    def _worker(self) -> None:
        while not self._stop.is_set():
            time.sleep(self._flush_interval)
            self.flush()

    def _post_lines(self, lines: List[str], mention_everyone: bool) -> None:
        # Build a compact payload; keep it small and readable
        joined = "\n".join(lines)
        blocks = _split_discord_content(joined, limit=1600)

        for block in blocks:
            content = f"```\n{block}\n```"
            if mention_everyone:
                content = "@everyone\n" + content
            send_panic_notification(
                content,
                webhook_urls=self._webhook_urls,
                mention_everyone=mention_everyone,
            )


# =======================
# Helpers
# =======================

def shorten_and_center(name: str, width: int) -> str:
    if len(name) > width:
        name = name[: width - 1] + "…"
    return center_cell(name, width)


def center_cell(text: str, width: int) -> str:
    text = text or ""
    if len(text) >= width:
        return text[:width]
    left = (width - len(text)) // 2
    right = width - len(text) - left
    return " " * left + text + " " * right


def _display_width(text: str) -> int:
    """Approximate monospace display width for Discord code blocks."""
    width = 0
    for ch in text:
        code = ord(ch)

        # Zero-width joiner and variation selectors should not add width.
        if code in (0x200D, 0xFE0E, 0xFE0F):
            continue
        if unicodedata.combining(ch):
            continue

        # Emoji and full-width chars are generally rendered as width 2.
        if 0x1F300 <= code <= 0x1FAFF:
            width += 2
        elif unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def pad_cell_display(text: str, width: int) -> str:
    """Pad cell content to a target display width (not Python len)."""
    text = (text or "").strip()
    pad = width - _display_width(text)
    if pad <= 0:
        return text
    return text + (" " * pad)


def reorder_emojis_by_race(row_data, sheet_conch_order, race_conch_order):
    """
    row_data: one row from sheet
    sheet_conch_order: LIST_CONCH (sheet column order)
    race_conch_order: OCR order (current race)
    """

    # strip timestamp + winner
    sheet_emojis = row_data[1:-1]

    sheet_map = {
        conch: emoji
        for conch, emoji in zip(sheet_conch_order, sheet_emojis)
        if emoji and emoji.strip()
    }

    return [sheet_map.get(conch, "") for conch in race_conch_order]

def format_ranking_with_gap(ranking):
    """
    ranking: List[(name, score)] sorted desc
    """
    winner_score = ranking[0][1]
    lines = []

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]

    for i, (name, score) in enumerate(ranking):
        gap = score - winner_score
        medal = medals[i] if i < len(medals) else f"{i+1}."
        if i == 0:
            lines.append(f"{medal} {name}  (score {score:.2f})")
        else:
            lines.append(f"{medal} {name}  ({gap:.2f})")

    return "\n".join(lines)

def format_wsi_padded(ranking, width_name=22, width_bar=20):
    """
    ranking: List[(name, score)] TOP-6 already
    returns: formatted string for Discord code block
    """
    # compute WSI
    scores = np.array([s for _, s in ranking], dtype=np.float32)
    max_s = scores.max()
    min_s = scores.min()

    if max_s == min_s:
        wsi_values = [100] * len(ranking)
    else:
        wsi_values = ((scores - min_s) / (max_s - min_s) * 100).round().astype(int)

    lines = []
    for (name, _), wsi in zip(ranking, wsi_values):
        bar_filled = int(round(wsi / 100 * width_bar))
        bar = "█" * bar_filled + "░" * (width_bar - bar_filled)

        line = (
            f"{name.ljust(width_name)} "
            f"{bar} "
            f"{str(wsi).rjust(3)}"
        )
        lines.append(line)

    return "```\n" + "\n".join(lines) + "\n```"

# =======================
# Main Discord Function
# =======================

def send_discord_notification(
    data,
    prediction,
    ranking,
    debug=False,
    matched_rows=None,
    allow_everyone_mentions=True,
):
    """
    data: OCR data
    prediction: predicted winner name
    ranking: List[(conch_name, rank_score)]
    """

    from config import LIST_CONCH, WEBHOOK_URL

    try:
        embed = {
            "title": "🏁 Conch Race Results",
            "description": "A new race has been processed!",
            "color": 0x00FF00,
            "fields": [],
            "footer": {"text": "Conch Race OCR Bot"},
        }

        if debug:
            embed["title"] = "🐞 Debug Mode — " + embed["title"]
            embed["color"] = 0xFF0000

        # =======================
        # OCR Results
        # =======================
        for name, info in data.items():
            _add_embed_field(
                embed,
                name,
                f"Rate: {info['rate']} {info['emoji']}",
                inline=True,
            )

        # =======================
        # Prediction
        # =======================
        if prediction:
            _add_embed_field(embed, "🔮 Predicted Winner", prediction, inline=False)

        # =======================
        # Duplicate Detection
        # =======================
        num_conch = len(data)
        PERFECT_MATCH_SCORE = num_conch
        has_perfect_match = False
        has_winner_conflict = False

        # =======================
        # Ranking Probabilities
        # =======================
        if ranking:
            top_ranking = ranking[:6]
            ranking_text = format_ranking_with_gap(top_ranking)

            _add_embed_field(embed, "📊 Rank & Confidence Gap", ranking_text, inline=False)
            
            wsi_tables = format_wsi_padded(top_ranking)

            _add_embed_field(embed, "💪 Win Strength Index (WSI)", wsi_tables, inline=False)
            
        # =======================
        # Historical Match Table
        # =======================
        if matched_rows:
            MAX_COL_WIDTH = 5
            # Emoji often render as double-width in Discord code blocks.
            # Keep empty cells aligned with emoji cells.
            MAX_COL_EMOJI_WIDTH = 2
            MAX_WINNER_WIDTH = 5

            conch_names = list(data.keys())
            short_names = [
                shorten_and_center(name, MAX_COL_WIDTH)
                for name in conch_names
            ]

            header_cols = (
                [center_cell("Row", 3)]
                + short_names
                + [center_cell("Winner", MAX_WINNER_WIDTH)]
                + [center_cell("Score", 5)]
            )

            header_line = " | ".join(header_cols)
            table_lines = [header_line, "-" * len(header_line)]

            perfect_match_winners = []

            for m in matched_rows:
                if m.get("score") == PERFECT_MATCH_SCORE:
                    has_perfect_match = True
                    winner_name = m["row_data"][-1] if m["row_data"] else ""
                    if winner_name:
                        perfect_match_winners.append(winner_name)

                row_num = center_cell(str(m["row_number"]), 3)
                score = center_cell(f"{m['score']}/{num_conch}", 5)

                emojis = reorder_emojis_by_race(
                    m["row_data"],
                    LIST_CONCH,
                    list(data.keys()),
                )

                emoji_cells = [
                    pad_cell_display(e or "", MAX_COL_EMOJI_WIDTH)
                    for e in emojis
                ]

                winner = m["row_data"][-1] if m["row_data"] else ""
                winner_cell = shorten_and_center(winner, MAX_WINNER_WIDTH)

                row_line = " | ".join(
                    [row_num]
                    + emoji_cells
                    + [winner_cell]
                    + [score]
                )

                table_lines.append(row_line)

            table_chunks = _chunk_codeblock_lines(table_lines, max_chars=1000)
            for i, table_text in enumerate(table_chunks):
                field_name = "📜 Historical Match Table" if i == 0 else "📜 Historical Match Table (cont.)"
                _add_embed_field(embed, field_name, table_text, inline=False)

            if has_perfect_match:
                distinct_winners = sorted(set(perfect_match_winners))
                winners_text = ", ".join(f"**{winner}**" for winner in distinct_winners) or "(unknown)"

                _add_embed_field(
                    embed,
                    "⚠️ Duplicate Detected",
                    f"Perfect match winners: {winners_text}",
                    inline=False,
                )

                if len(distinct_winners) >= 2:
                    has_winner_conflict = True
                    _add_embed_field(
                        embed,
                        "🚨 Winner Conflict Warning",
                        (
                            "Found multiple winners in perfect-match rows: "
                            f"{winners_text}. Please verify historical data."
                        ),
                        inline=False,
                    )

            if has_winner_conflict:
                embed["color"] = 0xFF0000
            elif has_perfect_match:
                embed["color"] = 0xFFFF00

        payload = {"embeds": [embed]}
        can_mention_everyone = bool(allow_everyone_mentions and not debug)

        if has_perfect_match and can_mention_everyone:
            payload["allowed_mentions"] = {"parse": ["everyone"]}

        # =======================
        # Send to Discord
        # =======================
        for url in WEBHOOK_URL:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code >= 400:
                logging.error(
                    "Discord webhook failed (%s): %s",
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()

            if has_perfect_match and can_mention_everyone:
                conflict_text = ""
                if matched_rows:
                    perfect_winners = sorted({
                        (m["row_data"][-1] if m["row_data"] else "")
                        for m in matched_rows
                        if m.get("score") == PERFECT_MATCH_SCORE
                        and (m["row_data"][-1] if m["row_data"] else "")
                    })
                    if len(perfect_winners) >= 2:
                        conflict_text = "\n🚨 Multiple winners found: " + ", ".join(perfect_winners)

                requests.post(
                    url,
                    json={
                        "content": _truncate_discord_text(
                            "@everyone\n⚠️ Duplicate data detected!" + conflict_text,
                            1900,
                        )
                    },
                    timeout=15,
                )

            logging.info(f"Discord notification sent successfully to {url}")

    except Exception:
        logging.error(traceback.format_exc())
