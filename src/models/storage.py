"""Salvataggio/caricamento sessioni (JSON) ed export in TXT, SRT e resoconto."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .session import Session


def save_session(session: Session, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after the complete JSON has reached disk. A failed save must
    # never truncate the user's existing transcript.
    payload = json.dumps(session.to_dict(), ensure_ascii=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_session(path: str | Path) -> Session:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Session.from_dict(data)


def _format_timestamp(seconds: float, *, sep: str = ",") -> str:
    """Converte secondi in 'HH:MM:SS,mmm' (sep ',' per SRT, '.' altrove)."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


import re


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def transcript_lines(session: Session) -> list[str]:
    """Genera le righe della trascrizione formattate con gestione delle sovrapposizioni.

    Se segmenti di voci differenti si sovrappongono temporalmente, vengono
    raggruppati in formato 'copione teatrale':
        [In contemporanea]:
          - Voce 1 (HH:MM:SS): testo...
          - Voce 2 (HH:MM:SS): testo...
    Altrimenti viene emesso il formato lineare [Nome HH:MM:SS] testo.
    """
    if session.schema_version >= 6 and session.utterances:
        session.sync_segments()
    valid_segs = sorted([s for s in session.segments if s.text and s.text.strip()], key=lambda s: (s.start, s.end, s.id))
    if not valid_segs:
        return []

    lines: list[str] = []
    i = 0
    while i < len(valid_segs):
        group = [valid_segs[i]]
        max_end = valid_segs[i].end
        j = i + 1
        while j < len(valid_segs):
            next_seg = valid_segs[j]
            if next_seg.start < max_end - 0.2 and any(next_seg.speaker != s.speaker for s in group):
                group.append(next_seg)
                max_end = max(max_end, next_seg.end)
                j += 1
            else:
                break

        if len(group) > 1:
            overlap_block = "[In contemporanea]:\n" + "\n".join(
                f"  - {session.display_name(s.speaker)} ({_format_hms(s.start)}): {s.text.strip()}"
                for s in group
            )
            lines.append(overlap_block)
            i = j
        else:
            seg = group[0]
            lines.append(f"[{session.display_name(seg.speaker)} {_format_hms(seg.start)}] {seg.text.strip()}")
            i += 1

    return lines


def export_txt(session: Session) -> str:
    """Trascrizione leggibile: ogni segmento con nome voce e timestamp d'inizio."""
    if session.schema_version >= 6 and session.utterances:
        session.sync_segments()
    lines: list[str] = []
    for seg in session.segments:
        name = session.display_name(seg.speaker)
        stamp = _format_timestamp(seg.start, sep=".")[:-4]  # HH:MM:SS
        lines.append(f"[{name} {stamp}] {seg.text.strip()}")
    return "\n".join(lines) + ("\n" if lines else "")


def export_txt_by_speaker(session: Session, speaker_id: str) -> str:
    """Esporta la trascrizione completa per un singolo parlante specificato."""
    name = session.display_name(speaker_id)
    items = [
        u for u in (session.utterances or [])
        if u.speaker == speaker_id and not getattr(u, "suppressed_bleed", False)
    ]
    if not items and session.segments:
        items = [s for s in session.segments if s.speaker == speaker_id]

    sorted_items = sorted(
        items,
        key=lambda x: (
            getattr(x, "start", getattr(x, "start_s", 0.0)),
            getattr(x, "end", getattr(x, "end_s", 0.0)),
        ),
    )

    prof = session.speaker_transcription_profile(speaker_id)
    lines: list[str] = [
        f"# PROFILO TRASCRIZIONE VOCE: {name} ({speaker_id})",
        f"# Interventi: {prof['turn_count']} | Parole totali: {prof['word_count']} | Durata parlato: {prof['total_speaking_seconds']}s",
        "",
    ]
    for it in sorted_items:
        start = float(getattr(it, "start", getattr(it, "start_s", 0.0)))
        end = float(getattr(it, "end", getattr(it, "end_s", 0.0)))
        text = str(getattr(it, "text", "")).strip()
        if not text:
            continue
        stamp_start = _format_timestamp(start, sep=".")[:-4]
        stamp_end = _format_timestamp(end, sep=".")[:-4]
        lines.append(f"[{stamp_start} - {stamp_end}] {text}")

    return "\n".join(lines) + ("\n" if lines else "")


def export_all_speaker_transcripts(session: Session, output_dir: str | Path) -> list[Path]:
    """Genera file TXT individuali per ciascun parlante nella directory di output."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    base_name = session.name.replace(" ", "_") if session.name else session.id

    for spk_id in session.speaker_ids():
        name = session.display_name(spk_id).replace(" ", "_")
        safe_name = re.sub(r'[\\/*?:"<>|]', "", f"{base_name}_{name}.txt")
        file_path = out_p / safe_name
        content = export_txt_by_speaker(session, spk_id)
        file_path.write_text(content, encoding="utf-8")
        saved_files.append(file_path)

    return saved_files


# Limiti di sottotitolazione: due righe da ~42 caratteri e pochi secondi per
# battuta sono la pratica corrente. Servono qui perche' il raggruppamento per
# voce unisce segmenti contigui, e un turno lungo diventerebbe una sola battuta
# di minuti, illeggibile e fuori standard.
SRT_MAX_SECONDS = 6.0
SRT_MAX_CHARS = 84


def _split_text(text: str, max_chars: int) -> list[str]:
    """Spezza su spazi, senza mai tagliare una parola a meta'."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _segment_cues(segment, max_seconds: float, max_chars: int) -> list[tuple[float, float, str]]:
    """Divide un segmento nelle battute che rispettano durata e lunghezza.

    Con le parole allineate i tempi delle battute sono quelli reali; senza (o
    dopo una modifica manuale, che invalida l'allineamento) si ripartisce la
    durata in proporzione al testo, che e' una stima dichiarata, non un dato.
    """
    text = segment.text.strip()
    if not text:
        return []
    duration = max(0.0, segment.end - segment.start)
    if duration <= max_seconds and len(text) <= max_chars:
        return [(segment.start, segment.end, text)]

    if segment.words_valid and segment.words:
        cues: list[tuple[float, float, str]] = []
        pending: list[tuple[str, float]] = []  # (testo, fine) delle parole in corso
        start = None
        for word in segment.words:
            piece = word.text.strip()
            if not piece:
                continue
            if start is None:
                start = word.start
            candidate = " ".join([*(item[0] for item in pending), piece])
            if pending and (len(candidate) > max_chars or (word.end - start) > max_seconds):
                cues.append((start, pending[-1][1],
                             " ".join(item[0] for item in pending)))
                pending, start = [], word.start
            pending.append((piece, word.end))
        if pending and start is not None:
            cues.append((start, pending[-1][1], " ".join(item[0] for item in pending)))
        if cues:
            return cues

    # Senza allineamento affidabile: righe per lunghezza, tempi proporzionali.
    pieces = _split_text(text, max_chars)
    total = sum(len(piece) for piece in pieces) or 1
    cues = []
    cursor = segment.start
    for piece in pieces:
        span = duration * (len(piece) / total)
        cues.append((cursor, min(segment.end, cursor + span), piece))
        cursor += span
    return cues


def export_srt(session: Session, max_seconds: float = SRT_MAX_SECONDS,
               max_chars: int = SRT_MAX_CHARS) -> str:
    """Sottotitoli SRT con il nome della voce, entro i limiti di durata e lunghezza.

    Preserva le battute simultanee ordinandole cronologicamente per tempo d'inizio,
    senza scartare alcuno speaker sovrapposto.
    """
    all_cues: list[tuple[float, float, str]] = []
    # Raccoglie i blocchi da tutti i segmenti (o utterances)
    if session.schema_version >= 6 and session.utterances:
        session.sync_segments()
    items = session.segments
    for seg in items:
        name = session.display_name(seg.speaker)
        cues = _segment_cues(seg, max_seconds, max_chars)
        for position, (start, end, text) in enumerate(cues):
            # Il nome della voce sulla prima battuta del turno
            prefix = f"[{name}] " if position == 0 else ""
            body = "\n".join(_split_text(prefix + text, max_chars))
            all_cues.append((start, max(end, start), body))

    # Ordina cronologicamente i sottotitoli per evitare salti temporali tra voci sovrapposte
    all_cues.sort(key=lambda c: (c[0], c[1]))

    blocks: list[str] = []
    for index, (start, end, body) in enumerate(all_cues, 1):
        blocks.append(
            f"{index}\n{_format_timestamp(start)} --> {_format_timestamp(end)}\n{body}\n"
        )
    return "\n".join(blocks)

