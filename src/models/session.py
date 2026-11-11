"""Modello dati di una sessione di trascrizione.

Una Session raccoglie i segmenti trascritti (con timestamp e voce), la mappa
voce->nome assegnata dall'utente e i riferimenti ai file audio registrati.
Tutto e' basato su dataclass + tipi standard, cosi' il modello e' serializzabile
in JSON senza dipendenze esterne ed e' usabile sia dalla pipeline che dalla GUI.
"""

from __future__ import annotations

import uuid
import math
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Word:
    """Singola parola con i suoi timestamp, per l'evidenziazione sincronizzata."""

    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    """Un segmento di parlato continuo attribuito a una voce."""

    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    words: list[Word] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    words_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AudioSources:
    """Percorsi dei file audio associati alla sessione (relativi alla cartella dati).

    Salvarli relativi (e non assoluti come "C:\\Users\\...") e' cio' che permette
    di aprire la stessa sessione su un altro PC, o dopo aver spostato la cartella
    dati, e ritrovare comunque il suo audio.
    """

    mic: str | None = None
    system: str | None = None
    mixed: str | None = None

    def primary(self) -> str | None:
        """File principale da usare per playback/elaborazione (come salvato)."""
        return self.mixed or self.mic or self.system

    def resolved_primary(self):
        """Percorso reale del file principale su questo PC, o None se introvabile."""
        from .. import paths

        for candidate in (self.mixed, self.mic, self.system):
            found = paths.resolve_media(candidate)
            if found is not None:
                return found
        return None


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transcript_artifacts import Utterance


@dataclass
class Session:
    """Una sessione completa: audio, segmenti e mappa delle voci."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "Sessione senza nome"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    language: str | None = None
    model: str | None = None
    # Profilo scelto prima dell'elaborazione: serve a riprodurre la pipeline
    # corretta quando si riapre la sessione.
    audio_profile: str | None = None
    audio: AudioSources = field(default_factory=AudioSources)
    segments: list[Segment] = field(default_factory=list)
    # Copia immutata dell'output prima del controllo di qualità. Viene usata
    # per confrontare le attribuzioni e non viene mostrata negli export salvo
    # richiesta esplicita dell'utente.
    raw_segments: list[Segment] | None = None
    # Metadati del controllo automatico dell'attribuzione, separati dal
    # resoconto LLM in ``analysis``.
    transcript_review: dict | None = None
    # Mappa id-voce -> nome leggibile assegnato dall'utente, es. "SPEAKER_00" -> "Marco".
    speaker_names: dict[str, str] = field(default_factory=dict)
    # Esito dell'analisi con l'LLM locale (riassunto, decisioni, azioni), come
    # dizionario serializzabile: vedi processing.analysis.Analysis. Resta None
    # finche' l'utente non chiede l'analisi.
    analysis: dict | None = None
    schema_version: int = 6
    revision: int = 0
    provenance: dict = field(default_factory=dict)
    diarization: dict = field(default_factory=dict)
    chat_history: list[dict] = field(default_factory=list)
    edit_history: list[dict] = field(default_factory=list)
    redo_history: list[dict] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    raw_utterances: list[Utterance] | None = None
    audio_assets: list[dict] = field(default_factory=list)
    revisions: list[dict] = field(default_factory=list)
    pipeline_strategy: str = "standard"
    capture_manifest: dict | None = None
    active_view: str = "timeline"
    timeline_visible_speakers: list[str] = field(default_factory=list)
    media_duration_s: float = 0.0
    recovery_candidates: list[dict] = field(default_factory=list)
    speaker_evidence: list[dict] = field(default_factory=list)

    def fingerprint(self) -> str:
        """Also detects edits made by older clients that do not bump revision."""
        value = [
            self.id,
            self.revision,
            [asdict(s) for s in self.segments],
            [asdict(u) for u in self.utterances],
            self.speaker_names,
            self.pipeline_strategy,
            self.active_view,
        ]
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False).encode("utf-8")).hexdigest()

    def _edit_state(self, reason: str) -> dict:
        canonical = self.schema_version >= 6 and bool(self.utterances)
        state = {
            "reason": reason,
            "segments": [] if canonical else [asdict(s) for s in self.segments],
            "utterances": [asdict(u) for u in self.utterances],
            "raw_utterances": None if canonical else [asdict(u) for u in self.raw_utterances] if self.raw_utterances is not None else None,
            "raw_segments": None if canonical else [asdict(s) for s in self.raw_segments] if self.raw_segments is not None else None,
            "speaker_names": dict(self.speaker_names),
            "transcript_review": deepcopy(self.transcript_review),
            "pipeline_strategy": self.pipeline_strategy,
            "active_view": self.active_view,
            "timeline_visible_speakers": list(self.timeline_visible_speakers),
            "schema_version": self.schema_version,
            "recovery_candidates": deepcopy(self.recovery_candidates),
            "speaker_evidence": deepcopy(self.speaker_evidence),
            "diarization": deepcopy(self.diarization),
        }
        if canonical:
            # Immutable raw outputs stay once on Session. Compress snapshots so
            # a two-hour review does not multiply transcript JSON twentyfold.
            import base64
            import zlib
            encoded = json.dumps(state, ensure_ascii=False, allow_nan=False).encode("utf-8")
            return {"reason": reason, "codec": "zlib-json-v1", "state": base64.b64encode(zlib.compress(encoded, 1)).decode("ascii")}
        return state

    def begin_edit(self, reason: str) -> None:
        if self.raw_segments is None:
            self.raw_segments = deepcopy(self.segments)
        if self.raw_utterances is None:
            self.raw_utterances = deepcopy(self.utterances)
        self.edit_history.append(self._edit_state(reason))
        self.edit_history = self.edit_history[-20:]
        self.redo_history.clear()
        self.revision += 1

    def _restore_edit(self, source: list[dict], destination: list[dict]) -> bool:
        if not source:
            return False
        state = source[-1]
        if state.get("codec") == "zlib-json-v1":
            import base64
            import zlib
            state = json.loads(zlib.decompress(base64.b64decode(state["state"])))
        restored = Session.from_dict(state)
        destination.append(self._edit_state(state.get("reason", "Modifica")))
        source.pop()
        self.segments = restored.segments
        self.utterances = restored.utterances
        if restored.raw_utterances is not None:
            self.raw_utterances = restored.raw_utterances
        if restored.raw_segments is not None:
            self.raw_segments = restored.raw_segments
        self.speaker_names = restored.speaker_names
        self.transcript_review = restored.transcript_review
        self.pipeline_strategy = getattr(restored, "pipeline_strategy", "standard")
        self.active_view = getattr(restored, "active_view", "conversation")
        self.timeline_visible_speakers = list(getattr(restored, "timeline_visible_speakers", []))
        self.recovery_candidates = restored.recovery_candidates
        self.speaker_evidence = restored.speaker_evidence
        if "diarization" in state:
            self.diarization = restored.diarization
        if self.schema_version >= 6:
            self.sync_segments()
        self.revision += 1
        return True

    def undo(self) -> bool:
        return self._restore_edit(self.edit_history, self.redo_history)

    def redo(self) -> bool:
        return self._restore_edit(self.redo_history, self.edit_history)

    def edit_segment(self, index: int, *, text: str | None = None,
                     speaker: str | None = None) -> bool:
        if self.schema_version >= 6:
            from .timeline import TimelineEditor, UNSET
            event_id = self.segments[index].id
            return TimelineEditor(self).edit_event(event_id, text=text, speaker_id=speaker if speaker is not None else UNSET)
        segment = self.segments[index]
        if ((text is None or text == segment.text) and
                (speaker is None or speaker == segment.speaker)):
            return False
        self.begin_edit("Correzione segmento")
        if text is not None and text != segment.text:
            segment.text = text
            segment.words_valid = False
            if "timing_da_riallineare" not in segment.warnings:
                segment.warnings.append("timing_da_riallineare")
        if speaker is not None:
            segment.speaker = speaker
        return True

    def split_segment(self, index: int, seconds: float) -> bool:
        if self.schema_version >= 6:
            from .timeline import TimelineEditor
            return TimelineEditor(self).split_event(self.segments[index].id, seconds)
        segment = self.segments[index]
        if not segment.words_valid or not segment.words:
            raise ValueError("Serve l'allineamento delle parole prima di dividere il segmento.")
        left = [w for w in segment.words if (w.start + w.end) / 2 < seconds]
        right = segment.words[len(left):]
        if not left or not right:
            return False
        self.begin_edit("Divisione segmento")
        parts = []
        for words in (left, right):
            part = deepcopy(segment)
            part.id = uuid.uuid4().hex
            part.source_ids = [segment.id]
            part.start, part.end = words[0].start, words[-1].end
            part.words = deepcopy(words)
            part.text = " ".join(w.text.strip() for w in words)
            parts.append(part)
        self.segments[index:index + 1] = parts
        return True

    # --- voci ---------------------------------------------------------------

    def speaker_ids(self) -> list[str]:
        """Id voce distinti presenti nei segmenti e negli eventi timeline, in ordine di prima comparsa."""
        ordered: list[str] = []
        for seg in self.segments:
            if seg.speaker and seg.speaker not in ordered:
                ordered.append(seg.speaker)
        for utt in self.utterances:
            spk = utt.speaker_id
            if spk and spk not in ordered:
                ordered.append(spk)
        for spk in self.speaker_names:
            if spk not in ordered:
                ordered.append(spk)
        return ordered

    def display_name(self, speaker_id: str | None) -> str:
        """Nome assegnato alla voce, oppure l'id grezzo se non rinominata."""
        if not speaker_id or str(speaker_id).upper() in ("UNKNOWN", "UNASSIGNED", "NONE"):
            return "Non assegnato"
        return self.speaker_names.get(speaker_id, speaker_id)

    def speaker_transcription_profile(self, speaker_id: str) -> dict:
        """Estrae il profilo completo di trascrizione per una specifica voce."""
        name = self.display_name(speaker_id)
        # Raccoglie interventi da utterances se presenti, altrimenti da segments
        items = [u for u in self.utterances if u.speaker == speaker_id and not getattr(u, "suppressed_bleed", False)]
        if not items and self.segments:
            items = [s for s in self.segments if s.speaker == speaker_id]

        sorted_items = sorted(
            items,
            key=lambda x: (
                getattr(x, "start", getattr(x, "start_s", 0.0)),
                getattr(x, "end", getattr(x, "end_s", 0.0)),
            ),
        )

        texts = [getattr(x, "text", "").strip() for x in sorted_items if getattr(x, "text", "").strip()]
        full_transcript = " ".join(texts)
        words = sum(len(t.split()) for t in texts)

        intervals = [
            [
                round(float(getattr(x, "start", getattr(x, "start_s", 0.0))), 3),
                round(float(getattr(x, "end", getattr(x, "end_s", 0.0))), 3),
            ]
            for x in sorted_items
        ]
        total_speaking_time = sum(max(0.0, b - a) for a, b in intervals)
        dur = self.duration
        speaking_ratio = (total_speaking_time / dur) if dur > 0 else 0.0

        vp_meta = self.diarization.get("voice_profiles", {}).get(speaker_id) if isinstance(self.diarization, dict) else None
        is_known = bool(vp_meta.get("is_known_profile", False)) if isinstance(vp_meta, dict) else False

        res = {
            "speaker_id": speaker_id,
            "display_name": name,
            "name": name,
            "turn_count": len(sorted_items),
            "word_count": words,
            "total_speaking_seconds": round(total_speaking_time, 3),
            "speaking_ratio": round(speaking_ratio, 4),
            "transcript": full_transcript,
            "turn_intervals": intervals,
            "is_known_profile": is_known,
        }
        return res

    def all_speaker_transcription_profiles(self) -> dict[str, dict]:
        """Estrae i profili di trascrizione per ciascuna voce identificata nella sessione."""
        return {spk: self.speaker_transcription_profile(spk) for spk in self.speaker_ids()}


    def adjacent_same_speaker_groups(self) -> int:
        """Numero di accorpamenti possibili tra segmenti consecutivi.

        Il conteggio considera ogni confine tra due segmenti con la stessa voce.
        Per esempio ``A, A, A`` restituisce 2: sono necessari due accorpamenti
        per ottenere un solo segmento.  La lista grezza non viene mai toccata.
        """
        if self.schema_version >= 6 and self.utterances:
            ordered = sorted(self.utterances, key=lambda u: (u.start_s, u.end_s, u.event_id))
            return sum(a.speaker_id == b.speaker_id and a.source_audio_ref == b.source_audio_ref
                       and 0 <= b.start_s - a.end_s <= 2 for a, b in zip(ordered, ordered[1:]))
        return sum(
            1
            for left, right in zip(self.segments, self.segments[1:])
            if left.speaker == right.speaker
        )

    def group_adjacent_speaker_segments(self) -> int:
        """Accorpa i segmenti consecutivi attribuiti alla stessa voce.

        L'operazione lavora esclusivamente su ``segments`` e crea copie dei
        segmenti/parole, così ``raw_segments`` resta una fotografia immutata
        dell'output originale. Restituisce quanti segmenti sono stati rimossi.
        """
        if self.schema_version >= 6 and self.utterances:
            from .timeline import TimelineEditor
            temporary = deepcopy(self)
            editor = TimelineEditor(temporary)
            removed = 0
            while True:
                ordered = sorted(temporary.utterances, key=lambda u: (u.start_s, u.end_s, u.event_id))
                pair = next(((a, b) for a, b in zip(ordered, ordered[1:])
                             if a.speaker_id == b.speaker_id and a.source_audio_ref == b.source_audio_ref
                             and 0 <= b.start_s - a.end_s <= 2), None)
                if pair is None:
                    break
                editor.merge_events(pair[0].event_id, pair[1].event_id)
                removed += 1
            if removed:
                self.begin_edit("Raggruppamento eventi compatibili")
                self.utterances = temporary.utterances
                self.sync_segments()
            return removed
        if len(self.segments) < 2:
            return 0

        grouped: list[Segment] = []
        merged_count = 0
        for source in self.segments:
            current = deepcopy(source)
            if grouped and grouped[-1].speaker == current.speaker:
                previous = grouped[-1]
                previous.end = max(previous.end, current.end)
                left = previous.text.strip()
                right = current.text.strip()
                if left and right:
                    # Se il segmento precedente era troncato con '…' e il successivo continua in minuscolo,
                    # ricuce la frase rimuovendo l'ellissi intermedia.
                    if left.endswith("…") and right and right[0].islower():
                        left = left[:-1].rstrip()
                    # Evita lo spazio prima della punteggiatura che spesso
                    # apre un segmento Whisper (",", ".", ";", ecc.).
                    separator = "" if right[0] in ",.;:!?%)]}" else " "
                    previous.text = left + separator + right
                else:
                    previous.text = left or right
                previous.words.extend(current.words)
                # Un segmento corretto a mano ha l'allineamento non piu' valido:
                # unirlo a uno intatto non lo rende valido, quindi lo stato
                # peggiore vince e gli avvisi si conservano entrambi.
                previous.words_valid = previous.words_valid and current.words_valid
                for warning in current.warnings:
                    if warning not in previous.warnings:
                        previous.warnings.append(warning)
                for cand in current.candidates:
                    if cand not in previous.candidates:
                        previous.candidates.append(cand)
                merged_count += 1
            else:
                grouped.append(current)

        if merged_count:
            self.begin_edit("Raggruppamento voci consecutive")
            self.segments = grouped
            review = dict(self.transcript_review or {})
            review["grouped_same_speaker"] = True
            try:
                previous_count = int(review.get("grouped_count", 0) or 0)
            except (TypeError, ValueError):
                previous_count = 0
            review["grouped_count"] = previous_count + merged_count
            self.transcript_review = review
        return merged_count

    def rename_speaker(self, speaker_id: str, name: str) -> None:
        if self.speaker_names.get(speaker_id, "") == name.strip():
            return
        self.begin_edit("Nome voce")
        if name.strip():
            self.speaker_names[speaker_id] = name.strip()
        else:
            self.speaker_names.pop(speaker_id, None)

    def migrate_to_v3(self) -> None:
        """Migra esplicitamente la sessione allo schema v3 garantendo la presenza di utterances."""
        from .transcript_artifacts import Utterance
        if not self.utterances and self.segments:
            self.utterances = [
                Utterance(
                    start_s=s.start,
                    end_s=s.end,
                    utterance_id=s.id,
                    stream_id=0,
                    speaker=s.speaker,
                    candidate_speakers=deepcopy(s.candidates),
                    text=s.text,
                    words=deepcopy(s.words),
                    hypothesis_ids=list(s.source_ids),
                )
                for s in self.segments
            ]
        self.schema_version = 3

    def migrate_to_v4(self) -> None:
        """Migra esplicitamente la sessione allo schema v4."""
        self.migrate_to_v3()
        if not self.timeline_visible_speakers:
            self.timeline_visible_speakers = self.speaker_ids()[:6]
        if not self.pipeline_strategy:
            self.pipeline_strategy = "standard"
        if not self.active_view:
            self.active_view = "conversation"
        self.schema_version = 4

    upgrade_to_v4 = migrate_to_v4

    def migrate_to_v5(self) -> None:
        """Migra la sessione allo schema v5 con modello multitraccia overlap-aware."""
        from ..processing.merge import materialize_overlap_utterances, utterances_to_segments

        if self.schema_version < 4 or not self.utterances:
            self.migrate_to_v4()
        primary_ref = self.audio.primary()
        seg_by_id = {s.id: s for s in self.segments}
        for utt in self.utterances:
            if not utt.source_audio_ref and primary_ref:
                utt.source_audio_ref = primary_ref
            if not utt.candidate_speakers and utt.utterance_id in seg_by_id:
                utt.candidate_speakers = deepcopy(seg_by_id[utt.utterance_id].candidates)
            if utt.original_prediction is None:
                utt.original_prediction = {
                    "start_s": round(utt.start_s, 3),
                    "end_s": round(utt.end_s, 3),
                    "speaker": utt.speaker,
                    "text": utt.text,
                    "stream_id": utt.stream_id,
                    "overlap_group_id": utt.overlap_group_id,
                }

        diar = self.diarization if isinstance(self.diarization, dict) else {}
        raw_overlaps = diar.get("overlaps") or []
        overlaps = [
            (float(pair[0]), float(pair[1]))
            for pair in raw_overlaps
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
        ordinary_turns = diar.get("turns") or []
        exclusive_turns = diar.get("exclusive") or []

        self.utterances = materialize_overlap_utterances(
            self.utterances,
            overlaps=overlaps,
            ordinary_turns=ordinary_turns,
            exclusive_turns=exclusive_turns,
            source_audio_ref=primary_ref,
        )
        self.segments = utterances_to_segments(self.utterances)
        if not self.raw_utterances and self.utterances:
            self.raw_utterances = deepcopy(self.utterances)
        self.schema_version = 5

    upgrade_to_v5 = migrate_to_v5

    def sync_segments(self) -> None:
        """Refresh the compatibility projection; speech events remain authoritative."""
        from ..processing.merge import utterances_to_segments
        self.utterances.sort(key=lambda u: (u.start_s, u.end_s, u.event_id))
        self.segments = utterances_to_segments(self.utterances)

    def migrate_to_v6(self) -> None:
        """Expand legacy data without rerunning models or replacing corrections.

        Raw snapshots preserve the pre-migration state. Existing v3/v4 segment
        edits win over stale duplicate utterances with the same event ID.
        """
        from .transcript_artifacts import Utterance, recompute_overlap_relationships
        from ..processing.merge import materialize_overlap_utterances
        if self.schema_version >= 6 and (self.utterances or not self.segments):
            recompute_overlap_relationships(self.utterances)
            self.sync_segments()
            return
        if self.raw_segments is None:
            self.raw_segments = deepcopy(self.segments)
        if self.raw_utterances is None and self.utterances:
            self.raw_utterances = deepcopy(self.utterances)
        if not self.utterances:
            self.migrate_to_v3()
            if not self.raw_utterances:
                self.raw_utterances = deepcopy(self.utterances)
        elif self.schema_version < 5:
            by_id = {u.event_id: u for u in self.utterances}
            for seg in self.segments:
                event = by_id.get(seg.id)
                if event is None:
                    self.utterances.append(Utterance(start_s=seg.start, end_s=seg.end, utterance_id=seg.id,
                                                     text=seg.text, speaker=seg.speaker, words=deepcopy(seg.words)))
                else:
                    text_changed = event.text != seg.text
                    for key, value in (("text", seg.text), ("speaker", seg.speaker), ("start_s", seg.start), ("end_s", seg.end)):
                        if getattr(event, key) != value:
                            event.record_user_correction(key, getattr(event, key), value, reason="legacy_segment_edit")
                            setattr(event, key, value)
                    event.words = [] if text_changed or not seg.words_valid else deepcopy(seg.words)
        for event in self.utterances:
            event.source_audio_ref = event.source_audio_ref or self.audio.primary()
        self.utterances = materialize_overlap_utterances(self.utterances, ordinary_turns=self.diarization.get("turns", []),
                            exclusive_turns=self.diarization.get("exclusive", []), overlaps=self.diarization.get("overlaps", []),
                            source_audio_ref=self.audio.primary())
        recompute_overlap_relationships(self.utterances)
        self.schema_version = 6
        self.active_view = "timeline"
        self.sync_segments()

    upgrade_to_v6 = migrate_to_v6

    @property
    def duration(self) -> float:
        seg_max = max((seg.end for seg in self.segments), default=0.0)
        utt_max = max((utt.end_s for utt in self.utterances), default=0.0)
        return max(seg_max, utt_max, self.media_duration_s)

    # --- serializzazione ----------------------------------------------------

    def to_dict(self) -> dict:
        if self.schema_version >= 6 and self.utterances:
            self.sync_segments()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        if not isinstance(data, dict):
            raise ValueError("La sessione deve essere un oggetto JSON")
        version = data.get("schema_version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2, 3, 4, 5, 6):
            raise ValueError("Versione della sessione non supportata")
        revision = data.get("revision", 0)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("Revisione della sessione non valida")
        for field_name in ("id", "name", "created_at"):
            if field_name in data and not isinstance(data[field_name], str):
                raise ValueError(f"Campo sessione non valido: {field_name}")
        names = data.get("speaker_names", {})
        if not isinstance(names, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in names.items()):
            raise ValueError("Nomi delle voci non validi")
        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            raise ValueError("Elenco segmenti non valido")
        raw_backup = data.get("raw_segments")
        if raw_backup is not None and not isinstance(raw_backup, list):
            raise ValueError("Elenco segmenti grezzi non valido")

        def validate_segment_list(items: list, label: str) -> None:
            for seg in items:
                if not isinstance(seg, dict) or not isinstance(seg.get("words", []), list):
                    raise ValueError(f"Segmento {label} non valido")
                if not isinstance(seg.get("speaker", "SPEAKER_00"), str):
                    raise ValueError("Identificativo voce non valido")
                if not isinstance(seg.get("id", "legacy"), str) or not seg.get("id", "legacy"):
                    raise ValueError("ID segmento non valido")
                if not isinstance(seg.get("words_valid", True), bool):
                    raise ValueError("Stato allineamento non valido")
                for key in ("warnings", "source_ids"):
                    if not isinstance(seg.get(key, []), list) or any(
                            not isinstance(v, str) for v in seg.get(key, [])):
                        raise ValueError(f"Campo segmento non valido: {key}")
                if not isinstance(seg.get("candidates", []), list):
                    raise ValueError("Candidati non validi")
                for item in [seg, *seg.get("words", [])]:
                    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                        raise ValueError("Testo segmento/parola non valido")
                    timestamps = [item.get("start"), item.get("end")]
                    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or
                           not math.isfinite(t) or t < 0 for t in timestamps):
                        raise ValueError("Timestamp non valido")
                    if timestamps[1] < timestamps[0]:
                        raise ValueError("La fine del segmento precede l'inizio")
                    probability = item.get("probability")
                    if probability is not None and (isinstance(probability, bool) or
                            not isinstance(probability, (int, float)) or
                            not math.isfinite(probability) or not 0 <= probability <= 1):
                        raise ValueError("Probabilità parola non valida")

        validate_segment_list(raw_segments, "principale")
        if raw_backup is not None:
            validate_segment_list(raw_backup, "grezzo")
        analysis = data.get("analysis")
        if analysis is not None and not isinstance(analysis, dict):
            raise ValueError("Analisi della sessione non valida")
        transcript_review = data.get("transcript_review")
        if transcript_review is not None and not isinstance(transcript_review, dict):
            raise ValueError("Metadati della revisione non validi")
        audio = AudioSources(**data.get("audio", {}))
        if any(value is not None and not isinstance(value, str) for value in asdict(audio).values()):
            raise ValueError("Percorso audio non valido")
        audio_profile = data.get("audio_profile")
        if audio_profile is not None and not isinstance(audio_profile, str):
            raise ValueError("Profilo audio non valido")
        if audio_profile not in (None, "mic_system", "stereo_ambient"):
            raise ValueError("Profilo audio non riconosciuto")
        def parse_segment_list(items: list | None) -> list[Segment] | None:
            if items is None:
                return None
            return [
                Segment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    speaker=s.get("speaker", "SPEAKER_00"),
                    words=[Word(**w) for w in s.get("words", [])],
                    id=s.get("id") or uuid.uuid5(uuid.NAMESPACE_URL,
                        f"tessitura:{data.get('id', 'legacy')}:{i}:{s['start']}:{s['text']}").hex,
                    words_valid=s.get("words_valid", True),
                    warnings=list(s.get("warnings", [])),
                    candidates=deepcopy(s.get("candidates", [])),
                    source_ids=list(s.get("source_ids", [])),
                )
                for i, s in enumerate(items)
            ]

        segments = parse_segment_list(raw_segments) or []
        if len({s.id for s in segments}) != len(segments):
            raise ValueError("ID segmento duplicato")
        for key in ("provenance", "diarization"):
            if not isinstance(data.get(key, {}), dict):
                raise ValueError(f"Campo sessione non valido: {key}")
        for key in ("chat_history", "edit_history", "redo_history"):
            if not isinstance(data.get(key, []), list) or any(
                    not isinstance(v, dict) for v in data.get(key, [])):
                raise ValueError(f"Campo sessione non valido: {key}")

        from .transcript_artifacts import Utterance

        raw_utts = data.get("utterances", [])
        if not isinstance(raw_utts, list):
            raise ValueError("Elenco utterances non valido")
        parsed_utterances = [
            Utterance.from_dict(u) if isinstance(u, dict) else u
            for u in raw_utts
        ]
        if len({u.event_id for u in parsed_utterances}) != len(parsed_utterances):
            raise ValueError("ID evento duplicato")

        raw_backup_utts = data.get("raw_utterances")
        parsed_raw_utterances = (
            [Utterance.from_dict(u) if isinstance(u, dict) else u for u in raw_backup_utts]
            if raw_backup_utts is not None and isinstance(raw_backup_utts, list)
            else None
        )

        if not parsed_utterances and segments:
            # Migrazione automatica da v1/v2 a v3: popola utterances
            parsed_utterances = [
                Utterance(
                    start_s=s.start,
                    end_s=s.end,
                    utterance_id=s.id,
                    stream_id=0,
                    speaker=s.speaker,
                    candidate_speakers=deepcopy(s.candidates),
                    text=s.text,
                    words=deepcopy(s.words),
                    hypothesis_ids=list(s.source_ids),
                )
                for s in segments
            ]
        elif parsed_utterances and not segments:
            # Proiezione verso segments per consumatori legacy
            from ..processing.merge import utterances_to_segments
            segments = utterances_to_segments(parsed_utterances)

        audio_assets = list(data.get("audio_assets", []))
        revisions = list(data.get("revisions", []))
        pipeline_strategy = data.get("pipeline_strategy", "standard")
        capture_manifest = data.get("capture_manifest")
        active_view = data.get("active_view", "conversation")
        timeline_visible_speakers = list(data.get("timeline_visible_speakers", []))

        result = cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            name=data.get("name", "Sessione senza nome"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            language=data.get("language"),
            model=data.get("model"),
            audio_profile=audio_profile,
            audio=audio,
            segments=segments,
            raw_segments=parse_segment_list(raw_backup),
            raw_utterances=parsed_raw_utterances,
            transcript_review=transcript_review,
            speaker_names=data.get("speaker_names", {}),
            analysis=analysis,
            revision=revision,
            provenance=deepcopy(data.get("provenance", {})),
            diarization=deepcopy(data.get("diarization", {})),
            chat_history=deepcopy(data.get("chat_history", [])),
            edit_history=deepcopy(data.get("edit_history", [])[-20:]),
            redo_history=deepcopy(data.get("redo_history", [])[-20:]),
            utterances=parsed_utterances,
            audio_assets=audio_assets,
            revisions=revisions,
            pipeline_strategy=pipeline_strategy,
            capture_manifest=capture_manifest,
            active_view=active_view,
            timeline_visible_speakers=timeline_visible_speakers,
            schema_version=int(data.get("schema_version", 2)),
            media_duration_s=float(data.get("media_duration_s", 0)),
            recovery_candidates=deepcopy(data.get("recovery_candidates", [])),
            speaker_evidence=deepcopy(data.get("speaker_evidence", [])),
        )
        if not math.isfinite(result.media_duration_s) or result.media_duration_s < 0:
            raise ValueError("Durata audio non valida")
        if version >= 6:
            result.sync_segments()
        return result
