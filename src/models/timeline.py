"""Atomic edits to the canonical speech events, with an immutable prediction trail.

Legacy Segment objects are projections for old consumers. No edit is recovered by
matching timestamps: simultaneous events are addressed exclusively by their ID.
"""
from __future__ import annotations

import math
import uuid
from copy import deepcopy

from .transcript_artifacts import Utterance, recompute_overlap_relationships

UNSET = object()


class TimelineEditor:
    def __init__(self, session):
        self.session = session
        session.migrate_to_v6()

    def event(self, event_id: str) -> Utterance:
        return next(u for u in self.session.utterances if u.event_id == event_id)

    def _finish(self):
        recompute_overlap_relationships(self.session.utterances)
        self.session.sync_segments()

    def edit_event(self, event_id, *, text=None, speaker_id=UNSET, start_s=None,
                   end_s=None, processing_state=None, uncertain=None) -> bool:
        event = self.event(event_id)
        changes = {}
        for key, value in (("text", text), ("start_s", start_s), ("end_s", end_s),
                           ("processing_state", processing_state), ("uncertain", uncertain)):
            if value is not None and getattr(event, key) != value:
                changes[key] = value
        if speaker_id is not UNSET and event.speaker_id != speaker_id:
            changes["speaker"] = speaker_id or None
        start, end = changes.get("start_s", event.start_s), changes.get("end_s", event.end_s)
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("Intervallo non valido: serve 0 <= inizio < fine.")
        if not changes:
            return False
        self.session.begin_edit("Correzione evento")
        for key, value in changes.items():
            event.record_user_correction(key, deepcopy(getattr(event, key)), deepcopy(value))
            setattr(event, key, value)
        if any(key in changes for key in ("text", "start_s", "end_s")):
            event.words = []
        if "text" in changes:
            event.processing_state = "reviewed" if event.text.strip() else "missing_transcription"
        self._finish()
        return True

    def create_event(self, start_s, end_s, text="", speaker_id=None) -> Utterance:
        if not all(math.isfinite(t) for t in (start_s, end_s)) or not 0 <= start_s < end_s:
            raise ValueError("Intervallo non valido.")
        self.session.begin_edit("Nuovo intervento manuale")
        event = Utterance(start_s=start_s, end_s=end_s, text=text,
                          speaker=speaker_id, source_audio_ref=self.session.audio.primary(),
                          edit_status="user_created", processing_state="reviewed" if text.strip() else "missing_transcription",
                          original_prediction={"origin": "manual", "text": "", "speaker": None},
                          user_corrections=[{"field": "create", "author": "user", "text": text}])
        self.session.utterances.append(event)
        self._finish()
        return event

    def create_speaker(self, name: str) -> str:
        if not name.strip():
            raise ValueError("Inserisci un nome per la voce.")
        self.session.begin_edit("Nuova voce")
        speaker_id = f"USER_{uuid.uuid4().hex[:8]}"
        self.session.speaker_names[speaker_id] = name.strip()
        return speaker_id

    def split_event(self, event_id, seconds, left_text=None, right_text=None) -> bool:
        event = self.event(event_id)
        if not event.start_s < seconds < event.end_s:
            return False
        left_words = [w for w in event.words if (w.start + w.end) / 2 < seconds]
        right_words = [w for w in event.words if (w.start + w.end) / 2 >= seconds]
        if left_text is None or right_text is None:
            if not left_words or not right_words:
                raise ValueError("Indica il testo delle due parti: allineamento assente al punto di divisione.")
            left_text = " ".join(w.text.strip() for w in left_words)
            right_text = " ".join(w.text.strip() for w in right_words)
        self.session.begin_edit("Divisione evento")
        parts = []
        for start, end, text, words in ((event.start_s, seconds, left_text, left_words),
                                        (seconds, event.end_s, right_text, right_words)):
            part = deepcopy(event)
            part.utterance_id = uuid.uuid4().hex
            part.parent_event_ids = [event_id]
            part.start_s, part.end_s, part.text = start, end, text
            # Manual text or a cut through a word invalidates alignment. Keeping
            # a partial word list would make SRT export omit the corrected text.
            aligned_text = " ".join(w.text.strip() for w in words)
            part.words = deepcopy(words) if (text == aligned_text and all(start <= w.start <= w.end <= end for w in words)) else []
            part.record_user_correction("split", event_id, {"start_s": start, "end_s": end, "text": text})
            parts.append(part)
        index = self.session.utterances.index(event)
        self.session.utterances[index:index + 1] = parts
        self._finish()
        return True

    def merge_events(self, left_id, right_id) -> bool:
        left, right = self.event(left_id), self.event(right_id)
        if left.start_s > right.start_s:
            left, right = right, left
        lane = sorted((u for u in self.session.utterances if u.speaker_id == left.speaker_id), key=lambda u: (u.start_s, u.end_s, u.event_id))
        if (left is right or left.speaker_id != right.speaker_id or left.source_audio_ref != right.source_audio_ref
                or right.start_s < left.end_s or right.start_s - left.end_s > 2
                or lane.index(right) != lane.index(left) + 1):
            raise ValueError("Unisci solo eventi adiacenti della stessa voce e sorgente, senza sovrapposizione e con pausa <= 2 s.")
        self.session.begin_edit("Unione eventi")
        merged = deepcopy(left)
        merged.utterance_id = uuid.uuid4().hex
        merged.parent_event_ids = [left.event_id, right.event_id]
        merged.end_s = right.end_s
        merged.text = " ".join(x.strip() for x in (left.text, right.text) if x.strip())
        merged.words = deepcopy(left.words + right.words) if left.words and right.words else []
        merged.user_corrections.extend(deepcopy(right.user_corrections))
        merged.record_user_correction("merge", [left.to_dict(), right.to_dict()], merged.event_id)
        self.session.utterances = [u for u in self.session.utterances if u.event_id not in merged.parent_event_ids] + [merged]
        self._finish()
        return True

    def confirm_identity(self, event_id, speaker_id) -> bool:
        event = self.event(event_id)
        self.session.begin_edit("Conferma identità locale")
        event.record_user_correction("speaker", event.speaker, speaker_id, reason="identity_confirmed")
        event.record_user_correction("identity_confirmation", None, {"speaker_id": speaker_id, "scope": "event"}, reason="user_confirmed")
        event.speaker_id = speaker_id
        event.uncertain = False
        event.diarization_provenance["identity_confirmation_scope"] = "event"
        self.session.speaker_evidence.append({"event_id": event_id, "speaker_id": speaker_id,
                                              "start_s": event.start_s, "end_s": event.end_s,
                                              "confirmed": True, "scope": "event", "source_audio_ref": event.source_audio_ref})
        self._finish()
        return True

    def accept_candidate(self, candidate_id, speaker_id=None) -> Utterance:
        candidate = next(c for c in self.session.recovery_candidates if c["candidate_id"] == candidate_id)
        if candidate.get("status") != "proposed":
            raise ValueError("Candidato già esaminato.")
        if candidate.get("kind") == "speaker_assignment":
            event = self.event(candidate["proposed_event_id"])
            if candidate.get("event_snapshot") != event.to_dict():
                raise ValueError("L'evento è cambiato dopo la proposta. Riesamina l'identità prima di applicarla.")
            target_speaker = speaker_id or candidate.get("proposed_speaker")
            if not target_speaker:
                raise ValueError("Identità ancora irrisolta.")
            self.session.begin_edit("Accetta identità globale proposta")
            event.record_user_correction("speaker", event.speaker, target_speaker, reason="global_acoustic_evidence_reviewed")
            event.speaker_id = target_speaker
            candidate["status"] = "accepted"
            self._finish()
            return event
        event = Utterance.from_dict(deepcopy(candidate["event"]))
        normalized = " ".join(event.text.casefold().split()).strip(" .?!,…")
        digest = event.asr_provenance.get("source_sha256")
        if digest and any(u.processing_state == "reviewed" and u.speaker_id == speaker_id
                          and u.asr_provenance.get("source_sha256") == digest
                          and " ".join(u.text.casefold().split()).strip(" .?!,…") == normalized
                          and u.start_s < event.end_s and u.end_s > event.start_s
                          for u in self.session.utterances):
            raise ValueError("Contenuto già accettato per questa voce e regione; confronta le alternative.")
        self.session.begin_edit("Accetta candidato come evento distinto")
        event.utterance_id = uuid.uuid4().hex
        event.speaker_id = speaker_id
        event.record_user_correction("accept_candidate", candidate_id, event.event_id)
        event.processing_state = "reviewed"
        self.session.utterances.append(event)
        candidate["status"] = "accepted"
        candidate["accepted_event_id"] = event.event_id
        self._finish()
        return event

    def reject_candidate(self, candidate_id) -> bool:
        candidate = next(c for c in self.session.recovery_candidates if c["candidate_id"] == candidate_id)
        if candidate.get("status") != "proposed":
            raise ValueError("Candidato già esaminato.")
        self.session.begin_edit("Rifiuta candidato di recupero")
        candidate["status"] = "rejected"
        return True
