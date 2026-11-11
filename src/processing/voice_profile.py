"""Profilazione acustica delle voci presenti nella registrazione.

Costruisce profili coerenti (centroidi acustici, durata, stabilità e canale)
per ogni voce prima dell'attribuzione, correlando eventuali profili già noti
nel progetto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any

from .adaptation import normalize_embedding, match_speaker_profiles
from .diarize import Turn


@dataclass
class VoiceProfile:
    """Profilo acustico e temporale di una voce identificata con trascrizione associata."""

    speaker_id: str
    name: str | None = None
    embedding: list[float] | None = None
    total_duration: float = 0.0
    turn_count: int = 0
    channel_dominance: int | None = None
    confidence: float = 1.0
    is_known_profile: bool = False
    metadata: dict = field(default_factory=dict)
    transcript: str = ""
    word_count: int = 0
    speaking_ratio: float = 0.0
    turn_intervals: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "embedding": self.embedding,
            "total_duration": round(self.total_duration, 3),
            "turn_count": self.turn_count,
            "channel_dominance": self.channel_dominance,
            "confidence": round(self.confidence, 4),
            "is_known_profile": self.is_known_profile,
            "metadata": dict(self.metadata),
            "transcript": self.transcript,
            "word_count": self.word_count,
            "speaking_ratio": round(self.speaking_ratio, 4),
            "turn_intervals": [[round(a, 3), round(b, 3)] for a, b in self.turn_intervals],
        }

    @classmethod
    def from_dict(cls, data: dict) -> VoiceProfile:
        return cls(
            speaker_id=str(data.get("speaker_id", "SPEAKER_00")),
            name=data.get("name"),
            embedding=data.get("embedding"),
            total_duration=float(data.get("total_duration", 0.0)),
            turn_count=int(data.get("turn_count", 0)),
            channel_dominance=data.get("channel_dominance"),
            confidence=float(data.get("confidence", 1.0)),
            is_known_profile=bool(data.get("is_known_profile", False)),
            metadata=dict(data.get("metadata", {})),
            transcript=str(data.get("transcript", "")),
            word_count=int(data.get("word_count", 0)),
            speaking_ratio=float(data.get("speaking_ratio", 0.0)),
            turn_intervals=[[float(a), float(b)] for a, b in data.get("turn_intervals", [])],
        )


def enrich_voice_profiles_with_transcript(
    profiles: dict[str, VoiceProfile],
    items: Sequence[Any],
    *,
    total_duration: float | None = None,
    speaker_names: Mapping[str, str] | None = None,
) -> dict[str, VoiceProfile]:
    """Arricchisce o sintetizza i profili vocali con la trascrizione e le metriche per voce."""
    enriched = dict(profiles)
    names = speaker_names or {}

    spk_items: dict[str, list[Any]] = {}
    for item in items:
        if getattr(item, "suppressed_bleed", False):
            continue
        spk = getattr(item, "speaker", "SPEAKER_00")
        spk_items.setdefault(spk, []).append(item)

    max_dur = total_duration if total_duration and total_duration > 0 else 0.0
    if max_dur <= 0.0:
        for it_list in spk_items.values():
            for it in it_list:
                end_t = getattr(it, "end", getattr(it, "end_s", 0.0))
                if end_t > max_dur:
                    max_dur = end_t

    for spk, it_list in spk_items.items():
        sorted_it = sorted(
            it_list,
            key=lambda x: (
                getattr(x, "start", getattr(x, "start_s", 0.0)),
                getattr(x, "end", getattr(x, "end_s", 0.0)),
            ),
        )

        texts = [getattr(x, "text", "").strip() for x in sorted_it if getattr(x, "text", "").strip()]
        full_transcript = " ".join(texts)
        words = sum(len(t.split()) for t in texts)
        intervals = [
            [
                float(getattr(x, "start", getattr(x, "start_s", 0.0))),
                float(getattr(x, "end", getattr(x, "end_s", 0.0))),
            ]
            for x in sorted_it
        ]
        spk_duration = sum(max(0.0, b - a) for a, b in intervals)
        ratio = (spk_duration / max_dur) if max_dur > 0.0 else 0.0

        if spk not in enriched:
            enriched[spk] = VoiceProfile(
                speaker_id=spk,
                name=names.get(spk),
                total_duration=spk_duration,
                turn_count=len(sorted_it),
                transcript=full_transcript,
                word_count=words,
                speaking_ratio=ratio,
                turn_intervals=intervals,
            )
        else:
            p = enriched[spk]
            p.transcript = full_transcript
            p.word_count = words
            p.turn_intervals = intervals
            if p.total_duration <= 0.0:
                p.total_duration = spk_duration
            if p.turn_count <= 0:
                p.turn_count = len(sorted_it)
            p.speaking_ratio = ratio
            if not p.name and spk in names:
                p.name = names[spk]

    return enriched


def build_voice_profiles(
    turns: list[Turn],
    *,
    speaker_embeddings: Mapping[str, Sequence[float]] | None = None,
    channel_dominance: Sequence[int] | None = None,
    frame_s: float = 0.02,
    known_profiles: Mapping[str, Sequence[float]] | None = None,
    match_threshold: float = 0.70,
) -> dict[str, VoiceProfile]:
    """Crea un profilo per ciascuna voce presente nei turni di diarizzazione."""
    profiles: dict[str, VoiceProfile] = {}
    if not turns:
        return profiles

    # Calcolo durata e numero turni per ciascun interlocutore
    for turn in turns:
        dur = max(0.0, turn.end - turn.start)
        if turn.speaker not in profiles:
            profiles[turn.speaker] = VoiceProfile(
                speaker_id=turn.speaker,
                total_duration=dur,
                turn_count=1,
            )
        else:
            p = profiles[turn.speaker]
            p.total_duration += dur
            p.turn_count += 1

    # Assegnazione centroidi acustici normalizzati
    if speaker_embeddings:
        for spk, raw_vec in speaker_embeddings.items():
            if spk in profiles:
                norm_vec = normalize_embedding(raw_vec)
                if norm_vec:
                    profiles[spk].embedding = [round(x, 8) for x in norm_vec]

    # Stima del canale dominante se disponibile
    if channel_dominance is not None and len(channel_dominance) > 0 and frame_s > 0:
        counts_per_spk: dict[str, dict[int, int]] = {spk: {} for spk in profiles}
        for turn in turns:
            i0 = max(0, int(turn.start / frame_s))
            i1 = min(len(channel_dominance), max(i0 + 1, int(turn.end / frame_s)))
            spk_counts = counts_per_spk[turn.speaker]
            for ch in channel_dominance[i0:i1]:
                ch_int = int(ch)
                if ch_int >= 0:
                    spk_counts[ch_int] = spk_counts.get(ch_int, 0) + 1

        for spk, counts in counts_per_spk.items():
            if counts:
                best_ch = max(counts, key=lambda k: counts[k])
                total = sum(counts.values())
                if total > 0 and (counts[best_ch] / total) >= 0.60:
                    profiles[spk].channel_dominance = best_ch

    # Correlazione con profili noti del progetto
    if known_profiles:
        active_embeddings = {
            spk: p.embedding
            for spk, p in profiles.items()
            if p.embedding is not None
        }
        if active_embeddings:
            matches = match_speaker_profiles(
                active_embeddings,
                known_profiles,
                threshold=match_threshold,
            )
            for spk, match_info in matches.items():
                if spk in profiles:
                    profiles[spk].name = str(match_info["name"])
                    profiles[spk].is_known_profile = True
                    profiles[spk].confidence = float(match_info.get("similarity", 1.0))

    return profiles


def match_with_known_profiles(
    profiles: Mapping[str, VoiceProfile],
    known_profiles: Mapping[str, Sequence[float]],
    threshold: float = 0.70,
) -> dict[str, dict]:
    """Confronta i profili attuali con i profili noti del progetto."""
    active_embeddings = {
        spk: p.embedding
        for spk, p in profiles.items()
        if p.embedding is not None
    }
    if not active_embeddings or not known_profiles:
        return {}
    return match_speaker_profiles(active_embeddings, known_profiles, threshold=threshold)


@dataclass
class GlobalSpeakerMapping:
    """Mappatura globale delle identita' tra finestre multiple con evidenze."""

    global_speakers: dict[str, VoiceProfile]
    local_to_global: dict[tuple[int, str], str]  # (window_idx, local_spk_id) -> global_spk_id
    evidence: list[dict[str, Any]] = field(default_factory=list)


def filter_clean_speech_turns(
    turns: list[Turn],
    *,
    min_duration_s: float = 0.8,
    exclude_overlaps: bool = True,
) -> list[Turn]:
    """Seleziona solo turni puliti (senza overlap e di durata sufficiente) per calcolo centroidi affidabili."""
    if not turns:
        return []

    clean: list[Turn] = []
    for i, t in enumerate(turns):
        dur = max(0.0, t.end - t.start)
        if dur < min_duration_s:
            continue

        if exclude_overlaps:
            has_overlap = False
            for j, o in enumerate(turns):
                if i != j and o.speaker != t.speaker:
                    if min(t.end, o.end) > max(t.start, o.start):
                        has_overlap = True
                        break
            if has_overlap:
                continue

        clean.append(t)
    return clean


def _cosine_sim(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calcola la similarita' coseno tra due vettori normalizzati."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    return float(sum(a * b for a, b in zip(v1, v2)))


def reconcile_window_speakers(
    windows: list[dict[str, VoiceProfile] | list[VoiceProfile]],
    *,
    known_profiles: Mapping[str, Sequence[float]] | None = None,
    similarity_threshold: float = 0.70,
    ambiguity_margin: float = 0.05,
) -> GlobalSpeakerMapping:
    """Riconcilia le identita' locali tra finestre temporali multiple.

    Invariante fondamentale (sezione 6.3 e gate P05):
    - Permutare gli ID locali in una finestra non altera le identita' globali finali.
    - Se l'accoppiamento e' ambiguo (differenza di score < ambiguity_margin tra due candidati),
      evita attribuzioni forzate e assegna UNKNOWN o identita' distinta.
    - Se una voce nuova non supera similarity_threshold rispetto a nessun profilo globale,
      crea un nuovo profilo globale anziche' forzarla in un cluster esistente.
    """
    global_speakers: dict[str, VoiceProfile] = {}
    local_to_global: dict[tuple[int, str], str] = {}
    evidence: list[dict[str, Any]] = []

    global_counter = 0

    # Inizializza profili noti se forniti
    if known_profiles:
        for name, emb in known_profiles.items():
            norm_e = normalize_embedding(emb)
            if norm_e:
                spk_id = f"GLOBAL_{name}"
                global_speakers[spk_id] = VoiceProfile(
                    speaker_id=spk_id,
                    name=name,
                    embedding=[round(x, 8) for x in norm_e],
                    is_known_profile=True,
                )

    for w_idx, win in enumerate(windows):
        local_dict = win if isinstance(win, dict) else {p.speaker_id: p for p in win}
        if not local_dict:
            continue

        # Ordina per speaker_id per garantire determinismo assoluto
        sorted_locals = sorted(local_dict.items(), key=lambda item: item[0])

        for local_id, l_prof in sorted_locals:
            if not l_prof.embedding:
                # Senza embedding, assegna nuovo ID locale o UNKNOWN
                global_counter += 1
                g_id = f"SPEAKER_G{global_counter:02d}"
                global_speakers[g_id] = VoiceProfile(
                    speaker_id=g_id,
                    name=l_prof.name,
                    total_duration=l_prof.total_duration,
                    turn_count=l_prof.turn_count,
                    channel_dominance=l_prof.channel_dominance,
                )
                local_to_global[(w_idx, local_id)] = g_id
                evidence.append({
                    "window": w_idx, "local": local_id, "global": g_id,
                    "reason": "no_embedding_new_cluster"
                })
                continue

            # Calcola similarita' con tutti i profili globali esistenti che hanno un embedding
            scores: list[tuple[str, float]] = []
            for g_id, g_prof in global_speakers.items():
                if g_prof.embedding:
                    sim = _cosine_sim(l_prof.embedding, g_prof.embedding)
                    scores.append((g_id, sim))

            scores.sort(key=lambda s: s[1], reverse=True)

            matched_gid: str | None = None
            if scores and scores[0][1] >= similarity_threshold:
                best_gid, best_score = scores[0]
                # Controllo ambiguita': se c'e' un secondo candidato troppo vicino
                if len(scores) > 1 and (best_score - scores[1][1]) < ambiguity_margin:
                    # Ambigua: non forzare matching errato
                    evidence.append({
                        "window": w_idx, "local": local_id, "top_candidates": scores[:2],
                        "reason": "ambiguous_match"
                    })
                    g_id = "UNKNOWN"
                    local_to_global[(w_idx, local_id)] = g_id
                    continue
                else:
                    matched_gid = best_gid
                    evidence.append({
                        "window": w_idx, "local": local_id, "global": best_gid,
                        "similarity": round(best_score, 4), "reason": "similarity_match"
                    })

            if matched_gid:
                local_to_global[(w_idx, local_id)] = matched_gid
                # Aggiorna metriche globale
                gp = global_speakers[matched_gid]
                gp.total_duration += l_prof.total_duration
                gp.turn_count += l_prof.turn_count
            else:
                # Nuova voce non presente nei profili globali
                global_counter += 1
                new_gid = f"SPEAKER_G{global_counter:02d}"
                global_speakers[new_gid] = VoiceProfile(
                    speaker_id=new_gid,
                    name=l_prof.name,
                    embedding=list(l_prof.embedding),
                    total_duration=l_prof.total_duration,
                    turn_count=l_prof.turn_count,
                    channel_dominance=l_prof.channel_dominance,
                )
                local_to_global[(w_idx, local_id)] = new_gid
                evidence.append({
                    "window": w_idx, "local": local_id, "global": new_gid,
                    "reason": "below_threshold_new_speaker"
                })

    return GlobalSpeakerMapping(
        global_speakers=global_speakers,
        local_to_global=local_to_global,
        evidence=evidence,
    )
