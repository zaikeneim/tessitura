"""Profili audio espliciti usati dalla pipeline.

Un file stereo non identifica da solo due sorgenti indipendenti: una
registrazione ambientale stereo e un file mic+audio possono avere la stessa
forma tecnica. Il profilo viene quindi scelto dall'utente prima di caricare i
modelli, così la pipeline non può applicare per errore il channel hint a una
registrazione ambientale.
"""

from __future__ import annotations

from enum import Enum


class AudioProfile(str, Enum):
    """Layout semantico delle sorgenti audio."""

    MIC_SYSTEM = "mic_system"
    STEREO_AMBIENT = "stereo_ambient"

    @classmethod
    def parse(cls, value: "AudioProfile | str | None") -> "AudioProfile | None":
        if value is None or value == "":
            return None
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(
                f"Profilo audio non valido: {value!r}. "
                f"Valori ammessi: {', '.join(item.value for item in cls)}."
            ) from exc

    @property
    def label(self) -> str:
        return {
            AudioProfile.MIC_SYSTEM: "Microfono + audio di sistema",
            AudioProfile.STEREO_AMBIENT: "Ambientale stereo",
        }[self]

    @property
    def description(self) -> str:
        return {
            AudioProfile.MIC_SYSTEM: (
                "Due sorgenti separate: il canale microfono e il canale "
                "audio remoto/sistema."
            ),
            AudioProfile.STEREO_AMBIENT: (
                "Un recorder o microfono stereo che registra le persone "
                "nella stessa stanza. I canali non sono speaker separati."
            ),
        }[self]
