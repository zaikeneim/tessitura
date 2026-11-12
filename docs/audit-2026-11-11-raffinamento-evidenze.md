# Tessitura — Audit del codice guidato dalle evidenze (2026-11-11)

Branch di lavoro: `audit/evidence-driven-refinement` (tagliato da `feat/global-overlap-timeline`
a `e8d7fce`, con il lavoro non committato di quel ramo portato avanti intatto; snapshot di
sicurezza pre-audit creato con `git stash create`: `c3a474ac`). Consegna: commit locale sul ramo
di audit del repository di lavoro (che non ha un ramo `main`) e, su richiesta esplicita, commit
su `main` della copia pubblicabile dopo `sync-clean`; nessun push.

> Ripresa e nuove verifiche: [rapporto della ripresa](audit-2026-11-11-ripresa.md).
> Lavoro ancora da fare: [lavoro residuo](audit-2026-11-11-lavoro-residuo.md).
> Le metriche WER/cpWER sotto riportate sono state calcolate con il manifest gold dei primi
> 60 minuti, che non è nel checkout: si trova solo nel backup locale dei dati. Per questo la
> ripresa non ha potuto riprodurle; script e risultati sono conservati in
> `artifacts/audit-20261111/` (cartella ignorata da Git).

Hardware: Intel i7-13700HX, RTX 4070 Laptop 8 GB, 32 GB RAM, Windows 11 x64.
Runtime verificato: Python 3.11.16, torch 2.13.0+cu126 (cuDNN 9.10.02), torchaudio 2.11.0+cu126,
torchcodec 0.14.0, CTranslate2 4.8.1, faster-whisper 1.2.1, pyannote.audio 4.0.7, PySide6 6.11.1,
PyInstaller 6.22.3. Modelli: `Systran/faster-whisper-large-v3` (`int8_float16`),
`pyannote/speaker-diarization-community-1`.

Registrazione di riferimento: il file ambientale stereo di 5 890,8 s (sha256 `7f3653af…`),
già elaborato per intero prima dell'audit con checkpoint ASR/allineamento/diarizzazione nella
cache utente (`cache/benchmarks/full-raw-…/checkpoints/a4f1…`). Riferimento annotato: il manifest
gold dei primi 60 minuti (501 turni, 4 voci, 4 883 parole), recuperato dal backup locale del
progetto perché `data/evaluation/` nel repository era stato svuotato. **Limite del riferimento**:
i turni gold hanno i confini del trascritto sorgente (granularità ~7 s) e il testo deriva da una
precedente uscita di Tessitura corretta a mano; i numeri di attribuzione parola-per-parola sono
quindi indicativi, non una verità di riferimento verificata a livello di parola. Le metriche
della sezione 7 vanno lette come confronti relativi sulla stessa registrazione.

---

## 1. Architettura e moduli esaminati

Flusso reale (profilo ambientale, GUI e CLI):

`audio_utils.mix_sources` → `transcribe.Transcriber` (faster-whisper, VAD Silero, retry mirati,
recupero copertura) → `alignment.align_segments` → `diarize.Diarizer.diarize_detailed`
(turni ordinari, esclusivi, overlap, embedding) → `voice_profile.build_voice_profiles` →
`merge.assign_speakers` (voto per frase, `_smooth`, `_absorb_alternations`) →
`transcript_review.review_labels` → `global_diarization.reconstruct_events` →
`merge.utterances_to_segments` → `Session` (schema 6) → `storage.save_session` →
`TranscriptController` / `SpeakerTimelineView` / export TXT, SRT.

Moduli letti integralmente o nelle parti rilevanti: `src/processing/{pipeline, transcribe,
diarize, merge, global_diarization, transcript_review, separation, region_recovery, coverage,
alignment, verified_evaluation, evaluation, checkpoints, resources, capture_sync, autocorrect}.py`,
`src/models/{session, transcript_artifacts, timeline, storage}.py`,
`src/gui/{speaker_timeline_view, transcript_controller, main_window}.py`,
`src/audio/{capture, recording_metadata}.py`, `scripts/{verify, run_pipeline, evaluate_sessions,
measure_fragmentation, audit_global_timeline, build-windows.ps1}`,
`.github/workflows/release-windows.yml`, `requirements.txt`, `environment.yml`, `AGENTS.md`,
`README.md`, `docs/{baseline, benchmark-results, architecture-target, architecture-audit}.md`.

GitNexus: indice presente e aggiornato al commit `e8d7fce` (896 simboli). Usato per
`impact` su `reconstruct_events` (CRITICAL, 12 simboli a monte, 7 flussi), `review_labels`
(CRITICAL, 9), `assign_speakers` (CRITICAL, 19). `detect-changes --scope all` sul working tree:
64 file e 248 simboli cambiati, rischio CRITICAL — il numero include il lavoro non committato
del ramo `feat/global-overlap-timeline`, non solo le modifiche dell'audit. L'indice è stato
rigenerato (`analyze --index-only`) prima del commit; la variante con `--repo` di `analyze`
non è accettata da questa versione CLI (1.6.12).

Baseline test prima di qualsiasi modifica: `python -m scripts.verify` → 67 passi, tutti superati.

---

## 2. Difetti confermati

### D1 — P0 · La revisione dell'attribuzione non arriva mai alla sessione (profilo ambientale)
`src/processing/pipeline.py:735-738` (stato prima della correzione, codice non committato del
ramo `feat/global-overlap-timeline`; in `HEAD` il percorso era ancora
`build_utterances_from_segments(segments)`).

Sequenza osservata: `assign_speakers` → `review_labels` → `result.segments` (+ autocorrezione) →
poi `reconstruct_events(tsegments, ordinary_turns, exclusive_turns)` che **ricalcola ogni parola dai
soli turni ordinari** e ignora sia `labels` sia `segments`. Conseguenze misurate sui checkpoint
della registrazione di 2 ore (stesse parole ASR, stessa diarizzazione):

| percorso | eventi testuali | parole non assegnate | catene ping-pong (segmenti intrappolati) |
|---|---|---|---|
| merge + review (HEAD) | 908 | 0 | 19 (95 / 908 = 10,5 %) |
| `reconstruct_events` senza etichette (WIP) | 1 220 | 628 (387 eventi `UNASSIGNED`) | 60 (337 / 1 220 = 27,6 %) |

Sulla finestra gold di 60 minuti (sezione 7): confusione diarizzazione 3,38 % → 8,93 %, errore di
attribuzione parola 4,09 % → 11,23 % (2,15 % errate + 9,09 % non assegnate). Inoltre
`transcript_review` veniva salvato con `changed_words`/`selected` riferiti a etichette mai usate,
e `raw_segments` ("Trascrizione grezza") conteneva l'uscita *più* levigata di quella mostrata:
semantica invertita rispetto al README. La revisione costa 7,1 s CPU per poi essere scartata.

Classificazione: difetto di integrazione del lavoro in corso, non scelta intenzionale: la
docstring di `global_diarization` descrive l'obiettivo "nessuno smoothing forzato", ma il
risultato elimina anche il voto per frase che aveva risolto la frammentazione (`merge.py`,
caso "Si. Per / rendere una linea") e contraddice README e metadati salvati.

**Correzione (minima)**: `reconstruct_events(..., word_labels=...)` — le etichette finali
(merge, sostituite da quelle della revisione quando attiva) diventano la voce dell'evento; l'attività
ordinaria continua a decidere `uncertain`, `candidate_speakers` e gli eventi simultanei non
trascritti (`materialize_overlap_utterances`), con provenienza esplicita
(`assignment`: `reviewed_label_supported | reviewed_label_in_overlap | reviewed_label_without_activity`).
Senza `word_labels` il comportamento precedente è invariato (test esistenti verdi).
`pipeline.py` passa `final_labels`. File: `src/processing/global_diarization.py`,
`src/processing/pipeline.py`.

### D2 — P0 · CLI documentata inutilizzabile
`scripts/run_pipeline.py:40-43`: `--speakers` e `--no-denoise` dichiarati due volte →
`argparse.ArgumentError: conflicting option string` prima di qualunque elaborazione. Introdotto
dal commit `23bc2b4` (2026-09-21). Il comando del README `python -m scripts.run_pipeline …`
non poteva partire. Corretto rimuovendo i duplicati; test `tests/test_run_pipeline_cli.py`.

### D3 — P1 · Evento di durata zero non modificabile
`src/processing/global_diarization.py` (`reconstruct_events`): la prima parola ASR della
registrazione reale è " Gli" a 0,000–0,000 s e precede il primo turno (0,03 s); isolata in un
evento con `start_s == end_s`, che `TimelineEditor.edit_event` rifiuta ("Intervallo non valido").
L'utente non poteva correggerla dalla timeline. Il percorso legacy non aveva il problema (la parola
restava nella frase 0,0–5,32 s). Correzione: fine evento `max(end, start + 0,001)`; le parole
mantengono i loro timestamp. Test: `test_zero_length_first_word_before_any_activity_stays_editable`.

### D4 — P2 · Gate CI incompleto
`.github/workflows/release-windows.yml` e `scripts/build-windows.ps1` eseguivano
`unittest discover`, che raccoglie solo le classi `TestCase`: `tests/test_merge.py` (9 funzioni,
il modulo sulle regressioni storiche dell'attribuzione) non veniva mai eseguito. Inoltre il job
installava solo `requirements.txt`, che per scelta documentata (`environment.yml`) non contiene
PySide6: 17 moduli di test importano `PySide6` a livello di modulo e sarebbero falliti in import.
Nessuna esecuzione del workflow risulta (nessun tag `v*` sul remoto). Correzione: `pip install
PySide6` nel runner (nessuna conda) e `python -m scripts.verify` sia nel workflow sia nello
script di build; `--full` escluso perché il runner non ha GPU né modelli. Non validato su GitHub
Actions (nessuna esecuzione reale possibile da qui).

### D5 — P2 · Semantica di `_overlapping_spans`
`src/processing/diarize.py:85`: contava gli *intervalli* attivi, non le *voci distinte*: due turni
della stessa voce accavallati davano una sovrapposizione fittizia. Sulla diarizzazione reale non
si verifica (0 coppie stessa-voce intersecanti su 2 091 turni; 178 intervalli / 83,8 s identici
con entrambe le definizioni), quindi nessun impatto osservato. Corretto con conteggio per voce,
fine-prima-di-inizio allo stesso istante conservato. Test: `tests/test_overlap_semantics.py`
(due voci, adiacenti, tre voci, stessa voce intersecante, annotazioni duplicate/frammentate,
confronto con `activity_spans`).

### D6 — P2 · `candidate_speakers` solo dell'ultima parola
`reconstruct_events` sovrascriveva `candidate_speakers` a ogni parola: l'evento conservava i
candidati dell'ultima parola invece dell'attività accumulata. Ora accumulata sull'evento.

---

## 3. Falsi positivi e questioni investigate senza modifica

- **Euristiche di `merge.py`** (voto per frase, `_smooth`, `_absorb_alternations`). A/B sui
  checkpoint di 2 ore, finestra gold 60 min (errore attribuzione parola / segmenti intrappolati):
  completo 4,09 % / 95; senza `_absorb_alternations` 4,25 % / 105; senza `_smooth` 4,09 % / 95;
  `preserve_short_turns=False` 3,94 % / 65; voto parola-per-parola 4,23 % / 105. Le 10 parole
  cambiate da `_absorb_alternations` sull'intera registrazione concordano col gold 9 a 1.
  Una variante con guardia di durata sulle "isole" (≥ 0,65 s non è frammento) è stata provata e
  **scartata**: +8 parole errate, +10 segmenti intrappolati. Scenari costruiti a mano (turni
  esclusivi come li passa la pipeline): interruzione di 3 parole / 1,1 s senza pausa né
  punteggiatura → assorbita (limite noto, voluto dalla regola "isola"; conservata se attestata in
  overlap o con ≥ 3 voci); risposta breve "sì." punteggiata tra due turni dello stesso parlante →
  assorbita da `_smooth` con 2 voci, recuperata da `review_labels` (alternativa
  `preserve_short_turns`) ora che la revisione arriva alla sessione; alternanza rapida con
  punteggiatura e ≥ 3 voci → conservata; senza punteggiatura e pause < 0,35 s → collassata dalla
  regola "catena" (per progetto); falso cambio di 0,3 s dentro una frase → assorbito. Nessuna
  modifica: la regola attuale è netta positiva sul corpus disponibile.
- **`review_labels` su questa registrazione**: seleziona `baseline` (margine −0,0057,
  71 parole ambigue, 0 cambiate). Non è un no-op per costruzione: con 2 voci e interiezioni
  brevi sceglie l'alternativa (test esistenti e nuovo test di persistenza).
- **Sovrapposizione**: `turns` ordinari restano autorevoli; `exclusive` usati solo per il voto
  parola. Sul file reale: massimo 2 voci simultanee, 0 turni degeneri, 0 duplicati.
- **`transcribe.py`**: parametri empirici confermati diversi dai default della libreria installata
  (VAD: soglia 0,35 vs 0,5, silenzio minimo 400 ms vs 2 000, padding 600 ms vs 400;
  `condition_on_previous_text=False` vs True; `hallucination_silence_threshold=5.0` vs None;
  `repetition_penalty=1.05`). Non modificati. Retry: `clip_timestamps` assoluti, accettazione solo
  se `_quality_score` migliora di > 0,05, parole filtrate sul punto medio nella finestra originale;
  recupero copertura: le parole già coperte da `existing_spans` vengono scartate (modifica non
  committata del ramo, corretta). Sull'uscita reale di 2 ore: 1 453 segmenti, 0 inizi non
  monotoni, 0 parole con inizio arretrato, 0 riparazioni di allineamento, ultimo timestamp
  5 881,94 s su 5 890,78 s. Osservazione non corretta (P3): `_quality_score` non pesa il numero di
  parole, quindi un retry più corto ma con logprob migliore può sostituire un segmento sospetto più
  lungo; richiede A/B su audio (vedi lavoro residuo).
- **Separazione** (`separation.py`): le tre guardie (correlazione > 0,88, traccia < −60 dBFS,
  caduta > 12 dB tra grezzo e somma) restano; la somma delle tracce non è prova di parole perse
  (fase/scala), come già annotato nel codice; il fallback grezzo è sempre conservato e
  `region_recovery` non accetta mai automaticamente un candidato. Nessuna modifica; diagnostica
  aggiuntiva (es. SI-SDR fra tracce e miscela) proposta come residuo.
- **Timeline** (`speaker_timeline_view.py`), sessione reale di 2 235 eventi, offscreen:
  caricamento 0,23 s (8,9 MB), layout+render finestra 60 s 5,9 ms, finestra 2 h intera 43,9 ms
  (2 235 blocchi), `update_playback_position` 0,07 ms; solo la finestra visibile viene disposta e
  disegnata; la forma d'onda è letta fuori dal thread GUI con timer di debounce (80 ms). Nessun
  ridisegno eccessivo trovato; nessuna estrazione da `main_window.py` proposta: i test di ciclo
  di vita esistono e passano, non è emerso un difetto misurabile.
- **Registrazione / sorgenti**: modalità ambientale, cuffie e altoparlanti coperte da
  `tests/test_recording_capture_modes.py`, `test_recording_sources.py`, `test_capture_sync.py`,
  `test_loopback_timing_regressions.py` (offset separati dal ritardo acustico, drift su 2 h solo
  con clock affidabile, doppio parlato non cancellato, originali immutabili). Nessuna modifica.
- **Dipendenze**: versioni installate coerenti con la documentazione ufficiale (sezione 4):
  CTranslate2 4.8.1 richiede CUDA 12 + cuDNN 9 (torch cu126 fornisce cuDNN 9.10); torchaudio
  2.11 è dichiarato compatibile con torch 2.11+ (2.13 incluso); PySide6 6.11.1 nel range
  Python ≥ 3.10 < 3.16. Nessun aggiornamento eseguito.
- **Default `denoise`**: nel lavoro non committato del ramo `PipelineOptions.denoise` è passato da
  `True` a `False` (`pipeline.py`). Non è una modifica dell'audit; segnalata perché cambia il
  comportamento predefinito della GUI rispetto alla baseline documentata (`docs/baseline.md`,
  `denoise=true`).
- **Checkpoint e annullamento**: `checkpoints.CheckpointStore` è usato solo da test e harness
  di benchmark, non da `process_recording`; il worker GUI non espone un annullamento. Limiti
  reali ma non difetti del codice corrente; vedi lavoro residuo.

---

## 4. Ricerca sulle buone pratiche (fonti ufficiali)

| tema | implementazione attuale | fonte | perché si applica | beneficio / rischio | verifica |
|---|---|---|---|---|---|
| Requisiti GPU CTranslate2 | ct2 4.8.1 con DLL cuDNN/cuBLAS di torch cu126 | [faster-whisper README](https://github.com/SYSTRAN/faster-whisper) | "le ultime versioni di ctranslate2 supportano solo CUDA 12 e cuDNN 9"; ct2 4.4.0 per cuDNN 8 | configurazione già corretta; rischio solo se si aggiorna torch a un cuDNN diverso | `check_env`, `runtime_health verify`, elaborazione 2 h riuscita |
| VAD Silero e `vad_parameters` | dizionario con soglie empiriche | README faster-whisper (default: rimuove solo silenzi > 2 s) | i default tagliano parlato basso e pause brevi (misure nel codice) | mantenere; A/B solo su campioni verificati | test `test_transcribe_retry`, `test_coverage_recovery` |
| `condition_on_previous_text` | `False` | default libreria `True`, README mostra `False` per distil | evita loop di ripetizione documentati nel codice | mantenere | idem |
| Diarizzazione esclusiva | `exclusive` per il voto parola, `turns` per overlap | [scheda pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | la scheda indica l'uscita esclusiva per "semplificare la riconciliazione con i timestamp della trascrizione"; ingresso mono 16 kHz | uso conforme; overlap conservato dai turni ordinari | `tests/test_overlap_semantics.py`, `test_transcript_review.DiarizationOutputsTests` |
| Metriche | WER/cpWER/confusione (`evaluation.py`), DER/JER senza collar su regioni verificate (`verified_evaluation.py`) | pyannote.metrics / scheda community-1 ("nessun collar, overlap non escluso") | il modulo già allinea la definizione DER a quella del modello | nessuna modifica | `tests/test_verified_evaluation.py` |
| torch / torchaudio | 2.13.0+cu126 / 2.11.0+cu126, torchcodec 0.14.0 | [installazione torchaudio](https://docs.pytorch.org/audio/stable/installation.html) | "TorchAudio 2.11 funziona con PyTorch 2.11 e tutte le versioni successive (2.12, 2.13, …)"; Python 3.10–3.14 | configurazione valida; torchaudio è in manutenzione, torchcodec consigliato per I/O | `scripts.update.ps1 -Check`, diarizzazione reale |
| PySide6 | 6.11.1 da conda-forge | [PySide6 su PyPI](https://pypi.org/project/PySide6/) | richiede Python ≥ 3.10 < 3.16; wheel Windows x86-64 disponibili (necessarie nel runner CI senza conda) | installazione pip nel solo runner CI | `QT_QPA_PLATFORM=offscreen` + suite GUI |
| Runner GitHub | `windows-latest`, Python 3.11 | [immagine Windows Server 2025](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md) | include Inno Setup 6.7.1 e Python 3.11.9; nessuna GPU | i test GPU restano fuori dal gate (`--full` locale) | da verificare al primo run reale |
| PyInstaller | 6.22.3 | [requisiti PyInstaller](https://pyinstaller.org/en/stable/requirements.html) | Windows 8+ supportato | invariato | smoke test `--smoke-test` già nel workflow |

---

## 5. Modifiche applicate, in ordine di priorità

1. **P0** `global_diarization.reconstruct_events(word_labels=…)` + `pipeline.py` passa le
   etichette finali (D1, D6).
2. **P0** `scripts/run_pipeline.py`: rimossi gli argomenti duplicati (D2).
3. **P1** evento di durata zero (D3).
4. **P2** `diarize._overlapping_spans` per voci distinte (D5).
5. **P2** CI e build: `scripts.verify` + PySide6 nel runner (D4).

Nessuna altra modifica funzionale: parametri di trascrizione, euristiche di `merge.py`,
separazione, cattura audio, timeline, dipendenze e aggiornamenti invariati.

---

## 6. Test aggiunti o modificati

- `tests/test_review_persistence.py` (nuovo, 8 test): pipeline completa su fixture ambientale
  con `Transcriber`, `Diarizer` e `align_segments` simulati; la revisione cambia "sì" da A a B
  e la decisione viene seguita in ricostruzione eventi, `Session`, salvataggio, ricaricamento,
  timeline (`TimelineCanvas`: blocco nella corsia B), export TXT/SRT, correzione manuale con
  undo/redo; coerenza fra `transcript_review.changed_words` e contenuto; comportamento senza
  etichette invariato; etichette non supportate o in overlap segnalate e non nascoste; frase non
  spezzata ai cambi di candidato; parola di durata zero modificabile; errore su conteggio
  etichette incoerente. Prima della correzione la fixture produceva `speaker=None` per "sì"
  (replay registrato).
- `tests/test_overlap_semantics.py` (nuovo, 7 test).
- `tests/test_run_pipeline_cli.py` (nuovo, 2 test).
- Nessun test esistente modificato. Suite finale: `python -m scripts.verify` → vedi sezione 7.

---

## 7. Misure prima / dopo (stessi checkpoint ASR + diarizzazione, finestra gold 0–3 600 s)

| configurazione | WER | cpWER | confusione | errore attribuzione parola (errate + non assegnate) | ping-pong |
|---|---|---|---|---|---|
| HEAD `e8d7fce` (merge + review → segmenti) | 9,32 % | 17,90 % | 3,38 % | 4,09 % (4,09 + 0) | 95 / 908 |
| WIP non committato (`reconstruct_events` senza etichette) | 9,34 % | 21,22 % | 8,93 % | 11,23 % (2,15 + 9,09) | 337 / 1 220 |
| **dopo l'audit** (etichette revisionate + attività) | 9,32 % | 17,90 % | 3,37 % | 4,09 % (4,09 + 0) | 71 / 1 090 |
| **dopo l'audit, esecuzione reale completa** (`run_pipeline`, 2 h, retry 0, denoise off) | 9,32 % | 17,90 % | 3,37 % | 4,09 % | 71 / 1 090 |

Il WER (0,02 punti) differisce fra i percorsi solo per l'ordinamento di una parola a cavallo di
eventi; l'uscita reale dell'intera pipeline coincide con il replay offline. La baseline storica di
`docs/baseline.md` (WER 7,86 %, cpWER 16,37 %) usava `float16` e `denoise=true` su un estratto di
60 min: non confrontabile direttamente con queste esecuzioni (`int8_float16`, `denoise=false`).
DER/JER a collar zero non riportati: il manifest non è al livello `human_verified` richiesto da
`verified_evaluation` (gate `BLOCKED_PENDING_HUMAN_VERIFICATION`).

Sull'intera registrazione (7 813 parole, 2 091 turni): eventi testuali 908 → 1 220 → 1 090;
parole non assegnate 0 → 628 → 0; eventi testuali marcati incerti dopo l'audit: 232 (416 parole:
145 in overlap attestato, 87 senza attività della voce assegnata); eventi di attività non
trascritta conservati: 1 145.

---

## 8. Prestazioni e VRAM

Esecuzione reale completa con il codice corretto (`scripts.run_pipeline`, profilo ambientale,
`large-v3 int8_float16`, retry 0, denoise off), campionamento `nvidia-smi` e RSS ogni 5 s:

| misura | valore |
|---|---|
| audio | 5 890,8 s |
| tempo totale (preparazione, ASR, allineamento, diarizzazione, merge, revisione, salvataggio) | 548 s (RTF 0,093) |
| picco memoria GPU usata (intero dispositivo) | 2 418 MiB di 8 188 |
| picco RSS processo | 4 092 MiB |
| esito | 0, sessione 2 235 eventi, 7 813 parole |

Fasi dalla baseline con checkpoint (stessa configurazione, codice pre-audit): ASR 350 s,
allineamento 0,02 s (cross-attention già validata), diarizzazione 189 s. I modelli sono caricati in
sequenza e rilasciati (`close()` → `gc` → `empty_cache`), mai residenti insieme: il picco di
2,4 GB lascia ampio margine sugli 8 GB. Costi CPU del post-processing sul file intero:
`assign_speakers` 0,19 s, `review_labels` 7,06 s, `reconstruct_events` con etichette 1,44 s,
`build_utterances_from_segments` 1,1 s.

---

## 9. Modifiche scartate o non applicate

- Guardia di durata per le "isole" in `_absorb_alternations`: peggiora sul gold (+8 errate,
  +10 segmenti intrappolati) → scartata.
- Disattivare `_smooth` o `_absorb_alternations`: nessun miglioramento o peggioramento misurato
  → non toccati.
- Forzare l'uscita della revisione sui segmenti senza passare dagli eventi canonici: avrebbe
  perso provenienza, candidati e attività simultanea → scartata a favore di `word_labels`.
- Lock-file delle dipendenze o aggiornamenti di torch/ct2/PySide6: nessuna evidenza di difetto,
  cambio non validabile senza build e smoke test completi → solo raccomandazione (sezione 10).
- Estrazione di parti da `main_window.py` (1 979 righe): nessun problema misurabile.

---

## 10. Limiti residui e lavoro da implementare

1. **Riferimento verificato**: il manifest gold è `legacy_gold_unverified` con confini a ~7 s.
   Servono regioni annotate a mano (`reference-review-template.json` in `artifacts/global-timeline`)
   per DER/JER a collar zero e per decidere i casi di interiezioni brevi. Finché manca, nessuna
   modifica alle euristiche di `merge.py` è dimostrabile.
2. **Pipeline senza checkpoint e senza annullamento**: `CheckpointStore` esiste ma
   `process_recording` non lo usa; un crash a 9 minuti su 10 perde tutto e il worker GUI non è
   interrompibile. Lavoro proposto: salvare ASR/allineamento/diarizzazione con `cache_key`
   (già content-addressed) e un `cancel_event` controllato in `transcribe`/`diarize`.
3. **Retry e numero di parole**: aggiungere in `_quality_score` o nell'accettazione del retry una
   guardia sul rapporto di parole (es. rifiutare un retry che perde > 50 % delle parole di un
   segmento non ripetitivo); da validare con A/B su audio con `--quality-retries 24` vs `0`
   (non eseguito: solo retry 0 misurati in questa sessione).
4. **Autocorrezione sugli eventi**: `autocorrect_truncations` (puntini di sospensione, avviso
   `frase_troncata`) agisce sui segmenti di compatibilità e non sugli eventi canonici; da
   riportare come annotazione sugli eventi se la si vuole in timeline ed export.
5. **CI reale**: il workflow corretto non è mai stato eseguito su GitHub; serve un
   `workflow_dispatch` di prova. Possibili attriti: tempo di installazione di torch CPU,
   test che toccano la cache modelli (`test_model_store_safety`).
6. **Riproducibilità**: generare un file di vincoli (`pip freeze` dell'ambiente validato, indice
   cu126) da usare in `setup.ps1`/`update.ps1` come `--constraint`, dopo smoke test pacchettizzato.
7. **Diagnostica di separazione**: aggiungere una misura di separazione fra tracce (es. SI-SDR
   rispetto alla miscela) ai metadati `SeparationResult`, senza cambiare le decisioni.
8. **Memoria**: `coverage` e `mix_sources` caricano l'intero audio in RAM (picco 4,1 GB su 2 h);
   accettabile su 32 GB, da rivedere per file > 4 h con lettura a blocchi.
9. **Vincoli dei rapidi scambi senza punteggiatura**: la regola "catena" collassa alternanze
   1–2 parole con pause < 0,35 s se Whisper omette la punteggiatura; mitigazione possibile solo
   con evidenza acustica per parola (energia per turno) o con dati annotati.
10. **Default `denoise=False`** nel lavoro del ramo: da decidere esplicitamente e documentare
    (README e `docs/baseline.md` parlano di denoise attivo).

---

## File modificati dall'audit

- `src/processing/global_diarization.py` — `reconstruct_events(word_labels=…)`, candidati
  accumulati, durata minima evento.
- `src/processing/pipeline.py` — `final_labels` passate alla ricostruzione.
- `src/processing/diarize.py` — `_overlapping_spans` per voci distinte.
- `scripts/run_pipeline.py` — argomenti duplicati rimossi.
- `scripts/build-windows.ps1`, `.github/workflows/release-windows.yml` — gate `scripts.verify`,
  PySide6 nel runner.
- `tests/test_review_persistence.py`, `tests/test_overlap_semantics.py`,
  `tests/test_run_pipeline_cli.py` — nuovi.
- `docs/audit-2026-11-11-raffinamento-evidenze.md` — questo rapporto.
