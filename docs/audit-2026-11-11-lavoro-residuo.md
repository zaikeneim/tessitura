# Tessitura — lavoro residuo dopo l'audit (stato al 12 novembre 2026)

Elenco unico di ciò che resta da implementare o da attestare, raccolto da
[audit guidato dalle evidenze](audit-2026-11-11-raffinamento-evidenze.md),
[ripresa dell'audit](audit-2026-11-11-ripresa.md) e
[avanzamento delle fasi](PROCESSING_PROGRESS.md). Ogni voce indica perché serve,
da dove partire e come si verifica. Nessuna voce è stata implementata: sono
state lasciate fuori perché mancava l'evidenza per farle senza rischio.

Priorità: **P1** incide su correttezza o sulla possibilità di misurare;
**P2** affidabilità, rilascio, manutenzione; **P3** miglioramento facoltativo.

## Cosa è già fatto (per non rifarlo)

- Le etichette della revisione arrivano agli eventi canonici
  (`reconstruct_events(word_labels=…)`), con incertezza e candidati conservati.
- CLI `scripts.run_pipeline` di nuovo utilizzabile (argomenti duplicati rimossi).
- Evento di durata zero reso modificabile.
- `_overlapping_spans` conta voci distinte.
- Gate `scripts.verify` in CI e nello script di build, PySide6 in entrambi i job,
  `--full` con invocazione valida di `runtime_health`.
- Avanzamento per fasi (`on_stage`, pannello «Fasi di elaborazione»).

## A. Misurare l'accuratezza (blocca tutte le modifiche alle euristiche)

| # | P | Cosa manca | Da dove partire | Verifica |
|---|---|---|---|---|
| A1 | P1 | Riferimento annotato a mano. Il template ha 26 regioni `unverified_candidate`, senza revisore; `evaluate_verified_reference` lo rifiuta correttamente. Senza di esso DER, JER e WER in sovrapposizione non sono misurabili. | `artifacts/global-timeline/reference-review-template.json`; compilare eventi per voce con tempi assoluti, `verification_level="human_verified"`, `reviewed_by`, `annotation_complete=true`. | `src/processing/verified_evaluation.py` restituisce le metriche invece di rifiutare. |
| A2 | P1 | Il manifest gold dei primi 60 minuti non è nel checkout: `data/evaluation/` è vuota, la copia sta solo nel backup locale dei dati. Le misure WER/cpWER dell'audit dipendono da quel file. | Ripristinare manifest, RTTM e STM in `data/evaluation/` (cartella ignorata da Git). | `python -m scripts.evaluate_sessions <manifest>` mostra `turns: 501`. |
| A3 | P2 | `scripts/evaluate_sessions.py` confronta l'intera sessione con un riferimento di 60 minuti e conta come parole il testo segnaposto degli eventi non trascritti. Sulle sessioni schema 6 il WER risulta gonfiato. L'audit ha usato una variante con finestra e senza segnaposto, rimasta fuori dal repository. | Aggiungere a `evaluate_sessions.py` il taglio sulla finestra del manifest, l'esclusione dei segnaposto e delle parole non assegnate dal testo per voce, e l'errore di attribuzione per parola. Copia degli script usati in `artifacts/audit-20261111/` (`eval_window.py`). | Stessi numeri del rapporto: WER 9,32 %, cpWER 17,90 %, attribuzione errata 4,09 %. |
| A4 | P2 | Replay offline dei percorsi di attribuzione dai checkpoint ASR/diarizzazione (10 s, senza GPU) esiste solo come script esterno. È lo strumento che ha trovato il difetto principale. | Portare `replay_paths.py` / `replay_fixed.py` / `ab_heuristics.py` (in `artifacts/audit-20261111/`) in `scripts/`, con percorsi come argomenti. | Il replay riproduce 1 090 eventi testuali, 0 parole non assegnate, 71 segmenti intrappolati. |

## B. Affidabilità sulle registrazioni lunghe

| # | P | Cosa manca | Da dove partire | Verifica |
|---|---|---|---|---|
| B1 | P2 | `process_recording` non usa i checkpoint: un errore a fine diarizzazione perde anche i 6 minuti di ASR. `CheckpointStore` e `cache_key` esistono e sono usati solo da test e harness. | Salvare e riusare le fasi `asr`, `alignment`, `diarization` in `src/processing/pipeline.py` con la chiave già content-addressed (audio, opzioni, revisioni modello). | Test: errore simulato dopo l'ASR, seconda esecuzione senza richiamare `Transcriber`; chiave diversa al cambio di un'opzione. |
| B2 | P2 | Nessun annullamento globale dell'elaborazione dalla GUI; la coda si ferma solo dopo il file in corso. | `src/processing/resources.py` ha già `check_cancel`; passare un `cancel_event` a `transcribe` (tra i segmenti del generatore) e al hook di pyannote, esporlo in `ProcessWorker`. | Test: annullamento a metà ASR rilascia il modello e non emette una sessione. |
| B3 | P3 | Memoria: copertura vocale e mix caricano tutto l'audio in RAM (picco misurato 4,1 GB su 98 minuti). Accettabile con 32 GB, non verificato oltre le 4 ore. | `coverage.detect_speech_activity` e `audio_utils.mix_sources` a blocchi. | Misura RSS su un file sintetico di 4 ore. |
| B4 | P3 | `review_labels` impiega 7 s sul file intero perché `_agreement` scorre tutti i turni per ogni parola. | Ricerca per finestra come in `assign_speakers` (`bisect`). | Etichette e punteggi identici, tempo < 1 s. |

## C. Trascrizione

| # | P | Cosa manca | Da dove partire | Verifica |
|---|---|---|---|---|
| C1 | P2 | Il comportamento con i retry attivi (default 24) non è stato misurato: tutte le esecuzioni integrali dell'audit usano `--quality-retries 0`. | Due esecuzioni integrali, retry 0 e 24, sugli stessi modelli. | Confronto con A3: nessuna parola persa, WER non peggiore. |
| C2 | P3 | `_quality_score` non considera il numero di parole: un retry più corto con logprob migliore può sostituire un segmento più lungo. | `src/processing/transcribe.py`, accettazione del retry: rifiutare se il retry perde oltre metà delle parole di un segmento non ripetitivo. | A/B su C1 più test unitario con segmento finto. |
| C3 | P3 | `autocorrect_truncations` (puntini di sospensione, avviso `frase_troncata`) agisce sui segmenti di compatibilità e non raggiunge gli eventi canonici, quindi non si vede in timeline ed export. | Riportarlo come annotazione sull'evento in `reconstruct_events` o dopo, senza modificare il testo riconosciuto. | Test di persistenza: avviso presente dopo salvataggio e ricaricamento. |

## D. Attribuzione delle voci

| # | P | Cosa manca | Da dove partire | Verifica |
|---|---|---|---|---|
| D1 | P1 | Con due sole voci, un'interruzione fino a 3 parole senza pausa né punteggiatura viene assorbita dalla voce circostante; uno scambio rapido senza punteggiatura con pause < 0,35 s collassa su una voce. Sul gold disponibile la regola è positiva (9 parole giuste contro 1), ma il gold non distingue le interiezioni brevi. | Dopo A1: evidenza acustica per parola (energia per turno o embedding breve) prima di assorbire. La guardia di durata semplice è già stata provata e scartata. | Meno errori sulle regioni annotate senza aumentare i segmenti intrappolati (`scripts.measure_fragmentation`). |
| D2 | P2 | 232 eventi testuali restano marcati incerti sul file reale (145 in sovrapposizione attestata, 87 senza attività della voce assegnata). L'interfaccia li mostra, ma non c'è un flusso di revisione dedicato. | Filtro «da verificare» in `src/gui/speaker_timeline_view.py` sui valori di `diarization_provenance.assignment`. | Test GUI offscreen: il filtro elenca solo quegli eventi. |

## E. Separazione delle voci

| # | P | Cosa manca | Da dove partire | Verifica |
|---|---|---|---|---|
| E1 | P3 | Le tre guardie (correlazione, silenzio, caduta di livello) non misurano la qualità della separazione; fase, scala e ritardo possono alterarle. | Aggiungere ai metadati di `SeparationResult` una misura diagnostica (SI-SDR rispetto alla miscela, correlazione dopo riallineamento) senza cambiare le decisioni. | Test con due sinusoidi e con una traccia duplicata ritardata. |

## F. Pacchetto, CI e riproducibilità

| # | P | Cosa manca | Da dove partire | Verifica |
|---|---|---|---|---|
| F1 | P2 | Il workflow di rilascio corretto non è mai stato eseguito su GitHub Actions. | `workflow_dispatch` di prova con una versione beta. | Job verify, build e smoke verdi; nessuna pubblicazione senza tag. |
| F2 | P2 | Lo smoke timeline del bundle leggero fallisce: `ModuleNotFoundError: scipy` dal percorso `MainWindow` → `pipeline` → `audio_utils`. Togliere SciPy dalle esclusioni non è bastato; il tentativo è stato ripristinato. La release completa include SciPy ma non è stata provata allo stesso modo. | `packaging/windows/tessitura.spec`, build console per leggere il traceback successivo. | `--timeline-smoke` con exit code 0 sul bundle. |
| F3 | P2 | Installazione e disinstallazione del pacchetto completo e inferenza nell'eseguibile congelato non attestate in locale. | Build completa, installazione silenziosa in una cartella temporanea, `--smoke-test`, disinstallazione. | Come nel job CI, eseguito in locale. |
| F4 | P3 | Dipendenze con soli minimi: nessun ambiente risolto. | File di vincoli generato dall'ambiente validato (`runtime_health.py --constraints`), usato con `--constraint` in `setup.ps1` e `update.ps1`, indice cu126. | `scripts.verify --full` e smoke del pacchetto sull'ambiente ricreato da zero. |

## G. Decisioni di prodotto da prendere

| # | P | Decisione | Nota |
|---|---|---|---|
| G1 | P2 | `PipelineOptions.denoise` è passato a `False` nel lavoro sulla timeline globale, mentre README e `docs/baseline.md` descrivono il denoise attivo. | Scegliere il default e allineare documentazione e GUI; la baseline storica (WER 7,86 %) usava denoise attivo e `float16`. |
| G2 | P3 | `transcript_review` e «Trascrizione grezza» ora confrontano uscita del merge e uscita revisionata; sul file reale coincidono. | Valutare se mostrare il pulsante solo quando `changed_words > 0`. |

## H. Registrazione

| # | P | Cosa manca | Verifica |
|---|---|---|---|
| H1 | P2 | Nessuna cattura reale lunga in ciascuna modalità (ambientale, cuffie, altoparlanti, Bluetooth): le prove sono sintetiche o brevi. | Una registrazione di almeno 30 minuti per modalità, controllo di offset, drift e sidecar con `docs/RECORDING_VALIDATION.md`. |

## I. Repository

| # | P | Cosa manca | Nota |
|---|---|---|---|
| I1 | P2 | Il repository di lavoro non ha un ramo `main`: il lavoro è sul ramo di audit. `main` esiste solo nella copia pubblicabile. | Decidere se creare `main` anche nel repository di lavoro o continuare con rami per funzionalità. |
| I2 | P2 | Il commit pubblicato della copia pubblicabile contiene ancora collegamenti locali assoluti in `docs/architecture-target.md`; i commit locali successivi li rendono relativi, ma non sono stati inviati. | `git push` manuale dalla copia pubblicabile dopo la revisione. |
| I3 | P3 | Analisi GitNexus nella copia pubblicabile prima del commit, come richiesto da `AGENTS.md`. | Rigenerare l'indice in quella cartella. |

## Ordine consigliato

1. A2, A3, A4: rendono ripetibile ciò che l'audit ha misurato a mano.
2. A1: senza annotazioni verificate D1 e C2 restano ipotesi.
3. B1, B2: riducono il costo di un errore su file lunghi.
4. F1, F2, F3: prima del primo rilascio pubblico.
5. C1 e G1: decidono la configurazione predefinita con dati.
