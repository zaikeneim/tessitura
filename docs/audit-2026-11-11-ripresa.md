# Tessitura — audit di ripresa, 11 novembre 2026

## Ambito e stato del lavoro

Repository locale: `C:\Users\<utente>\Documents\projects\kisp-like`.
Ramo mantenuto: `audit/evidence-driven-refinement`, HEAD
`e8d7fcec6de9144daa9a9d21375a40a783a8b459`. Le numerose modifiche già in staging
appartengono alle attività precedenti e sono state preservate. Nessun commit,
merge su `main`, push o pubblicazione di release eseguito nella ripresa.

La ripresa conferma le correzioni del precedente audit e aggiunge due patch al
gate di verifica/rilascio. Non cambia motori, pesi, parametri ASR, smoothing,
cattura, separazione o updater. Le misure del precedente rapporto restano
storiche: il manifest che ne supportava WER/cpWER non è reperibile nel checkout
attuale o nel backup ZIP esaminato. Non vengono presentate come misure verificate.

## Architettura e percorso prioritario

GUI PySide6 e CLI chiamano `process_recording`. Il percorso ambientale esegue
preparazione audio, faster-whisper/CTranslate2, allineamento, Community-1,
assegnazione per frase e revisione deterministica. `reconstruct_events` crea
gli eventi canonici; i `Segment` sono una proiezione di questi eventi.

Traccia verificata: `merge.assign_speakers` → `transcript_review.review_labels`
→ `pipeline.final_labels` → `reconstruct_events(word_labels=...)`
→ `utterances_to_segments` → `Session` schema 6 → `storage.save_session`
→ `Session.from_dict` → `TranscriptController` / `TimelineCanvas` → TXT/SRT.
Riferimenti: `src/processing/pipeline.py:713`,
`src/processing/global_diarization.py:40`, `src/models/storage.py:13`,
`src/models/session.py:583`, `src/gui/transcript_controller.py:103`.

Letture e verifiche aggiuntive: `src/audio/{capture,devices}.py`,
`src/processing/{merge,transcribe,diarize,source_transcription,separation,
separation_backends,region_recovery,alignment,overlap_diagnostics,
verified_evaluation,checkpoints}.py`, `src/models/{timeline,transcript_artifacts}.py`,
`src/gui/{speaker_timeline_view,main_window}.py`, README, documenti architetturali,
setup, manifest dipendenze e packaging Windows.

Le sorgenti mic/sistema conservano elaborazione e provenienza distinte.
Le sorgenti ambientali non sono considerate automaticamente monovoce.
La diarizzazione ordinaria rimane autorevole per l'attività simultanea;
l'esclusiva aiuta l'attribuzione delle parole.

GitNexus: repository esplicitamente selezionato come `tessitura`; indice aggiornato
prima delle letture. L'enumerazione globale dei processi segnala limiti di budget:
un flusso assente non prova l'assenza di un chiamante. Contesti e sorgenti sono
stati incrociati. `reconstruct_events` ha rischio **CRITICAL**: 16 simboli a monte,
8 processi; nessuna modifica ulteriore nella ripresa. `scripts.verify.main` ha
rischio LOW, un riferimento diretto. Workflow YAML e classe di test restituiscono
UNKNOWN: controllati anche i comandi effettivi, le dipendenze dei job e la raccolta
dinamica dei test. UNKNOWN non è stato interpretato come assenza di impatto.

`detect_changes --scope all` finale rileva 132 file, 1.227 simboli, 261 processi e rischio
CRITICAL includendo tutto il lavoro precedente; il primo risultato era troncato.
Rieseguito su 11 gruppi isolati: tutti i 132 file coperti senza `partial` o
`truncated`. Il confronto isolato con lo staging trovato all'avvio contiene
soltanto i sei file della ripresa, 19 simboli, rischio LOW, risultato completo.
Il contenuto dello staging reale è rimasto identico; i confronti usano indici
temporanei. I risultati del grafo non sostituiscono i test né risolvono la
precedente limitazione di enumerazione globale dei flussi.

## Difetti confermati e decisioni

| Priorità | Difetto e prova | Esito |
|---|---|---|
| P0, precedente | La ricostruzione senza etichette perdeva la decisione del reviewer (`pipeline.py:748`, `global_diarization.py:40`). La fixture cambia una parola da A a B: senza `word_labels`, metadati `changed_words=1` ma speaker finale nullo. | Correzione precedente confermata; la stessa fixture conserva B in eventi, JSON, reload, corsia timeline e export. Provenienza e attività simultanea restano disponibili. |
| P2, precedente | `unittest discover` omette i test a funzione, incluso `tests/test_merge.py`. | CI e build usano già `python -m scripts.verify`; aggiunto test che forza il fallimento di quel modulo e controlla exit code 1. |
| P2, nuovo | `scripts/verify.py:62` invocava `runtime_health.py verify`, ma il parser richiede `--snapshot` o `--verify FILE`. Baseline `--full`: 1 passo fallito su 72, exit code 2 del sottocomando. | Snapshot temporaneo e successiva verifica offline con l'interfaccia esistente; pulizia del file alla fine. Lo snapshot è un controllo operativo, non un lock o una baseline di accuratezza. |
| P2, nuovo | `.github/workflows/release-windows.yml:80`: il job build installava requirements e PyInstaller senza PySide6. I job hanno ambienti separati; l'installazione nel job verify non raggiunge il builder. | Aggiunto PySide6 anche al builder. Test YAML su entrambi i job, comando effettivo del gate e dipendenze di pubblicazione. Nessuna versione locale aggiornata. |
| P2, precedente | `_overlapping_spans`, `diarize.py:85`, contava intervalli invece di voci distinte. | Confermati i sette test: voci diverse, adiacenze, triplo parlato, stessa voce, duplicati/frammenti e input degenere. |

I due nuovi casi di regressione fallivano prima delle patch e passano dopo.
La correzione CLI per argomenti duplicati e la durata minima degli eventi,
già presenti, passano rispettivamente nei test CLI e di persistenza.

## Baseline e confronto riprodotto

`python` nei comandi indica `env\python.exe`. Prima delle patch:
`python -m scripts.verify`: **70 passi superati**.
Il primo `--full` dimostra l'errore del parser sopra descritto.

Dopo le patch mantenute: **23 test mirati superati**; `python -m scripts.verify --full`:
**74 passi superati**, compresi diagnostica ambiente, snapshot e smoke offline
ASR/decoder/pyannote/Qt. È stata selezionata esplicitamente la cache completa
`C:\Users\<utente>\.cache\huggingface\hub`, senza download: quella configurata nel
progetto contiene tiny ma non Community-1. Il fallimento iniziale per cache
incompleta è conservato nel log; non è stato nascosto o saltato.
Una verifica degli hash Git conferma che i quattro file di codice/test mantenuti
coincidono con lo snapshot della prima esecuzione completa riuscita. Anche
`tessitura.spec` coincide con quello snapshot dopo il tentativo scartato sotto.

Replay degli stessi checkpoint integrali ASR/allineamento/diarizzazione:

| Misura strutturale | Senza etichette revisionate | Con correzione precedente |
|---|---:|---:|
| Parole ASR | 7.813 | 7.813 |
| Eventi testuali | 1.220 | 1.090 |
| Parole non assegnate | 628 | 0 |
| Eventi testuali incerti | 387 | 232 |
| Eventi totali, attività senza testo inclusa | 2.651 | 2.235 |
| Identità del multinsieme parola + timestamp | sì | sì |

Questi numeri dimostrano conservazione delle parole e passaggio delle decisioni;
zero parole non assegnate non dimostra zero errori di attribuzione.
Sul file reale la revisione seleziona baseline, cambia zero parole e segnala
71 parole ambigue. Il caso con una decisione diversa è coperto dalla fixture.

Il template disponibile contiene 26 regioni, `verification_level=unverified_candidate`,
`annotation_complete=false`, nessun revisore. `evaluate_verified_reference` lo
rifiuta correttamente. WER, DER, JER, errori di parola attribuita e omissioni:
**non misurabili con un riferimento verificato disponibile**.

## Registrazione completa e prestazioni

Sorgente: `C:\Users\<utente>\Downloads\260824_2105.mp3`, 141.392.583 byte,
SHA-256 `7f3653af3608130a805a447726a3f74347438efa8382fb904fe8f0ad9d823f12`.
Durata reale circa 5.890,8 s, ossia 98 minuti; la denominazione “due ore” è
approssimativa. Il sorgente resta immutato.

La nuova esecuzione integrale è terminata con exit code 0. Configurazione:
ambientale, standard, italiano, large-v3, CUDA `int8_float16`, batch 1,
conteggio voci automatico, denoise off, retry 0, revisione safe.

| Misura del run integrale | Risultato |
|---|---:|
| Tempo del comando CLI, preparazione e salvataggio inclusi | 561,50 s |
| Rapporto tempo di elaborazione / durata audio | 0,0953 |
| Massimo RSS campionato del processo | 3.609,97 MiB |
| Massimo memoria GPU campionata, intero dispositivo | 2.424 MiB su 8.188 MiB |
| Intervallo obiettivo del campionamento | 2,5 s |
| Eventi / eventi testuali / parole | 2.235 / 1.090 / 7.813 |
| Parole, timestamp e speaker identici al replay corretto | sì, confronto esatto |
| SHA-256 MP3 prima e dopo identico | sì |

Le misure campionate non rappresentano un picco istantaneo garantito; la memoria
GPU comprende gli altri processi del dispositivo. Il supervisor può aggiungere
fino a un intervallo di campionamento al tempo riportato. Il precedente harness
di baseline registrava 574,9 s, ma condizioni di cache e carico non sono un A/B
controllato: non si attribuisce alcuna accelerazione alle patch dei gate.
Durante parte del run erano attive anche verifiche CPU e build diagnostiche:
il tempo riportato non è un benchmark in condizioni di carico isolato.
Il file è stato elaborato per intero; non è stato ascoltato e annotato per intero.
Prove: `artifacts/audit-resume-20261111/full-run.json` e `full-pipeline.log`.

Sulla sessione reale appena generata passa anche `main.py --timeline-smoke` in
ambiente sorgente: 7.813 parole conservate, due eventi simultanei a 99 s,
controller collegato agli eventi canonici, correzione su una copia, salvataggio,
reload, undo/redo ed export. Il JSON originale di elaborazione non è sostituito.

### Build pacchettizzata e modifica scartata

PyInstaller 6.22.3 completa la build `LightBundle`; il relativo `--smoke-test`
restituisce exit code 0 e `frozen=true`. Lo smoke di base controlla percorsi e
migrazione, ma non importa la GUI. **Lo smoke timeline del bundle leggero fallisce**:
una build console diagnostica produce `ModuleNotFoundError: No module named 'scipy'`,
attraverso `MainWindow` → `pipeline` → `audio_utils` → `scipy.signal`.
Il plugin Computer Use non autorizzava la lettura del dialogo di Tessitura;
il traceback è stato quindi raccolto dalla build console, senza usare altre API UI.

Tentativo minimo: rimuovere SciPy dalla lista di esclusione del solo bundle leggero.
Il test mirato sulla configurazione passa, ma la nuova build apre ancora un
dialogo di eccezione e non completa lo smoke timeline. **Tentativo e relativo test
di configurazione scartati e ripristinati**, perché insufficienti alla prova reale.
I processi diagnostici rimasti sul dialogo sono stati chiusi.

Classificazione: limite P2 del percorso di build leggero usato per diagnostica;
non è dimostrato lo stesso difetto nella release completa, che già include SciPy.
Non si dichiara riuscito lo smoke GUI del pacchetto né l'inferenza congelata.
Il packaging completo e l'updater restano invariati. Log:
`packaging.log`, `packaged-smoke.json`, `packaged-timeline-traceback.log`,
`packaging-fixed.log`, `packaged-fixed-results.json` nella cartella delle prove.

Profilo timeline offscreen su 2.235 eventi: reload 0,370 s; layout + render,
mediana di dieci prove, 5,13 ms sulla finestra di 60 s (12 blocchi), 83,95 ms
sull'intero file (2.235 blocchi). Tutti gli ID sopravvivono al reload.
Non emerge una ragione misurata per estrarre o riscrivere `main_window.py`.

Forma d'onda: scansione di tutti i campioni stereo nativi in 17,49 s; reload
della cache in 14,64 ms. Su 58 finestre distribuite lungo il file, lettura di
1.024 bin in mediana 3,68 ms; layout + forma d'onda + render durante lo scorrimento
8,67 ms in mediana, massimo osservato 35,07 ms. Sono prove Qt offscreen, non una
sessione manuale di ascolto o una misura della latenza del dispositivo audio.

Benchmark CPU esistente, `python -m scripts.benchmark_pipeline`: risultati esatti
identici al riferimento interno; matching 0,561 s contro 0,0153 s; preparazione
audio 60 s, 0,304 s contro 0,162 s. Sono confronti fra implementazioni già
esistenti nello script, **non accelerazioni prodotte dalle nuove patch**.
L'artefatto originale `artifacts/performance.json` è stato ripristinato.

## Questioni investigate senza modifica

- **Smoothing:** test esistenti coprono isole, pause, frasi chiuse, interiezioni
  in overlap, catene e tre voci. Le catene di due voci possono assorbire scambi
  rapidi, anche in overlap. Senza annotazioni adeguate non è dimostrato che una
  modifica migliori il risultato senza riframmentare le frasi.
- **Retry/copertura:** conservati VAD 0,35/0,25, padding 600 ms, silenzio 400 ms,
  temperature e controllo allucinazioni. Passano deduplicazione, clipping,
  contesto, recupero ai confini e rifiuto del silenzio. La funzione di qualità
  può favorire un candidato più corto: il run integrale usa retry 0 e non prova
  l'equivalenza con il default 24. Nessuna nuova soglia arbitraria introdotta.
- **Separazione:** correlazione, silenzio e livello della somma sono guardie
  conservative, non prove di parole perse o qualità ASR. Fase, scala e ritardo
  possono alterarle. Restano fallback originale e candidati da approvare;
  nessuna accettazione basata sul numero di parole. SI-SDR o correlazione con
  riallineamento restano diagnostiche P3 da misurare, non criteri aggiunti.
- **Recording e timeline:** passano test sorgenti ambientali/multivoce,
  cuffie, altoparlanti, echo e double-talk, offset, drift sintetico su due ore,
  annullamento locale, correzioni, undo/redo e reprocessing mirato. Non è stata
  svolta una nuova cattura reale di due ore in ciascuna modalità hardware.
- **Affidabilità lunga:** modelli sequenziali, originale preservato, salvataggio
  JSON atomico. Il checkpoint store non è integrato nella pipeline GUI e il
  worker non offre annullamento globale: limiti da affrontare con prove di
  crash/ripresa, senza ampliare questa patch. Non è dimostrata la ripresa della
  pipeline completa dopo un crash.

## Runtime, ricerca e riproducibilità

Windows x64, i7-13700HX, RTX 4070 Laptop 8 GB, 32 GB RAM. Runtime rilevato:
Python 3.11.16; torch 2.13.0+cu126; torchaudio 2.11.0+cu126; torchcodec 0.14.0;
CUDA 12.6; cuDNN 9.10.02; CTranslate2 4.8.1; faster-whisper 1.2.1;
pyannote.audio 4.0.7; pyannote.metrics 4.1; PySide6 6.11.1; PyInstaller 6.22.3.
Revisioni locali: faster-whisper-large-v3
`edaa852ec7e145841d8ffdb056a99866b5f0a478`; Community-1
`3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.

| Implementazione / decisione | Fonte ufficiale e motivo | Beneficio, rischio e verifica |
|---|---|---|
| Conservare CUDA 12/cuDNN 9 e ASR esistente | [faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu), [CTranslate2 4.5](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.5.0) documentano il requisito recente cuDNN 9. | Evita incompatibilità; nessun upgrade. Smoke e run locale verificano la configurazione effettiva. |
| Esclusiva per allineamento, ordinaria per overlap | [Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1#exclusive-speaker-diarization) espone entrambe. | Conserva contemporaneità e provenienza; test di overlap e persistenza. |
| Rifiutare metriche senza riferimento verificato | [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/reference.html) distingue miss, falso allarme, confusione e JER, con collar/overlap espliciti. | Evita confronti ingannevoli; il gate delle annotazioni resta bloccante. |
| Non intensificare denoise/separazione automaticamente | [Audacity](https://manual.audacityteam.org/man/noise_reduction.html) descrive limiti e distorsioni nella riduzione del rumore. | Protegge parlato debole; richiede A/B su campioni verificati prima di cambiare parametri. |
| Mantenere torch/torchaudio installati | [Compatibilità TorchAudio](https://docs.pytorch.org/audio/stable/installation.html): 2.11 supporta torch 2.11 e successivi. | La coppia rilevata è supportata; test operativi passano. Nessuna migrazione. |
| PySide6 in entrambi i job CI | [Qt/PyInstaller](https://doc.qt.io/qtforpython-6.11/deployment/deployment-pyinstaller.html), [PyInstaller](https://pyinstaller.org/en/stable/hooks-config.html) richiedono un ambiente con i moduli da raccogliere. | Builder autonomo; test YAML e packaging locale. Nessuna reinstallazione pip nel Conda dell'utente. |
| Conservare update/debounce della timeline | [QWidget.update](https://doc.qt.io/qt-6.11/qwidget.html#update) pianifica e combina i ridisegni. | Nessun passaggio indiscriminato a repaint sincrono; profilo offscreen sopra. |
| Rilascio dipendente dal gate | [GitHub Actions needs](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-jobs) propaga fallimenti ai job dipendenti. | Build dipende da verify; publish da verify/build. Test CPU senza `--full`; nessun rilascio con check obbligatori falliti. |

`requirements.txt` conserva minimi ampi; `environment.yml` vincola Python/ffmpeg
e usa Qt da conda-forge. La riproducibilità completa richiede un ambiente risolto
Conda + vincoli pip + revisioni dei modelli, seguito da build e smoke di quel
preciso ambiente. Generato solo un inventario diagnostico locale con l'interfaccia
esistente `runtime_health.py --constraints`; non promosso a lock universale.

## File modificati nella ripresa e prove locali

- `.github/workflows/release-windows.yml`: PySide6 nel builder.
- `scripts/verify.py`: snapshot temporaneo e invocazione CLI valida.
- `tests/test_packaging_and_installer.py`: dipendenze e gate fra job.
- `tests/test_verify_gate.py`: nuovo; modulo a funzioni obbligatorio e CLI reale.
- `docs/audit-2026-11-11-raffinamento-evidenze.md`: chiarimento sul ramo e sulle misure storiche.
- `docs/audit-2026-11-11-ripresa.md`: questo rapporto.

Log, JSON di confronto, inventario ambiente, campioni di memoria e output privati
restano nella cartella ignorata `artifacts/audit-resume-20261111/`.
I file sorgente modificati dagli agenti precedenti sono elencati nel precedente
rapporto; non vanno confusi con i sei file di questa ripresa.

Restano da attestare un run reale GitHub Actions, accuratezza su annotazioni umane
complete, installazione/disinstallazione del pacchetto completo e comportamento
dei retry su audio verificato. Nessuna sostituzione di motori o ristrutturazione
estesa è giustificata dalle prove raccolte.
