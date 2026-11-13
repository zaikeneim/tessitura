# Tessitura — recupero selettivo delle voci e contatore dei profili

Rapporto tecnico del 13 novembre 2026. Il recupero è opzionale e produce proposte da verificare: non applica automaticamente testo o identità alla timeline. Il contatore dei profili è disponibile anche con il recupero disattivato.

**Stato della validazione:** la funzionalità e le protezioni vengono verificate con test di regressione e con audio reale. Non è dimostrato un miglioramento dell'accuratezza. L'utente ha confermato l'assenza di una trascrizione umanamente verificata; sono preparati materiali per la revisione. Il criterio di accettazione relativo alla riduzione misurata delle omissioni resta aperto.

## Flusso esistente e motivi delle fasi saltate

| Percorso | Esecuzione effettiva e riferimenti |
|---|---|
| Standard | [`process_recording`](../src/processing/pipeline.py) valida il profilo, prepara un mix mono normalizzato a 16 kHz, esegue ASR, allineamento, diarizzazione, attribuzione, revisione deterministica e ricostruzione degli eventi. Whisper e pyannote ricevono il mix; «raw» per la diarizzazione significa senza denoise, non necessariamente i byte originali del file importato. |
| Riduzione rumore | `PipelineOptions.denoise=False` e la casella GUI inizialmente disattivata sono scelte operative. [`denoise.py`](../src/processing/denoise.py) usa spectral gating conservativo e un ripiego se perde energia; il gating può comunque danneggiare parole deboli e timbro. Non è stato riattivato per impostazione predefinita. |
| Selezione strategia | [`MainWindow._build_options`](../src/gui/main_window.py) inoltra la strategia scelta per la registrazione; l'importazione usa `standard`. Modalità cuffie/altoparlanti e sorgenti mic+sistema possono attivare `source_aware`. `profile_first` resta un percorso esplicito delle opzioni di pipeline. |
| Profile-first | La diarizzazione precede `build_voice_profiles`. Con più voci, [`extract_speaker_tracks`](../src/processing/speaker_tracks.py) può mascherare il mix secondo i turni e restituire `status="sliced"`. Gli intervalli sovrapposti contengono ancora la miscela. |
| Speaker-isolated | La strategia esplicita tenta il backend ONNX sull'intero mix. Solo un risultato acustico accettato diventa `status="separated"`; il fallimento conserva il percorso standard. Non è la modalità predefinita. |
| Sorgenti note | [`transcribe_sources`](../src/processing/source_transcription.py) trascrive mic e sistema separatamente, conserva sincronizzazione, provenienza e attività concorrente. Salta intenzionalmente il denoise anche quando richiesto. L'eventuale cancellazione eco resta un derivato candidato e non sostituisce l'ASR primario del microfono. |
| Recupero precedente | [`RegionRecoveryWorker`](../src/gui/region_worker.py) e i pulsanti della timeline eseguivano `recover_region` solo su richiesta manuale. I retry di Whisper e il recupero della copertura già esistenti non erano una seconda passata acustica automatica sui parlanti simultanei. |
| Timeline | [`reconstruct_events`](../src/processing/global_diarization.py) conserva parole, etichette revisionate, incertezza e attività senza testo. Le utterance canoniche possono sovrapporsi; i segmenti sono una proiezione di compatibilità. |

La fase «separazione saltata» nel percorso standard descrive quindi una decisione reale. Le descrizioni aggiornate distinguono esplicitamente il ritaglio per turni dalla separazione acustica. `SpeakerTracksResult.separated` resta una proprietà di instradamento compatibile con i chiamanti precedenti; per l'esito acustico si usa `acoustic`, oltre a `status` e ai metadati protetti dalle sovrascritture diagnostiche.

Il precedente confronto globale è conservato in [benchmark C, §9.05](plans/2026-09-21-gitnexus-plan-trascrizione-conversazione-timeline.md). Non aveva giustificato la promozione di MossFormer2 o SepFormer al posto del percorso standard. Quelle misure storiche non sono nuove misure di questo intervento; inoltre il loro riferimento ASR non sostituisce un riferimento umano completo.

## Seconda passata opzionale

La voce **Recupero avanzato delle voci**, nelle opzioni avanzate, offre `Disattivato`, `Adattivo` e `Sperimentale`. Il valore iniziale rimane `off`. Tutti i rami della pipeline terminano prima la loro trascrizione e costruiscono la sessione di base; [`_recover_session`](../src/processing/pipeline.py) richiama poi [`recover_session_regions`](../src/processing/region_recovery.py).

Per una sorgente ambientale singola senza offset si usa il file originale, conservando il contesto nei canali e nel sample rate nativi. Nei percorsi con sorgenti sincronizzate o offset si usa il mix di elaborazione allineato alla timeline: la semantica dei canali è già stata dichiarata e gli originali restano nel manifest di cattura. Questa seconda passata sul mix non equivale a un nuovo trattamento indipendente di ciascun microfono.

`plan_recovery` esegue una scansione a blocchi di otto secondi. Combina turni non esclusivi, attività energetica indipendente dall'ASR, copertura delle parole, eventi senza testo, probabilità basse, attribuzioni incerte e cambi rapidi di voce. Classifica separatamente sovrapposizioni a due o almeno tre candidati, possibili omissioni, parlato debole, rumore e coda riverberante. Attività energetica, stima SNR e riverbero sono indicatori non calibrati: non provano presenza di parole, distanza fisica o RT60.

La coda ordinaria contiene al massimo otto regioni, con un budget di 120 secondi di audio comprensivo del contesto. Una regione dura al massimo 18 secondi, con due secondi di contesto per lato. Episodi vicini vengono accorpati solo con lo stesso insieme di voci e senza superare il limite. La selezione distribuisce le regioni sulla registrazione e riserva fino a due controlli a basso rischio, quando disponibili. Anche questi controlli richiedono verifica umana.

La passata confronta, per ciascuna regione selezionata:

1. Audio originale decodificato, con copia lossless del contesto nativo.
2. Normalizzazione Tessitura della stessa finestra.
3. Gain statico conservativo, quando richiesto dal problema e dal livello rilevato.
4. Separazione acustica della miscela nelle regioni sovrapposte.
5. Gain seguito da separazione, quando il gain è effettivamente applicabile.

Le viste sono WAV float a 16 kHz; il contesto nativo resta separato. Il gain è limitato a 12 dB, con limitazione morbida dei picchi: non introduce gate né guadagno variabile nel tempo. I livelli, l'attività, gli indizi di rumore, il centroide spettrale, il rapporto delle alte frequenze e la coda energetica vengono registrati anche per le viste elaborate. Non sono criteri di promozione qualitativa.

La separazione avviene prima di caricare l'ASR della regione. Ogni sorgente accettata viene trascritta indipendentemente con lo stesso snapshot ASR, lingua e prompt delle altre viste del confronto. Il modello viene chiuso prima della regione successiva. Questo limita la contemporaneità dei motori pesanti; il costo di caricamento per regione è una scelta conservativa ancora ottimizzabile.

## Validazione, identità e revisione

Le protezioni esistenti della separazione restano attive: duplicazione delle forme d'onda, silenzio, collasso energetico e perdita di livello. Si aggiungono diagnostici di attività e correlazione degli inviluppi, provenienza del backend e checksum dei pesi. Output non finiti o con durata incompatibile vengono respinti. Questi controlli non dimostrano conservazione delle parole.

Le predizioni complete rimangono nei checkpoint, separate per strategia e sorgente. Le proposte automatiche escludono testo riconosciuto come duplicato, parole di solo contesto e predizioni ASR degeneri. Conservano evidenza dei possibili duplicati e delle frasi ripetute fra tracce; sequenze duplicate di almeno tre parole fra sorgenti indipendenti vengono respinte come possibile leakage. Una separazione a due uscite con almeno tre voci candidate resta esplicitamente incompleta.

Ogni proposta contiene hash dell'audio sorgente, tempi assoluti, configurazione di enhancement, modello e checksum della separazione, snapshot ASR, indice di sorgente, voci candidate, controlli di duplicazione, stato di validazione e tempo di elaborazione. L'indice di uscita del separatore non diventa un'identità. La fase di riconciliazione segnala l'indisponibilità di un'identificazione acustica verificata delle nuove tracce: i suggerimenti derivano dall'attività, la voce rimane non assegnata fino alla revisione.

Il confronto audio nella timeline permette di ascoltare contesto originale, normalizzazione, gain e sorgenti separate usando gli offset assoluti salvati. Il player differisce il seek se la durata del nuovo media non è ancora nota. L'accettazione aggiunge un evento distinto; il rifiuto cambia solo lo stato della proposta. Sono supportati undo/redo, salvataggio, riapertura e protezione della vista grezza in sola lettura. Un'alternativa identica già accettata per la stessa voce e regione non può essere aggiunta nuovamente.

Le correzioni dell'utente e le utterance preesistenti non vengono sostituite. Le euristiche di smoothing del primo passaggio non sono state modificate senza un riferimento verificato: la loro capacità di assorbire interventi brevi resta un limite noto. La seconda passata rende disponibili proposte indipendenti per la revisione di quei casi.

## Modelli e provider

| Componente | Verifica locale e decisione |
|---|---|
| MossFormer2 SS 16K ONNX | Presente, checksum del manifest verificato, input mono 16 kHz, due uscite; l'export installato limita le finestre a tre secondi. Licenza dichiarata Apache-2.0 nella [model card](https://huggingface.co/alibabasglab/MossFormer2_SS_16K). Valutato sulle regioni del file reale. |
| SepFormer WHAMR16K ONNX | Presente e verificato, mono 16 kHz, due sorgenti; modello destinato anche a miscele rumorose e riverberanti. Licenza dichiarata Apache-2.0 nella [model card](https://huggingface.co/speechbrain/sepformer-whamr16k). Valutato separatamente, con le stesse impostazioni ASR. |
| DeepFilterNet | Pacchetto assente. Nessun peso scaricato e nessuna inferenza dichiarata. Un enhancer monovoce potrebbe sopprimere il parlante minoritario: non è stato introdotto nelle sovrapposizioni. |
| WPE / nara_wpe | Pacchetto assente. Il riverbero rimane un indicatore diagnostico; non viene applicata dereverberazione non verificata. |
| WeSep / estrazione target | Esaminata la [documentazione ufficiale](https://github.com/wenet-e2e/wesep): toolkit di ricerca Apache-2.0 con eccezioni da verificare per componenti e checkpoint. Nessun estrattore locale convalidato disponibile. La modalità sperimentale mostra la fase come non disponibile; non crea enrollment né scarica modelli. |
| ONNX Runtime | Versione locale 1.27.0; provider disponibili `AzureExecutionProvider` e `CPUExecutionProvider`. I separatori usano CPU. CUDA e DirectML non sono disponibili in questo runtime: non è stato possibile confrontarli e non è stato cambiato provider. |

Gli embedding di identità della pipeline esistente continuano a usare l'audio senza denoise. Questo intervento non sostituisce quelle viste con l'enhancement e non aggiunge inferenza per il contatore. La disponibilità di un pacchetto o di un embedding non equivale alla disponibilità di un sistema TSE pronto per l'uso.

## Contatore «Profili vocali»

[`profile_stage_detail`](../src/processing/voice_profile.py) ricava il conteggio dalla collezione restituita da `build_voice_profiles`, senza usare `num_speakers`. I percorsi standard e profile-first mostrano `0 profili creati`, `1 profilo creato` o il plurale. Il percorso source-aware conta i profili effettivamente prodotti dall'arricchimento della trascrizione, una sola volta. Quando la generazione è disattivata, lo stato resta saltato e mostra anche il conteggio zero.

[`StageReporter`](../src/processing/progress.py) salva dettaglio ed extra nel manifest `provenance.processing_stages`. [`StageView`](../src/gui/stage_view.py) rende il conteggio nella colonna Dettaglio e la durata in una colonna distinta, con virgola decimale italiana. Al termine riporta la riga dei profili nell'area visibile, anche dopo molte fasi di recupero. `MainWindow._show_saved_stages` ricostruisce e mostra la tabella alla riapertura della sessione. Il tooltip distingue embedding numericamente validi, abbinamenti noti ed enrollment TSE esplicitamente verificati; la sola durata non rende un profilo idoneo al TSE.

Esempio verificato: `Profili vocali | Completata | 4 profili creati | 0,0 s`.

## Benchmark reale e materiali per la revisione

Il file di riferimento misura 5.890,782 secondi, ossia 98 minuti e 10,8 secondi. La sessione di confronto proviene dall'audit locale, è legata allo stesso SHA-256 e contiene 7.813 parole. Le conferme di voce esistenti non costituiscono una trascrizione completa verificata.

`scripts/benchmark_advanced_recovery.py` verifica il legame fra audio e baseline, esegue il confronto offline, campiona RSS del processo e VRAM totale del dispositivo e salva:

- `benchmark.json`, con impostazioni, provider, tempi, conteggi e limiti delle misure.
- `review-session.json`, apribile in Tessitura, con predizioni e proposte da ascoltare.
- `reference-review-template.json`, con finestre selezionate e annotazioni volutamente incomplete.
- Checkpoint di regione, viste lossless e contesti originali, senza pubblicare audio o trascrizioni.

I risultati operativi finali sono riportati qui al termine delle misure. Le prove non modificano né i byte dell'MP3 né gli eventi della baseline. Le proposte non sono «parole recuperate verificate», e i conteggi non misurano accuratezza.

| Separatore CPU; ASR large-v3 CUDA | Regioni elaborate | Controlli elaborati | Proposte non verificate | Regioni irrisolte | Tempo totale | RSS massimo campionato | VRAM totale massima campionata |
|---|---:|---:|---:|---:|---:|---:|---:|
| MossFormer2 | 12 | 2 | 40 | 2 | 278,8 s | 4,34 GiB | 2.756 MiB |
| SepFormer WHAMR16K | 12 | 2 | 42 | 2 | 156,6 s | 4,21 GiB | 2.732 MiB |

Entrambi i run hanno mantenuto invariati i 2.651 eventi e le 7.813 parole della baseline, oltre all'hash del file sorgente. Entrambi i controlli hanno prodotto zero proposte in ciascun run: questo non prova che le viste elaborate abbiano riconosciuto tutte le parole correttamente. I controlli ASR hanno scartato tre predizioni con MossFormer2 e sei con SepFormer. Una separazione accettata dai guardrail resta un esperimento da ascoltare.

Le dodici finestre includono controlli a 40 e 1.356 secondi, overlap a 97, 667, 1.484, 2.982 e 4.488 secondi, attività senza testo a 21 secondi, bassa confidenza a 99 secondi, incertezza d'identità a 108 secondi e parlato debole/rumoroso con possibile riverbero a 360 e 840 secondi. Queste etichette sono ipotesi automatiche. Non è stato attestato un caso reale a tre voci simultanee né un caso fisicamente distante mediante annotazione umana; l'incompletezza di un separatore a due uscite rispetto a tre voci candidate è coperta da regressione sintetica.

Il budget di audio selezionato, comprese le finestre di contesto, è identico nei due run. Le versioni e gli hash di implementazione e modelli sono nei rispettivi `benchmark.json`. La memoria è campionata ogni circa mezzo secondo: RSS riguarda il processo, la VRAM riguarda l'intero dispositivo e non è una misura isolata del solo modello. Picchi molto brevi possono non essere osservati. Non si deduce una superiorità qualitativa di SepFormer dalle 42 proposte o dal minor tempo.

Il primo esperimento di selezione, conservato separatamente, elaborava 12 regioni senza riuscire a riservare controlli: il filtro escludeva anche i soli indizi acustici di rumore o riverbero. È stato corretto prima del confronto finale. Quel primo run MossFormer2 ha prodotto 47 proposte in 352,2 secondi, con picchi campionati di 4,20 GiB RSS e 2.721 MiB di VRAM totale. È un controllo operativo preliminare, non una misura di accuratezza né un confronto conclusivo.

Per la revisione umana occorre ascoltare le stesse finestre in tutte le viste, annotare tutte le voci e le parole, mantenere i tempi assoluti e compilare revisore, completezza e checksum. Solo dopo si possono calcolare WER/CER, errori attribuiti alle voci, DER/JER con overlap e omissioni/duplicazioni reali tramite [`verified_evaluation.py`](../src/processing/verified_evaluation.py). Un transcript più lungo o un audio più pulito non soddisfano quel requisito.

## Affidabilità e limiti

I checkpoint atomici legano audio, baseline, turni, configurazione, snapshot ASR, manifest dei separatori, versioni runtime e codice eseguito. La ripresa controlla intervallo e hash dei file audio derivati; un file alterato o un checkpoint corrotto richiedono nuova elaborazione. Lo stesso candidato non viene inserito due volte in sessione.

Il limite ordinario di 180 secondi è cooperativo, controllato fra regioni e fra viste: una chiamata nativa già avviata può superarlo. La cancellazione programmatica conserva la prima passata; il pulsante batch esistente continua a fermarsi fra file. Non è stato aggiunto un pulsante di interruzione del primo passaggio. Il fallback CPU è verificato per esaurimento VRAM al caricamento dell'ASR; errori durante la decodifica rendono la regione irrisolta.

Audio e modelli restano locali. I file di confronto devono rimanere disponibili finché servono per la revisione e non vengono cancellati automaticamente da sotto una sessione salvata. Tentativi falliti possono lasciare diagnostica nella cache. Nessuna nuova prova di cattura hardware lunga, installazione Windows o aggiornamento GitHub Releases è stata eseguita; quelle implementazioni non sono state modificate.

## Verifiche e consegna

Le regressioni coprono baseline immutabile, sorgenti concorrenti, tempi assoluti, copie lossless, gain limitato, duplicati, output non finiti o collassati, tre voci candidate contro due sorgenti, modelli mancanti o falliti, fallback CPU, budget, cancellazione, checkpoint alterati, accettazione/rifiuto, undo/redo e preview GUI. Per i profili vengono verificati zero, uno e quattro risultati, un numero configurato diverso da quello prodotto, entrambi gli ordini della pipeline, le sorgenti note, serializzazione su disco, riapertura visibile e rendering con durata `0,0 s`.

Risultati del 13 novembre 2026 sull'albero consegnato:

- `python -m scripts.verify`: **76 passi, tutti superati** (raccolta `unittest` più ogni modulo di test eseguito singolarmente).
- Moduli dedicati: `tests/test_advanced_recovery.py` e `tests/test_profile_counter.py`, superati anche da soli.
- Gitleaks sulle modifiche in staging: nessuna credenziale rilevata.
- GitNexus `detect-changes --scope staged`: 17 file, 175 simboli, 76 processi, rischio **CRITICAL**, risultato né parziale né troncato. Il rischio riflette pipeline, avanzamento e timeline, toccati per costruzione.

Nota sull'identità delle tracce recuperate. Una verifica separata sugli artefatti esistenti mostra che i centroidi salvati da pyannote e gli embedding ricalcolati su finestre pulite di almeno tre secondi concordano con l'etichetta di diarizzazione in 196 casi su 198. È un'indicazione che un suggerimento acustico dell'identità è realizzabile senza nuovi modelli; non è stato implementato, perché le proposte in sovrapposizione sono brevi e l'affidabilità su tracce separate non è misurata. La fase di riconciliazione resta dichiarata non disponibile.

GitNexus è associato esplicitamente al repository di lavoro `tessitura`. Gli impact preventivi hanno segnalato HIGH su pipeline, recupero e profili, CRITICAL sui controlli di avanzamento, sull'aggiornamento dello stato GUI e sui metadati delle tracce. I risultati UNKNOWN dei callback Qt e dei nuovi helper sono stati verificati anche nel sorgente; non sono stati interpretati come basso rischio. L'estrazione globale dei processi segnala limiti di budget: l'assenza di un flusso non prova che non esista.

`node scripts/sync-clean.mjs` (anteprima) sull'albero consegnato: 17 modifiche, nessuna cancellazione, quattro file pubblici protetti lasciati invariati; controlli su percorsi personali e metadati di registrazione superati. `--apply` viene eseguito dopo il commit locale: esito, SHA dei commit e stato della copia pubblicabile sono riportati nella consegna, non in questo file. Nessun push e nessuna modifica del ramo `main`.
