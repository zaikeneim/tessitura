# Esecuzione e avanzamento delle fasi

La pipeline pubblica `process_recording` espone il callback facoltativo
`on_stage(StageEvent)`, oltre al callback compatibile `progress(message, fraction)`.
Il piano viene pubblicato prima delle operazioni costose. Ogni fase resta
`pending`, diventa `running` quando l'operazione inizia e termina come
`completed`, `skipped` con motivo, oppure `failed` con errore. Una callback
tardiva non riapre una fase terminata. Il 100% complessivo richiede che tutte le
fasi pianificate siano completate o esplicitamente saltate.

La barra conta avanzamento osservato delle fasi, non una previsione del tempo
residuo. L'ASR inoltra i secondi effettivamente decodificati; durante retry e
recupero omissioni la fase torna indeterminata e mostra l'operazione corrente.
I sottopassi pyannote mostrano la propria percentuale, senza trasformarla in
una stima della durata dell'intera diarizzazione. I tempi visualizzati vengono
aggiornati dagli eventi e consolidati al termine delle operazioni.

## Percorsi eseguiti

| Percorso | Fasi e condizioni |
|---|---|
| Standard | Validazione, preparazione audio, denoise opzionale, ASR con retry e controllo della copertura entro il budget, allineamento, diarizzazione opzionale, profili vocali, attribuzione, revisione deterministica opzionale, ricostruzione degli eventi simultanei, sessione e risultati intermedi. |
| Profilazione preventiva | Diarizzazione e profili precedono l'ASR. L'estrazione delle voci viene tentata quando sono state rilevate almeno due voci; il percorso completo resta disponibile quando la separazione non riesce. |
| Separazione esplicita | Tentativo di separazione, ASR per traccia, allineamento, attribuzione dalla traccia e ricomposizione. Le fasi del percorso standard non usate sono indicate come saltate, con motivo. |
| Sorgenti mic/sistema | Preparazione e sincronizzazione non distruttiva, verifica eco quando richiesta, caricamento ASR, trascrizione di ciascuna sorgente, allineamento e diarizzazione per sorgente, ricomposizione con sovrapposizioni, sessione. Solo un microfono dichiarato monovoce verificato salta la propria diarizzazione. Una sorgente assente o senza testo ha motivi di salto espliciti. |
| Diagnostica | Verifica sorgente/cache, scansione completa con conteggio dei frame, selezione ed estrazione dei campioni, salvataggio di rapporto e riferimento. Una cache verificata produce salti espliciti. |
| Confronti diagnostici | Verifica campioni e modelli locali, confronto delle varianti per canale con callback ASR, salvataggio. Gli errori delle singole varianti restano nel rapporto e producono un avviso visibile; le varianti riuscite restano riutilizzabili dalla cache. |

Il percorso sorgenti conserva l'ASR sugli originali: il denoise non viene
applicato e la fase lo dichiara. Eventuali candidati di cancellazione eco non
sostituiscono l'ingresso ASR primario. La revisione deterministica standard non
viene dichiarata eseguita sui percorsi che attribuiscono la voce dalla traccia.
Scansione e confronti diagnostici restano operazioni esplicite del dialogo
«Diagnostica audio importato…».

## Interfaccia e risultati

`ProcessWorker`, `BatchProcessWorker` e `DiagnosticWorker` inoltrano gli eventi
attraverso segnali Qt. La finestra principale e il dialogo diagnostico mostrano
fasi ordinate, stato, dettagli, avvisi e tempo osservato. «Fasi di elaborazione»
permette di richiudere l'elenco nella finestra principale senza perderlo.
La coda identifica il file corrente, segnala gli errori per file e rispetta
l'arresto dopo il file in corso; un arresto non viene presentato come coda
interamente completata.

Le sessioni conservano il riepilogo finale in
`provenance.processing_stages`, compatibile con salvataggio e ricaricamento.
Un errore fatale lascia le fasi successive in attesa e non emette successo.
I fallback già previsti conservano testo e identità sconosciuta, con avviso;
anche gli errori del recupero ASR precedentemente silenziosi vengono segnalati.

## Verifica del 12 novembre 2026

- Test automatici dedicati: ordine standard e profilazione preventiva,
  callback reali di retry/copertura, progressi di entrambe le sorgenti,
  salti motivati, errori ASR/allineamento, cache diagnostica, varianti fallite,
  segnali worker/interfaccia e arresto della coda.
- Gate `env\python.exe -m scripts.verify`: tutti i moduli di test, inclusi quelli
  a funzioni, oltre alla raccolta `unittest`.
- Smoke Qt offscreen: clic effettivo su «Trascrivi registrazione», audio vocale
  sintetico controllato, tiny e Community-1 locali, CUDA sulla RTX 4070 Laptop.
  Due segmenti e 25 parole candidate; nessun errore. Verificati pannello fasi,
  completamento e layout a 1000×700, con chiusura e riapertura disponibili.
- Smoke sorgenti con modelli reali: la stessa voce sintetica fornita come due
  sorgenti controllate; 59 eventi, quattro interventi, sovrapposizioni mantenute,
  microfono verificato saltato e sorgente sistema diarizzata.
- Controllo offline di Whisper, decoder, pyannote e Qt con cache locale;
  diagnostica dell'ambiente senza aggiornare pacchetti o modelli.
- GitNexus: rischio complessivo CRITICAL, comprendendo il lavoro precedente.
  L'analisi globale troncata è stata ripetuta in gruppi completi con indici Git
  temporanei; lo staging reale è rimasto invariato. L'enumerazione globale dei
  flussi del grafo mantiene i limiti di budget segnalati dall'indicizzatore.

Log, screenshot e risultati dello smoke restano in `artifacts/progress/`,
cartella ignorata da Git. Questi controlli provano esecuzione e avanzamento;
non misurano WER/DER e non attestano registrazioni fisiche con cuffie/Bluetooth,
accuratezza su riferimenti umani o installazione/disinstallazione del pacchetto.
