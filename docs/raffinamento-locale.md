# Raffinamento locale di testo e attribuzioni

Implementazione di `implementation_plan.md`, 16 settembre 2026.

## Uso

- La casella **Controlla e raffina con il modello locale** abilita il controllo
  acustico e la passata finale con il modello scelto in Impostazioni.
- La pipeline usa esclusivamente un modello già completo in cache. Se quello
  selezionato manca, il raffinamento viene saltato, senza download né sostituzioni
  silenziose. Da Python, `analysis_model_repo=None` sceglie tra i modelli del catalogo
  già disponibili; `llm_refine=False` disabilita il passaggio.
- **Modifica → Raffina trascrizione con modello locale** tratta il testo della
  sessione aperta. L'azione richiede il modello selezionato già scaricato.
- **Modifica → Annulla/Ripeti** ripristina le correzioni; **Analisi → Mostra
  trascrizione grezza** permette il confronto con l'originale.

## Comportamento e salvaguardie

Whisper e pyannote sono rilasciati prima del caricamento dell'LLM. Lo stesso LLM
esegue prima la revisione delle voci e poi quella del testo, e viene chiuso in
`finally`. La pipeline lavora su una copia: se caricamento o generazione falliscono,
mantiene l'output acustico e registra l'esito in `transcript_review.llm_refinement`.
La revisione delle voci conserva la soglia esistente di 0,8 e le etichette note.
Per microfono + sistema conserva le attribuzioni derivanti dai canali; con una sola
voce o diarizzazione disabilitata esegue soltanto il raffinamento testuale.

Il testo viene inviato come dati JSON, con indici espliciti e budget di contesto
comprensivo della risposta. Sono scartati indici estranei al blocco, risposte
malformate, conflitti, variazioni di lunghezza superiori al 50%, aggiunte/rimozioni
di parole e alterazioni di numeri. Sono ammesse punteggiatura, maiuscole, accenti e
correzioni molto limitate di refusi. Un segmento intero che eccede il budget viene
segnalato tra `skipped_indices` e conservato; non si applica mai una risposta parziale
al suo posto. Le risposte malformate sono conteggiate in `invalid_blocks`.

Ogni proposta è legata a ID e impronta della sessione, ID del segmento e testo
precedente. La GUI lavora su un'istantanea: risultati ormai superati sono rifiutati.
Una modifica testuale invalida l'allineamento delle parole; i tempi originali
restano conservati, senza inventare tempi per il nuovo testo. Le correzioni LLM
non diventano esempi di apprendimento umano. Le successive modifiche manuali
vengono confrontate con il testo già raffinato; dopo un raggruppamento ambiguo il
contributo viene escluso dalla memoria.

## Limiti rispetto alle intenzioni del piano

Un modello solo testuale non può ricostruire parole mancanti dall'audio né verificare
l'identità di una voce. Per le frasi troncate interviene quindi sulla punteggiatura,
senza completamenti inventati. I filtri riducono le alterazioni, ma non dimostrano
correttezza semantica; anche la confidenza dichiarata dal modello non è una misura
acustica. L'attivazione della funzione non certifica miglioramenti WER/DER/tcpWER.
Serve ancora il confronto su registrazioni con riferimento umano validato.

## Verifiche

- Test di validazione JSON, protezione del contenuto, budget, applicazione su
  sessione cambiata, salvataggio e annullamento/ripetizione.
- Test di pipeline senza modello, modello disabilitato, errore di caricamento,
  errore nella seconda passata, protezione dei canali, chiusura del motore e
  rilascio del trascrittore prima dell'LLM.
- Test del worker, collegamento GUI, apprendimento e metadati malformati.
- `env\python.exe -m scripts.verify`: esegue discover e tutti i moduli di test.
- Prova reale sulla GPU locale con Qwen3 4B Instruct già in cache, revisione
  `cdbee75f17c01a7cc42f958dc650907174af0554`: due brevi segmenti sintetici,
  entrambe le passate completate in 25,2 secondi, due correzioni di testo,
  nessun cambio voce. Dopo la chiusura PyTorch riportava circa 8,1 MiB allocati
  e 20 MiB riservati. È una verifica funzionale, non un benchmark di accuratezza
  né di prestazioni su registrazioni lunghe.

GitNexus 1.6.12 è stato usato per impatti e revisione delle modifiche. Il grafo
segnala alcuni flussi non enumerati per limiti interni: i risultati non sono una
prova di copertura completa; i collegamenti Qt sono stati controllati anche nel
codice e nei test della GUI.
