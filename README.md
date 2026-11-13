# Tessitura

App desktop per **registrare** audio (microfono e/o output del PC, es. riunioni o
sessioni Discord), **ridurre il rumore**, **trascrivere**, **allineare i timestamp**,
**separare le voci** (diarizzazione) e **rivedere** il risultato in una GUI
assegnando nomi alle voci.

Tutta l'elaborazione avviene **in locale** (nessun dato verso cloud).

## Requisiti
- Windows 10/11
- GPU NVIDIA con driver recenti (testato su RTX 4070 Laptop GPU, 8 GB VRAM)
- Miniconda

## Cartella autosufficiente (consigliato)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_portable.ps1
```

Mette **dentro la cartella dell'app** tutto il necessario:

```
env\      ambiente Python (torch, faster-whisper, pyannote, Qt, ffmpeg)  ~6 GB
models\   modelli scaricati (Whisper large-v3, pyannote community-1)     ~3 GB
data\     registrazioni e sessioni
```

Poi si avvia con **doppio clic su `Avvia Tessitura`** (il collegamento con l'icona, in
cartella e sul Desktop). Se conda ha gia' l'ambiente `tessitura`, lo script lo clona
con gli hardlink: pochi secondi e quasi zero spazio in piu'.

Copiando l'intera cartella su un disco esterno o su un altro PC l'app parte senza
installare niente. Se l'ambiente manca (hai copiato solo il codice), **il primo
avvio lo ricrea da solo** — serve solo la connessione.

Dopo averla resa portabile, la vecchia cache dei modelli in
`C:\Users\<utente>\.cache\huggingface` non serve piu' a quest'app: si puo'
cancellare per recuperare i 3 GB.

## Aggiornare le librerie

Dall'app: pulsante **Aggiornamenti…**, che controlla sia le librerie sia le
revisioni dei modelli scaricati e mostra cosa e' invecchiato. Non installa nulla:
l'aggiornamento vero va fatto ad app chiusa, con

```powershell
powershell -ExecutionPolicy Bypass -File scripts\update.ps1 -Check   # solo diagnosi
powershell -ExecutionPolicy Bypass -File scripts\update.ps1          # aggiorna
```

Lo script crea una copia **indipendente** in `.runtimes/` con `conda --copy`.
Installa e verifica gli aggiornamenti nella copia; solo dopo tutti i controlli
cambia atomicamente `.runtime.json`. Non modifica o cancella l'ambiente attivo.
Servono diversi GB liberi per la copia. Le copie fallite restano disponibili per
la diagnosi; non vengono eliminate automaticamente.

Per preservare il comportamento di trascrizione, il motore di inferenza e le sue
dipendenze installate restano vincolati alle versioni esistenti, inclusi i build
CUDA. Modelli e cache non vengono aggiornati o rimossi. Aggiornare il motore
richiede una validazione separata dell'accuratezza su audio di riferimento.

Il controllo prima dell'attivazione esegue `pip check`, verifica versioni e
backend, prova offline Whisper e diarizzazione con i modelli locali, verifica
Qt e i decoder e lancia i test. Devono essere già presenti in cache il modello
Whisper `tiny` e il bundle diarizzazione completo: se mancano, l'aggiornamento
non viene attivato e non vengono avviati download impliciti durante la verifica.

Avviare tramite **Avvia Tessitura** oppure `scripts/start.ps1`, che seguono l'ambiente
selezionato. Il comando diretto `env/python.exe main.py` usa sempre il vecchio
env, anche dopo un aggiornamento. Per tornare alla versione precedente:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\update.ps1 -Rollback
```

Il rollback funziona anche senza rete o con un ambiente candidato non avviabile.
Ripristina il precedente file di selezione se quello corrente è danneggiato;
al primo aggiornamento può ripartire da `env/`. Non spostare le cartelle dentro
`.runtimes/`: i percorsi interni di conda devono restare stabili. Lo script
richiede l'ambiente portabile; per un'installazione solo conda, eseguire prima
`scripts/make_portable.ps1`. Il vecchio parametro `-Cpu` viene rifiutato:
il cambio del backend non fa parte di un aggiornamento conservativo.

## Setup ambiente (alternativa: ambiente conda di sistema)

### Su un PC nuovo: un comando solo

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Lo script installa Miniconda se manca, crea (o aggiorna) l'ambiente `tessitura`,
installa le dipendenze e sceglie da solo il build di PyTorch: **CUDA se trova una
GPU NVIDIA, altrimenti CPU**. E' ripetibile: rilanciarlo aggiorna solo cio' che
serve. Con il token gia' a portata di mano:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -HfToken hf_xxxxx
```

### Setup manuale (equivalente)

```powershell
# 1. Crea l'ambiente conda (Python 3.11 + ffmpeg + PySide6 da conda-forge)
conda env create -f environment.yml
conda activate tessitura

# 2. Installa tutte le dipendenze (installa anche un torch CPU temporaneo)
pip install -r requirements.txt

# 3. Solo con GPU NVIDIA: forza i build CUDA di torch/torchaudio (indice cu126,
#    compatibile con faster-whisper/ctranslate2 che richiede le librerie CUDA 12)
pip install --force-reinstall --no-deps torch==2.13.0 torchaudio==2.11.0 `
    --index-url https://download.pytorch.org/whl/cu126
```

> Nota SSL: se conda/pip danno `CERTIFICATE_VERIFY_FAILED`, e' l'antivirus (AVG)
> che intercetta HTTPS. Soluzione senza disattivare la sicurezza:
> `conda config --set ssl_verify truststore` (lo fa gia' `setup.ps1`).
> A runtime ci pensa `src/bootstrap.py` (pacchetto `truststore`).

### Token Hugging Face (per la diarizzazione)
La diarizzazione usa `pyannote.audio`. Il primo download del modello
`pyannote/speaker-diarization-community-1` richiede un account Hugging Face gratuito;
dopo, il modello resta in cache e il token non serve più.
1. Crea un account su https://huggingface.co
2. Accetta le condizioni del modello `pyannote/speaker-diarization-community-1`
3. Crea un token di tipo «Read» su https://huggingface.co/settings/tokens
4. In Tessitura apri **Impostazioni → Hugging Face…**, incolla il token, poi
   **Salva token** e **Verifica connessione**. Il token va nel Gestore credenziali
   di Windows, mai in un file.

In alternativa imposta la variabile d'ambiente `HF_TOKEN`, che ha la precedenza
sul token salvato. Dettagli, ordine di precedenza ed errori: [docs/HUGGING_FACE.md](docs/HUGGING_FACE.md).

## Avvio

```powershell
conda activate tessitura
python main.py
```

oppure, senza attivare niente:

```powershell
.\scripts\start.ps1
.\scripts\start.ps1 -DataDir D:\tessitura-dati   # con cartella dati condivisa
```

L'app **si adatta da sola al PC su cui gira**: rileva la GPU e la VRAM, sceglie
di conseguenza la taglia del modello Whisper e la precisione, e lo scrive nella
finestra insieme alla stima dei tempi. Con GPU da 8 GB usa `large-v3` con
precisione `int8_float16`; senza GPU scende a `small`/`base` in `int8`, perche'
`large-v3` su CPU richiederebbe ore. La scelta resta modificabile a mano dal
menu a tendina.

Microfono e audio di sistema si registrano su due WAV separati, conservati per
impostazione predefinita senza compressione con perdita. Abilitando esplicitamente
**MP3 compatto (con perdita)**, a fine registrazione diventano **un unico MP3 stereo a 192 kbps** (mic a sinistra,
sistema a destra) e i WAV vengono cancellati: un file solo da gestire, e un'ora
passa da ~1.2 GB a ~80 MB. I due canali restano distinti perche' e' l'indizio
che distingue chi parla al microfono (`IO`) da chi arriva dalla chiamata
(`REMOTO`); la pipeline li riapre da sola. Lo sfasamento di partenza dei due
stream viene compensato al momento dell'unione.

Prima di trascrivere o diarizzare l'app richiede il **Profilo audio**:

- **Microfono + audio di sistema**: due sorgenti indipendenti, oppure un MP3
  stereo creato dall'app con microfono a sinistra e sistema a destra;
- **Ambientale stereo**: recorder o microfono stereo che registra tutte le voci
  nella stessa stanza. I canali non vengono interpretati come speaker distinti
  e il channel hint `IO/REMOTO` resta disattivato.

Un file stereo esterno non va classificato automaticamente: il numero di canali
non dice se si tratta di mic/sistema o di una registrazione ambientale. Senza
questa scelta la pipeline si arresta prima di caricare Whisper e pyannote.

Si registra su WAV e si converte dopo: un encoder dentro il ciclo di cattura
rischia di perdere frame e sfasare tutti i timestamp. I WAV si cancellano solo
dopo aver verificato che l'MP3 sia leggibile e della durata giusta.

La lingua della trascrizione si sceglie dal menu **Lingua** (Automatica, italiano,
inglese, portoghese, spagnolo, francese, tedesco). Whisper usa **una** lingua per
registrazione, decisa dai primi secondi di parlato, e l'automatico puo' sbagliare:
su una lezione in italiano ha rilevato l'occitano, e da li' in poi la trascrizione
degrada. Se la lingua la conosci, indicala. Se in una stessa registrazione si
alternano piu' lingue, il testo nella lingua non scelta esce impreciso: meglio
registrare sessioni separate per lingua.

Verifica rapida dell'ambiente (GPU, faster-whisper, dispositivi audio, token):
```powershell
python scripts/check_env.py
```

Per l'elaborazione da riga di comando il profilo è obbligatorio, ad esempio:

```powershell
python -m scripts.run_pipeline "registrazione.mp3" --profile stereo_ambient --speakers 4 --lang it
```

Per una sessione con nomi inventati si può aggiungere un glossario, ad esempio
`--prompt "Progetto Aurora, Relatore A, Relatore B"`.

I segmenti sospetti vengono ritentati automaticamente; `--quality-retries 0`
disattiva questa seconda passata quando serve una lavorazione strettamente
deterministica e più veloce.

Prima di restituire la sessione, il controllo qualità confronta l'attribuzione
conservativa con quella parola-per-parola. Sui file **ambientali** sceglie la
seconda solo quando l'accordo con i turni è chiaramente migliore; i casi incerti
restano invariati e vengono marcati per una revisione. L'output precedente è
conservato nel file di sessione: dal menu **Modifica -> Mostra trascrizione
grezza** (o dal pulsante **Trascrizione grezza** nella barra degli strumenti)
si può confrontare senza perdere la versione revisionata. Il controllo è attivo
di default nella GUI e nella CLI; per disattivarlo usare `--review off`.

Questo passaggio non riscrive il testo e non sovrascrive l'indizio osservato dei
file microfono+sistema. In questo modo una revisione automatica non può rendere
peggiore un'attribuzione già determinata dalla sorgente audio.

Quando la trascrizione contiene più righe consecutive della stessa voce, il menu
**File -> Raggruppa testo per speaker consecutivi…** (o il pulsante **Raggruppa
voci**) le unisce dopo conferma. Timestamp e parole restano nella loro sequenza,
mentre la trascrizione grezza rimane disponibile per il confronto.

Quando il numero dei partecipanti è noto, indicarlo in **Numero di voci** (per
esempio `4` per `Master` più tre partecipanti): la pipeline conserva in questo
caso anche gli interventi brevi invece di assimilarli automaticamente alla voce
dominante.

## Cartella dati e uso su piu' PC

Tutti i percorsi si risolvono a partire dalla cartella dell'app, non dalla
cartella da cui la lanci. La cartella dati si sceglie cosi', in ordine:

1. variabile d'ambiente `TESSITURA_DATA_DIR` (se impostata; `KISP_DATA_DIR` resta compatibile);
2. `data/` dentro il progetto (comportamento predefinito);
3. cartella dati utente del sistema (`%LOCALAPPDATA%\tessitura`, oppure la vecchia `kisp-like`), se la cartella
   del progetto non e' scrivibile.

Le sessioni salvano i riferimenti audio **relativi** alla cartella dati, e in
apertura l'app ritrova il file anche se e' stato spostato: quindi la stessa
sessione si apre su un PC diverso senza perdere l'audio sincronizzato col testo.

```powershell
# usare una cartella dati condivisa (sync, disco esterno, ...)
$env:TESSITURA_DATA_DIR = "D:\tessitura-dati"
python main.py
```

Quindi: `setup.ps1` su ogni PC, e `TESSITURA_DATA_DIR` su una cartella sincronizzata
se vuoi ritrovare le stesse sessioni ovunque.

## Modelli e motori acustici

Le informazioni e lo stato dei modelli locali sono raccolti in **Impostazioni →
Modelli e motori…**, con due schede: **Trascrizione** (`faster-whisper`) e
**Diarizzazione** (`pyannote/speaker-diarization-community-1`). Ogni scheda mostra il
modello attivo, repository, stato della cache, revisioni locale/remota e
dimensione stimata; da qui si possono verificare gli aggiornamenti o scaricare
il modello selezionato.

Su GPU NVIDIA da 8 GB VRAM (es. RTX 4070 Laptop GPU), Tessitura seleziona
automaticamente `large-v3` con quantizzazione `int8_float16` (pesi INT8,
attivazioni FP16), dimezzando il consumo di VRAM di CTranslate2 e liberando
esplicitamente la memoria GPU tra la fase di trascrizione, l'allineamento
forzato dei timestamp (`src/processing/alignment.py`) e la diarizzazione
(`pyannote/speaker-diarization-community-1`).

### Apprendimento locale del progetto

Tessitura migliora tra una seduta e la successiva senza modificare in modo
irreversibile i pesi dei modelli:

- una correzione manuale salvata aggiunge al prompt futuro soltanto sostituzioni
  lessicali brevi e simili a un errore ASR; i normali cambi di contenuto vengono
  scartati;
- un nome speaker diventa un profilo vocale solo dopo la conferma dell'utente e
  viene proposto in seguito soltanto sopra una soglia conservativa di similarità.

La memoria resta in `tessitura-progetto.json`, ha limiti di dimensione ed è
scritta atomicamente. Da **Dettagli progetto** si può metterla in pausa,
ispezionare quante correzioni/profili contiene oppure azzerarla. La
cancellazione non tocca glossario, note, audio o sessioni e viene applicata solo
premendo **Salva**. Un'operazione **Annulla**, al salvataggio successivo, rimuove
anche il contributo della correzione revocata.

### Revisione ed Export

- **Modifica → Annulla / Ripeti** (`Ctrl+Z` / `Ctrl+Y`): riporta indietro
  correzioni, rinomine e raggruppamenti.
- **Modifica → Mostra trascrizione grezza** (o pulsante **Trascrizione grezza**):
  confronta l'output revisionato con la trascrizione grezza senza perdere le
  modifiche.
- **Export JSON / TXT / SRT / per-speaker**: le battute SRT rispettano durata e
  lunghezza massime (`SRT_MAX_SECONDS`, `SRT_MAX_CHARS` in `src/models/storage.py`)
  anche dopo il raggruppamento per voce; con le parole allineate i tempi sono
  quelli reali, dopo una modifica manuale vengono ripartiti sul testo.

## Ripresa e valutazione

Le fasi della pipeline possono usare checkpoint content-addressed basati su hash
dell'audio, opzioni effettive, revisioni dei modelli e versione dell'algoritmo.
Un arresto o un cambio di impostazioni invalida solo la fase interessata e le
successive. Il modulo `src/processing/evaluation.py` fornisce WER e cpWER
deterministici per i test locali; con MeetEval e pyannote installati si possono
aggiungere tcpWER e DER ufficiali. Un confronto di qualità deve usare un gold
annotato separato: i risultati sintetici dei test non sono una promessa di WER o
DER su registrazioni future.

Il corpus di riferimento va creato da registrazioni e trascrizioni che si è
autorizzati a elaborare. Audio, annotazioni e manifest restano locali nelle
directory ignorate `data/recordings/`, `data/sessions/` e `data/evaluation/`.
Per valutare la diarizzazione, controllare manualmente anche i confini dei turni
e le sovrapposizioni fra voci.

Esempio di corpus sui primi 60 minuti di una registrazione ambientale stereo
di un'ora:

```powershell
python scripts/build_phase0_corpus.py `
  --audio "data\recordings\esempio.mp3" `
  --transcript "data\sessions\esempio.txt" `
  --out "data\evaluation\esempio_gold.json" `
  --readable-out "data\evaluation\esempio_gold.txt" `
  --stm-out "data\evaluation\esempio_gold.stm" `
  --rttm-out "data\evaluation\esempio_gold.rttm" `
  --duration 3600 --audio-profile stereo_ambient --minutes 60
```

## Combinare più registrazioni in un solo MP3

*File → Combina più audio in un MP3…* unisce due o più registrazioni. Due modi,
da non confondere:

- **In sequenza**: i file vengono messi uno dopo l'altro nell'ordine scelto. È il
  caso della sessione spezzata in più parti dal registratore; la durata finale è
  la somma.
- **Sovrapposti**: i file suonano insieme, come due microfoni sulla stessa
  scena; la durata è quella del più lungo. I segnali non vengono allineati fra
  loro: due riprese della stessa stanza possono cancellarsi in parte per
  sfasamento, quindi ascolta il risultato prima di trascriverlo.

Frequenze e numero di canali diversi vengono uniformati (48 kHz stereo). Il
livello non viene abbassato automaticamente; un limiter finale evita la
saturazione. Gli originali restano sul disco. A fine operazione l'app propone di
usare subito il file combinato per la trascrizione.

Da codice: `src.audio.encode.combine_to_mp3([...], destinazione, mode="concat")`
oppure `mode="mix"`.

## Test

Test veloci, senza GPU e senza scaricare modelli:

```powershell
# comando unico: esegue discover E ogni modulo di test, nessuno escluso
python -m scripts.verify

# con diagnostica ambiente e vincoli di versione
python -m scripts.verify --full

python scripts/benchmark_pipeline.py
# quando esiste un manifest gold locale:
python scripts/evaluate_sessions.py data/evaluation/manifest.json
```

Usa `scripts.verify` e non solo `unittest discover`: discover raccoglie soltanto
le classi `TestCase`, quindi i moduli scritti a funzioni risultavano "passati"
solo perche' non venivano mai eseguiti. Per un singolo modulo usa
`python -m tests.test_x` e non `python tests/test_x.py`: i file piu' recenti non
aggiungono la root a `sys.path` e come file fallirebbero con ModuleNotFoundError.

Anteprima della nuova interfaccia, senza registrare audio o caricare modelli:
`python scripts/preview_ui.py` (immagini in `artifacts/`).
Per creare e distribuire un pacchetto Windows, vedi
[installazione e release](docs/INSTALLATION_AND_RELEASES.md).

## Struttura
```
src/
  audio/       cattura mic + WASAPI loopback, enumerazione dispositivi, codifica/mix
  processing/  denoise, trascrizione (faster-whisper), allineamento forzato (alignment.py),
               diarizzazione (pyannote), merge trascrizione+voci, valutazione locale
               (model_store.py, checkpoints.py, evaluation.py)
  models/      modello dati sessione (schema v4), salvataggio/caricamento ed export
  gui/         interfaccia PySide6
  hardware.py  rilevamento GPU/CPU e scelta di modello e precisione
  paths.py     cartella dati e percorsi portabili
  app.py       entry point
data/
  recordings/  registrazioni WAV
  sessions/    sessioni salvate (JSON)
  work/        audio intermedio (mix, denoise)
scripts/       setup.ps1 (installazione), start.ps1 (avvio), diagnostica e benchmark
tests/         test di regressione
```

## Note sulla qualita' del risultato

Alcune scelte sono il risultato di errori misurati su registrazioni reali, non
default: cambiarle alla leggera fa tornare i bug.

- Il **denoise e' volutamente leggero** (`denoise_strength` 0.5, stima
  stazionaria) e il livello viene ripristinato dopo il filtro. Con i valori
  aggressivi di prima il gating toglieva fino al 66% dell'energia del parlato
  piu' basso, che spariva dalla trascrizione.
- La **diarizzazione gira sull'audio non denoised**: il denoise altera il timbro,
  cioe' proprio l'informazione su cui pyannote riconosce le voci.
- L'audio viene **normalizzato** a un livello da parlato prima della
  trascrizione: registrazioni basse facevano scartare il parlato al VAD.
- Whisper gira con `condition_on_previous_text=False`, altrimenti entra in loop
  di ripetizione della stessa frase.
- La voce si assegna **per frase, non per parola**: e' cio' che evita di
  attribuire meta' frase a una persona e meta' a un'altra.

## Stato
In sviluppo. Milestone: M0 ambiente · M1 cattura audio · M2 pipeline · M3 GUI.

## Licenza

Il codice originale di Tessitura è distribuito con la **PolyForm Strict License
1.0.0**. Il testo completo è in [LICENSE.md](LICENSE.md); titolare e contatti sono
indicati in [NOTICE.md](NOTICE.md).

La licenza consente gli usi non commerciali previsti dai suoi termini. Non concede
diritti di modifica, creazione di opere derivate o redistribuzione. Per ottenere
autorizzazioni ulteriori, contattare il titolare indicato in `NOTICE.md`.

Librerie di terze parti e modelli AI mantengono le rispettive licenze.
