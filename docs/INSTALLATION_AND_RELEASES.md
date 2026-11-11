# Installazione Windows, Packaging e Sistema di Aggiornamento GitHub Releases per Tessitura

Questo documento descrive l'architettura di distribuzione Windows di **Tessitura**, la separazione delle directory utente, il sistema di provisioning dei modelli al primo avvio, il meccanismo di aggiornamento basato su GitHub Releases e le istruzioni operative per generare `Tessitura.exe` e `Tessitura-Setup-{version}.exe`.

---

## 1. Architettura di Packaging e Installazione Windows

### Scelta Tecnologica
- **Application Freezer**: **PyInstaller (`onedir`)** (`packaging/windows/tessitura.spec`)
  - Produce `dist\Tessitura\Tessitura.exe` e la directory `_internal\` con l'interprete Python 3.11 incorporato, PySide6 (Qt6), librerie audio (`sounddevice`, `soundfile`, `pyaudiowpatch`), `faster-whisper`, `ctranslate2` e `truststore`.
  - Avvio immediato senza estrazione ad ogni esecuzione in `%TEMP%` (a differenza di `onefile`).
  - Non richiede alcuna installazione Python separata sul PC dell'utente.
- **Windows Installer**: **Inno Setup 6** (`packaging/windows/tessitura.iss`)
  - Produce `dist\Tessitura-Setup-{version}.exe`.
  - Installazione **per-utente (`PrivilegesRequired=lowest`)** in `%LOCALAPPDATA%\Programs\Tessitura`, senza richiedere privilegi di amministratore (UAC).
  - `AppId` stabile (`{{9B3F4C81-2A7E-4F1D-8E55-6D91A2B4C8E0}`) per riconoscere installazioni esistenti ed eseguire upgrade in-place.
  - Crea collegamento nel Menu Start (`Tessitura`), collegamento Desktop opzionale e voce in *App installate* di Windows.
  - Scrive il file marker `{app}\.installed` che attiva in modo deterministico la modalità installata in `src/paths.py`.
  - Impedisce downgrade accidentali tramite controllo Pascal Script (`InitializeSetup`), superabile solo passando esplicitamente `/ALLOWDOWNGRADE`.

---

## 2. Separazione delle Directory (Modalità Installata vs Sviluppo)

Quando Tessitura viene eseguita da un pacchetto installato (`sys.frozen == True` o presenza di `.installed`), nessun file mutabile viene mai scritto dentro la cartella del programma:

| Ruolo | Percorso Windows | Comportamento su Upgrade / Disinstallazione |
|---|---|---|
| **Binari applicazione** | `%LOCALAPPDATA%\Programs\Tessitura` | Aggiornato su upgrade; rimosso su disinstallazione |
| **Configurazione e backup** | `%APPDATA%\Tessitura` (`settings.json`, `.hf_token`, `backups\`) | **Preservato** su upgrade e disinstallazione |
| **Modelli AI** | `%LOCALAPPDATA%\Tessitura\models` (o cartella scelta dall'utente) | **Preservato** su upgrade e disinstallazione |
| **Registrazioni e sessioni** | `%LOCALAPPDATA%\Tessitura\data` (o cartella scelta dall'utente) | **Preservato** su upgrade e disinstallazione |
| **Cache API e installer** | `%LOCALAPPDATA%\Tessitura\cache` (`github_releases_cache.json`, `updates\`) | **Preservato**; pulibile senza perdita dati |
| **Log diagnostici** | `%LOCALAPPDATA%\Tessitura\logs` | **Preservato** |

---

## 3. Provisioning dei Modelli AI al Primo Avvio

I modelli AI (`large-v3` ~3.1 GB e `pyannote/speaker-diarization-community-1` ~450 MB) **non** vengono inclusi dentro `Tessitura-Setup-{version}.exe` per mantenere l'installer snello e veloce da scaricare:
- Il modulo `src/processing/first_run.py` e la finestra `Impostazioni → Cartelle dati e configurazione iniziale…` (`src/gui/first_run_dialog.py`) rilevano GPU/VRAM, spazio disco disponibile e stato della cache locale.
- L'utente può scegliere percorsi personalizzati per registrazioni e modelli AI, inserire facoltativamente il token Hugging Face (salvato in `%APPDATA%\Tessitura\.hf_token`) o rimandare il download usando subito la registrazione locale offline.
- Una volta scaricati in `%LOCALAPPDATA%\Tessitura\models`, i modelli vengono riutilizzati offline senza necessità di connessione e non vengono riscaricati durante gli aggiornamenti dell'applicazione.

---

## 4. Sistema di Aggiornamento GitHub Releases (`src/update_manager.py`)

### Canali di Rilascio e Risoluzione SemVer 2.0.0
- **Canale Stabile (`stable`)**:
  - Interroga `GET https://api.github.com/repos/{owner}/{repo}/releases/latest`.
  - Considera esclusivamente release pubblicate, non-draft e non-prerelease (`prerelease == false`).
- **Canale Beta (`beta`)**:
  - Interroga `GET https://api.github.com/repos/{owner}/{repo}/releases?per_page=30`.
  - Scarta le bozze (`draft == true`) e ordina tutte le release candidate usando il comparatore **SemVer 2.0.0** (`src/version.py:SemVer`), rispettando la precedenza `1.0.0 < 1.1.0-alpha.1 < 1.1.0-beta.1 < 1.1.0-beta.2 < 1.1.0-beta.10 < 1.1.0-rc.1 < 1.1.0` indipendentemente dalla data di pubblicazione.
- **Caching e Rate Limit**:
  - Salva `ETag` e `Last-Modified` in `%LOCALAPPDATA%\Tessitura\cache\github_releases_cache.json` ed esegue richieste condizionali (`If-None-Match` -> `HTTP 304 Not Modified`).
  - Limita il controllo automatico a una volta ogni 24 ore (`86400` s).
  - In assenza di rete o in caso di rate limit (`HTTP 403`/`429`), usa la cache locale e mostra un messaggio diagnostico senza mai bloccare l'applicazione.

### Verifica di Integrità e Autenticità del Pacchetto
Prima di consentire l'avvio assistito dell'installer scaricato in `%LOCALAPPDATA%\Tessitura\cache\updates\Tessitura-Setup-{version}.exe`:
1. **Protezione Downgrade**: rifiuta versioni inferiori a quella attualmente installata.
2. **Verifica Dimensione**: confronta i byte scaricati con il campo `size` dell'asset GitHub Release.
3. **Verifica Crittografica SHA-256**: scarica `SHA256SUMS.txt` dalla stessa release, calcola lo SHA-256 del file `.exe` e confronta i digest in tempo costante (`hmac.compare_digest`). Se il controllo fallisce, il file viene rimosso immediatamente.
4. **Verifica Firma Digitale Authenticode (`Get-AuthenticodeSignature`)**:
   - Con policy `strict` (predefinita): richiede firma Authenticode `Valid` (e corrispondenza con `trusted_publishers` se configurati). Se l'installer non è firmato (`NotSigned`), **blocca l'esecuzione automatica** e offre il pulsante per aprire la pagina GitHub Release ufficiale.
   - Con policy `checksum_only` (per ambienti di test/sviluppo privi di certificato code-signing): verifica obbligatoriamente lo SHA-256 rispetto a `SHA256SUMS.txt` e mostra un avviso esplicito prima della conferma utente.

### Ciclo di Vita Sicuro dell'Aggiornamento Assistito
1. **Protezione Operazioni Attive**: `UpdateManager.can_start_update()` blocca l'avvio dell'aggiornamento se è in corso una registrazione audio, una combinazione MP3, una trascrizione o una diarizzazione.
2. **Backup Pre-Aggiornamento**: `src/migrations.py:create_state_backup()` salva `settings.json` e `projects.json` in `%APPDATA%\Tessitura\backups\backup-{prev}_to_{target}-{timestamp}\`.
3. **Registrazione Stato `pending_update`**: scrive in `settings.json` la versione target, la versione precedente e il percorso del backup.
4. **Avvio Esterno e Chiusura Controllata**: avvia `Tessitura-Setup-{version}.exe` come processo indipendente (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`) e chiude `Tessitura.exe` rilascando i file lock.
5. **Validazione Post-Aggiornamento**: al successivo avvio, `validate_and_migrate_startup()` verifica se la versione in esecuzione corrisponde a `target_version` (`update_verified`) oppure se l'installazione è stata annullata (`update_cancelled_or_rolled_back`), eseguendo eventuali migrazioni di schema.

---

## 5. Istruzioni per Compilare e Installare Tessitura

### Compilazione Locale su Windows
Dalla root del progetto:

```powershell
# Compilazione completa (esegue i test, genera Tessitura.exe, Tessitura-Setup-{version}.exe e SHA256SUMS.txt)
powershell -ExecutionPolicy Bypass -File .\scripts\build-windows.ps1 -Version "1.0.0"

# Compilazione rapida di smoke test (senza includere i wheel CUDA opzionali nel bundle)
powershell -ExecutionPolicy Bypass -File .\scripts\build-windows.ps1 -Version "1.0.0" -LightBundle -SkipTests
```

Artefatti generati in `dist\`:
- `dist\Tessitura\Tessitura.exe` (bundle applicativo `onedir`)
- `dist\Tessitura-Setup-1.0.0.exe` (installer per-utente Windows)
- `dist\SHA256SUMS.txt` (checksum SHA-256 ufficiali)

### Configurazione del Repository GitHub Release
I parametri del repository GitHub vengono scritti in `build_config.json` da `scripts/build-windows.ps1`:
- `-GitHubOwner "<owner>"` (o variabile d'ambiente `TESSITURA_GITHUB_OWNER`)
- `-GitHubRepo "<repo>"` (o variabile d'ambiente `TESSITURA_GITHUB_REPO`)
- `-TrustPolicy "strict"` (oppure `"checksum_only"`)

### Pubblicazione Automatica tramite GitHub Actions
Il workflow `.github/workflows/release-windows.yml` viene attivato da:
- Tag SemVer `v*.*.*` (es. `git tag v1.0.0 && git push origin v1.0.0` per una release **Stable**, oppure `git tag v1.1.0-beta.1 && git push origin v1.1.0-beta.1` per una **Beta** pre-release).
- Avvio manuale (`workflow_dispatch`) dalla scheda Actions di GitHub.

