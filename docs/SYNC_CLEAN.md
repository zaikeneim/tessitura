# Aggiornare la copia CLEAN

`kisp-like` resta il repository di lavoro. `tessitura-github` riceve i file da
pubblicare e mantiene una propria cronologia Git.

Eseguire dalla cartella di lavoro:

```powershell
# Mostra aggiunte, modifiche e cancellazioni senza cambiare file.
node scripts/sync-clean.mjs

# Applica l'esportazione dopo una scansione locale con Gitleaks.
node scripts/sync-clean.mjs --apply
```

Il comando legge i file correnti, comprese le modifiche non ancora committate e
i nuovi file non ignorati. Applica le esclusioni di entrambi i repository e
ulteriori protezioni per dati privati, credenziali, media, modelli e file generati.
Non trasferisce directory `.git`, commit, branch o tag del repository di lavoro.
I collegamenti Markdown ai file del progetto che usano un percorso locale
`file:///` vengono convertiti in collegamenti relativi.

La CLEAN deve avere un indice e una directory di lavoro puliti prima di
`--apply`. Il comando aggiorna i file, aggiunge quelli nuovi e cancella soltanto
i file gia tracciati nella CLEAN che non appartengono piu all'esportazione.
Non elimina i dati locali non tracciati. I link simbolici vengono rifiutati.

Per applicare le modifiche serve Gitleaks nel PATH oppure il binario locale gia
scaricato in `.cache/publication-audit/gitleaks/gitleaks.exe`. Una scansione fallita
blocca la copia prima di modificare i file.

## File pubblici personalizzati

Quattro file hanno una versione specifica per la pubblicazione:

- `.gitignore`
- `README.md`
- `scripts/benchmark_dual_source.py`
- `scripts/benchmark_speaker_separation.py`

Il comando li segnala come `PROTECTED` quando differiscono dal repository di
lavoro e conserva la versione della CLEAN. Per modificarli, riportare nella CLEAN
soltanto le modifiche desiderate, mantenendo esclusioni, esempi generici e percorsi
anonimizzati. Non copiarli integralmente dal repository di lavoro.

## Revisionare e salvare nella CLEAN

Dopo la sincronizzazione:

```powershell
git -C ../tessitura-github status
git -C ../tessitura-github diff
```

`git diff` mostra i file gia tracciati; controllare anche i nuovi file elencati da
`git status`. Eseguire i test appropriati alle modifiche. Prima di committare,
eseguire anche l'analisi GitNexus richiesta da `AGENTS.md` nella CLEAN.

Quindi creare il commit dalla cartella CLEAN. Il push resta un'operazione manuale
separata. La sincronizzazione non effettua commit o push.

Per ripetere le verifiche dello script:

```powershell
node --test tests/sync-clean.test.mjs
```
