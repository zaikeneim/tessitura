# Token Hugging Face e modello di diarizzazione

Il token serve soltanto per il primo download del modello di diarizzazione
`pyannote/speaker-diarization-community-1` (circa 33 MB), un repository *gated*:
l'account del token deve aver accettato le condizioni d'uso del modello. Una volta
in cache, il modello si usa senza connessione e senza token.

## Dove si configura

- **Impostazioni → Hugging Face…**, oppure la scheda «Hugging Face» in
  **Impostazioni → Modelli e motori…**.
- La configurazione iniziale accetta il token facoltativo e lo salva allo stesso modo.

Controlli disponibili: campo «Hugging Face Access Token» mascherato, **Mostra/Nascondi**,
**Salva token**, **Verifica connessione**, **Rimuovi token** e il collegamento a
<https://huggingface.co/settings/tokens>.

Stati mostrati:

| Stato | Significato |
|---|---|
| Non configurato | Nessun token in variabili d'ambiente, Gestore credenziali o vecchio file. |
| Non verificato | Token presente ma non ancora verificato, oppure verifica impossibile (rete). |
| Verifica in corso | Controllo dell'account e dell'accesso al modello. |
| Valido | Token accettato da Hugging Face. Se manca l'accesso al modello, il messaggio sotto lo indica con il collegamento alla pagina del modello. |
| Non valido | Token rifiutato: errato, scaduto o revocato. |

«Verifica connessione» usa il token scritto nel campo, se presente, senza salvarlo;
altrimenti il token effettivamente configurato.

## Ordine di precedenza

1. Variabili d'ambiente `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`
   (nello stesso ordine). Come in `huggingface_hub`, l'ambiente prevale sul token
   salvato: la schermata lo segnala.
2. Gestore credenziali di Windows, credenziale generica `Tessitura/HuggingFace`,
   salvata per l'utente corrente su questo computer (non viene sincronizzata con altri PC).
3. Vecchio file in chiaro `.hf_token` delle versioni precedenti: letto solo per
   compatibilità e segnalato come sconsigliato. **Sposta nel Gestore credenziali**
   lo importa e cancella il file.

## Sicurezza

- Il token non viene mai scritto in file, impostazioni JSON, log o URL; i messaggi
  di errore passano da un filtro che rimuove stringhe con forma di token.
- L'interfaccia non rilegge mai il token salvato: mostra solo l'origine. Il campo
  si svuota dopo il salvataggio.
- Il token è usato solo da `src/processing/hf_access.py` verso Hugging Face per
  verificare l'accesso e scaricare il modello. Il modello in cache viene caricato
  senza credenziali.
- Su sistemi senza Gestore credenziali compatibile il salvataggio viene rifiutato:
  usare `HF_TOKEN`.

## Download e integrità

Quando il modello manca, prima di trascrivere con «Distingui le voci», prima di
una coda di progetto e prima della revisione globale delle identità si apre
«Modello di diarizzazione», che mostra i passi:

1. cache locale;
2. credenziali (presenza e validità del token);
3. accesso al repository;
4. condizioni d'uso accettate;
5. download, ripreso dai file parziali in caso di interruzione;
6. integrità: dimensioni e SHA-256 confrontati con i metadati di Hugging Face,
   poi un caricamento di prova su CPU. Solo allora lo snapshot viene attivato.

Se l'elaborazione o la revisione globale si fermano per un problema del modello,
dopo la configurazione riprendono con la stessa richiesta, senza riavviare
l'applicazione. Annullando, non parte nulla e nulla va perso.

## Errori distinti

| Caso | Messaggio (sintesi) | Azione proposta |
|---|---|---|
| Token mancante | Nessun token configurato | Configura Hugging Face |
| Token non valido o scaduto | Token rifiutato | Crea un nuovo token «Read» |
| Autorizzazione negata | Token valido senza permesso di lettura dei repository gated | Token «Read» o permessi fine-grained |
| Condizioni non accettate | Condizioni d'uso del modello non accettate | Apri pagina del modello |
| Errore di rete | Hugging Face non raggiungibile, o servizio non disponibile (HTTP 429/5xx) | Riprova |
| Download incompleto | Download interrotto | Riprova: riprende dai file parziali |
| Cache corrotta | Dimensioni o checksum non corrispondono | Elimina cache danneggiata e riscarica |
| Modello incompatibile | Non si carica con la versione installata di pyannote.audio | Aggiornamento dell'ambiente |
| Modello non in cache | Solo per operazioni che non scaricano, come la revisione globale | Verifica e scarica |

Nella trascrizione con sorgenti separate (microfono + sistema) un modello mancante
non blocca l'elaborazione: le voci restano non risolte e l'avviso riporta la causa precisa.
