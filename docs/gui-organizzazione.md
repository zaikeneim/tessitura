# Organizzazione della GUI

La finestra separa gli strumenti in tre schede laterali: Audio (acquisizione e combinazione), Trascrizione (profilo, lingua, partecipanti), Progetto (cartella, memoria, coda e aggiornamenti). Importazione e avvio della trascrizione rimangono fuori dallo scorrimento. Le opzioni avanzate conservano i propri valori quando vengono nascoste.

Il documento e il lettore occupano l'area principale. Chat, riassunto e revisione delle voci condividono le stesse QAction dei menu, quindi rispettano automaticamente disponibilità della sessione e stato di elaborazione. Il pannello voci può essere nascosto per dare spazio al testo. Modelli e motori mantengono la propria finestra di impostazioni.

## Riferimenti e scelte

- [Microsoft — Navigation basics](https://learn.microsoft.com/en-us/windows/apps/design/basics/navigation-basics): struttura poco profonda, destinazioni nominate e controlli standard. Applicazione: tre schede Qt, nessuna navigazione obbligata a più livelli.
- [Nielsen Norman Group — Progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/): opzioni secondarie mostrate su richiesta. Applicazione: glossario, trattamento e dettagli hardware espandibili; profilo obbligatorio sempre nella scheda principale di trascrizione.
- [Microsoft — Accessible Windows apps](https://learn.microsoft.com/en-au/windows/apps/design/accessibility/developing-inclusive-windows-apps): tastiera, nomi accessibili e focus. Applicazione: schede con acceleratori, controlli Qt nativi, nomi dei campi e focus visibile. Non costituisce una certificazione di accessibilità.

Verifica: `python -m tests.test_gui_navigation`, suite `python -m scripts.verify` e rendering `python scripts/preview_ui.py`. Le anteprime usano sessioni sintetiche e non avviano modelli. Controllare anche 1000×700 e opzioni avanzate aperte; gli screenshot da soli non verificano il comportamento di menu e worker.
