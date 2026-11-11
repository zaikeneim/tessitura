@echo off
rem Avvio di Tessitura. Se l'ambiente Python non c'e' (PC nuovo, cartella appena
rem copiata), lo prepara e poi avvia l'app.
setlocal
set "ROOT=%~dp0"
set "PYW=%ROOT%env\pythonw.exe"

if not exist "%ROOT%.runtime.json" if not exist "%PYW%" (
    echo Primo avvio su questo PC: preparo l'ambiente, ci vuole qualche minuto.
    echo.
    powershell -ExecutionPolicy Bypass -File "%ROOT%scripts\make_portable.ps1"
    if errorlevel 1 (
        echo.
        echo Preparazione non riuscita. Leggi i messaggi qui sopra.
        pause
        exit /b 1
    )
)

if not exist "%ROOT%.runtime.json" if not exist "%PYW%" (
    echo Ambiente non trovato in "%ROOT%env". Lancia scripts\make_portable.ps1 a mano.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\launch.ps1"
if errorlevel 1 pause
