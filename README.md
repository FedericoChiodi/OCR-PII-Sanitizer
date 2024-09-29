# PII Detector for Medical Images

Questo progetto utilizza un modello DistilBERT finetunato per rilevare e coprire informazioni personali identificabili (PII) in radiografie. La soluzione combina EasyOCR per l'estrazione del testo dalle immagini e un sistema di classificazione per identificare se il testo contiene PII.

## Caratteristiche

- Estrazione del testo da immagini mediche utilizzando EasyOCR.
- Rilevamento di PII in base al testo estratto.
- Censura del testo se rilevato come PII.
- Supporto per lingue multiple (inglese e italiano).
- Report dettagliato delle predizioni con metriche di classificazione.
- Tempo di elaborazione di ogni immagine <= 1s

## Installazione

1. Clona la repository:

   ```sh
   git clone https://github.com/tuo-username/PII-Detector-Medical-Images.git
   cd PII-Detector-Medical-Images
   ```

2. Crea un ambiente virtuale
    ```sh
    python -m venv nome_del_venv
     ```
     ```sh
    source nome_del_venv/bin/activate
    o
    .\nome_del_venv\Scripts\activate
    ```

3. Installa le dipendenze:
    ```sh
    pip install -r requirements.txt
    ```

4. Presta attenzione ai parametri dello script

5. Esegui lo script
    ```sh
    python app.py
    ```