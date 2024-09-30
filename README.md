# PIIDMI - PII Detector for Medical Images

Questo progetto utilizza un modello DistilBERT finetuned per il rilevamento e l'opzionale censura di informazioni personali (PII: Personally Identifiable Information).

Questa soluzione combina EasyOCR per l'estrazione del testo dalle immagini, ed un sistema di classificazione binario per identificare se il testo contiene PII o meno.

## Funzionalità

- Estrazione del testo da immagini mediche tramite EasyOCR.
- Rilevamento di PII analizzando il testo estratto.
- Opzionale censura del testo, guidata dalla bounding box provvista dall'OCR.
- Supporto multi-lingua (inglese e italiano).
- Report dettagliato delle predizioni con metriche di classificazione.
- Tempo di elaborazione per un'immagine <= 1s (misurato su RTX 3060)

## Argomenti disponibili

| Argomento                    | Short | Descrizione                                                                                   | Default                         | Tipo    |
| ---------------------------- | ----- | --------------------------------------------------------------------------------------------- | ------------------------------- | ------- |
| `--model`                    | `-m`  | Path o URL del modello                                                                        | `MrAB01/PersonalInfoClassifier` | `str`   |
| `--tokenizer`                | `-t`  | Path o URL del tokenizer                                                                      | `MrAB01/PersonalInfoClassifier` | `str`   |
| `--ocr_confidence_threshold` | `-T`  | Threshold di confidenza dell'OCR sotto alla quale il testo non viene analizzato. Default=0.75 | `0.75`                          | `float` |
| `--input_folder`             | `-I`  | Path della cartella che contiene le immagini da elaborare                                     | `dataset/images`                | `str`   |
| `--output_folder`            | `-O`  | Path della cartella dove salvare le immagini elaborate                                        | `output_images`                 | `str`   |
| `--do_censor`                | `-C`  | Presente: Censura PII; Assente: segnala solamente                                             | N/A                             | `bool`  |
| `--cover_all_text`                    | `-A`  | Presente: Copre tutto il testo se viene rilevato un PII; Assente: copre solo il PII rilevato                            | N/A                             | `bool`  |
| `--debug`                    | `-d`  | Presente: Abilita info di debug; Assente: disabilita info di debug                            | N/A                             | `bool`  |

## Installazione

1. Clona la repository:

   ```sh
   git clone https://github.com/FedericoChiodi/OCR-PII-Sanitizer.git
   cd PII-Detector-Medical-Images
   ```

2. Crea un ambiente virtuale (consigliato) ed attivalo

    ```sh
    python -m venv pii_detect
    ./pii_detect/Scripts/activate
    ```

3. Installa le dipendenze:

    ```sh
    pip install -r requirements.txt
    ```

4. Esegui

    ```sh
    python app.py --debug
    ```

## Credits

- [PersonalInfoClassifier](https://huggingface.co/MrAB01/PersonalInfoClassifier)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [DistilBERT](https://arxiv.org/pdf/1910.01108)