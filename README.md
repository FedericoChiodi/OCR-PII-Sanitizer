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
| `--ocr_confidence_threshold` | `-T`  | Threshold di confidenza dell'OCR sotto alla quale il testo non viene analizzato. | `0.65`                          | `float` |
| `--input_folder`             | `-I`  | Path della cartella che contiene le immagini da elaborare                                     | `dataset/images`                | `str`   |
| `--output_folder`            | `-O`  | Path della cartella dove salvare le immagini elaborate                                        | `output_images`                 | `str`   |
| `--do_censor`                | `-C`  | Presente: Censura PII; Assente: segnala solamente                                             | N/A                             | `bool`  |
| `--cover_all`                | `-A`  | Presente: Censura tutto il testo; Assente: censura solo PII                                             | N/A                             | `bool`  |
| `--log_level`                | `-ll` | Set the logging level                                                                         | WARNING                         | `str`   |

## Installazione

1. Clona la repository:

   ```sh
   git clone https://github.com/FedericoChiodi/PIIDMI-PII-Detector-for-Medical-Images
   cd PIIDMI-PII-Detector-for-Medical-Images
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

4. Installa [PyTorch](https://pytorch.org/get-started/locally/) per il tuo SO e [CUDA version](https://developer.nvidia.com/cuda-downloads)

5. Esegui

    ```sh
    python app.py
    ```

## Crediti, Attribuzioni e Licenze

- [PersonalInfoClassifier](https://huggingface.co/MrAB01/PersonalInfoClassifier) - [MIT License](https://choosealicense.com/licenses/mit/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - [Apache 2.0 License](https://choosealicense.com/licenses/apache-2.0/)
- [DistilBERT](https://arxiv.org/pdf/1910.01108)
