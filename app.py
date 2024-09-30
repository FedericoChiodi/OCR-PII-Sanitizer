"""
Script per il rilevamento e l'oscuramento di PII (Personally Identifiable Information) in radiografie in 
formato jpg/png utilizzando OCR ed un modello fine-tuned di BERT.
"""

import argparse
import os
import numpy as np

import cv2
import easyocr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SUPPORTED_FORMATS = ["jpg", "png"]


def main():
    """Funzione principale per rilevare ed opzionalmente censurare PII."""
    process_images(args.input_folder, args.output_folder)


# Parse parametri
parser = argparse.ArgumentParser(
    description=f"Script per il rilevamento e l'oscuramento di PII (Personally Identifiable Information) in radiografie estraendone il testo tramite OCR ed analizzandolo con un modello BERT fine-tuned\nFormati immagine supportati: {SUPPORTED_FORMATS}"
)

parser.add_argument(
    "--model",
    "-m",
    help="Path o URL del modello",
    default="MrAB01/PersonalInfoClassifier",
    type=str,
)
parser.add_argument(
    "--tokenizer",
    "-t",
    help="Path o URL del tokenizer",
    default="MrAB01/PersonalInfoClassifier",
    type=str,
)
parser.add_argument(
    "--ocr_confidence_threshold",
    "-T",
    help="Threshold di confidenza dell'OCR sotto alla quale il testo non viene analizzato. Default=0.75",
    default=0.75,
    type=float,
)
parser.add_argument(
    "--input_folder",
    "-I",
    help="Path della cartella che contiene le immagini da elaborare",
    default=os.path.join("dataset", "images"),
    type=str,
)
parser.add_argument(
    "--output_folder",
    "-O",
    help="Path della cartella dove salvare le immagini elaborate",
    default="output_images",
    type=str,
)
parser.add_argument(
    "--do_censor",
    "-C",
    help="Presente: Censura PII; Assente: segnala solamente",
    action="store_true",
)
parser.add_argument(
    "--debug",
    "-d",
    help="Presente: Abilita info di debug; Assente: disabilita info di debug",
    action="store_true",
)

args = parser.parse_args()

# Caricamento del modello e del tokenizer fine-tuned
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(args.model)

# Inizializzazione EasyOCR per l'estrazione del testo in 2 lingue
reader = easyocr.Reader(["en", "it"])


def cover_text(image, bounding_box):
    """Copre una bounding box letta dall'OCR con verde chiaro."""
    pts = np.array(bounding_box, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], (144, 238, 144))  # Verde chiaro


def cover_all_text(image, ocr_results):
    """Copre tutto il testo se rilevato almeno una PII."""
    for result in ocr_results:
        bounding_box = result[0]
        cover_text(image, bounding_box)


def process_image(image_path, output_dir):
    """Processa un'immagine e copre le PII se rilevate."""
    image = cv2.imread(image_path)

    ocr_results = reader.readtext(image_path, detail=1)

    pii_detected = False

    for result in ocr_results:
        text, confidence = result[1], result[2]
        if args.debug:
            print(f"\tTesto rilevato: {text}")
            print(f"\tConfidenza OCR: {confidence}")

        # Processa solo il testo rilevato con confidenza sopra la soglia
        if confidence >= args.ocr_confidence_threshold:
            # Tokenizzazione del testo estratto
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # Predizione con il modello
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
            if args.debug:
                print("\tPrevisione    :", "PII" if prediction else "ANON")
            # Viene rilevata PII
            if prediction == 1:
                pii_detected = True
        else:
            if args.debug:
                print(
                    f"\tAnalisi saltata (confidenza OCR < {args.ocr_confidence_threshold})"
                )

    # Se viene rilevata PII, tutto il testo nell'immagine viene coperto
    if pii_detected:
        if args.do_censor:
            cover_all_text(image, ocr_results)
        else:
            print(f"PII rilevato in: {image_path}")

    # Salva l'immagine processata
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)


def process_images(image_dir, output_dir):
    """Processa tutte le immagini e le salva nella cartella specificata."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image_filename in enumerate(os.listdir(image_dir)):
        if args.debug:
            print(f"Elaborando immagine {i + 1}: {image_filename}")
        ext = str(image_filename).split(".")[-1]
        image_path = os.path.join(image_dir, image_filename)
        if ext in SUPPORTED_FORMATS:
            process_image(image_path, output_dir)
        else:
            if args.debug:
                print(
                    f"\tWARNING: rilevato un formato immagine non supportato (.{ext}) : {image_path}"
                )


if __name__ == "__main__":
    main()
