"""
Script per il rilevamento e l'oscuramento di PII in radiografie in 
formato jpg utilizzando OCR e un modello fine-tuned di BERT.
"""

import argparse
import os
import numpy as np

import cv2
import easyocr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    """Funzione principale per elaborare immagini e rilevare PII."""
    process_images(args.input_folder, args.output_folder)

# Parse parametri
parser = argparse.ArgumentParser(
    description=
    "Rileva ed oscura PII in radiografie in formato jpg usando OCR ed un fine tune di BERT"
)

parser.add_argument(
    "--checkpoint",
    help="Path o l'URL dove risiede il checkpoint",
    default="MrAB01/PersonalInfoClassifier",
    type=str,
)
parser.add_argument(
    "--tokenizer",
    help="Path o l'URL dove risiede il tokenizer",
    default="MrAB01/PersonalInfoClassifier",
    type=str,
)
parser.add_argument(
    "--OCR_confidence",
    help="Threshold di confidenza dell'OCR",
    default=0.75,
    type=float,
)
parser.add_argument(
    "--input_folder",
    help="Path della cartella che contiene le immagini da elaborare",
    default="dataset/images",
    type=str,
)
parser.add_argument(
    "--output_folder",
    help="Path della cartella dove salvare le immagini elaborate",
    default="output_images",
    type=str,
)
parser.add_argument(
    "--do_cover",
    help="True -> Censura PII; False -> Segnala PII",
    default=True,
    type=bool,
)
parser.add_argument(
    "--debug",
    help="Abilita o disabilita info di debug",
    default=False,
    type=bool,
)

args = parser.parse_args()

# Caricamento il modello e tokenizer finetunato
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

# Inizializzazione EasyOCR per l'estrazione del testo in 2 lingue
reader = easyocr.Reader(['en', 'it'])


def cover_text(image, bounding_box):
    """Coprire una bounding box letta dall'OCR con verde chiaro."""
    pts = np.array(bounding_box, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], (144, 238, 144))  # Verde chiaro


def cover_all_text(image, ocr_results):
    """Coprire tutto il testo se rilevato almeno un PII."""
    for result in ocr_results:
        bounding_box = result[0]
        cover_text(image, bounding_box)


def process_image(image_path, output_dir):
    """Processare un'immagine e coprire il PII se rilevato."""
    image = cv2.imread(image_path)

    ocr_results = reader.readtext(image_path, detail=1)

    pii_detected = False

    for result in ocr_results:
        text, confidence = result[1], result[2]
        if args.debug: 
            print(f"Testo rilevato: {text} con confidenza: {confidence}")

        # Processa solo il testo rilevato con confidenza sopra la soglia
        if confidence >= args.OCR_confidence:
            # Tokenizzazione del testo estratto
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # Predizione con il modello
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
            if args.debug: 
                print(prediction)
            # Viene rilevato PII
            if prediction == 1:
                pii_detected = True

    # Se viene rilevato PII, tutto il testo nell'immagine viene coperto
    if pii_detected and args.do_cover:
        cover_all_text(image, ocr_results)
    elif pii_detected and not args.do_cover:
        print(f"PII rilevato in: {image_path}")

    # Salva l'immagine processata
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)


def process_images(image_dir, output_dir):
    """Processare tutte le immagini nella cartella fornita."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image_filename in enumerate(os.listdir(image_dir)):
        if args.debug:
            print(f"Elaborando immagine {i + 1}: {image_filename}")
        if image_filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_filename)
            process_image(image_path, output_dir)


if __name__ == "__main__":
    main()
