import argparse
import os
import cv2
from easyocr import Reader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import numpy as np

SUPPORTED_FORMATS = ["jpg", "png"]
FILL_COLOR = (144, 238, 144)  # Light Green

class ModelLabels:
    SAFE = 0
    PII = 1

# Set up global logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(levelname)s :: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

# Determine available torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    logger.warning(f"Could not use CUDA as device, falling back to {device}")
else:
    logger.info(f"Using CUDA as torch device.")

def get_filename_extension(filename: str):
    return str(filename).split(".")[-1].lower()

def cover_text(image, bounding_box):
    """Draws a rectangle over a given bounding box"""
    pts = np.array(bounding_box, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], FILL_COLOR)

def cover_all_text(image, ocr_results):
    """Covers all text in the image."""
    for result in ocr_results:
        bounding_box = result[0]
        cover_text(image, bounding_box)

def cover_pii_text(image, ocr_results, pii_indices):
    """Covers only the text detected as PII."""
    for idx in pii_indices:
        bounding_box = ocr_results[idx][0]
        cover_text(image, bounding_box)

def process_image(image_path, output_directory, tokenizer, model, reader, args):
    """Processes an image and censors PII if detected."""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"\tFailed to read image '{image_path}'. Skipping.")
        return

    ocr_results = reader.readtext(image_path, detail=1)
    consider_all_as_pii = False
    pii_indices = []

    if not ocr_results:
        logger.info(f"\tNo text detected in {image_path}")
    else:
        for i, result in enumerate(ocr_results):
            snippet, confidence = result[1], result[2]
            logger.info(f"\t[{confidence}] {snippet}")

            # Process only text with OCR confidence above the threshold
            if confidence < args.ocr_confidence_threshold:
                logger.warning(
                    f"\tSkipping analysis of text snippet '{snippet}' in {image_path} (OCR confidence < {args.ocr_confidence_threshold})"
                )
                continue

            # Tokenization of extracted text
            inputs = tokenizer(
                snippet, return_tensors="pt", truncation=True, padding=True
            )

            # Inference with selected model
            with torch.no_grad():
                outputs = model(**inputs.to(device))
                logits = outputs.logits

                # if binary classifier returns 1, PII was detected
                prediction = torch.argmax(logits, dim=-1).item()
                logger.info("\tPII" if prediction == ModelLabels.PII else "\tSAFE")

                if prediction == ModelLabels.PII:
                    consider_all_as_pii = True
                    pii_indices.append(i)

    if args.do_censor:
        if consider_all_as_pii:
            if args.cover_all:
                # Cover all text if PII is detected and --cover_all is active
                logger.info(f"\tCensoring ALL TEXT in: {image_path}")
                cover_all_text(image, ocr_results)
            else:
                # Cover only detected PII text
                logger.info(f"\tCensoring ONLY PII TEXT in: {image_path}")
                cover_pii_text(image, ocr_results, pii_indices)

    # Save the processed image
    output_path = os.path.join(output_directory, os.path.basename(image_path))
    try:
        cv2.imwrite(output_path, image)
    except Exception as e:
        logger.critical(
            f"Failed to write image '{output_path}' with exception '{e}'. CONTINUING"
        )

def process_images(input_directory, output_directory, tokenizer, model, reader, args):
    """Processes all images in the specified directory."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        logger.info(f"Output directory created: {output_directory}")

    for i, image_filename in enumerate(os.listdir(input_directory)):
        logger.info(f"Analyzing [{i+1}]: {image_filename} ...")
        ext = get_filename_extension(image_filename)
        image_path = os.path.join(input_directory, image_filename)

        if ext in SUPPORTED_FORMATS:
            process_image(image_path, output_directory, tokenizer, model, reader, args)
        else:
            logger.error(f"Unsupported image format, skipping ({image_path})")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description=f"Script per il rilevamento e l'oscuramento di PII in radiografie estraendone il testo tramite OCR ed analizzandolo con un modello BERT fine-tuned\nFormati immagine supportati: {SUPPORTED_FORMATS}"
    )

    # Register arguments
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
        help="Threshold di confidenza dell'OCR sotto alla quale il testo non viene analizzato. Default=0.65",
        default=0.65,
        type=float,
    )
    parser.add_argument(
        "--input_directory",
        "-I",
        help="Path della cartella che contiene le immagini da elaborare",
        default=os.path.join("dataset", "test_images"),
        type=str,
    )
    parser.add_argument(
        "--output_directory",
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
        "--cover_all",
        "-A",
        help="Presente: Censura tutto il testo; Assente: censura solo PII",
        action="store_true",
    )
    parser.add_argument(
        "--log_level",
        "-ll",
        help="Set the log level",
        default="WARNING",
        choices=logging._levelToName.values(),
        type=str,
    )

    # Parse command line args
    args = parser.parse_args()

    # Set log level
    logger.setLevel(logging._nameToLevel[args.log_level])

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        logger.critical(f"Failed to load tokenizer with exception '{e}'. ABORTING")
        exit(1)

    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model).to(
            device
        )
    except Exception as e:
        logger.critical(f"Failed to load model with exception '{e}'. ABORTING")
        exit(1)

    # Init OCR
    reader = Reader(["en", "it"])

    # Start processing
    process_images(
        args.input_directory, args.output_directory, tokenizer, model, reader, args
    )

if __name__ == "__main__":
    main()
