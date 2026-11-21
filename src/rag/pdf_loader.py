from pathlib import Path
from io import BytesIO

import pymupdf
import pytesseract
from PIL import Image


# Layout configurations for known pages.
# Add new document/page layouts here instead of writing new functions.

PAGE_LAYOUTS = {
    ("who_india_diabetes_protocol.pdf", 22): {
        "sections": [
            {
                "title": "DIAGNOSIS AND TREATMENT TARGETS",
                "bounds": (150, 1500, 430, 710)
            },
            {
                "title": "ENTRY CRITERIA",
                "bounds": (150, 1550, 730, 820)
            },
            {
                "title": "TREATMENT STEPS",
                "steps": [
                    (840, 970),
                    (970, 1140),
                    (1140, 1350),
                    (1350, 1530),
                    (1530, 1700),
                    (1700, 1900)
                ],
                "bounds": (250, 900)
            },
            {
                "title": "IMPORTANT CONSIDERATIONS",
                "bounds": (900, 1550, 850, 1900)
            },
            {
                "title": "METFORMIN NOTES",
                "bounds": (150, 850, 1900, 2120)
            },
            {
                "title": "HYPOGLYCEMIA",
                "bounds": (850, 1550, 1900, 2120)
            }
        ]
    },

    ("who_india_diabetes_protocol.pdf", 14): {
        "sections": [
            {
                "title": "BLOOD GLUCOSE ASSESSMENT",
                "bounds": (130, 1600, 330, 800)
            },
            {
                "title": "TREATMENT STEPS",
                "steps": [
                    (830, 1035),
                    (1035, 1220),
                    (1220, 1380),
                    (1380, 1570),
                    (1570, 1780),
                    (1780, 1950)
                ],
                "bounds": (150, 950)
            },
            {
                "title": "IMPORTANT CONSIDERATIONS",
                "bounds": (1000, 1650, 850, 1950)
            },
            {
                "title": "FOOTNOTES",
                "bounds": (130, 1650, 1950, 2250)
            }
        ]
    }
}


def get_text_blocks(page, bounds):
    x_min, x_max, y_min, y_max = bounds

    blocks = [
        block for block in page.get_text("blocks")
        if x_min <= block[0]
        and block[2] <= x_max
        and y_min <= block[1] < y_max
        and block[4].strip()
    ]

    return sorted(blocks, key=lambda b: (b[1], b[0]))


def extract_configured_page(page, config):
    sections = []

    for section in config["sections"]:
        sections.append(f"\n{section['title']}")

        if "steps" in section:
            x_min, x_max = section["bounds"]

            for number, (start_y, end_y) in enumerate(
                section["steps"], start=1
            ):
                sections.append(f"\nSTEP {number}")

                blocks = get_text_blocks(
                    page,
                    (x_min, x_max, start_y, end_y)
                )

                for block in blocks:
                    sections.append(block[4].strip())

        else:
            blocks = get_text_blocks(page, section["bounds"])

            for block in blocks:
                sections.append(block[4].strip())

    return "\n".join(sections)


def extract_text_from_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_pages = []

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            layout_key = (pdf_path.name, page_number)
            config = PAGE_LAYOUTS.get(layout_key)

            if config:
                text = extract_configured_page(page, config)
                method = "native"

            else:
                native_text = page.get_text("text").strip()

                if len(native_text) >= 100:
                    text = native_text
                    method = "native"

                else:
                    pix = page.get_pixmap(
                        matrix=pymupdf.Matrix(2, 2),
                        alpha=False
                    )

                    image = Image.open(
                        BytesIO(pix.tobytes("png"))
                    )

                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang="eng",
                        config="--psm 3"
                    ).strip()

                    if len(ocr_text) > len(native_text):
                        text = ocr_text
                        method = "ocr"

                    elif native_text:
                        text = native_text
                        method = "native_sparse"

                    else:
                        text = ""
                        method = "empty"

            extracted_pages.append({
                "page_number": page_number,
                "text": text,
                "extraction_method": method
            })

    return extracted_pages


def load_medical_documents(folder_path="documents/medical_knowledge"):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    documents = []
    pdf_files = sorted(folder_path.glob("*.pdf"))

    for pdf_file in pdf_files:

        pages = extract_text_from_pdf(pdf_file)

        documents.append({
            "document_name": pdf_file.name,
            "pages": pages
        })

        native_count = sum(
            p["extraction_method"] == "native" for p in pages
        )
        ocr_count = sum(
            p["extraction_method"] == "ocr" for p in pages
        )
        sparse_count = sum(
            p["extraction_method"] == "native_sparse" for p in pages
        )
        empty_count = sum(
            p["extraction_method"] == "empty" for p in pages
        )

        print(f"\nProcessed: {pdf_file.name}")
        print(f"Total pages: {len(pages)}")
        print(f"Native text pages: {native_count}")
        print(f"OCR pages: {ocr_count}")
        print(f"Sparse native pages: {sparse_count}")
        print(f"Empty pages: {empty_count}")

    if not pdf_files:
        print("No PDF documents found.")

    return documents
