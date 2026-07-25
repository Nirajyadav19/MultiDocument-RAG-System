from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        # If text extraction succeeded, return it
        if any(doc.page_content.strip() for doc in docs):
            return docs

    except Exception as e:
        print(f"PyMuPDF failed: {e}")

    # OCR fallback
    print("Using OCR...")

    images = convert_from_path(pdf_path,poppler_path=r"C:\poppler\poppler-26.02.0\Library\bin")

    documents = []

    for page_no, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        print(page_no)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_no + 1,
                    "source": pdf_path
                }
            )
        )

    return documents