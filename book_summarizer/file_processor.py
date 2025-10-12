import pypdf
import docx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import IO

def get_text_from_txt(file: IO[bytes]) -> str:
    """Lee y decodifica un archivo de texto plano."""
    return file.read().decode("utf-8", errors="ignore")

def get_text_from_pdf(file: IO[bytes]) -> str:
    """Extrae texto de un archivo PDF."""
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def get_text_from_docx(file: IO[bytes]) -> str:
    """Extrae texto de un archivo DOCX."""
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text])

def get_text_from_epub(file: IO[bytes]) -> str:
    """Extrae texto de un archivo EPUB."""
    book = epub.read_epub(file)
    text = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        text += soup.get_text() + "\n"
    return text