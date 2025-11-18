import pypdf
import docx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import IO
import tempfile
import os

def get_text_from_txt(file: IO[bytes]) -> str:
    return file.read().decode("utf-8", errors="ignore")

def get_text_from_pdf(file: IO[bytes]) -> str:
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def get_text_from_docx(file: IO[bytes]) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text])

def get_text_from_epub(file: IO[bytes]) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name
    
    try:
        book = epub.read_epub(tmp_path)
        text = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text() + "\n"
        return text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)