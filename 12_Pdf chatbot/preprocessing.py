from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
import re
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_pdf_content(raw_text: str) -> str:
    # Step 1: Remove non-ASCII characters (e.g., emojis or symbols)
    text = raw_text.encode("ascii", errors="ignore").decode()

    # Step 2: Normalize hyphenated line breaks (e.g., "exam-\nple" → "example")
    text = re.sub(r'-\s*\n\s*', '', text)

    # Step 3: Remove standalone line breaks within paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Step 4: Remove repeated newlines or multiple spaces
    text = re.sub(r'\n+', '\n', text)  # multiple newlines to one
    text = re.sub(r'[ \t]+', ' ', text)  # tabs and multi-spaces to one space

    # Step 5: Remove headers/footers like "Page 3 of 12", etc.
    text = re.sub(r'Page\s+\d+\s+(of|/)\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?i)(Confidential|Draft|Internal Use Only|©.*?\d{4})', '', text)
    text = re.sub(r'^\s*\d{1,2}\s*$', '', text, flags=re.MULTILINE)  # Remove isolated page numbers

    # Step 6: Fix bullet point formatting
    text = re.sub(r'•\s*', '- ', text)  # normalize bullets
    text = re.sub(r'^\s*[-–—]+\s*', '- ', text, flags=re.MULTILINE)

    # Step 7: Trim whitespace on lines and remove empty lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = ' '.join(lines)

    # Step 8: Convert to lowercase
    cleaned_text = cleaned_text.lower()

    return cleaned_text.strip()


def extract_and_clean_pdf(uploaded_file):

    print("extract_and_clean_pdf called with file_path:", uploaded_file.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    # Load PDF and split into pages
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Extract metadata from the first page (if available)
    meta_data_book = {
        "title": docs[0].metadata.get("title", ""),
        "author": docs[0].metadata.get("author", ""),
        "date": docs[0].metadata.get("date", ""),
        "publisher": docs[0].metadata.get("publisher", "")
    }

    # Concatenate content from all pages
    raw_text = " ".join(doc.page_content for doc in docs)

    # Clean the extracted text
    cleaned_text = clean_pdf_content(raw_text)

    return cleaned_text, meta_data_book

def split_into_chunks(cleaned_text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    chunks = text_splitter.create_documents([cleaned_text])
    return chunks
