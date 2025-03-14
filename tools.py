from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

import fitz  # PyMuPDF
from langchain.tools import Tool
import os

PDF_DIRECTORY = "docs"  # Directorul unde sunt plasate fișierele PDF

def extract_text_from_pdf(filename: str) -> str:
    """Extracts text from a given PDF file."""
    try:
        doc = fitz.open(filename)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text if text else "No text found in this document."
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"

def query_pdfs(folder_path: str = "docs/") -> str:
    """Searches all PDFs in the specified folder and extracts their content."""
    if not os.path.exists(folder_path):
        return f"Folder {folder_path} does not exist."

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        return "No PDF files found in the docs folder."

    extracted_data = []
    for pdf in pdf_files:
        pdf_path = os.path.join(folder_path, pdf)
        extracted_text = extract_text_from_pdf(pdf_path)
        extracted_data.append(f"### {pdf} ###\n{extracted_text}\n")

    return "\n".join(extracted_data)

pdf_query_tool = Tool(
    name="pdf_query",
    func=lambda _: query_pdfs(),  # Nu mai are nevoie de un parametru direct
    description="Extracts text from all PDFs in the docs folder.",
)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)