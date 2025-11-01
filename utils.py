"""
Utility functions for Resume Optimizer.
Provides PDF text extraction and web scraping functionality.
"""
import PyPDF2
import requests
from bs4 import BeautifulSoup
import tempfile
import logging
from typing import Any

# Setup logger
logger = logging.getLogger(__name__)

# Constants
REQUEST_TIMEOUT = 8


def extract_text_from_pdf(uploaded_file: Any) -> str:
    """
    Extract text from PDF file.
    
    Accepts either a file path (str) or a file-like object (like Streamlit's uploaded file).
    
    Args:
        uploaded_file: File path string or file-like object
        
    Returns:
        Concatenated text from all pages or empty string on failure
    """
    text = ""
    try:
        # If uploaded_file is file-like, write to temp file so PyPDF2 can read reliably
        if hasattr(uploaded_file, "read"):
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                with open(tmp.name, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
        else:
            # Assume it's a file path
            with open(uploaded_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return text
    
    return text


def extract_text_from_url(url: str) -> str:
    """
    Extract text content from a web page URL.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Extracted text content or error message
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            return soup.get_text(separator=" ", strip=True)
        else:
            logger.warning(f"Failed to fetch URL {url}: Status {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {e}")
        return str(e)

