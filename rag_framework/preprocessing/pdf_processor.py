"""PDF processing module for extracting and structuring content."""

from typing import List, Dict, Optional
import re


class PDFProcessor:
    """Process PDF files and extract hierarchical structure."""

    def __init__(self):
        """Initialize PDF processor."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 package is required. Install with: pip install pypdf2")
        
        self.PdfReader = PdfReader

    def extract_text(self, pdf_path: str) -> str:
        """Extract raw text from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        reader = self.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_hierarchical_structure(self, text: str) -> Dict[str, any]:
        """Create hierarchical structure from text.
        
        This function identifies sections, subsections, and paragraphs.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary representing hierarchical structure
        """
        # Simple hierarchical structure based on heading patterns
        lines = text.split('\n')
        structure = {
            "title": "",
            "sections": []
        }
        
        current_section = None
        current_subsection = None
        
        # Patterns for detecting headings
        title_pattern = re.compile(r'^[A-Z][A-Z\s]+$')  # ALL CAPS
        section_pattern = re.compile(r'^(\d+\.?\s+|[A-Z]\.\s+)')  # 1. or A.
        subsection_pattern = re.compile(r'^(\d+\.\d+\.?\s+)')  # 1.1.
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect title
            if not structure["title"] and title_pattern.match(line):
                structure["title"] = line
                continue
            
            # Detect section
            if section_pattern.match(line) and not subsection_pattern.match(line):
                current_section = {
                    "heading": line,
                    "content": "",
                    "subsections": []
                }
                structure["sections"].append(current_section)
                current_subsection = None
                continue
            
            # Detect subsection
            if subsection_pattern.match(line):
                if current_section is not None:
                    current_subsection = {
                        "heading": line,
                        "content": ""
                    }
                    current_section["subsections"].append(current_subsection)
                continue
            
            # Add content to current context
            if current_subsection is not None:
                current_subsection["content"] += line + " "
            elif current_section is not None:
                current_section["content"] += line + " "
        
        return structure

    def process_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Process PDF and return hierarchical structure.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Hierarchical structure dictionary
        """
        text = self.extract_text(pdf_path)
        structure = self.create_hierarchical_structure(text)
        return structure
