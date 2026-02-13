"""Text chunking module for splitting documents."""

from typing import List, Dict, Optional


class TextChunker:
    """Split text into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """Initialize text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: Primary separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split by separator first
        splits = text.split(self.separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split would exceed chunk_size, save current chunk
            if len(current_chunk) + len(split) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + split
                else:
                    current_chunk = split
            else:
                if current_chunk:
                    current_chunk += self.separator + split
                else:
                    current_chunk = split
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def chunk_hierarchical_structure(
        self, structure: Dict
    ) -> List[Dict[str, str]]:
        """Chunk hierarchical document structure.
        
        Args:
            structure: Hierarchical structure from PDFProcessor
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        title = structure.get("title", "")
        
        for section_idx, section in enumerate(structure.get("sections", [])):
            section_heading = section.get("heading", "")
            section_content = section.get("content", "")
            
            # Chunk section content
            if section_content:
                section_chunks = self.chunk_text(section_content)
                for chunk in section_chunks:
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "title": title,
                            "section": section_heading,
                            "section_idx": section_idx,
                            "type": "section"
                        }
                    })
            
            # Process subsections
            for subsection_idx, subsection in enumerate(section.get("subsections", [])):
                subsection_heading = subsection.get("heading", "")
                subsection_content = subsection.get("content", "")
                
                if subsection_content:
                    subsection_chunks = self.chunk_text(subsection_content)
                    for chunk in subsection_chunks:
                        chunks.append({
                            "text": chunk,
                            "metadata": {
                                "title": title,
                                "section": section_heading,
                                "subsection": subsection_heading,
                                "section_idx": section_idx,
                                "subsection_idx": subsection_idx,
                                "type": "subsection"
                            }
                        })
        
        return chunks
