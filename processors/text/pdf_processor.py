import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType

class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF documents"""
    
    def __init__(self, use_vision_for_complex: bool = True):
        """
        Initialize PDF processor
        
        Args:
            use_vision_for_complex: Whether to use vision processing for complex PDFs
        """
        super().__init__()
        self.use_vision_for_complex = use_vision_for_complex
        
    def process(self, file_path: str) -> StructuredDocument:
        """Process PDF document"""
        # First try text extraction
        document = self._extract_text(file_path)
        
        # If text extraction didn't yield much content and vision processing is enabled,
        # fall back to vision processing
        content_size = sum(len(element.content) for section in document.sections for element in section.elements)
        if content_size < 500 and self.use_vision_for_complex:
            from processors.vision.vision_processor import VisionDocumentProcessor
            try:
                return VisionDocumentProcessor().process(file_path)
            except Exception as e:
                # If vision processing fails, return what we got from text extraction
                print(f"Vision processing failed: {e}")
                pass
                
        return document
        
    def _extract_text(self, file_path: str) -> StructuredDocument:
        """Extract text from PDF using PyPDF2"""
        from PyPDF2 import PdfReader
        
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=DocumentType.PDF
        )
        
        try:
            reader = PdfReader(file_path)
            
            # Simple processing - create one section per page
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text.strip():
                    continue
                    
                section = DocumentSection(title=f"Page {i+1}")
                section.metadata["page"] = i+1
                
                # Simple paragraph splitting
                paragraphs = text.split("\n\n")
                for para in paragraphs:
                    if not para.strip():
                        continue
                        
                    # Try to detect if paragraph is a heading
                    element_type = "paragraph"
                    metadata = {}
                    
                    # Simple heuristic: if paragraph is short and ends with colon, it might be a heading
                    if len(para) < 100 and para.strip().endswith(":"):
                        element_type = "heading"
                        metadata["level"] = 2
                        
                    section.add_element(DocumentElement(
                        content=para.strip(),
                        element_type=element_type,
                        metadata=metadata
                    ))
                    
                document.add_section(section)
                
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            # Create error section
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to extract text: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)
            
        return document