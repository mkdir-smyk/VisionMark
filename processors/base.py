from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pathlib import Path
import os
import json

class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    IMAGE = "image"  # jpg, png, etc.
    WORD = "docx"
    POWERPOINT = "pptx"
    JUPYTER = "ipynb"
    PYTHON = "py"
    R_SCRIPT = "r"
    R_MARKDOWN = "rmd"
    MARKDOWN = "md"
    TEXT = "txt"
    EXCEL = "excel"  # xlsx, xls files
    UNKNOWN = "unknown"
    
    @classmethod
    def from_file_extension(cls, file_path: str) -> "DocumentType":
        """Determine document type from file extension"""
        ext = os.path.splitext(file_path.lower())[1][1:]
        
        if ext in ("jpg", "jpeg", "png", "bmp", "tiff"):
            return cls.IMAGE
        elif ext == "pdf":
            return cls.PDF
        elif ext == "docx":
            return cls.WORD
        elif ext == "pptx":
            return cls.POWERPOINT
        elif ext == "ipynb":
            return cls.JUPYTER
        elif ext == "py":
            return cls.PYTHON
        elif ext == "r":
            return cls.R_SCRIPT
        elif ext == "rmd":
            return cls.R_MARKDOWN
        elif ext == "md":
            return cls.MARKDOWN
        elif ext == "txt":
            return cls.TEXT
        elif ext in ("xlsx", "xls", "xlsm"):
            return cls.EXCEL
        else:
            return cls.UNKNOWN

class DocumentElement:
    """Base class for document elements"""
    def __init__(
        self, 
        content: str, 
        element_type: str,
        position: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.element_type = element_type
        self.position = position or {}
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "content": self.content,
            "type": self.element_type,
            "position": self.position,
            "metadata": self.metadata
        }
        
    def to_markdown(self) -> str:
        """Convert to markdown representation"""
        if self.element_type == "heading":
            level = self.metadata.get("level", 1)
            return f"{'#' * level} {self.content}\n\n"
        elif self.element_type == "paragraph":
            return f"{self.content}\n\n"
        elif self.element_type == "code":
            lang = self.metadata.get("language", "")
            return f"```{lang}\n{self.content}\n```\n\n"
        elif self.element_type == "list_item":
            return f"- {self.content}\n"
        elif self.element_type == "image":
            alt = self.metadata.get("alt", "Image")
            src = self.metadata.get("src", "")
            return f"![{alt}]({src})\n\n"
        elif self.element_type == "table":
            # Basic table rendering - in real implementation this would be more complex
            return f"**Table**: {self.content}\n\n"
        else:
            return f"{self.content}\n\n"

class DocumentSection:
    """Section of a document containing multiple elements"""
    def __init__(
        self,
        title: Optional[str] = None,
        elements: Optional[List[DocumentElement]] = None,
        level: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.title = title
        self.elements = elements or []
        self.level = level
        self.metadata = metadata or {}
        
    def add_element(self, element: DocumentElement) -> None:
        """Add an element to the section"""
        self.elements.append(element)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "title": self.title,
            "level": self.level,
            "elements": [element.to_dict() for element in self.elements],
            "metadata": self.metadata
        }
        
    def to_markdown(self) -> str:
        """Convert to markdown representation"""
        result = ""
        if self.title:
            result += f"{'#' * self.level} {self.title}\n\n"
            
        for element in self.elements:
            result += element.to_markdown()
            
        return result

class StructuredDocument:
    """Complete structured document"""
    def __init__(
        self,
        title: Optional[str] = None,
        source_file: Optional[str] = None,
        doc_type: Optional[DocumentType] = None,
        sections: Optional[List[DocumentSection]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.title = title
        self.source_file = source_file
        self.doc_type = doc_type or DocumentType.UNKNOWN
        self.sections = sections or []
        self.metadata = metadata or {}
        
    def add_section(self, section: DocumentSection) -> None:
        """Add a section to the document"""
        self.sections.append(section)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "title": self.title,
            "source_file": self.source_file,
            "doc_type": self.doc_type.value if self.doc_type else None,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata
        }
        
    def to_markdown(self) -> str:
        """Convert document to markdown format"""
        # If we have direct markdown in metadata, use that
        if "markdown" in self.metadata:
            return self.metadata["markdown"]
        
        # Otherwise, build markdown from structure
        parts = []
        
        # Add document title
        if self.title:
            parts.append(f"# {self.title}\n\n")
        
        # Add each section
        for section in self.sections:
            # Add section title with appropriate heading level
            if section.title:
                level = section.level if section.level else 1
                heading = "#" * (level + 1)  # +1 because document title is h1
                parts.append(f"{heading} {section.title}\n\n")
            
            # Add elements
            for element in section.elements:
                if element.element_type == "markdown":
                    # For markdown elements, just add the content directly
                    parts.append(f"{element.content}\n\n")
                else:
                    # For other elements, convert to markdown
                    if element.element_type == "paragraph":
                        parts.append(f"{element.content}\n\n")
                    elif element.element_type == "heading":
                        level = element.metadata.get("level", 2)
                        heading = "#" * (level + 1)  # +1 because doc title is h1
                        parts.append(f"{heading} {element.content}\n\n")
                    elif element.element_type == "list":
                        for item in element.content.split("\n"):
                            parts.append(f"- {item.strip()}\n")
                        parts.append("\n")
                    elif element.element_type == "code":
                        lang = element.metadata.get("language", "")
                        parts.append(f"```{lang}\n{element.content}\n```\n\n")
                    else:
                        # Default behavior for unknown element types
                        parts.append(f"{element.content}\n\n")
        
        return "".join(parts)
        
    def save_markdown(self, output_path: str) -> None:
        """Save as markdown file"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
            
    def get_direct_markdown(self) -> str:
        """
        Get direct markdown if available, otherwise generate from structure
        
        Returns:
            str: Markdown representation of the document
        """
        if "direct_markdown" in self.metadata:
            return self.metadata["direct_markdown"]
        else:
            return self.to_markdown()

class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    def __init__(self):
        """Initialize the document processor"""
        pass
        
    @abstractmethod
    def process(self, file_path: str) -> StructuredDocument:
        """
        Process a document and return structured content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            StructuredDocument: The processed document
        """
        pass

    @classmethod
    def get_processor(cls, file_path: str, force_vision: bool = False) -> "BaseDocumentProcessor":
        """Factory method to get appropriate processor for a file"""
        doc_type = DocumentType.from_file_extension(file_path)
        
        # Import here to avoid circular imports
        from processors.text.pdf_processor import PDFProcessor
        from processors.vision.vision_processor import VisionDocumentProcessor
        from processors.text.docx_processor import DocxProcessor
        from processors.text.pptx_processor import PowerPointProcessor
        from processors.text.ipynb_processor import JupyterNotebookProcessor
        from processors.text.code_processor import CodeFileProcessor
        from processors.text.markdown_processor import MarkdownProcessor
        from processors.text.text_processor import TextProcessor
        from processors.text.excel_processor import ExcelProcessor
        
        # Use vision processor if forced (applies to PDF)
        if force_vision:
            return VisionDocumentProcessor()
        
        # Return appropriate processor based on document type
        if doc_type == DocumentType.PDF:
            return PDFProcessor()
        elif doc_type == DocumentType.WORD:
            return DocxProcessor()
        elif doc_type == DocumentType.POWERPOINT:
            return PowerPointProcessor() 
        elif doc_type == DocumentType.JUPYTER:
            return JupyterNotebookProcessor()
        elif doc_type == DocumentType.IMAGE:
            return VisionDocumentProcessor()
        elif doc_type in (DocumentType.PYTHON, DocumentType.R_SCRIPT):
            return CodeFileProcessor()
        elif doc_type in (DocumentType.MARKDOWN, DocumentType.R_MARKDOWN):
            return MarkdownProcessor()
        elif doc_type == DocumentType.EXCEL:
            return ExcelProcessor()
        elif doc_type == DocumentType.TEXT:
            return TextProcessor()
        else:
            # Default to text processor for unknown types
            return TextProcessor()