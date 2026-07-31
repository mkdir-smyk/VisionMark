from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType
from pathlib import Path

class CodeFileProcessor(BaseDocumentProcessor):
    """Processor for code files (Python, R, etc.)"""
    
    # Language mapping for syntax highlighting
    LANGUAGE_MAP = {
        "py": "python",
        "r": "r",
        "js": "javascript",
        "ts": "typescript",
        "sh": "bash",
        "cpp": "cpp",
        "c": "c",
        "java": "java",
        "go": "go",
        "rb": "ruby",
        "php": "php",
    }
    
    def process(self, file_path: str) -> StructuredDocument:
        """Process code file"""
        file_ext = Path(file_path).suffix.lower()[1:]
        doc_type = DocumentType.from_file_extension(file_path)
        
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=doc_type
        )
        
        try:
            # Read code file
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Get language for syntax highlighting
            language = self.LANGUAGE_MAP.get(file_ext, file_ext)
            
            # Create markdown representation
            markdown = f"# {Path(file_path).name}\n\n"
            markdown += f"```{language}\n{code_content}\n```\n"
            
            # Create document structure
            main_section = DocumentSection(title=Path(file_path).name)
            main_section.add_element(DocumentElement(
                content=code_content,
                element_type="code",
                metadata={"language": language}
            ))
            document.add_section(main_section)
            
            # Store markdown
            document.metadata["markdown"] = markdown
            
        except Exception as e:
            print(f"Error processing code file: {e}")
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process code file: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)
        
        return document