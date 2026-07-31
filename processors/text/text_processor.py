from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType
from pathlib import Path

class TextProcessor(BaseDocumentProcessor):
    """Processor for plain text files"""
    
    def process(self, file_path: str) -> StructuredDocument:
        """Process text file"""
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=DocumentType.TEXT
        )
        
        try:
            # Read text content
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Create document structure
            main_section = DocumentSection(title="Text Content")
            main_section.add_element(DocumentElement(
                content=text_content,
                element_type="paragraph"
            ))
            document.add_section(main_section)
            
            # Generate markdown representation
            markdown = f"# {Path(file_path).name}\n\n{text_content}\n"
            document.metadata["markdown"] = markdown
            
        except Exception as e:
            print(f"Error processing text file: {e}")
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process text file: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)
        
        return document