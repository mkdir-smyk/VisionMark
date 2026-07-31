from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType
from pathlib import Path
import re

class MarkdownProcessor(BaseDocumentProcessor):
    """Processor for Markdown and R Markdown files"""
    
    def process(self, file_path: str) -> StructuredDocument:
        """Process markdown file"""
        doc_type = DocumentType.from_file_extension(file_path)
        
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=doc_type
        )
        
        try:
            # Read markdown content
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Extract sections based on headings
            section_pattern = r'^(#{1,6})\s+(.+)$'
            current_section = DocumentSection(title="Main Content")
            current_content = []
            last_position = 0
            
            for match in re.finditer(section_pattern, markdown_content, re.MULTILINE):
                # If we have content before this heading, add it to current section
                if last_position < match.start():
                    content_before = markdown_content[last_position:match.start()].strip()
                    if content_before:
                        current_section.add_element(DocumentElement(
                            content=content_before,
                            element_type="markdown"
                        ))
                        current_content.append(content_before)
                
                # Add current section to document if it has content
                if current_content:
                    document.add_section(current_section)
                    current_content = []
                
                # Create new section for this heading
                level = len(match.group(1))  # Number of # characters
                title = match.group(2)
                current_section = DocumentSection(title=title, level=level)
                
                last_position = match.end()
            
            # Add any remaining content to the last section
            if last_position < len(markdown_content):
                remaining_content = markdown_content[last_position:].strip()
                if remaining_content:
                    current_section.add_element(DocumentElement(
                        content=remaining_content,
                        element_type="markdown"
                    ))
                    current_content.append(remaining_content)
            
            # Add the last section if it has content
            if current_content:
                document.add_section(current_section)
            
            # For R Markdown, we could additionally parse and extract R code chunks
            # but for now we'll just handle it as regular markdown
            
            # Store the original markdown content
            document.metadata["markdown"] = markdown_content
            
        except Exception as e:
            print(f"Error processing markdown file: {e}")
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process markdown file: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)
        
        return document