from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType
from pathlib import Path
import json
import nbformat

class JupyterNotebookProcessor(BaseDocumentProcessor):
    """Processor for Jupyter Notebooks"""
    
    def process(self, file_path: str) -> StructuredDocument:
        """Process Jupyter Notebook"""
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=DocumentType.JUPYTER
        )
        
        try:
            # Parse the notebook
            notebook = nbformat.read(file_path, as_version=4)
            
            # Create a section for each cell
            markdown_parts = []
            
            # Add notebook title if available
            if 'title' in notebook.metadata:
                markdown_parts.append(f"# {notebook.metadata['title']}\n\n")
            
            # Process cells
            for i, cell in enumerate(notebook.cells):
                cell_type = cell.cell_type
                
                # For markdown cells, add the content directly
                if cell_type == 'markdown':
                    markdown_parts.append(cell.source + "\n\n")
                    
                    # Also add to document structure
                    section = DocumentSection(title=f"Markdown Cell {i+1}")
                    section.add_element(DocumentElement(
                        content=cell.source,
                        element_type="markdown"
                    ))
                    document.add_section(section)
                    
                # For code cells, format as code blocks with output
                elif cell_type == 'code':
                    # Code
                    markdown_parts.append(f"```python\n{cell.source}\n```\n\n")
                    
                    # Add code to document structure
                    section = DocumentSection(title=f"Code Cell {i+1}")
                    section.add_element(DocumentElement(
                        content=cell.source,
                        element_type="code",
                        metadata={"language": "python"}
                    ))
                    
                    # Process output if available
                    if hasattr(cell, 'outputs') and cell.outputs:
                        outputs = []
                        for output in cell.outputs:
                            if output.output_type == 'stream':
                                outputs.append(f"```\n{output.text}\n```\n\n")
                            elif output.output_type == 'execute_result':
                                if 'text/plain' in output.data:
                                    outputs.append(f"```\n{output.data['text/plain']}\n```\n\n")
                        
                        if outputs:
                            output_content = "".join(outputs)
                            markdown_parts.append(f"**Output:**\n\n{output_content}")
                            section.add_element(DocumentElement(
                                content=output_content,
                                element_type="code_output"
                            ))
                    
                    document.add_section(section)
            
            # Store combined markdown in metadata
            document.metadata["markdown"] = "".join(markdown_parts)
            
        except Exception as e:
            print(f"Error processing notebook: {e}")
            # Create error section
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process notebook: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)
        
        return document