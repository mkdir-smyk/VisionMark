from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType
from pathlib import Path
import pandas as pd
import os

class ExcelProcessor(BaseDocumentProcessor):
    """Processor for Excel files (xlsx, xls, xlsm)"""
    
    def process(self, file_path: str) -> StructuredDocument:
        """
        Process Excel file and convert to markdown
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            StructuredDocument: The processed document
        """
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=DocumentType.EXCEL
        )
        
        try:
            # Get all sheet names
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            
            # Process all sheets and create complete markdown
            all_markdown = []
            
            # Add document title - make it more descriptive
            # Clean up the filename to remove temporary path prefixes
            filename = Path(file_path).stem
            # Remove temporary filename prefixes like med_temp_XXX_
            if 'med_temp_' in filename:
                clean_filename = filename.split('_', 3)[-1] if len(filename.split('_')) > 3 else filename
            else:
                clean_filename = filename
                
            all_markdown.append(f"# Excel Document: {clean_filename}\n\n")
            
            # Track sheet data for document metadata
            sheets_data = {}
            
            # Process each sheet
            for sheet_name in sheet_names:
                # Read the sheet into a DataFrame
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                # Create markdown for this sheet with better heading
                sheet_markdown = f"## Sheet: {sheet_name}"
                
                # Add row and column count for better context
                row_count = len(df)
                col_count = len(df.columns)
                sheet_markdown += f" ({row_count} rows × {col_count} columns)\n\n"
                
                # Convert DataFrame to markdown table
                table_markdown = df.to_markdown(index=False)
                if table_markdown:
                    sheet_markdown += table_markdown + "\n\n"
                else:
                    # Fallback for complex tables
                    sheet_markdown += self._complex_df_to_markdown(df) + "\n\n"
                
                # Add sheet data to metadata
                sheets_data[sheet_name] = {
                    "rows": row_count,
                    "columns": col_count,
                    "column_names": df.columns.tolist()
                }
                
                # Create section for this sheet with improved title
                section = DocumentSection(title=f"Sheet: {sheet_name} ({row_count} rows × {col_count} columns)", level=2)
                section.metadata["sheet_name"] = sheet_name
                section.metadata["rows"] = row_count
                section.metadata["columns"] = col_count
                
                # Add the table as a markdown element
                section.add_element(DocumentElement(
                    content=table_markdown,
                    element_type="markdown"
                ))
                
                # Add section to document
                document.add_section(section)
                
                # Add to combined markdown
                all_markdown.append(sheet_markdown)
            
            # Store the combined markdown in metadata
            document.metadata["markdown"] = "\n".join(all_markdown)
            document.metadata["sheets"] = sheets_data
            document.metadata["sheet_count"] = len(sheet_names)
            
        except Exception as e:
            print(f"Error processing Excel file: {str(e)}")
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process Excel file: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)
            document.metadata["error"] = str(e)
            
        return document
    
    def _complex_df_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert complex DataFrames to markdown tables
        Handles cases where the standard to_markdown might fail
        
        Args:
            df: DataFrame to convert
            
        Returns:
            str: Markdown table representation
        """
        # Create header row
        header = "| " + " | ".join(str(col) for col in df.columns) + " |"
        
        # Create separator row
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        
        # Create data rows
        rows = []
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(val) if not pd.isna(val) else "" for val in row) + " |"
            rows.append(row_str)
        
        # Combine all parts
        return header + "\n" + separator + "\n" + "\n".join(rows)