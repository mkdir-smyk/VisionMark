import os
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from io import BytesIO

from PIL import Image
from openai import OpenAI

from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType

class VisionDocumentProcessor(BaseDocumentProcessor):
    """Document processor using Qwen2.5-VL via OpenAI-compatible API"""
    
    # Configurable remote model and API settings
    MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct:featherless-ai"
    BASE_URL = "https://router.huggingface.co/v1"
    API_KEY = None
    
    @classmethod
    def configure_api(
        cls,
        api_key=None,
        base_url=None,
        model=None,
    ):
        """
        Configure class-level API settings for the remote vision model.
        """
        if api_key:
            cls.API_KEY = api_key
            
        if base_url:
            cls.BASE_URL = base_url
            
        if model:
            cls.MODEL_PATH = model
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize remote vision document processor with OpenAI SDK
        
        Args:
            api_key: API key for the remote service
            base_url: Base URL for the OpenAI compatible endpoint
            model: Remote model identifier
            temperature: Temperature for generation (0.0-1.0).
            max_tokens: Maximum tokens to generate.
        """
        super().__init__()
        
        self.temperature = float(temperature) if temperature is not None else 0.0
        self.max_tokens = int(max_tokens) if max_tokens is not None else 4000
        
        self.model = model or self.MODEL_PATH
        api_key = (
            api_key
            or self.API_KEY
            or os.getenv("HF_TOKEN")
        )
        self.client = OpenAI(
            base_url=base_url or self.BASE_URL,
            api_key=api_key,
        )
            
    def process(self, file_path: str, max_concurrent: int = 2, images_per_batch: int = 1, 
                dynamic_batching: bool = True, max_tokens_per_batch: int = 4000) -> StructuredDocument:
        """
        Process document using remote vision model
        
        Args:
            file_path: Path to document file
            max_concurrent: Maximum number of concurrent API calls for PDFs
            images_per_batch: Number of consecutive pages to process in a single API call
                (1 = traditional single page processing, 2+ = multi-image processing)
            dynamic_batching: Whether to dynamically determine batch sizes based on image complexity
            max_tokens_per_batch: Maximum tokens per batch when using dynamic batching
            
        Returns:
            StructuredDocument: Processed document
        """
        # Get document type
        doc_type = DocumentType.from_file_extension(file_path)
        
        # Create document with basic metadata
        document = StructuredDocument(
            title=Path(file_path).stem,
            source_file=file_path,
            doc_type=doc_type
        )
        
        # If file is PDF, process each page separately
        if (doc_type == DocumentType.PDF):
            if images_per_batch > 1:
                # Use multi-image processing if specified
                self._process_pdf_multi(file_path, document, max_concurrent, images_per_batch, 
                                        dynamic_batching, max_tokens_per_batch)
            else:
                # Use original single-page processing
                self._process_pdf(file_path, document, max_concurrent)
                
            # Ensure page markers are removed from final document output
            if "markdown" in document.metadata:
                document.metadata["markdown"] = self._remove_page_markers(document.metadata["markdown"])
                
                # Also update section content to remove page markers
                for section in document.sections:
                    for element in section.elements:
                        if element.element_type == "markdown":
                            element.content = self._remove_page_markers(element.content)
                            
        else:
            # For images and other document types
            image_content = self._prepare_image(file_path)
            
            # Get markdown response directly
            markdown_response = self._call_api(image_content)
            
            # Create a section for the entire document
            section = DocumentSection(title=Path(file_path).stem, level=1)
            section.add_element(DocumentElement(
                content=markdown_response,
                element_type="markdown"
            ))
            document.add_section(section)
            
            # Store raw markdown in metadata
            document.metadata["markdown"] = markdown_response
        
        return document
    
    def _prepare_image(self, file_path: str) -> str:
        """Read image file and encode as base64"""
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _process_pdf(self, file_path: str, document: StructuredDocument, max_concurrent: int = 2) -> None:
        """
        Process PDF document with parallel page processing
        """
        from pdf2image import convert_from_path
        import tempfile
        import concurrent.futures
        
        # Storage for combined markdown from all pages
        all_markdown = [None] * 0  # Will resize based on page count
        all_sections = [None] * 0  # Will resize based on page count
        
        try:
            # Convert PDF to images
            with tempfile.TemporaryDirectory() as path:
                # Check if we can convert the PDF
                try:
                    print("Converting PDF to images...")
                    images = convert_from_path(file_path)
                    page_count = len(images)
                    print(f"Converted {page_count} pages")
                    
                    # Resize result arrays
                    all_markdown = [None] * page_count
                    all_sections = [None] * page_count
                    
                except Exception as e:
                    print(f"Error converting PDF to images: {str(e)}")
                    error_section = DocumentSection(title="Error")
                    error_section.add_element(DocumentElement(
                        content=f"Failed to convert PDF to images: {str(e)}",
                        element_type="paragraph"
                    ))
                    document.add_section(error_section)
                    return
                
                # Save all images first
                image_paths = []
                for i, image in enumerate(images):
                    temp_image_path = os.path.join(path, f"page_{i+1}.jpg")
                    image.save(temp_image_path, "JPEG")
                    image_paths.append((i, temp_image_path))
                
                # Process pages in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    # Submit all tasks and store the futures
                    future_to_page = {
                        executor.submit(self._process_single_page, path, index, temp_path): (index, temp_path) 
                        for index, temp_path in image_paths
                    }
                    
                    # Process completed tasks as they finish
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_page):
                        index, temp_path = future_to_page[future]
                        try:
                            section, markdown = future.result()
                            all_sections[index] = section
                            all_markdown[index] = markdown
                            completed += 1
                            print(f"Completed page {index+1}/{page_count} ({completed}/{page_count} done)")
                        except Exception as e:
                            print(f"Error processing page {index+1}: {str(e)}")
                            error_section = DocumentSection(title=f"Error on Page {index+1}")
                            error_section.add_element(DocumentElement(
                                content=f"Failed to process page: {str(e)}",
                                element_type="paragraph"
                            ))
                            all_sections[index] = error_section
                            all_markdown[index] = f"## Page {index+1}\n\nError processing page: {str(e)}"
                
                # Add all sections to document in correct order
                for section in all_sections:
                    if section:
                        document.add_section(section)
                        
                # Combine all markdown
                valid_markdown = [md for md in all_markdown if md]
                if valid_markdown:
                    document.metadata["markdown"] = "\n\n".join(valid_markdown)
                    
        except Exception as e:
            print(f"Error in PDF processing: {str(e)}")
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process PDF: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)

    def _process_pdf_multi(self, file_path: str, document: StructuredDocument, max_concurrent: int = 2, images_per_batch: int = 2, 
                           dynamic_batching: bool = True, max_tokens_per_batch: int = 4000) -> None:
        """
        Process PDF document with multi-image batches for improved context
        """
        from pdf2image import convert_from_path
        import tempfile
        import concurrent.futures
        
        # Storage for combined markdown from all pages
        all_markdown = []
        all_sections = []
        
        try:
            # Convert PDF to images
            with tempfile.TemporaryDirectory() as path:
                # Check if we can convert the PDF
                try:
                    print("Converting PDF to images...")
                    images = convert_from_path(file_path)
                    page_count = len(images)
                    print(f"Converted {page_count} pages")
                    
                    # Initialize result arrays
                    all_markdown = [None] * page_count
                    all_sections = [None] * page_count
                    
                except Exception as e:
                    print(f"Error converting PDF to images: {str(e)}")
                    error_section = DocumentSection(title="Error")
                    error_section.add_element(DocumentElement(
                        content=f"Failed to convert PDF to images: {str(e)}",
                        element_type="paragraph"
                    ))
                    document.add_section(error_section)
                    return
                
                # Save all images first
                image_paths = []
                for i, image in enumerate(images):
                    temp_image_path = os.path.join(path, f"page_{i+1}.jpg")
                    image.save(temp_image_path, "JPEG")
                    image_paths.append((i, temp_image_path))
                
                # Create batch groups
                batch_tasks = []
                
                for i in range(0, len(image_paths), images_per_batch):
                    batch = image_paths[i:i+images_per_batch]
                    if batch:
                        # For each batch, all indices and paths
                        indices = [idx for idx, _ in batch]
                        paths = [p for _, p in batch]
                        batch_tasks.append((indices, paths))
                
                print(f"Processing PDF with {len(batch_tasks)} batches of up to {images_per_batch} pages each")
                
                # Process batches in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    # Submit all batch tasks
                    future_to_batch = {
                        executor.submit(self._process_multi_page_batch, path, indices, paths): (indices, paths) 
                        for indices, paths in batch_tasks
                    }
                    
                    # Process completed batch tasks
                    completed_batches = 0
                    for future in concurrent.futures.as_completed(future_to_batch):
                        indices, paths = future_to_batch[future]
                        try:
                            results = future.result()  # List of (section, markdown) tuples
                            
                            # Store results in correct positions
                            for i, (idx, result) in enumerate(zip(indices, results)):
                                if result:
                                    section, markdown = result
                                    all_sections[idx] = section
                                    all_markdown[idx] = markdown
                            
                            completed_batches += 1
                            processed_pages = min(completed_batches * images_per_batch, page_count)
                            print(f"Completed batch {completed_batches}/{len(batch_tasks)} ({processed_pages}/{page_count} pages)")
                            
                        except Exception as e:
                            print(f"Error processing batch {indices}: {str(e)}")
                            # Create error sections for failed pages
                            for idx in indices:
                                page_num = idx + 1
                                error_section = DocumentSection(title=f"Error on Page {page_num}")
                                error_section.add_element(DocumentElement(
                                    content=f"Failed to process page batch: {str(e)}",
                                    element_type="paragraph"
                                ))
                                all_sections[idx] = error_section
                                all_markdown[idx] = f"## Page {page_num}\n\nError processing page: {str(e)}"
                
                # Add all sections to document in correct order
                for section in all_sections:
                    if section:
                        document.add_section(section)
                        
                # Combine all markdown
                valid_markdown = [md for md in all_markdown if md]
                if valid_markdown:
                    document.metadata["markdown"] = "\n\n".join(valid_markdown)
                    
        except Exception as e:
            print(f"Error in PDF processing: {str(e)}")
            error_section = DocumentSection(title="Error")
            error_section.add_element(DocumentElement(
                content=f"Failed to process PDF: {str(e)}",
                element_type="paragraph"
            ))
            document.add_section(error_section)

    def _process_single_page(self, temp_dir: str, page_index: int, image_path: str) -> tuple:
        """
        Process a single page in the PDF
        """
        print(f"Started processing page {page_index+1}")
        page_num = page_index + 1  # Convert to 1-based page numbers for display
        
        # Process image
        image_content = self._prepare_image(image_path)
        markdown_response = self._call_api(image_content)
        
        # Create section for this page
        section = DocumentSection(title=f"Page {page_num}", level=1)
        section.metadata["page"] = page_num
        section.add_element(DocumentElement(
            content=markdown_response,
            element_type="markdown"
        ))
        
        # Format markdown with page indicator
        page_markdown = f"## Page {page_num}\n\n{markdown_response}"
        
        return section, page_markdown
        
    def _call_api(self, image_content: str) -> str:
        """
        Run remote vision model inference using OpenAI compatible API
        
        Args:
            image_content: Base64 encoded image
            
        Returns:
            str: Markdown representation of the document
        """
        prompt_text = (
            "Convert this document into clean Markdown.\n"
            "Requirements:\n"
            "- preserve headings\n"
            "- preserve tables\n"
            "- preserve lists\n"
            "- preserve code blocks\n"
            "- preserve formatting\n"
            "- preserve hierarchy\n"
            "- do not include page numbers\n"
            "- output Markdown only\n"
            "- do not explain your reasoning"
        )
        
        print("Running remote model inference...")
        try:
            data_uri = f"data:image/jpeg;base64,{image_content}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            markdown_response = response.choices[0].message.content
            print("\nMarkdown response generated successfully")
            return markdown_response
            
        except Exception as e:
            print(f"Error during remote inference: {str(e)}")
            return f"# Model Error\n\nFailed to process document: {str(e)}"

    def _call_api_multi_image(self, image_contents: list, page_numbers: list) -> str:
        """
        Run remote vision model inference with multiple images 
        
        Args:
            image_contents: List of base64 encoded images
            page_numbers: List of page numbers for these images
            
        Returns:
            str: Markdown representation of the document across multiple pages
        """
        print(f"Processing {len(image_contents)} images remotely")
        
        try:
            content = []
            
            for img in image_contents:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                })
                
            prompt_text = (
                f"These are consecutive pages from a document (pages {', '.join(map(str, page_numbers))}). "
                "Convert this document into clean Markdown.\n"
                "Requirements:\n"
                "- preserve headings\n"
                "- preserve tables\n"
                "- preserve lists\n"
                "- preserve code blocks\n"
                "- preserve formatting\n"
                "- preserve hierarchy\n"
                "- do not include page numbers\n"
                "- output Markdown only\n"
                "- do not explain your reasoning"
            )
            
            content.append({
                "type": "text",
                "text": prompt_text
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Multi-image inference failed ({str(e)}). Falling back to sequential processing.")
            
            # Fallback to sequential execution if context window or multi-image processing fails
            results = []
            for img_content, page_num in zip(image_contents, page_numbers):
                print(f"Running fallback inference for page {page_num}...")
                page_markdown = self._call_api(img_content)
                results.append(f"## Page {page_num}\n\n{page_markdown}")
                
            return "\n\n".join(results)

    def _process_multi_page_batch(self, temp_dir: str, page_indices: list, image_paths: list) -> list:
        """
        Process multiple pages in a single API call
        """
        page_numbers = [idx+1 for idx in page_indices]  # Convert to 1-based page numbers
        print(f"Processing batch with pages: {page_numbers}")
        
        # Encode all images in the batch
        image_contents = []
        for path in image_paths:
            image_contents.append(self._prepare_image(path))
        
        # Call API with multiple images
        markdown_response = self._call_api_multi_image(image_contents, page_numbers)
        
        # Split response by page markers
        results = []
        
        try:
            # Check if it's a multi-page response with page markers
            if "## Page" in markdown_response:
                # Attempt to split by page markers
                parts = []
                current_part = []
                current_page_idx = None
                
                for line in markdown_response.split("\n"):
                    if line.strip().startswith("## Page"):
                        # If we already have content for a page, save it
                        if current_part and current_page_idx is not None:
                            parts.append((current_page_idx, "\n".join(current_part)))
                        
                        # Start a new page
                        current_part = [line]
                        
                        # Extract page number from heading
                        try:
                            page_text = line.strip().replace("## Page", "").strip()
                            current_page_idx = int(page_text) - 1  # Convert to 0-based index
                        except:
                            # If we can't extract page number, use position in batch
                            current_page_idx = len(parts)
                    else:
                        current_part.append(line)
                        
                # Add the last part
                if current_part and current_page_idx is not None:
                    parts.append((current_page_idx, "\n".join(current_part)))
                    
                # Create sections for each identified page
                for page_idx, content in parts:
                    if 0 <= page_idx < len(page_indices):
                        idx = page_indices[page_idx]
                        page_num = idx + 1
                        
                        section = DocumentSection(title=f"Page {page_num}", level=1)
                        section.metadata["page"] = page_num
                        section.add_element(DocumentElement(
                            content=content,
                            element_type="markdown"
                        ))
                        
                        # Include page heading if not already there
                        if not content.strip().startswith("## Page"):
                            page_markdown = f"## Page {page_num}\n\n{content}"
                        else:
                            page_markdown = content
                            
                        results.append((section, page_markdown))
            else:
                # If no page markers are found, distribute content evenly
                # This is a fallback and likely to be less accurate
                for i, idx in enumerate(page_indices):
                    page_num = idx + 1
                    
                    # Simple approach - split content by number of pages
                    chunk_size = max(1, len(markdown_response) // len(page_indices))
                    start_pos = i * chunk_size
                    end_pos = start_pos + chunk_size if i < len(page_indices)-1 else len(markdown_response)
                    
                    content = markdown_response[start_pos:end_pos]
                    
                    section = DocumentSection(title=f"Page {page_num}", level=1)
                    section.metadata["page"] = page_num
                    section.add_element(DocumentElement(
                        content=content,
                        element_type="markdown"
                    ))
                    
                    page_markdown = f"## Page {page_num}\n\n{content}"
                    results.append((section, page_markdown))
        except Exception as e:
            print(f"Error parsing multi-page response: {str(e)}")
            # Fallback - create an error section for each page
            for idx in page_indices:
                page_num = idx + 1
                error_content = f"Error parsing multi-page response: {str(e)}"
                
                section = DocumentSection(title=f"Page {page_num}", level=1)
                section.metadata["page"] = page_num
                section.add_element(DocumentElement(
                    content=error_content,
                    element_type="paragraph"
                ))
                
                page_markdown = f"## Page {page_num}\n\n{error_content}"
                results.append((section, page_markdown))
        
        # Ensure we have a result for each page in the batch
        while len(results) < len(page_indices):
            idx = page_indices[len(results)]
            page_num = idx + 1
            error_message = "No content was generated for this page in the batch."
            
            section = DocumentSection(title=f"Page {page_num}", level=1)
            section.metadata["page"] = page_num
            section.add_element(DocumentElement(
                content=error_message,
                element_type="paragraph"
            ))
            
            page_markdown = f"## Page {page_num}\n\n{error_message}"
            results.append((section, page_markdown))
        
        return results

    def _remove_page_markers(self, content: str) -> str:
        """
        Remove page marker headings from the markdown content
        """
        if not content:
            return content
            
        # Split by lines and filter out page marker headings
        lines = content.split('\n')
        filtered_lines = []
        skip_next_empty = False
        
        for line in lines:
            # Check if line is a page marker (## Page X)
            if line.strip().startswith('## Page '):
                skip_next_empty = True  # Skip the next empty line if it exists
                continue
                
            # Skip empty line after page marker
            if skip_next_empty and not line.strip():
                skip_next_empty = False
                continue
                
            filtered_lines.append(line)
            
        # Rejoin the filtered lines
        cleaned_content = '\n'.join(filtered_lines)
        
        return cleaned_content