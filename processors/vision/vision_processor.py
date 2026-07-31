import os
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from io import BytesIO

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

from processors.base import BaseDocumentProcessor, StructuredDocument, DocumentSection, DocumentElement, DocumentType

class VisionDocumentProcessor(BaseDocumentProcessor):
    """Document processor using Qwen2.5-VL-3B-Instruct via official Transformers implementation"""
    
    # Configurable local path or HF model ID
    MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # Class-level cache for the loaded model and processor to avoid reloading
    _model_instance = None
    _processor_instance = None
    
    @classmethod
    def configure_api(cls, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        No-op for backward compatibility. 
        API configuration is no longer needed as we use local Hugging Face models.
        """
        pass
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize local vision document processor with Hugging Face Transformers
        
        Args:
            api_key: Ignored (kept for compatibility)
            base_url: Ignored (kept for compatibility)
            model: Local path or HF ID for the Qwen2.5-VL model
            temperature: Temperature for generation (0.0-1.0).
            max_tokens: Maximum tokens to generate.
        """
        super().__init__()
        
        self.temperature = float(temperature) if temperature is not None else 0.0
        self.max_tokens = int(max_tokens) if max_tokens is not None else 4000
        
        model_path = model or self.MODEL_PATH
            
        # Initialize the model and processor once and cache them at the class level
        if VisionDocumentProcessor._model_instance is None:
            print(f"Loading Qwen2.5-VL model from {model_path}...")
            # Use device_map="auto" and torch_dtype="auto" for optimal automatic device placement
            VisionDocumentProcessor._model_instance = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
            print(f"Loading AutoProcessor from {model_path}...")
            VisionDocumentProcessor._processor_instance = AutoProcessor.from_pretrained(model_path)
            
            print("Model and processor loaded successfully.")
            
        self.model = VisionDocumentProcessor._model_instance
        self.processor = VisionDocumentProcessor._processor_instance
            
    def process(self, file_path: str, max_concurrent: int = 2, images_per_batch: int = 1, 
                dynamic_batching: bool = True, max_tokens_per_batch: int = 4000) -> StructuredDocument:
        """
        Process document using local vision model
        
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
        
        Args:
            file_path: Path to PDF file
            document: Document to add content to
            max_concurrent: Maximum number of concurrent API calls
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
        
        Args:
            file_path: Path to PDF file
            document: Document to add content to
            max_concurrent: Maximum number of concurrent API calls
            images_per_batch: Number of consecutive pages to process in a single API call
            dynamic_batching: Whether to dynamically determine batch sizes based on image complexity
            max_tokens_per_batch: Maximum tokens per batch when using dynamic batching
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
        
        Args:
            temp_dir: Temporary directory 
            page_index: Index of the page (0-based)
            image_path: Path to the image file
            
        Returns:
            tuple: (section, markdown) for the page
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
        Run local vision model inference using official Transformers implementation
        
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
        
        print("Running official Hugging Face model inference...")
        try:
            data_uri = f"data:image/jpeg;base64,{image_content}"
            
            # Build messages exactly like the official Qwen documentation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": data_uri
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            # Prepare inputs using the official template layout
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to the correct device
            inputs = inputs.to(self.model.device)
            
            # Setup generation kwargs
            gen_kwargs = {"max_new_tokens": self.max_tokens}
            if self.temperature > 0.0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False
                
            # Execute generation
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
            # Remove input tokens from generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode the generated output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            markdown_response = output_text[0]
            print("\nMarkdown response generated successfully")
            return markdown_response
            
        except Exception as e:
            print(f"Error during local inference: {str(e)}")
            return f"# Model Error\n\nFailed to process document: {str(e)}"

    def _call_api_multi_image(self, image_contents: list, page_numbers: list) -> str:
        """
        Run local vision model inference with multiple images 
        using the official Qwen multi-image format.
        
        Args:
            image_contents: List of base64 encoded images
            page_numbers: List of page numbers for these images
            
        Returns:
            str: Markdown representation of the document across multiple pages
        """
        print(f"Processing {len(image_contents)} images locally via Transformers")
        
        try:
            content_list = []
            
            # Add all images sequentially
            for img_content in image_contents:
                data_uri = f"data:image/jpeg;base64,{img_content}"
                content_list.append({
                    "type": "image",
                    "image": data_uri
                })
                
            # Construct multi-page aware prompt
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
            
            content_list.append({
                "type": "text",
                "text": prompt_text
            })
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
            
            # Prepare and tokenize inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            gen_kwargs = {"max_new_tokens": self.max_tokens}
            if self.temperature > 0.0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False
                
            # Execute generation
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
            # Remove input tokens from generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
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
        
        Args:
            temp_dir: Temporary directory
            page_indices: List of page indices (0-based)
            image_paths: List of paths to image files
            
        Returns:
            list: List of (section, markdown) tuples for each page
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
        
        Args:
            content: Markdown content with page markers
            
        Returns:
            str: Cleaned markdown without page markers
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