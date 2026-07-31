import os
import argparse
from pathlib import Path
from processors.base import BaseDocumentProcessor, DocumentType
from processors.vision.vision_processor import VisionDocumentProcessor

def process_file(file_path, output_dir, force_vision=False, max_concurrent=2, images_per_batch=1, 
               temperature=0.0, max_tokens=None, dynamic_batching=True, max_tokens_per_batch=4000):
    """Process a single file"""
    try:
        print(f"Processing: {file_path}")
        
        # Determine document type first
        doc_type = DocumentType.from_file_extension(file_path)
        
        # Select appropriate processor based on type and options
        if force_vision and doc_type == DocumentType.PDF:
            print(f"Using Vision processor for PDF: {file_path}")
            processor = VisionDocumentProcessor(temperature=temperature, max_tokens=max_tokens)
        elif doc_type == DocumentType.IMAGE:
            print(f"Using Vision processor for image: {file_path}")
            processor = VisionDocumentProcessor(temperature=temperature, max_tokens=max_tokens)
        else:
            # Use standard document detection
            processor = BaseDocumentProcessor.get_processor(file_path)
            
        print(f"Using {processor.__class__.__name__} for {file_path}")
        
        # Process document - with max_concurrent and images_per_batch for PDFs
        if isinstance(processor, VisionDocumentProcessor) and doc_type == DocumentType.PDF:
            document = processor.process(
                file_path, 
                max_concurrent=max_concurrent, 
                images_per_batch=images_per_batch,
                dynamic_batching=dynamic_batching,
                max_tokens_per_batch=max_tokens_per_batch
            )
            print(f"Processed PDF with {max_concurrent} concurrent workers, dynamic batching: {dynamic_batching}")
        else:
            document = processor.process(file_path)
        
        # Generate output filename
        base_name = Path(file_path).stem
        output_base = os.path.join(output_dir, base_name)
        
        md_path = f"{output_base}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(document.to_markdown())
        print(f"Markdown saved to: {md_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_directory(input_dir, output_dir, force_vision=False, max_concurrent=2, images_per_batch=1,
                    temperature=0.0, max_tokens=None, dynamic_batching=True, max_tokens_per_batch=4000):
    """Process all files in directory"""
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path, output_dir, force_vision, max_concurrent, images_per_batch,
                        temperature, max_tokens, dynamic_batching, max_tokens_per_batch)

def main():
    parser = argparse.ArgumentParser(description="Convert documents to structured markdown")
    
    parser.add_argument("input", nargs="?", help="Input file or directory path")
    parser.add_argument("--output", "-o", help="Output directory path", default="output")
    parser.add_argument("--ui", action="store_true", help="Launch graphical user interface")
    parser.add_argument("--force-vision", action="store_true", 
                       help="Force using vision model for PDFs instead of text extraction")
    parser.add_argument("--max-concurrent", type=int, default=2,
                       help="Maximum number of concurrent workers for PDF page processing (default: 2)")
    parser.add_argument("--api-key", help="API key for Qwen vision processor")
    parser.add_argument("--base-url", help="Base URL for API")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--images-per-batch", type=int, default=1,
                       help="Maximum number of PDF pages to process in a single API call (default: 1, 2+ enables multi-image processing)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for vision model generation (0.0-1.0, lower is more deterministic)")
    parser.add_argument("--max-tokens", type=int, 
                       help="Maximum tokens for vision model generation (default uses model's limit)")
    parser.add_argument("--dynamic-batching", action="store_true", default=True,
                       help="Automatically determine optimal batch size based on image complexity. The optimal batch size will not exceed the number set by --images-per-batch.")
    parser.add_argument("--no-dynamic-batching", action="store_false", dest="dynamic_batching",
                       help="Disable dynamic batching and use fixed images-per-batch")
    parser.add_argument("--max-tokens-per-batch", type=int, default=4000,
                       help="Maximum tokens per batch when using dynamic batching (default: 4000)")
    
    args = parser.parse_args()
    
    # Launch UI if requested
    if args.ui:
        from ui.app import create_ui
        app = create_ui()
        # Allow access to any path
        app.launch(allowed_paths=["/"])
        return
    
    # If no input is provided and not in UI mode, print help
    if not args.input:
        parser.print_help()
        return
    
    # Set API configuration if provided
    if args.api_key:
        from processors.vision.vision_processor import VisionDocumentProcessor
        VisionDocumentProcessor.configure_api(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.input):
        # Process directory
        process_directory(
            args.input, args.output, args.force_vision, args.max_concurrent, args.images_per_batch,
            args.temperature, args.max_tokens, args.dynamic_batching, args.max_tokens_per_batch
        )
    else:
        # Process single file
        process_file(
            args.input, args.output, args.force_vision, args.max_concurrent, args.images_per_batch,
            args.temperature, args.max_tokens, args.dynamic_batching, args.max_tokens_per_batch
        )

if __name__ == "__main__":
    main()