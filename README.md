VisionMark

VisionMark is a local-first document conversion pipeline that transforms various file formats into clean, structured Markdown optimized for Large Language Model (LLM) context windows.

While other tools rely on cloud APIs and external services, VisionMark is built to run entirely on your local machine. It uses a hybrid approach: rule-based parsing for standard text formats and native, local execution of the Qwen2.5-VL vision model for complex, scanned, or visually-heavy documents. This ensures your data remains private and you incur zero API costs.

Core Features

Fully Local Execution: Runs Qwen2.5-VL directly via Hugging Face Transformers. No third-party API keys or cloud services required.

Format Agnostic: Native support for PDFs, Office documents (DOCX, PPTX, XLSX), images, code files, and Jupyter Notebooks.

Hybrid Pipeline: Automatically routes text-heavy documents through fast rule-based parsers and uses the vision model for complex layouts or images.

Smart PDF Batching: Dynamically adjusts batch sizes when processing large PDFs to manage VRAM and token limits efficiently.

Granular Control: Offers CLI and UI options to control generation temperature, token limits, and concurrent processing threads.

Supported Formats

Category

Formats

Documents

PDF, DOCX, PPTX, XLSX

Images

PNG, JPG, JPEG, BMP

Code

Python, R, JavaScript, C++, etc.

Notebooks

Jupyter Notebooks (.ipynb)

Markdown

MD, RMD

Text

TXT

Installation

# Clone the repository
git clone https://github.com/mkdir-smyk/VisionMark.git
cd VisionMark

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt




Command Line Interface

For automated workflows, use the CLI. Note that since VisionMark runs the vision model natively, you specify the Hugging Face model pat.

python main.py sample_pdf.pdf \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --force-vision \
    --max-concurrent 2 \
    --output ./processed_docs \
    --images-per-batch 1 \
    --dynamic-batching \
    --max-tokens-per-batch 2000 \
    --temperature 0.0


Key CLI Options

Option

Description

Default

--output, -o

Output directory for markdown files

output

--ui

Launch the Gradio interface

False

--force-vision

Force the vision model for PDFs instead of text extraction

False

--max-concurrent

Number of concurrent workers for PDF processing

2

--images-per-batch

Number of PDF pages per inference batch

1

--dynamic-batching

Auto-adjust batch size based on complexity

True

--max-tokens-per-batch

Maximum tokens per batch

4000

--temperature

Generation temperature (0.0-1.0)

0.0

--model

Local path or HF ID for the Qwen2.5-VL model

Qwen/Qwen2.5-VL-3B-Instruct