VisionMark

VisionMark is a local-first document conversion pipeline that transforms various file formats into clean, structured Markdown optimized for Large Language Model (LLM) context windows.

Unlike other tools that rely on external APIs and cloud services, VisionMark is built to run entirely on your local machine. It uses a hybrid approach: fast rule-based parsing for standard text formats and native, local execution of the Qwen2.5-VL vision model (via Hugging Face Transformers) for complex, scanned, or visually heavy documents. This ensures your data remains private and you incur zero API costs.

Core Features

Fully Local Execution: Runs Qwen2.5-VL directly via Hugging Face Transformers. No third-party API keys or cloud services required.

Format Agnostic: Native support for PDFs, Office documents (DOCX, PPTX, XLSX), images, code files, and Jupyter Notebooks.

Hybrid Pipeline: Automatically routes text-heavy documents through fast rule-based extractors and uses the vision model for complex layouts or images.

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
git clone https://github.com/RoffyS/VisionMark.git
cd VisionMark

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Usage

Command Line Interface

For automated workflows, use the CLI. VisionMark runs the vision model natively, so you only need to specify the local path or Hugging Face model ID.

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

Web Interface

If you prefer a graphical interface, you can launch the local web UI via Gradio:

python main.py --ui


The UI allows you to drag and drop files, configure processing parameters (like concurrent workers and temperature), and view the conversion progress in real-time.

Roadmap

Recently Implemented

Temperature Control: Added temperature parameter for controlling the determinism of AI output.

Max Tokens Setting: Implemented customizable token limits for generation.

Multi-Page Processing: Support for processing multiple PDF pages in a single inference call.

Dynamic Batch Sizing: Intelligent adjustment of batch sizes based on page complexity.

Enhanced Office Support: Improved preservation of table structures in Word (DOCX) and Excel (XLSX) files.

Planned Features

CSV and TSV Support: Native parsing for plain-text tabular data files.

Custom Templates: User-defined output formats for different document types.

Multi-model Support: Integration with additional local vision-language models beyond Qwen.

API Mode: Headless operation for integration with other local applications.

Project Structure

VisionMark/
├── main.py               # Entry point
├── ui/
│   └── app.py            # Gradio UI implementation
├── processors/
│   ├── base.py           # Base extraction classes
│   ├── text/             # Rule-based text processors
│   └── vision/           # Native HF vision processor
├── test_docs/            # Example documents
├── test_output/          # Example processed results
└── requirements.txt      # Dependencies


Acknowledgements

The core philosophy of this project was inspired by a post from Andrej Karpathy noting that most documentation is still formatted for human reading rather than LLM context windows. VisionMark aims to bridge that gap by converting complex formats into flat, LLM-readable Markdown.

Special thanks to the Qwen team and the open-source community for the Qwen2.5-VL models and the Hugging Face Transformers library.

License

MIT License
