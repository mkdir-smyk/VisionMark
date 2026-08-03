# VisionMark

VisionMark is a document conversion pipeline that transforms PDFs, images, Office documents, notebooks, code files, and other formats into clean, structured Markdown optimized for Large Language Model (LLM) workflows.

It combines traditional parsing for text-based documents with Vision Language Models (VLMs) for scanned or visually complex documents, producing high-quality Markdown while preserving document structure.

---

## Features

- 📄 **Multi-format Support** – Convert PDFs, Office documents, images, notebooks, code, markdown, and plain text.
- 👁️ **Vision-based Document Understanding** – Uses **Qwen2.5-VL-32B-Instruct** via Hugging Face Inference Providers for OCR and layout understanding.
- 📝 **Structured Markdown Output** – Preserves headings, tables, lists, code blocks, and document hierarchy.
- 🔀 **Hybrid Processing Pipeline** – Uses lightweight parsers for text documents and vision models for scanned or image-based content.
- 💻 **Command Line Interface** – Process individual files or entire directories.
- 🌐 **Gradio Web Interface** – Simple browser-based interface for uploading and converting documents.
- ☁️ **Hugging Face Spaces Ready** – Designed for easy deployment on CPU-only Hugging Face Spaces using remote inference.

---

## Supported Formats

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, PPTX, XLSX |
| Images | PNG, JPG, JPEG, BMP |
| Code | Python, Java, JavaScript, C/C++, Go, Rust, R, and more |
| Notebooks | Jupyter Notebook (`.ipynb`) |
| Markdown | `.md`, `.rmd` |
| Text | `.txt` |

---

## Installation

```bash
git clone https://github.com/mkdir-smyk/VisionMark.git
cd VisionMark

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Configuration

VisionMark uses the Hugging Face Router for remote model inference.

Create a Hugging Face Access Token from:

https://huggingface.co/settings/tokens

Export it as an environment variable.

Linux/macOS

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

Windows PowerShell

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

---

## Running the CLI

Process a document:

```bash
python main.py sample.pdf
```

Force vision processing:

```bash
python main.py sample.pdf --force-vision
```

Process an image:

```bash
python main.py receipt.png
```

---

## Running the Web Interface

Launch the Gradio application:

```bash
python app.py
```

Then open:

```
http://127.0.0.1:7860
```

Upload a document and click **Convert** to receive structured Markdown.

---

## Project Structure

```
VisionMark/
│
├── app.py                  # Gradio entry point
├── main.py                 # CLI entry point
├── processors/
│   ├── base.py
│   ├── text/
│   └── vision/
├── ui/
│   ├── __init__.py
│   └── gradio_app.py
├── requirements.txt
└── README.md
```

---

## Model

The default Vision Language Model is:

```
Qwen/Qwen2.5-VL-32B-Instruct:featherless-ai
```

served through the Hugging Face Router using the OpenAI-compatible API.

---

## License

MIT License
