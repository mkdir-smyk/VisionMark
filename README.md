# VisionMark

VisionMark is a local-first document conversion pipeline that transforms various file formats into clean, structured Markdown optimized for Large Language Model (LLM) context windows.

While other tools rely on cloud APIs and external services, VisionMark is built to run entirely on your local machine. It uses a hybrid approach: rule-based parsing for standard text formats and native, local execution of the Qwen2.5-VL vision model for complex, scanned, or visually-heavy documents. This ensures your data remains private and you incur zero API costs.

---

## Core Features

- **Fully Local Execution:** Runs Qwen2.5-VL directly via Hugging Face Transformers. No third-party API keys or cloud services required.
- **Format Agnostic:** Native support for PDFs, Office documents (DOCX, PPTX, XLSX), images, code files, and Jupyter Notebooks.
- **Hybrid Pipeline:** Automatically routes text-heavy documents through fast rule-based parsers and uses the vision model for complex layouts or images.
- **Smart PDF Batching:** Dynamically adjusts batch sizes when processing large PDFs to manage VRAM and token limits efficiently.
- **Granular Control:** Offers CLI and UI options to control generation temperature, token limits, and concurrent processing threads.

---


## Supported Formats
   Category   | Formats                          |
 |------------|----------------------------------|
 | Documents  | PDF, DOCX, PPTX, XLSX            |
 | Images     | PNG, JPG, JPEG, BMP              |
 | Code       | Python, R, JavaScript, C++, etc.|
 | Notebooks  | Jupyter Notebooks (.ipynb)      |
 | Markdown   | MD, RMD                          |
 | Text       | TXT                              |

---

## Installation

```bash
git clone https://github.com/mkdir-smyk/VisionMark.git
cd VisionMark

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
