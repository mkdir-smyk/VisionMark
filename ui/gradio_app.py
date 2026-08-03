import gradio as gr

from processors.base import BaseDocumentProcessor, DocumentType
from processors.vision.vision_processor import VisionDocumentProcessor

MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct:featherless-ai"


def convert_document(
    uploaded_file,
    force_vision,
    temperature,
):
    if uploaded_file is None:
        return "Please upload a document."

    try:
        doc_type = DocumentType.from_file_extension(uploaded_file)

        if force_vision or doc_type == DocumentType.IMAGE:
            processor = VisionDocumentProcessor(
                temperature=temperature,
            )
        else:
            processor = BaseDocumentProcessor.get_processor(uploaded_file)

        document = processor.process(uploaded_file)

        return document.to_markdown()

    except Exception as e:
        return f"Error:\n\n{str(e)}"


def create_ui():

    with gr.Blocks(title="VisionMark") as demo:

        gr.Markdown("# VisionMark")
        gr.Markdown(
            "Convert PDFs and images into structured Markdown using a Vision Language Model."
        )

        file_input = gr.File(
            label="Upload Document",
            type="filepath",
        )

        force_vision = gr.Checkbox(
            value=False,
            label="Force Vision Processing",
        )

        gr.Markdown(f"**Model:** `{MODEL_NAME}`")

        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.1,
            label="Temperature",
        )

        convert_btn = gr.Button("Convert")

        output = gr.Code(
            label="Markdown Output",
            language="markdown",
        )

        convert_btn.click(
            fn=convert_document,
            inputs=[
                file_input,
                force_vision,
                temperature,
            ],
            outputs=output,
        )

    return demo