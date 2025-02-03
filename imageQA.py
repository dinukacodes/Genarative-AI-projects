from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

from transformers import BlipForQuestionAnswering, AutoProcessor
from PIL import Image
import gradio as gr
import os

# Load the model and processor
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

def blip_vqa(image, question):
    """Process the image and question to generate an answer using BLIP."""
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Define the Gradio interface
demo = gr.Interface(
    fn=blip_vqa,
    inputs=[
        gr.Image(label="Upload Image", type="pil"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering with BLIP",
    description="Upload an image and ask a question about it. The BLIP model will generate an answer.",
)

# Launch the Gradio app
demo.launch(share=True, server_port=int(os.getenv('PORT1', 7860)))
