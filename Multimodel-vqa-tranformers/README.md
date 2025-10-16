# README: Visual Question Answering with BLIP

## Overview
This project utilizes the BLIP (Bootstrapped Language-Image Pretraining) model for Visual Question Answering (VQA). Users can upload an image and ask a question about it, and the model will generate an answer based on the image content.

## Features
- Uses the `Salesforce/blip-vqa-base` model for image-based question answering.
- Accepts image inputs and textual questions.
- Provides text-based answers using BLIP's generation capabilities.
- Simple web interface built using Gradio.
- Supports public sharing of the application via Gradio's sharing link.

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install transformers gradio pillow
```

## Usage
Run the script using:

```bash
python script.py
```

The Gradio interface will launch, allowing users to upload an image and ask a question.

## Environment Variables
- `PORT1`: Specifies the port on which the Gradio server runs (defaults to `7860` if not set).

## Model Details
- **Model**: `Salesforce/blip-vqa-base`
- **Processor**: `AutoProcessor` for processing images and text

## Dependencies
- `transformers`
- `gradio`
- `pillow`
- `os`

## License
This project is released under an open-source license. Refer to `Salesforce/blip-vqa-base` licensing terms for model usage.