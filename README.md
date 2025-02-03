# Automatic Speech Recognition (ASR) with DistilWhisper

This project demonstrates the use of **Automatic Speech Recognition (ASR)** with the `distil-whisper` model to transcribe audio to text. The project leverages the Hugging Face `transformers` library and Gradio for real-time transcription from both microphone input and uploaded audio files.

## Requirements

Before running the code, you need to install the necessary dependencies. You can install them by running the following commands:

```bash
pip install transformers
pip install -U datasets
pip install soundfile
pip install librosa
pip install gradio
```

## Data Preparation

The dataset used in this example is from the [LibriSpeech ASR dataset](https://www.openslr.org/12/), specifically the clean training subset. We are using the `train.clean.100` split for demonstration purposes. 

- The dataset is streamed directly from Hugging Face using the `datasets` library.
- A sample is displayed, and its audio is played back to demonstrate the ASR system's input format.

## ASR Pipeline

The **DistilWhisper** model is used for automatic speech recognition in this project. It is a smaller, distilled version of the Whisper model by OpenAI.

### Model Setup

```python
from transformers import pipeline

# Initialize the ASR pipeline with the distil-whisper model
asr = pipeline(task="automatic-speech-recognition", model="distil-whisper/distil-small.en")

# Verify the feature extractor's sampling rate
asr.feature_extractor.sampling_rate
```

### Example Transcription

The following code demonstrates transcribing a sample audio from the dataset:

```python
transcription = asr(example["audio"]["array"])
print(transcription["text"])
```

## Microphone and File Upload Interface with Gradio

Gradio is used to create interactive interfaces for the ASR system. There are two interfaces:

1. **Microphone Input**: This allows users to transcribe speech using a microphone.
2. **File Upload**: This interface allows users to upload an audio file for transcription.

### Microphone Transcription

```python
mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Textbox(label="Transcription", lines=3),
    allow_flagging="never"
)
```

### File Upload Transcription

```python
file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Textbox(label="Transcription", lines=3),
    allow_flagging="never"
)
```

### Combining Interfaces

Both the microphone and file upload interfaces are combined into a tabbed layout using Gradio:

```python
with gr.Blocks() as demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"]
    )

# Launch the Gradio interface
demo.launch(share=True)
```

## Usage

To run the project:

1. Install the necessary dependencies.
2. Run the Python script.
3. The Gradio interface will launch in your browser, where you can choose between the microphone transcription or uploading an audio file.

## Notes

- The **DistilWhisper** model supports English audio.
- Ensure that your microphone is properly set up if using the microphone interface.
- The project is designed to run on a local environment, but it also supports sharing via Gradio's `share=True` option.

## Closing

If you'd like to stop the demo, you can close the Gradio interface by calling `demo.close()`.