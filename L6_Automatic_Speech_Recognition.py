#!/usr/bin/env python
# coding: utf-8

# # Lesson 6: Automatic Speech Recognition

# Install necessary libraries if not already present.
# ```
# !pip install transformers
# !pip install -U datasets
# !pip install soundfile
# !pip install librosa
# !pip install gradio
# ```

# Setup the environment.
from transformers.utils import logging
logging.set_verbosity_error()

# ### Data preparation

# Import the necessary libraries.
from datasets import load_dataset

# Load the dataset.
dataset = load_dataset("librispeech_asr", split="train.clean.100", streaming=True, trust_remote_code=True)

# Display a sample from the dataset.
example = next(iter(dataset))

# Playback an audio sample.
from IPython.display import Audio as IPythonAudio
IPythonAudio(example["audio"]["array"], rate=example["audio"]["sampling_rate"])
# ### Build the pipeline

# Import necessary libraries.
from transformers import pipeline

# Initialize the ASR pipeline.
asr = pipeline(task="automatic-speech-recognition", model="distil-whisper/distil-small.en")

# Verify the feature extractor's sampling rate.
asr.feature_extractor.sampling_rate
# Example transcription.
transcription = asr(example["audio"]["array"])
print(transcription["text"])

# # Function to transcribe speech.
# def transcribe_speech(filepath):
#     if filepath is None:
#         gr.Warning("No audio found, please retry.")
#         return ""
#     output = asr(filepath)
#     return output["text"]
# import os
# import gradio as gr

# # Interface for microphone input.
# mic_transcribe = gr.Interface(
#     fn=transcribe_speech,
#     inputs=gr.Audio(sources="microphone", type="filepath"),
#     outputs=gr.Textbox(label="Transcription", lines=3),
#     allow_flagging="never"
# )

# # Interface for file upload.
# file_transcribe = gr.Interface(
#     fn=transcribe_speech,
#     inputs=gr.Audio(sources="upload", type="filepath"),
#     outputs=gr.Textbox(label="Transcription", lines=3),
#     allow_flagging="never"
# )

# # Combining interfaces into a tabbed layout.
# with gr.Blocks() as demo:
#     gr.TabbedInterface(
#         [mic_transcribe, file_transcribe],
#         ["Transcribe Microphone", "Transcribe Audio File"]
#     )

# # Launch the Gradio interface.
# demo.launch(share=True)

# # To stop the demo:
# #demo.close()
# # import os
# # import gradio as gr

# # # Interface for microphone input.
# # mic_transcribe = gr.Interface(
# #     fn=transcribe_speech,
# #     inputs=gr.Audio(sources="microphone", type="filepath"),
# #     outputs=gr.Textbox(label="Transcription", lines=3),
# #     allow_flagging="never"
# # )

# # # Interface for file upload.
# # file_transcribe = gr.Interface(
# #     fn=transcribe_speech,
# #     inputs=gr.Audio(sources="upload", type="filepath"),
# #     outputs=gr.Textbox(label="Transcription", lines=3),
# #     allow_flagging="never"
# # )

# # # Combining interfaces into a tabbed layout.
# # with gr.Blocks() as demo:
# #     gr.TabbedInterface(
# #         [mic_transcribe, file_transcribe],
# #         ["Transcribe Microphone", "Transcribe Audio File"]
# #     )

# # # Launch the Gradio interface.
# # demo.launch(share=True)

# # # To stop the demo:
# # # demo.close()
# # # , server_port=int(os.environ['PORT1'])