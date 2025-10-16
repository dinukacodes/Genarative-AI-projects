#!/bin/bash
# Start VLLM in the background
vllm serve facebook/opt-125m &

# Start the Python frontend
python frontend.py
