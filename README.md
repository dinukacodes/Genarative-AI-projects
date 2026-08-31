# Generative AI Projects

A curated collection of small, focused generative-AI example projects and experiments. Each subfolder contains a standalone project demonstrating a particular model, technique, or integration pattern (speech recognition, VQA, RAG, containerized inference, and simple web scraping with LLMs).

This repository is intended as a learning playground: code is typically minimal and focused on demonstrating ideas rather than production-ready packaging. Use the READMEs inside each subproject for detailed setup and usage.

## Contents

- `containerized_inference-engine/` — Minimal example showing how to containerize a small inference script (includes a `Dockerfile` and `frontend.py`). Useful for learning how to build a self-contained inference image.
- `Distil-Whisper-audio-recognition/` — Demonstrates using Distil-Whisper (or Whisper-like models) for automatic speech recognition in a compact example script.
- `Multimodel-vqa-tranformers/` — Image-based VQA (Visual Question Answering) demo using transformers and multimodal inputs. Includes a sample image `beach.jpg` and `imageQA.py`.
- `RAG-from-scratch/` — Retrieval-Augmented Generation (RAG) experiments. Several scripts show how to build a simple retrieval pipeline (FAISS experiments and a Gradio demo script).
- `smart-chatbot-transformers/` — Small chatbot/NLP transformer examples.
- `web-scraper-withLLM/` — Example of simple web-scraping assisted by an LLM (or used to preprocess content for LLMs).

## Quickstart

1. Clone the repository:

   git clone <your-fork-or-url>

2. Pick a subproject you want to try and follow its README. Each subfolder usually contains a short README or `readme.txt` with instructions and any environment notes.

3. Typical steps you'll see across projects:

- Create and activate a Python virtual environment (recommended):

  - Windows PowerShell:

    python -m venv .venv; .\.venv\Scripts\Activate.ps1

  - Linux / macOS:

    python3 -m venv .venv
    source .venv/bin/activate

- Install dependencies (look for requirements in each subfolder or top of the script):

    pip install -r requirements.txt

- Run the example script, for example:

    python Multimodel-vqa-tranformers\imageQA.py

    or

    python RAG-from-scratch\rag_with_gradio.py

  On Linux / macOS use forward slashes:

    python Multimodel-vqa-tranformers/imageQA.py

    or

    python RAG-from-scratch/rag_with_gradio.py

Note: Some projects may require model downloads (transformers, whisper weights, etc.) which can be large. Run scripts while connected to the internet so they can cache models locally.

## Docker (containerized_inference-engine)

If you want to try the containerized example, open `containerized_inference-engine/README.md` and `Dockerfile`. A typical flow:

  docker build -t inference-example ./containerized_inference-engine
  docker run --rm -p 7860:7860 inference-example

(Use PowerShell or your preferred shell — commands above are shell-agnostic.)

## Contributing

Contributions are welcome. A few suggestions:

- Open an issue describing the enhancement or bug you want to address.
- Send small PRs that change one logical thing (fix a typo, add a README note, or improve a script's error handling).
- If you add heavy dependencies or large data files, prefer adding instructions to download them separately rather than committing large binaries.

## License

Each subproject may include its own LICENSE file. See the `containerized_inference-engine/LICENSE` for the repo-level license included in that folder.

## Notes & Tips

- Most scripts are example-focused and may lack production hardening (e.g., limited error handling, simple CLI). Treat them as learning material.
- If a script fails due to missing packages, check the script top comments or the subfolder README for dependency hints.
- When experimenting with models that require GPU acceleration, ensure your environment has the correct CUDA/cuDNN versions and matching PyTorch/transformers builds.


