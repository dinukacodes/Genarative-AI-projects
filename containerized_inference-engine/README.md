# Self-Hosted AI Chatbot with vLLM and Gradio  

![Docker](https://img.shields.io/badge/Docker-Enabled-blue)  
![vLLM](https://img.shields.io/badge/vLLM-Inference-orange)  
![Gradio](https://img.shields.io/badge/Gradio-UI-green)  

A self-hosted AI chatbot that runs locally on your machine using **vLLM** for inference and **Gradio** for the frontend. This project is containerized with **Docker**, making it easy to set up and run.  

---

## **Features**  
- **Local Inference**: No need for external APIs—everything runs on your machine.  
- **GPU Support**: Optimized for CUDA-enabled GPUs for faster inference.  
- **User-Friendly Interface**: A simple and intuitive chat interface powered by Gradio.  
- **Dockerized**: Easy to set up and run with Docker.  

---

## **How It Works**  
1. **vLLM Server**:  
   - The vLLM server runs the `facebook/opt-125m` model and exposes an API endpoint at `http://localhost:8000/v1`.  
   - It processes user prompts and generates responses using the model.  

2. **Gradio Frontend**:  
   - The Gradio frontend provides a web-based chat interface.  
   - It sends user messages to the vLLM server and displays the generated responses.  

3. **Docker Container**:  
   - The entire system is packaged into a Docker container for easy deployment.  

---

## **Getting Started**  

### **Prerequisites**  
- Docker installed on your machine.  
- NVIDIA GPU with CUDA support (optional but recommended for faster inference).  

### **Steps to Run**  
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/self-hosted-chatbot.git
   cd self-hosted-chatbot
