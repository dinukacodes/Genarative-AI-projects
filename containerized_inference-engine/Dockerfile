# Use an official Python image
FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Install VLLM and other required libraries
RUN pip install --no-cache-dir vllm openai flask gradio

# Copy your frontend script
COPY frontend.py .

# Expose ports for both services
EXPOSE 8000 5000

# Use a simple shell script to run both backend (VLLM) and frontend
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
