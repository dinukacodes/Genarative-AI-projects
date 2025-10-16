import gradio as gr
from openai import OpenAI

# Initialize OpenAI client with vLLM configuration
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

def respond(message, history):
    # Create completion using vLLM
    try:
        completion = client.completions.create(
            model="facebook/opt-125m",
            prompt=message,
            max_tokens=100
        )
        return completion.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface with updated parameters
demo = gr.ChatInterface(
    fn=respond,
    title="AI Chat Interface",
    description="Chat with vLLM-served models",
    examples=[
        "Tell me about yourself",
        "What's the weather like?",
        "Write a short story"
    ]
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True if you want to generate a public URL
        server_name="0.0.0.0",  # Listens on all network interfaces
        server_port=7860  # Default Gradio port
    )