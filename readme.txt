# README: Chatbot Client

## Overview

This script initializes a chatbot client using `gradio_client`. It interacts with a remote chatbot by maintaining a conversation history and making API requests.

## Features

- Maintains memory of the conversation.
- Communicates with an external chatbot API using `gradio_client`.
- Allows users to chat interactively.
- Supports exit commands (`exit` or `quit`).

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install gradio_client transformers 
```

## Usage

Run the script using:

```bash
python chatbot_client.py
```

The script will prompt for user input, interact with the chatbot, and maintain a conversation history.

## API Configuration

- **Endpoint**: `rodandegulle/experimental_1`
- **API Name**: `/chat`
- **Parameters**:
  - `message`: User input message
  - `system_message`: Contextual system message with conversation history
  - `max_tokens`: Limits response length (default `512`)
  - `temperature`: Controls response randomness (default `0.7`)
  - `top_p`: Probability sampling parameter (default `0.95`)

## Dependencies

- `gradio_client`

## License

This project is released under an open-source license. Ensure compliance with `rodandegulle/experimental_1` API terms.

