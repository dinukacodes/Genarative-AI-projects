

from transformers.utils import logging
logging.set_verbosity_error()


from transformers import pipeline , Conversation

from transformers import Conversation

chatbot = pipeline(task="conversational",
                   model="facebook/blenderbot-400M-distill")
user_message = """
What are some fun activities I can do in the winter?
"""
conversation = Conversation(user_message)
conversation = chatbot(conversation)
conversation.add_message(
    {"role": "user",
     "content": """What else do you recommend?"""
    })

conversation = chatbot(conversation)
conversation.add_message(
    {"role": "user",
     "content": """tell me a bad joke"""
    })

conversation = chatbot(conversation)
print(conversation)

