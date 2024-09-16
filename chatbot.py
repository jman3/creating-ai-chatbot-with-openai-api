from openai import OpenAI
from dotenv import load_dotenv
import os

# load the .env file
# this reads the .env file in my application's root directory and loads the variables into my environment
load_dotenv()

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = 'https://api.openai.com/v1'
DEFAULT_MODEL = 'gpt-3.5-turbo'

class ConversationManager():
    def __init__(self, api_key=None, base_url=DEFAULT_BASE_URL):
        if not api_key:
            api_key = DEFAULT_API_KEY
        if not base_url:
            base_url = DEFAULT_BASE_URL
        # Initialize the OpenAI client with the API key from the environment variables
        # os.getenv("OPENAI_API_KEY") fetches the API key stored in the .env file
        # This key is required to authenticate API requests to OpenAI's services
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(self, prompt):
        # There are three message roles: system, user, and assistant.
        # The "system" message is the first entry in the conversation history. It defines the AI's role, such as a helpful marketing assistant, which guides the AI in understanding the context of the interaction
        # "User" messages represent the questions or instructions given by the user
        # "Assistant" messages are the AI's responses to the user's inputs
        messages = [
            {
                'role': 'system',
                'content': 'You are a marketing specialist at an online graphic design platform that allows users to easily create designs using templates and tools'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages = messages
        )
        return response.choices[0].message.content

conv_manager = ConversationManager()
prompt = 'Can you help craft a marketing message that highlights our simplicity and vast template library?'

response = conv_manager.chat_completion(prompt)
print(response)