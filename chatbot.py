from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from database_schema_info import schema_info
from prompt_template import prompt_template
import os
import tiktoken
import json

# load the .env file
# this reads the .env file in my application's root directory and loads the variables into my environment
load_dotenv()

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = 'https://api.openai.com/v1'
DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKEN_BUDGET = 4096

class ConversationManager():
    def __init__(self, api_key=None, base_url=None, model=None, history_file=None, temperature=None, max_tokens=None, token_budget=None):
        if not api_key:
            api_key = DEFAULT_API_KEY
        if not base_url:
            base_url = DEFAULT_BASE_URL
        # Initialize the OpenAI client with the API key from the environment variables
        # os.getenv("OPENAI_API_KEY") fetches the API key stored in the .env file
        # This key is required to authenticate API requests to OpenAI's services
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        if history_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.history_file = f'conversation_history_{timestamp}.json'
        else:
            self.history_file = history_file

        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        
        self.system_messages = {
            'blogger': 'You are a creative blogger specializing in engaging and informative content for ABC Roasters.',
            'social_media_expert': 'You are a social media expert, crafting catchy and shareable posts for ABC Roasters.',
            'creative_assistant': 'You are a creative assistant skilled in crafting engaging marketing content for ABC Roasters.',
            'data_analyst': 'You are a data analyst skilled in writing SQL queries to analyze Canva\'s database.',
            'custom': 'Enter your custom system message here.',
        }
        self.system_message = self.system_messages['creative_assistant'] # set default persona
        self.load_conversation_history()

    # calculate how many tokens needed to process the given text
    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except:
            encoding = tiktoken.get_encoding('cl100k_base')
        
        tokens = encoding.encode(text)
        return len(tokens)
    
    # calculate how many tokens are being used to process messages in conversation_history
    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.conversation_history)

    def enforce_token_budget(self):
        # Keep removing messages until the token count is within the limit
        while self.total_tokens_used() > self.token_budget:
            # break if only system message is left in the conversation_history list
            if len(self.conversation_history) <= 1:
                break
            # Remove the oldest messages one by one, excluding the system message
            else:
                self.conversation_history.pop(1)
    
    # Change the system message persona based on a predefined list
    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f'Unknown persona: {persona}. Available persona list: {list(self.system_messages.keys())}')
    
    # Update or insert the system message at the start of the conversation history
    def update_system_message_in_history(self):
        if self.conversation_history and self.conversation_history[0]['role'] == 'system':
            self.conversation_history[0]['content'] = self.system_message
        else:
            self.conversation_history.insert(0, {
                'role': 'system',
                'content': self.system_message
            })

    # Set a custom system message and update the persona to use it
    def set_custom_system_message(self, custom_message):
        if not custom_message:
            raise ValueError('Custom message cannot be empty')
        self.system_messages['custom'] = custom_message
        self.set_persona('custom')


    def chat_completion(self, prompt, temperature=None, max_tokens=None):
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # add user's prompt to the conversation history
        self.conversation_history.append({'role': 'user', 'content': prompt})

        self.enforce_token_budget()
        # There are three message roles: system, user, and assistant.
        # The "system" message is the first entry in the conversation history. It defines the AI's role, such as a helpful marketing assistant, which guides the AI in understanding the context of the interaction
        # "User" messages represent the questions or instructions given by the user
        # "Assistant" messages are the AI's responses to the user's inputs
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f'An error occurred while generating a response: {e}')
            return None

        ai_response = response.choices[0].message.content
        
        # add AI's response to the history
        self.conversation_history.append({'role': 'assistant', 'content': ai_response})

        # save the conversation history after the chat completion
        self.save_conversation_history()
        return ai_response
    
    # Loads the conversation history from a JSON file if it exists
    def load_conversation_history(self):
        try:
            with open(self.history_file, 'r') as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{
                'role': 'system', 
                'content': self.system_message, 
            }]
        except json.JSONDecodeError:
            print("Error reading the conversation history file. Starting with an empty history.")
            self.conversation_history = [{
                'role': 'system', 
                'content': self.system_message, 
            }]

    # Saves the current conversation history to a JSON file
    def save_conversation_history(self):
        try:
            with open(self.history_file, 'w') as file:
                json.dump(self.conversation_history, file, indent=4)
        except IOError as e:
            print(f'An I/O error occurred while saving the conversation history: {e}')
        except Exception as e:
            print(f'An unexpected error occurred while saving the conversation history: {e}')

conv_manager = ConversationManager()
conv_manager.set_persona('data_analyst')

# Simulate chat completion
question = input("Enter your question: ")

prompt = f'tbd' # create prompt using question, prompt_template and schema_info
response = conv_manager.chat_completion(prompt)

print(response)

