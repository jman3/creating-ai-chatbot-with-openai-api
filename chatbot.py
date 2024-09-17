from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken

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
    def __init__(self, api_key=None, base_url=None, model=None, temperature=None, max_tokens=None, token_budget=None):
        if not api_key:
            api_key = DEFAULT_API_KEY
        if not base_url:
            base_url = DEFAULT_BASE_URL
        # Initialize the OpenAI client with the API key from the environment variables
        # os.getenv("OPENAI_API_KEY") fetches the API key stored in the .env file
        # This key is required to authenticate API requests to OpenAI's services
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        
        self.conversation_history = []
        self.system_messages = {
            'blogger': 'You are a creative blogger specializing in engaging and informative content for ABC Roasters.',
            'social_media_expert': 'You are a social media expert, crafting catchy and shareable posts for ABC Roasters.',
            'creative_assistant': 'You are a creative assistant skilled in crafting engaging marketing content for ABC Roasters.',
            'custom': 'Enter your custom system message here.',
        }
        self.system_message = self.system_messages['blogger'] # set default persona

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=temperature,
            max_tokens=max_tokens
        )
        ai_response = response.choices[0].message.content
        
        # add AI's response to the history
        self.conversation_history.append({'role': 'assistant', 'content': ai_response})
        return ai_response

conv_manager = ConversationManager(token_budget=80)
conv_manager.set_persona('creative_assistant')

prompt = 'Can you help craft a marketing message that highlights our simplicity and vast template library?'
response = conv_manager.chat_completion(prompt, temperature=0.5, max_tokens=100)

print(f'creative_assistant\'s answer: ')
print(response)

conv_manager.set_custom_system_message('You always generate witty and humorous content for ABC roasters ')
prompt = 'Can you help craft a marketing message that highlights our simplicity and vast template library?'
response = conv_manager.chat_completion(prompt, temperature=0.5, max_tokens=100)

print(f'\ncustom persona\'s answer: ')
print(response)