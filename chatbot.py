from openai import OpenAI
from dotenv import load_dotenv
import os

# load the .env file
# this reads the .env file in my application's root directory and loads the variables into my environment
load_dotenv()

# Initialize the OpenAI client with the API key from the environment variables
# os.getenv("OPENAI_API_KEY") fetches the API key stored in the .env file
# This key is required to authenticate API requests to OpenAI's services
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model = 'gpt-3.5-turbo',
    messages = [
        {
            'role': 'system',
            'content': 'You are a marketing specialist at an online graphic design platform that allows users to easily create designs using templates and tools'
        },
        {
            'role': 'user',
            'content': 'Can you help craft a marketing message that highlights our simplicity and vast template library?'
        }
    ]
)

print(response.choices[0].message.content)