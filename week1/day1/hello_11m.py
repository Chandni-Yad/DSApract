import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
my_api_key = os.getenv("GROQ_api_key")

if not my_api_key:
     raise ValueError("api error")


client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-120b"
role = "user"
prompt = "Do you know chandni"
message = {
     "role":role ,
     "content" :prompt
}
messages = [message]
response = client.chat.completions.create(model = model,messages = messages)
print (response)