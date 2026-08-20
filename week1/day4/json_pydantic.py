import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"


class Ticket(BaseModel):
    name: str
    email: str
    issue: str
    contact_number: str


schema = Ticket.model_json_schema()
response_format ={
"type":"json_object"
}

system_prompt = f"""
Extract the personal information from the ticket.Return the information according to this JSON schema:

{schema}

Return only a valid JSON object.
"""

message_system = {
    "role": "system",
    "content": system_prompt
}

text = """
Hello my name is Chandni.
I have an iPhone which is not working at all.
My email id is chandni.yadav@example.com.
My contact number is 89123.
"""

prompt = f"""
Please extract the required information from this ticket.and give a json output

{text}
"""

message = {
    "role": "user",
    "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format= response_format

)

answer = response.choices[0].message.content

print(answer)
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)
print(ticket.name)
print(ticket.email)