from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

while True:
    query = input("Ask ChatGPT: ")

    if not query:
        break

    response = client.responses.create(
        model="gpt-4.1",
        input=query
    )

    print(response.output_text)