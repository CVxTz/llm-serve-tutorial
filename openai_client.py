import os
import timeit

from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "DEFAULT")
MODEL = os.getenv("MODEL", "models/7B/llama-2-7b.Q5_K_M.gguf")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

start_time = timeit.default_timer()
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "What are the names of the four main characters of South park?",
        }
    ],
    temperature=0.0,
    max_tokens=20

)

print(response.choices[0].message.content)
elapsed = timeit.default_timer() - start_time

print(f"{elapsed=}")
