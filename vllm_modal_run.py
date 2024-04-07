import timeit

import modal

APP_NAME = "example-vllm-llama-chat"
f = modal.Function.lookup(APP_NAME, "generate")

start_time = timeit.default_timer()

print(f.remote("What are the names of the four main characters of South park?"))

elapsed = timeit.default_timer() - start_time

print(f"{elapsed=}")
