# Source: https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_mixtral.py

import os
import time

from modal import Image, Stub, enter, exit, gpu, method

APP_NAME = "example-vllm-llama-chat"
MODEL_DIR = "/model"
BASE_MODEL = "TheBloke/Llama-2-7B-Chat-AWQ"
GPU_CONFIG = gpu.T4(count=1)


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt"],  # Using safetensors
    )
    move_cache()


vllm_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "vllm==0.3.2",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, timeout=60 * 20)
)

stub = Stub(APP_NAME)


@stub.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60,
    allow_concurrent_inputs=10,
    image=vllm_image,
)
class Model:
    @enter()  # Lifecycle functions
    def start_engine(self):
        import time

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            disable_log_stats=True,  # disable logging so we can stream tokens
            disable_log_requests=True,
            quantization="awk" if "awk" in BASE_MODEL.lower() else None,
        )
        self.template = "<s> [INST]System: You are a helpful assistant.[/INST] [INST] {user} [/INST] "

        # this can take some time!
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=100,
        )

        request_id = random_uuid()
        result_generator = self.engine.generate(
            self.template.format(user=user_question),
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        start = time.monotonic_ns()
        async for output in result_generator:
            if output.outputs[0].text and "\ufffd" == output.outputs[0].text[-1]:
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta
        duration_s = (time.monotonic_ns() - start) / 1e9

        yield f"\n\tGenerated {num_tokens} tokens from {BASE_MODEL} in {duration_s:.1f}s, throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.\n"

    @exit()
    def stop_engine(self):
        pass


@stub.function()
def generate(user_question: str):
    model = Model()

    print("Sending new request:", user_question, "\n\n")

    result = ""
    for text in model.completion_stream.remote_gen(user_question):
        print(text, end="", flush=True)
        result += text

    return result
