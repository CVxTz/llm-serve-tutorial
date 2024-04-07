# LLM serving tutorial

Different ways to serve LLMs, examples applied to LLama 2 7B

* [Setup Anaconda Env](#setup-anaconda-env)
* [Local - CPU](#local---cpu)
  + [Install pre-compiled lib](#install-pre-compiled-lib)
  + [Download 5-bit quantized model in GGUF format](#download-5-bit-quantized-model-in-gguf-format)
  + [Run Server](#run-server)
    
## Setup Anaconda Env

Install Anaconda: https://docs.anaconda.com/free/anaconda/install/linux/

```bash
conda create -n llm-serve-tutorial python=3.10
conda activate llm-serve-tutorial
pip install -r requiremets.txt
```

## Local Server - Anaconda + CPU
Using llama-cpp-python | https://github.com/abetlen/llama-cpp-python 

### Install pre-compiled lib

```
pip install llama-cpp-python[server] \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
### Download 5-bit quantized model in GGUF format

```
mkdir -p models/7B
wget -O models/7B/llama-2-7b-chat.Q5_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true
```
### Run Server

```
python3 -m llama_cpp.server --model models/7B/llama-2-7b-chat.Q5_K_M.gguf
```

### Query server

```
export MODEL="models/7B/llama-2-7b-chat.Q5_K_M.gguf"
python openai_client.py
```
Prompt:
```
messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "What are the names of the four main characters of South park?",
    },
]
```

Response:
```
Ah, a simple question! The four main characters of South Park are:
1. Stan Marsh
2. Kyle Broflovski
3. Eric Cartman
4. Kenny McCormick

These four boys have been the central characters of the show since its debut in 1997 and have been the source of countless laughs and controversy over the years!
```
Processing time: 13.4s (Intel® Core™ i9-10900F CPU @ 2.80GHz × 20)

## Local Server - Anaconda + GPU
Using vllm | https://github.com/vllm-project/vllm?tab=readme-ov-file

### Install lib

```
pip install vllm
```

### Run Server

```
python -m vllm.entrypoints.openai.api_server --model TheBloke/Llama-2-7B-Chat-AWQ --api-key DEFAULT --quantization awq --enforce-eager
```

### Query server

```
export MODEL="TheBloke/Llama-2-7B-Chat-AWQ"
python openai_client.py
```
Prompt:
```
messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "What are the names of the four main characters of South park?",
    },
]
```

Response:
```
Of course! The four main characters of South Park are:

1. Stan Marsh
2. Kyle Broflovski
3. Eric Cartman
4. Kenny McCormick

These four characters have been the central figures of the show since its debut in 1997 and have been the main focus of most episodes throughout the series.
```
Processing time: 0.79s (Nvidia RTX 3080 + Intel® Core™ i9-10900F CPU @ 2.80GHz × 20)


## Local Server - Docker + GPU
Using vllm | https://github.com/vllm-project/vllm?tab=readme-ov-file

I assume you already have docker. If not, install it: https://docs.docker.com/engine/install/
Install Nvidia Docker runtime: (Ubuntu)
```
sudo apt install nvidia-cuda-toolkit
```
Then
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Then:
```
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo pkill -SIGHUP dockerd
```
Source: https://stackoverflow.com/questions/59008295/add-nvidia-runtime-to-docker-runtimes and https://github.com/NVIDIA/nvidia-docker/issues/1238

Run Docker:

```
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model TheBloke/Llama-2-7B-Chat-AWQ \
    --quantization awq --enforce-eager
```

### Query server

```
export MODEL="TheBloke/Llama-2-7B-Chat-AWQ"
python openai_client.py
```
Prompt:
```
messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "What are the names of the four main characters of South park?",
    },
]
```

Response:
```
Of course! The four main characters of South Park are:

1. Stan Marsh
2. Kyle Broflovski
3. Eric Cartman
4. Kenny McCormick

These four characters have been the central figures of the show since its debut in 1997 and have been the main focus of most episodes throughout the series.
```
Processing time: 0.81s (Nvidia RTX 3080 + Intel® Core™ i9-10900F CPU @ 2.80GHz × 20)


## Modal
Docs: https://modal.com/docs/
Example: https://modal.com/docs/examples/vllm_mixtral

### install modal

```
pip install modal
```

### Auth

```
modal setup
```

### Deploy
```
modal deploy vllm_modal_deploy.py
```

### Run
```
python vllm_modal_run.py
```
Response:

```
The four main characters of South Park are:

1. Stan Marsh
2. Kyle Broflovski
3. Eric Cartman
4. Kenny McCormick

These four characters have been the central characters of the show since its premiere in 1997 and have been the main focus of the series throughout its many seasons.
```
* Generated 77 tokens from TheBloke/Llama-2-7B-Chat-AWQ in 2.2s, throughput = 35 tokens/second on GPU(T4, count=1).

Processing time (Cold start): 37.5 s 
Processing time (Warm start): 2.87 s

Cold start doc: https://modal.com/docs/guide/cold-start 

## AnyScale

```
export API_KEY="CHANGEME"
export MODEL="meta-llama/Llama-2-7b-chat-hf"
export BASE_URL="https://api.endpoints.anyscale.com/v1"
python openai_client.py
```

Response:
```
Ah, a question that is sure to bring a smile to the faces of South Park fans everywhere! The four main characters of South Park are:

1. Stan Marsh
2. Kyle Broflovski
3. Eric Cartman
4. Kenny McCormick

These four boys have been the center of attention in South Park since the show first premiered in 1997, and their antics and adventures have kept audiences la

```
Processing time: 3.71 s