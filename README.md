# llm-serve-tutorial

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

## Local - CPU
Using llama-cpp-python | https://github.com/abetlen/llama-cpp-python 

### Install pre-compiled lib

```
pip install llama-cpp-python[server] \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
### Download 5-bit quantized model in GGUF format

```
mkdir -p models/7B
wget -O models/7B/llama-2-7b.Q5_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf?download=true
```
### Run Server

```
python3 -m llama_cpp.server --model models/7B/llama-2-7b.Q5_K_M.gguf
```

### Query server

```
export BASE_URL="http://localhost:8000"
export MODEL="models/7B/llama-2-7b.Q5_K_M.gguf"
python openai_client.py
```