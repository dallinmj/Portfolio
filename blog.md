---
title: "Running Large Language Models (LLMs): GPUs, VRAM, and Making Big Models Fit"
author: "Dallin Jacobs"
date: 2026-02-20
---

Open-source LLMs (like [Llama](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [Qwen](https://huggingface.co/Qwen), etc.) can run on your own hardware or cloud GPUs. This post is a practical guide to *why* you’d do that, *how* it works, and *how to estimate the GPU memory (VRAM) you’ll need*.

## Why run open-source LLMs instead of just using an API?

Using ChatGPT or a hosted API is convenient, but running your own model can be worth it when you care about:

* **Privacy / compliance**: you keep prompts and data within your environment.
* **Cost predictability**: heavy usage can get expensive with per-token pricing.
* **Latency control**: local inference (generating text) can be faster and more consistent.
* **Customization**: you can fine-tune or adapt models (e.g., [LoRA](https://www.ibm.com/think/topics/lora)) for your domain.
* **Reproducibility**: you can pin a model version and run the same workflow long-term.

If your use case is occasional Q&A or classification, APIs are still the easiest option. But if you’re building a system, doing research, or working with sensitive data, local/cloud GPUs are often a better fit.

## Why GPUs beat CPUs for LLMs

LLMs spend most of their time doing **large amounts of matrix multiplication**. GPUs are built for *tons of parallel math at once*, while CPUs are built to be great at a wide variety of tasks (but not nearly as many parallel math operations).

In practice:

* **CPU inference** can work for smaller models, but it’s almost always much slower.
* **GPU inference** is the standard approach because it’s dramatically faster.

### A simple mental model

* **Compute**: GPUs can do many more math operations at the same time.
* **Bandwidth**: GPUs can move data around inside VRAM very quickly (this matters constantly during inference).
* **Memory (VRAM)**: VRAM is limited, so “does the model fit?” is often the main constraint.

## Where to get models (and the types you’ll see)

The most common distribution hub is **Hugging Face**, which hosts model weights and other needed information.

![Hugging Face logo](better_huggingface.png)

Useful starting points:

* [Hugging Face Models](https://huggingface.co/models) (search + filters)
* [Transformers documentation](https://huggingface.co/docs/transformers/index) (loading and running models)

### Common model “types”

You’ll usually see variations like:

* **Base** models: raw pre-trained models (not specifically trained to chat).
* **Instruct / Chat** models: fine-tuned to follow instructions and have conversations.
* **Code** models: tuned for programming tasks.
* **Vision-language** models: accept images + text.
* and many others.

For most people trying to run an LLM, **instruct/chat** is the most immediately useful.

## Model sizes: why bigger isn’t always better

Model size is often described by how many **parameters** a model has (think: how many stored adjustable numbers the model learned during training). You’ll see names like **7B**, **13B**, **70B** where **B** means “billion parameters.”

* **Larger models** tend to be more capable (better reasoning, more knowledge, more robust answers).
* **Smaller models** are cheaper and faster and can be “good enough” for many tasks.

A practical way to choose:

* Start with **7B–8B** for local experimentation.
* Move to **13B–34B** when you want noticeably better quality and have more VRAM.
* Use **70B+** when you have access to powerful GPUs and need a substantial quality increase.

## VRAM basics: what actually uses GPU memory?

During inference (generating text), VRAM is mainly consumed by:

1. **Model weights** (the learned parameters — the “brain” stored as numbers)
2. **KV cache** (the model’s short-term memory while it’s reading your prompt and generating)
3. **Runtime overhead** (extra temporary memory used by the GPU/software to run efficiently)

### 1) Model weights (the big, constant chunk)

This is usually the largest and most predictable part. If your weights don’t fit, nothing runs.

### 2) KV cache (the “context length tax”)

When the model reads your prompt and generates new tokens, it stores some intermediate information so it doesn’t have to “re-think” the entire prompt from scratch every single token. That saved information is the **KV cache**.

**Key point:** KV cache grows as your conversation gets longer.

So:

* Short prompt → small KV cache → lower VRAM usage
* Long prompt / long chat history → big KV cache → higher VRAM usage

### 3) Runtime overhead (the “stuff around the model”)

Even if the weights fit perfectly, the system still needs VRAM for things like:

* temporary work buffers (space for fast math)
* token buffers and bookkeeping

You don’t usually manage this directly — you just budget some extra VRAM for it.

### Estimating weight memory (quick math)

A rough estimate for weight storage is:

**Weight memory (GB) ≈ (parameters × bytes_per_parameter) / 1e9**

Or if you want the more “computer-ish” version:

**Weight memory (GiB) ≈ (parameters × bytes_per_parameter) / (1024³)**

Typical bytes per parameter (depending on precision):

* FP32: 4 bytes
* FP16 / BF16: 2 bytes
* INT8: 1 byte
* 4-bit: 0.5 bytes

**Rule of thumb:** after you estimate the weights, add **~10–20% extra** for runtime overhead.

**Example (very rough):**
A 7B model in FP16:

* weights ≈ 7,000,000,000 × 2 bytes ≈ 14 GB
* plus overhead → ~16–18 GB total (typical)

### KV cache: why context length and batching matter

KV cache grows with:

* **Context length**: how many tokens are in the prompt + generated output.
* **Batch size**: how many requests you’re running at once.

#### Batch size

Batch size is basically: “how many prompts am I processing at the same time?”

* If you’re chatting with one model instance interactively, batch size is usually **1**.
* If you’re serving multiple users or processing a pile of prompts at once, batch size might be **8, 16, 32**, etc.

Higher batch size improves throughput (more total tokens/second across users), but it increases VRAM usage.

#### “Model architecture details” (simplified)

Different models have different internal shapes (like how wide the layers are and how many layers there are). Bigger models usually have:

* more layers
* wider layers

That tends to increase KV cache size too. You don’t need to memorize the math here — just know that **bigger models usually pay a bigger KV cache cost** at the same context length.

**This is why a model can “fit” for short prompts but crash with out-of-memory errors on long prompts.**

## Practical VRAM estimates for inference (weights-only baseline)

These estimates are **weights + typical overhead** (not worst-case KV cache). Real usage depends on context length and batching, but this gets you in the right ballpark.

| Model Size | FP16/BF16 (≈2 bytes/param) | INT8 (≈1 byte/param) | 4-bit (≈0.5 bytes/param) |
| ---------: | -------------------------: | -------------------: | -----------------------: |
|         7B |                  ~16–18 GB |             ~9–10 GB |                  ~5–6 GB |
|        13B |                  ~28–32 GB |            ~15–18 GB |                 ~8–10 GB |
|        34B |                  ~70–80 GB |            ~40–45 GB |                ~20–24 GB |
|        70B |                ~140–160 GB |            ~80–90 GB |                ~38–45 GB |

**Interpretation:**

* A single **24 GB** GPU can often run **7B–13B** comfortably (with INT8 or 4-bit).
* **70B** typically needs **multi-GPU**, or strong quantization + careful settings.

## Loading time: disk → RAM → VRAM (why startup can feel slow)

When you “load a model,” data often moves through stages:

1. **Disk → CPU RAM** (reading weight files from storage)
2. **CPU RAM → GPU VRAM** (copying weights onto the GPU)

This can feel slow for big models because you might be moving **tens or hundreds of gigabytes**.

### Super simple ways to speed this up / avoid pain

* **Use an SSD** (huge difference vs hard drives or slow network storage).
* **Download once, reuse many times** (keep models cached locally instead of re-downloading).
* **Don’t load what you don’t need** (use one model at a time when possible).
* **Be patient on the first load** (subsequent loads are often faster if the files are already cached by the OS).

## Minimal example: run inference on GPU with Transformers

You don’t need to memorize code to understand the workflow. Conceptually, running inference looks like this:

1. Pick a model ID from Hugging Face.
2. Load tokenizer (turns text into tokens).
3. Load the model weights onto GPU (or split across devices).
4. Send in a prompt.
5. Generate tokens until you hit a stop condition (max tokens, end token, etc.).

If you’re new, the main ideas to keep in mind are almost always:

* VRAM (does it fit?)
* context length (KV cache grows fast)
* loading time (big files)

## Making models fit on smaller GPUs

Below are common techniques to reduce VRAM usage or improve throughput.

| Technique                 | What it does            | Pros                           | Cons                                     | Example use                           |
| ------------------------- | ----------------------- | ------------------------------ | ---------------------------------------- | ------------------------------------- |
| **FP16/BF16**             | Half-precision weights  | Good quality, standard on GPUs | Still large memory                       | Default inference for mid-size models |
| **INT8**                  | 8-bit weights           | Big memory savings             | Sometimes slower / small quality changes | When FP16 barely doesn’t fit          |
| **4-bit quantization**    | 4-bit weights           | Huge VRAM savings              | Quality drop risk; extra complexity      | Fit 13B+ on 24GB GPUs                 |
| **Reduce context length** | Smaller KV cache        | Immediate OOM fix              | Less long-context ability                | Chatbots with shorter prompts         |
| **Lower batch size**      | Less KV + overhead      | Easy                           | Lower throughput                         | Single-user interactive runs          |
| **CPU offload**           | Some weights on CPU RAM | Can run bigger models          | Slower; needs fast CPU↔GPU transfer      | When VRAM is the hard limit           |
| **Multi-GPU parallelism** | Split model across GPUs | Run very large models          | Setup complexity                         | 70B+ models                           |

## Conclusion + what to do next

Running LLMs yourself is mainly a **hardware + workflow** problem:

* choose a model type (instruct/chat),
* choose a size that matches your VRAM,
* and use techniques like **quantization** when you need to fit bigger models.

**Next steps (CTA):**

1. Pick a **7B instruct** model from [Hugging Face Models](https://huggingface.co/models) and try running it locally or on a cloud GPU.
2. Watch VRAM usage with `nvidia-smi` while generating short vs long outputs (you’ll see the difference in VRAM usage).
3. Try **4-bit quantization** and compare (a) VRAM usage, (b) speed, and (c) response quality on the same prompt.

Here is a quick example to get you started!

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1) Pick a small instruct model to start
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# 2) Load tokenizer + model (device_map="auto" puts it on your GPU if available)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 3) Run a simple prompt
prompt = "In one paragraph, explain what VRAM is and why it matters for running LLMs."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
