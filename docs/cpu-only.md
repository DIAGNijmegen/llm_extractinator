# Running on CPU-only Hardware

LLM Extractinator works fine on machines **without a GPU** – as long as you pick a small enough model and are okay with slower runtimes.

This page explains:

- What to change (Docker / local)
- How to run with **CPU-only**
- Why you should **pick a smaller model**, e.g. `qwen3:8b`

!!! Note
      Using a GPU is **strongly recommended**, as it allows you to run larger models which generally perform better.

---

## 1. Core idea

LLM Extractinator itself doesn’t “talk” to your GPU. It just talks to **Ollama**.

Whether inference runs on **GPU or CPU** is controlled by:

- How Ollama is installed/configured
- Whether your Docker container is started with GPU access

So to go CPU-only you mainly need to:

1. Start Ollama without GPU access, and
2. Use a **small model** such as `qwen3:8b`.

Everything else (tasks, parsers, CLI flags) stays the same.

---

## 2. Choosing a CPU-friendly model

For larger models, Ollama will throw an error if no compatible GPU is found. These errors can be seen by running the model using the `verbose` flag.

Therefore for CPU-only hardware:

- Prefer models smaller than **10B parameters**
- Use the default **quantized** variants from Ollama
- Good starting point:
  - `qwen3:8b`

Example CLI call:

```bash
extractinate --task_id 1 --model_name "qwen3:8b" --reasoning_model
```

!!! Note
      We use the `--reasoning_model` flag here because `qwen3` is a reasoning-capable model. If you use a different model that does not emit intermediate reasoning, you can omit this flag.

If this still throws an error or inference is too slow, you can choose an even smaller model such as `qwen3:4b`, `qwen3:1.7b`, or even `qwen3:0.6b`. Alternatively, you can further tune `--num_predict`, `--max_context_len`, and `--num_examples` (see below).

!!! Warning
      Very small models may struggle to follow complex instructions or produce high-quality outputs. Always spot-check the results to ensure they meet your requirements.

---

## 3. Docker: GPU vs CPU-only

When running LLM Extractinator via Docker, GPU access is controlled by the `--gpus` flag in the `docker run` command.

### 3.1 Run **CPU-only** in Docker

To run on CPU only, simply **omit** the GPU flag:

```bash
docker run --rm \
-p 127.0.0.1:8501:8501 \
-p 11434:11434 \
-v $(pwd)/data:/app/data \
-v $(pwd)/examples:/app/examples \
-v $(pwd)/tasks:/app/tasks \
-v $(pwd)/output:/app/output \
lmmasters/llm_extractinator:latest
```

Inside the container:

- Ollama will run in CPU mode.
- The Studio (`launch-extractinator`) and CLI (`extractinate`) work exactly the same.
- The main difference is **speed**, so use a smaller model such as `qwen3:8b`.

---

## 4. Local installation: CPU-only

If you run everything directly on your machine instead of Docker:

1. Install **Ollama**.
2. Make sure the Ollama service is running.
3. Pull a small model:

   ```bash
   ollama pull qwen3:8b
   ```

4. Run Extractinator with that model:

   ```bash
   extractinate --task_id 1 --model_name "qwen3:8b" --reasoning_model
   ```

On a machine without a compatible GPU, Ollama will automatically fall back to CPU-only.

If you *do* have a GPU but want to force CPU, check Ollama’s configuration (e.g. setting GPU usage to 0) so it does not try to use the GPU.

---

## 5. Tweaking settings for CPU runs

On CPU you pay more dearly for every token, so it’s worth dialling a few knobs back.

### 5.1 Limit generation length

Use a smaller `--num_predict`:

```bash
extractinate --task_id 1   --model_name "qwen3:8b"   --num_predict 256 --reasoning_model
```

This caps how long the model’s response can be.

### 5.2 Limit context length

If your inputs are very long, you can reduce the effective context via `--max_context_len`:

```bash
extractinate --task_id 1   --model_name "qwen3:8b"   --max_context_len 2048 --reasoning_model
```

This can significantly reduce compute on very long texts.

### 5.3 Use fewer examples

Each example you add with `--num_examples` increases prompt size and compute.

For CPU-only runs:

- Prefer `--num_examples 0`

---

## 6. Quick CPU-only checklist

If you only have a CPU and want a sane setup:

1. **Pick a small model**, e.g. `qwen3:8b`.
2. If using Docker, **remove `--gpus all`** from the run command.
3. If running locally, make sure Ollama is installed and running; pull the CPU-friendly model.
4. Start with:

   ```bash
   extractinate --task_id 1 --model_name "qwen3:8b" --num_predict 256 --max_context_len 2048 --num_examples 0 --reasoning_model
   ```

5. If that’s fast enough, you can gradually increase `--num_predict` or context length as needed.
