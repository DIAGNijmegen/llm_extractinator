# Running with Docker

!!! tip "This is the recommended way to run LLM Extractinator"
    Docker bundles Python, Ollama, and the Studio into one container, so it's the simplest and most reliable setup — nothing to install on your host but Docker itself. Prefer a local Python install? See [Installation](installation.md).

This project ships with a GPU-ready Docker image so you can run everything in a consistent environment without installing all dependencies on your host.

Below is how to:

1. understand what Docker is,
2. install Docker,
3. create the local folders that will be mounted into the container,
4. run the container (Windows/PowerShell and Linux/macOS),
5. switch between the two modes the image supports (`app` and `shell`).

---

## 1. What is Docker?

Docker lets you run apps in *containers*: lightweight, isolated environments that bundle the OS libraries and dependencies your app needs. You get the same setup everywhere (your laptop, CI, a server), so “it works on my machine” stops being a problem.

In this repo, the image is built on top of an NVIDIA CUDA runtime image and already contains:

- Python 3.11
- your package (installed with `pip install -e .`)
- Ollama (started automatically in the container)
- an entrypoint script that can start the Streamlit app **or** drop you into a shell

---

## 2. Install Docker

**Desktop users (Windows / macOS):**

- Install **Docker Desktop** from the official Docker site.
- On Windows, make sure you can run `docker` from PowerShell.
- If you want **GPU** support on Windows, you also need a recent NVIDIA driver and the Docker + WSL2 stack that supports GPU.

**Linux users (Ubuntu, etc.):**

- Install the Docker Engine from your distro or from Docker’s official docs.
- For **GPU** support, install the **NVIDIA Container Toolkit** so `--gpus all` works.

> If `--gpus all` fails, check your driver/toolkit install.

---

## 3. Create local folders

The container expects to **mount** several folders from your host into `/app/...` inside the container. Create them once:

```bash
mkdir -p data examples tasks output ollama_models
```

These will map to:

- `./data` → `/app/data`
- `./examples` → `/app/examples`
- `./tasks` → `/app/tasks`
- `./output` → `/app/output`
- `./ollama_models` → `/root/.ollama` (optional, but recommended)

Anything the app writes there will persist on your machine.

### Persisting Ollama models (recommended)

By default, Ollama models are stored inside the container at `/root/.ollama`. This means **every time you start a new container, you'll need to pull the models again**.

To avoid this, mount a local directory to persist the Ollama models between container runs:

```bash
mkdir -p ollama_models
```

Then add `-v ${PWD}/ollama_models:/root/.ollama` (Windows/PowerShell) or `-v $(pwd)/ollama_models:/root/.ollama` (Linux/macOS) to your `docker run` command (see examples in section 4).

---

## 4. Run the container

### 4.1 Windows / PowerShell example

```powershell
# Remove `--gpus all` if you don't have a GPU
docker run --rm --gpus all `
  -p 127.0.0.1:8501:8501 `
  -p 11434:11434 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/examples:/app/examples `
  -v ${PWD}/tasks:/app/tasks `
  -v ${PWD}/output:/app/output `
  -v ${PWD}/ollama_models:/root/.ollama `
  lmmasters/llm_extractinator:latest
```

### 4.2 Linux / macOS variant

```bash
# Remove `--gpus all` if you don't have a GPU
docker run --rm --gpus all \
  -p 127.0.0.1:8501:8501 \
  -p 11434:11434 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/examples:/app/examples \
  -v $(pwd)/tasks:/app/tasks \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/ollama_models:/root/.ollama \
  lmmasters/llm_extractinator:latest
```

Open: <http://127.0.0.1:8501>

---

## 5. Connecting to an existing Ollama instance

By default the container runs its **own** Ollama and stores models in the `ollama_models` mount. If you already have Ollama running somewhere — on your host machine, or a shared GPU server — you can point LLM Extractinator at that instead, and skip the container's own model server and storage.

In that case, run a **leaner container**: drop the Ollama port (`11434`) and the `ollama_models` volume, since the container won't be serving or storing models itself.

**Windows / PowerShell:**

```powershell
docker run --rm --gpus all `
  -p 127.0.0.1:8501:8501 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/examples:/app/examples `
  -v ${PWD}/tasks:/app/tasks `
  -v ${PWD}/output:/app/output `
  lmmasters/llm_extractinator:latest
```

**Linux / macOS:**

```bash
docker run --rm --gpus all \
  -p 127.0.0.1:8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/examples:/app/examples \
  -v $(pwd)/tasks:/app/tasks \
  -v $(pwd)/output:/app/output \
  lmmasters/llm_extractinator:latest
```

Then tell LLM Extractinator where your Ollama server is:

- **In the Studio** — open the **Run** tab and set **Ollama server URL** to your instance.
- **On the CLI** — pass `--ollama_host http://host:11434`.

Typical URLs:

- `http://host.docker.internal:11434` — Ollama running on **your host machine** (Docker Desktop on Windows/macOS; on Linux add `--add-host=host.docker.internal:host-gateway` to the `docker run` command).
- `http://<server-ip>:11434` — Ollama on **another machine** (e.g. a shared GPU server).

!!! note "Since inference runs elsewhere, the container needs no GPU"
    When you connect to an external Ollama, the container itself doesn't do any inference — you can drop `--gpus all` from the command above.

!!! warning "The model must already be pulled there"
    Pointed at an externally managed server, LLM Extractinator **only connects** — it won't start the server, pull models, or unload them. Make sure the model you request is already available on that instance (`ollama pull <model>` on the machine running Ollama). See [`--ollama_host`](settings-reference.md) for the full behaviour.

---

## 6. The two modes (from the Dockerfile)

Your Dockerfile defines an entrypoint script:

- it **always starts Ollama** in the background: `ollama serve ...`
- it then looks at the **first argument** to decide the mode

```bash
/entrypoint.sh app   # default
/entrypoint.sh shell # drop into a shell
```

So, by default, when you run:

```bash
docker run ... lmmasters/llm_extractinator:latest
```

it uses `CMD ["app"]` → starts the Streamlit “extractinator”.

If you want to drop into the container and poke around (with the package already installed and Ollama running), just pass `shell` at the end:

**Windows / PowerShell:**

```powershell
docker run --rm --gpus all `
  -p 127.0.0.1:8501:8501 `
  -p 11434:11434 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/examples:/app/examples `
  -v ${PWD}/tasks:/app/tasks `
  -v ${PWD}/output:/app/output `
  -v ${PWD}/ollama_models:/root/.ollama `
  lmmasters/llm_extractinator:latest shell
```

**Linux / macOS:**

```bash
docker run --rm --gpus all \
  -p 127.0.0.1:8501:8501 \
  -p 11434:11434 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/examples:/app/examples \
  -v $(pwd)/tasks:/app/tasks \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/ollama_models:/root/.ollama \
  lmmasters/llm_extractinator:latest shell
```

That will **not** start the Streamlit UI; instead you’ll get a bash shell inside the container with `llm_extractinator` installed.

---

## 7. Notes

- The image exposes **two ports**: `8501` (Streamlit) and `11434` (Ollama).
- If you don’t have a GPU, you can try omitting `--gpus all`, but the image is CUDA-based, so GPU is the intended path.
- If your Docker Desktop uses different volume mappings (e.g., Windows drive letters), adjust the `-v` paths accordingly.
