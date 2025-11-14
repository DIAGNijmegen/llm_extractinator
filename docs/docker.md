# Running with Docker

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
mkdir -p data examples tasks output
```

These will map to:

- `./data` → `/app/data`
- `./examples` → `/app/examples`
- `./tasks` → `/app/tasks`
- `./output` → `/app/output`

Anything the app writes there will persist on your machine.

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
  lmmasters/llm_extractinator:latest
```

Open: <http://127.0.0.1:8501>

---

## 5. The two modes (from the Dockerfile)

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
  lmmasters/llm_extractinator:latest shell
```

That will **not** start the Streamlit UI; instead you’ll get a bash shell inside the container with `llm_extractinator` installed.

---

## 6. Notes

- The image exposes **two ports**: `8501` (Streamlit) and `11434` (Ollama).
- If you don’t have a GPU, you can try omitting `--gpus all`, but the image is CUDA-based, so GPU is the intended path.
- If your Docker Desktop uses different volume mappings (e.g., Windows drive letters), adjust the `-v` paths accordingly.
