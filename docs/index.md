# LLM Extractinator

![Overview of the LLM Data Extractor](images/doofenshmirtz.jpg)

LLM Extractinator is a prototype tool for **LLM-based structured extraction**. You give it text (CSV/JSON with one text field), tell it what structure you want (Pydantic model), and it will try to produce JSON that matches.

> ⚠️ Because this relies on LLMs, **always verify the output**. Models can hallucinate fields or misinterpret text.

## When to use

- you have many similar reports and want tabular/JSON output
- you want a no-code way (Studio) to define output models
- you want a CLI so the same task can run on a server/cron
- you want to experiment with different local models (Ollama)

## How to run

1. **Install** the package and start **Ollama**
2. Run **Studio** with `launch-extractinator` to design a parser and create a task JSON
3. Run the same task from the **CLI** with `extractinate --task_id 1 --model_name "phi4"`

## What’s in these docs

- **Installation** — set up Python + Ollama
- **Preparing Data** — what the input should look like
- **Parser** — how to design the Pydantic output
- **CLI Usage** — all flags
- **Studio** — UI walkthrough
- **Manual** — for power users
