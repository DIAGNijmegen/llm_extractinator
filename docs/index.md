# LLM Extractinator

![The Extractinator](images/doofenshmirtz.jpg)

**LLM Extractinator** turns messy, unstructured text into clean, **schema-validated structured data** — one `-inator` for your reports, notes, and text columns. You give it text, tell it what shape you want (a Pydantic model), and it prompts a local LLM to produce JSON that matches, validating every result against your schema.

> ⚠️ Because this relies on LLMs, **always verify the output**. Models can hallucinate fields or misread text. Nothing here is a substitute for a human check on anything that matters.

## What it's good for

- You have **many similar documents** (reports, notes, rows of text) and want them as a table or JSON.
- You want a **no-code** way to define the output shape and run extraction — that's the Studio.
- You want the *same* task to run **unattended** on a server or in a cron job — that's the CLI.
- You want to **experiment** with different local models through Ollama without rewiring anything.

## The mental model

Three ingredients, tied together by a small **task file**:

1. **Dataset** — a CSV or JSON file with a text field to read from.
2. **Output schema** — a Pydantic model (`OutputParser`) naming the fields to extract.
3. **Task JSON** — points at the dataset, the text field, and the schema.

Run the task, and each input row comes back with your extracted fields attached and a `status` telling you whether validation succeeded.

## The 60-second version

1. **Get it running.** The easiest way is [Docker](docker.md), which bundles Python, Ollama, and the Studio in one container. (Prefer a local install? See [Installation](installation.md).)
2. Open the **Studio** and build a schema and a task with no code.
3. Run it — from the Studio, or from the terminal with `extractinate --task_id 1 --model_name "phi4"`.
4. Inspect the results in the Studio's **Results** tab, or open the output JSON directly.

New here? The [Quickstart](quickstart.md) walks through a complete first extraction. Otherwise, jump to whichever piece you need:

- **[Installation](installation.md)** — Python + Ollama
- **[Preparing data](preparing-data.md)** — what your input should look like
- **[Output schema](parser.md)** — designing the Pydantic output
- **[Studio](studio.md)** — the app, tab by tab
- **[CLI usage](cli.md)** / **[Settings reference](settings-reference.md)** — every flag and field
- **[Few-shot prompting](examples.md)** — steering the model with examples
- **[Understanding output](output.md)** — where results land and what's in them
- **[Troubleshooting](troubleshooting.md)** — when something goes sideways
