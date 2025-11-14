# LLM Extractinator Studio

The Studio is the **interactive** way to use this project.

Run:

```bash
launch-extractinator
```

This will start a Streamlit app (usually at `http://localhost:8501`).

## What you can do

- **Manage data**: point to your CSV/JSON
- **Design parsers**: same builder as `build-parser`, but in-app
- **Create tasks**: fill in description, data path, input field, parser file
- **Run tasks**: trigger extraction from the UI and watch logs

## Why use the Studio

- you don’t have to remember CLI flags
- it guides you through the required fields
- you can try different parsers quickly

After you are happy with a task, you can run it from the CLI on a server.
