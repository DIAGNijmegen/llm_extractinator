# Quickstart

This page takes you from nothing to your first structured output in about five minutes. We'll extract a **product name** and **price** from a tiny CSV.

By the end you'll have created a dataset, an output schema, and a task — and run it.

!!! tip "Two ways to follow along"
    Every step below can be done **in the Studio** (no code) or **by hand** (plain files). We show both. Pick whichever you prefer — they produce identical files.

---

## 0. Before you start

Make sure you've [installed the package](installation.md) and that **Ollama is running**. You don't need to pull a model in advance; the tool downloads the one you request on first use.

Create a working folder with the standard layout:

```bash
mkdir -p myproject/{data,tasks/parsers,examples,output}
cd myproject
```

The tool looks for `data/`, `tasks/`, `tasks/parsers/`, and writes to `output/` by default.

---

## 1. Add a dataset

Save this as `data/products.csv`:

```text
id,text
1,"A 250 ml bottle of extra-virgin olive oil, on sale for €4.99."
2,"Stainless steel water bottle (750ml) — now only 18.50 euro."
```

The important part is that there's a **column with the text to read** — here, `text`. See [Preparing data](preparing-data.md) for JSON input and other details.

---

## 2. Define the output schema

The schema is a Pydantic model whose top-level class is named `OutputParser`.

=== "In the Studio"

    Launch the Studio (`launch-extractinator`), go to the **Task** tab, choose **Build a new task**, and click **🛠️ Build new** next to *Output schema*. Add two fields — `product_name` (`str`) and `price` (`float`) — then **Save & use this schema**. The Studio writes the file into `tasks/parsers/` for you.

=== "By hand"

    Save this as `tasks/parsers/product_schema.py`:

    ```python
    from pydantic import BaseModel

    class OutputParser(BaseModel):
        product_name: str
        price: float
    ```

See [Output schema](parser.md) for optional fields, lists, and nested models.

---

## 3. Create the task

The task JSON ties the dataset, the text column, and the schema together.

=== "In the Studio"

    Still on the **Task** tab under **Build a new task**: pick `products.csv` as the dataset, choose `text` as the text column, make sure `product_schema.py` is selected, write a short description, and click **💾 Save task**. The task is now marked *ready*.

=== "By hand"

    Save this as `tasks/Task001.json`:

    ```json
    {
      "Description": "Extract the product name and price from each row of text.",
      "Data_Path": "products.csv",
      "Input_Field": "text",
      "Parser_Format": "product_schema.py"
    }
    ```

---

## 4. Run it

=== "In the Studio"

    Open the **Run** tab, pick a model (e.g. `phi4`), and click **🚀 Run**. Watch the log stream; when it finishes you'll get a success summary and a nudge to open **Results**.

=== "On the CLI"

    ```bash
    extractinate --task_id 1 --model_name "phi4"
    ```

    The `--task_id 1` matches `Task001.json`. On first run, Ollama downloads `phi4` if you don't have it yet.

---

## 5. Look at the results

Results land in:

```text
output/run/Task001-run0/nlp-predictions-dataset.json
```

Each record has your **original columns** (`id`, `text`), the **extracted fields** (`product_name`, `price`), and a `status`:

```json
[
  {
    "id": 1,
    "text": "A 250 ml bottle of extra-virgin olive oil, on sale for €4.99.",
    "product_name": "Extra-virgin olive oil 250 ml",
    "price": 4.99,
    "status": "success"
  }
]
```

In the Studio, the **Results** tab shows the same data with success/failure counts, a searchable table, and a per-record detail view. More on the format in [Understanding output](output.md).

---

## Where to go next

- Guide the model with [few-shot examples](examples.md) when the output style matters.
- Tune speed and quality with the [settings reference](settings-reference.md).
- Running without a GPU? See [CPU-only](cpu-only.md).
- Something not working? Check [Troubleshooting](troubleshooting.md).
