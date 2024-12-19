
# LLM Extractinator

![Overview of the LLM Data Extractor](images/doofenshmirtz.jpg)

> [!Important]
> This tool is a prototype which is in active development and is still undergoing major changes. Please always check the results!

---

## Overview

This project enables the efficient extraction of structured data from unstructured text using large language models (LLMs). It provides a flexible configuration system and supports a variety of tasks.

### Tool Workflow

![Overview of the LLM Data Extractor](images/overview.png)

---

## 2. Installing the Package

### Option 1: Install from PyPI

The package is installable via PyPI using:

```bash
pip install llm_extractinator
```

### Option 2: Install using local clone

For contributing to or developing the package, clone this repository and install it using:

```bash
pip install -e .
```

---

## Setting Up Task Configuration

Create a JSON file in the `tasks` folder for each task, following the naming convention:

```bash
TaskXXX_taskname.json
```

Where `XXX` is a 3-digit number, and `taskname` is a brief descriptor of the task.

The JSON file should always include the following fields:

- **Task**: The name of the task.
- **Type**: The type of task.
- **Description**: A detailed description of the task.
- **Data_Path**: The filename of the data file in the data folder.
- **Input_Field**: The column name containing the text data.
- **Parser_Format**: The JSON format you want the output to be in. See `Task999_example.json` for an example.

The following fields are only mandatory if you want to have the model automatically generate examples:

- **Example_Path**: The path to data used for creating examples (only required if `num_examples > 0` when running the model).
- **Label_Field**: The column name containing the ground truth labels (only required if `num_examples > 0`).

---

## Input Flags for `extractinate`

The following input flags can be used to configure the behavior of the `extractinate` script:

| Flag                      | Type          | Default Value        | Description                                                                 |
|---------------------------|---------------|----------------------|-----------------------------------------------------------------------------|
| `--task_id`               | `int`         | **Required**         | Task ID to generate examples for.                                           |
| `--model_name`            | `str`         | `"mistral-nemo"`     | Name of the model to use for prediction tasks. See [https://ollama.com/search](https://ollama.com/search) for the options.                              |
| `--num_examples`          | `int`         | `0`                  | Number of examples to generate for each task.                               |
| `--n_runs`                | `int`         | `5`                  | Number of runs to perform.                                                  |
| `--temperature`           | `float`       | `0.3`                | Temperature for text generation.                                            |
| `--max_context_len`       | `int`         | `8192`               | Maximum context length for input text.                                      |
| `--num_predict`           | `int`         | `1024`               | Maximum number of tokens to predict.                                        |
| `--run_name`              | `Path`        | `"run"`              | Name of the run for logging purposes.                                       |
| `--output_dir`            | `Path`        | `<project_root>/output` | Path to the directory for output files.                                      |
| `--task_dir`              | `Path`        | `<project_root>/tasks` | Path to the directory containing task configuration files.                   |
| `--log_dir`               | `Path`        | `<project_root>/output` | Path to the directory for log files.                                        |
| `--data_dir`              | `Path`        | `<project_root>/data` | Path to the directory containing input data.                                 |
| `--chunk_size`            | `int`         | `None`               | Number of examples to generate in a single chunk. When None, use dataset size as chunksize.|
| `--translate`             | `bool`        | `False`              | Translate the generated examples to English.                                |

---

## Example `Task.json`

Below is an example configuration file for a task:

```json
{
    "Task": "Text Summarization",
    "Type": "Summarization",
    "Description": "Generate summaries for long-form text documents.",
    "Data_Path": "data/documents.csv",
    "Example_Path": "data/summaries_examples.csv",
    "Input_Field": "text",
    "Label_Field": "summary",
    "Parser_Format": {
        "summary": {
            "type": "str",
            "description": "The summary of the text document."
        }
    }
}
```

---

# Running the Extractor

To run the data extraction process, you can either use the command line or import the function in Python.

## Option 1: Using the Command Line

Use the following command:

```bash
extractinate --task_id 001 --model_name "mistral-nemo" --num_examples 0 --max_context_len 8192 --num_predict 8192 --translate
```

Customize the flags based on your task requirements.

## Option 2: Using the Function in Python

You can also call the extractor programmatically:

```python
from llm_extractinator import extractinate

extractinate(
    task_id=1,
    model_name="mistral-nemo",
    num_examples=0,
    max_context_len=8192,
    num_predict=8192,
    translate=True
)
```

---

## Output

The output will be saved in the specified `--output_dir`. Ensure that the directory structure and paths specified in the `Task.json` file match your project's organization.

For further details, check the logs in the directory specified by `--log_dir`.

---

## Enhancements and Contributions

Feel free to enhance this project by improving configurations, adding more task types, or extending model compatibility. Open a pull request or file an issue for discussions!
