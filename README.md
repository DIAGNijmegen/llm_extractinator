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

## Installing Ollama

For the package to work, Ollama needs to be installed on your machine.
For Linux, use the following command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

For Windows or macOS, install via [this link](https://ollama.com/download)

## Installing the Package

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

## Data Structure

The data structure for the input data should be as follows:

- The data should be in a CSV or a JSON file.
- The text data should be in a column specified by the `Input_Field` in the task configuration. The default is `text`.
- The name of the data file should be specified in the `Data_Path` field in the task configuration. The default location is the `data` folder, but this can be changed using the `--data_dir` flag when running the model.

When running the model with examples (`num_examples > 0`), the examples should be provided in a separate CSV or JSON file. The path to this file should be specified in the `Example_Path` field in the task configuration. The default location is the `data` folder, but this can be changed using the `--example_dir` flag when running the model.

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

The following field is only mandatory if you want to have the model use examples in its prompt:

- **Example_Path**: The path to data used for creating examples (only required if `num_examples > 0` when running the model).

> [!Important]
> If you don't want to use examples, omit the `Example_Path` field from the task configuration completely. Do not set it to an empty string!
---

## Input Flags for `extractinate`

The following input flags can be used to configure the behavior of the `extractinate` script:

| Flag                      | Type          | Default Value        | Description                                                                 |
|---------------------------|---------------|----------------------|-----------------------------------------------------------------------------|
| `--task_id`               | `int`         | **Required**         | Task ID to generate examples for.                                           |
| `--run_name`              | `str`         | "run"               | Name of the run for logging purposes.                                       |
| `--n_runs`                | `int`         | `5`                  | Number of runs to perform.                                                  |
| `--num_examples`          | `int`         | `0`                  | Number of examples to generate for each task.                               |
| `--num_predict`           | `int`         | `512`                | Maximum number of tokens to predict.                                        |
| `--chunk_size`            | `int`         | `None`               | Number of examples to generate in a single chunk. When None, use dataset size as chunksize.|
| `--overwrite`             | `bool`        | `False`              | Overwrite existing files instead of skipping them.                          |
| `--translate`             | `bool`        | `False`              | Translate the generated examples to English.                                |
| `--verbose`               | `bool`        | `False`              | Enable verbose logging.                                                     |
| `--reasoning_model`       | `bool`        | `False`              | Whether or not the model is a reasoning model.                              |
| `--model_name`            | `str`         | "mistral-nemo"      | Name of the model to use for prediction tasks.                              |
| `--temperature`           | `float`       | `0.0`                | Temperature for text generation.                                            |
| `--max_context_len`       | `str`         | `max`               | Maximum context length; 'split' splits data into short and long cases and does a run for them seperately (good if your dataset distribution has a tail with long reports and a bulk of short ones), 'max' uses the maximum token length of the dataset, or a number sets a fixed length.                                     |
| `--top_k`                 | `int`         | `None`               | Limits the sampling to the top K tokens.                                    |
| `--top_p`                 | `float`       | `None`               | Nucleus sampling probability threshold.                                     |
| `--seed`                  | `int`         | `None`               | Random seed for reproducibility.                                            |
| `--output_dir`            | `Path`        | `<project_root>/output` | Path to the directory for output files.                                      |
| `--task_dir`              | `Path`        | `<project_root>/tasks` | Path to the directory containing task configuration files.                   |
| `--log_dir`               | `Path`        | `<project_root>/output` | Path to the directory for log files.                                        |
| `--data_dir`              | `Path`        | `<project_root>/data` | Path to the directory containing input data.                                 |
| `--example_dir`           | `Path`        | `<project_root>/examples` | Path to the directory containing example data.                               |
| `--translation_dir`       | `Path`        | `<project_root>/translations` | Path to the directory containing translated examples.                        |

---

## Running the Extractor

To run the data extraction process, use either the command line or import the function in Python.

### Using the Command Line

```bash
extractinate --task_id 001 --model_name "mistral-nemo" 
```

### Using the Function in Python

```python
from llm_extractinator import extractinate

extractinate(
    task_id=1,
    model_name="mistral-nemo",
)
```

---

## Enhancements and Contributions

Feel free to contribute by improving configurations, adding more task types, or extending model compatibility. Open a pull request or file an issue for discussions!
