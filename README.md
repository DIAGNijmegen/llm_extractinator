# LLM_data_extractor

A tool for extracting data from text using LLMs with ollama

## Setup environment

```bash
conda create --name=data_extractor python=3.12
conda activate data_extractor

pip install -e .
```

## Setting up Task.json

Create a JSON file in the tasks folder called TaskXXX_taskname.json where XXX is a 3 digit number. In the folder should be the following information:

- Task: name of the task
- Type: type of task
- Description: detailed description of the task
- Data_Path: path to the data to perform the task on
- Example_Path: path to data used to create examples (only necessary if examples > 0 when running the model)
- Input_Field: the column name in which to find the text data
- Label_Field: the column name in which to find the ground truth label (only necessary if examples > 0 when running the model)
- Parser_Format: the JSON format that you want the output to be in. See Task999_example.json for an example