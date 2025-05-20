# 🛠️ Task Configuration

Create a JSON task file in the `tasks/` folder using the following naming format:

```bash
TaskXXX_taskname.json
```

---

## 📌 Required Fields

Your JSON file must include the following keys:

- `Task`: Name of the task  
- `Type`: Task type  
- `Description`: Description of the task  
- `Data_Path`: Filename of the dataset  
- `Input_Field`: Column containing the text data  
- `Parser_Format`: JSON format for the expected output  

---

## 🔄 Optional (for using examples)

- `Example_Path`: Path to the file with examples to include in the prompt

> **Note:** If you're not using examples, simply omit the `Example_Path` field.  
> Do **not** set it to an empty string!
