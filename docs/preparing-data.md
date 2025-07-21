
# 📂 Preparing Your Data for Extractinator

To run extraction tasks smoothly, your data needs to follow a simple, structured format.

---

## ✅ Supported Formats

Your dataset must be in **CSV** or **JSON** format.

---

## 🧾 Dataset Requirements

Each record should include a **single column or key** containing the text you want the model to extract from.

**Requirements**:

- A column/key with **raw text**
- This must contain **only strings**
- You’ll reference this column name in your Task file under the `Input_Field` field
  
Any other columns/keys can be included but are not processed by the model. They will be passed through as-is in the output.

---

## 🧪 Adding Examples (Optional)

Few-shot examples help guide the model’s output.

Create a separate **CSV** or **JSON** file with the following columns:

| Column   | Description                         |
|----------|-------------------------------------|
| `input`  | The example input text              |
| `output` | The expected output for that input  |

These examples will be shown to the model during extraction if the `num_examples` parameter is set to a non-zero value. Beware that adding examples increases the prompt size which can significantly impact the inference speed.

---

## 🧰 Output Parser

You must define a **Pydantic class** to describe the structure of the expected output.

- This is called the **OutputParser**
- You’ll import it in your Task file via the `Parser_Format` field

For details on building one visually or by code, see [parser.md](parser.md).
