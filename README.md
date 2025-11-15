# 🌟 NLP Dual Fine-Tuning Suite
### BERT-based Sentiment Classification + GPT-2/LLaMA Pseudo-code → Python Code Generation

This repository contains two advanced NLP tasks using Transformer architectures:

1. **Task 1 — Encoder-Only (BERT): Customer Feedback Sentiment Classification**  
2. **Task 2 — Decoder-Only (GPT-2/LLaMA): Pseudo-code to Python Code Generation**

Both tasks include preprocessing, model fine-tuning, evaluation, and example predictions.  
This project is ideal for learning end-to-end NLP workflows and model training.

---

# 📁 Project Structure
NLP-Dual-Finetuning-Suite/
│
├── task1_bert_sentiment/
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── samples/
│
├── task2_gpt2_code_generation/
│ ├── preprocess_pairs.py
│ ├── train_gpt2.py
│ ├── evaluate_code.py
│ ├── app.py # Streamlit/Gradio Interface
│ └── examples/
│
├── README.md
└── requirements.txt

markdown
Copy code

---

# 🧠 Task 1: BERT — Customer Feedback Sentiment Classification

## Objective
Fine-tune a BERT model to classify customer feedback into:
- Positive
- Negative
- Neutral

### Dataset
Kaggle: *Customer Feedback Dataset*  
https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset

### Preprocessing & Tokenization
- Removed missing/duplicate text
- Cleaned text (lowercase, punctuation handling)
- Tokenized using BERT Tokenizer
- Max length: 128 tokens
- Train/Validation split: 80/20

### Training Pipeline
- Model: **bert-base-uncased**
- Loss: CrossEntropyLoss
- Batch size: 16
- Optimizer: AdamW
- Warmup steps + linear learning rate scheduler
- Training for 3–5 epochs

### Evaluation Metrics
- Accuracy
- F1-score (macro & weighted)
- Confusion Matrix

Example output:
Accuracy: 0.91
F1-Score: 0.90

shell
Copy code

### Example Predictions
Input: "The service was amazing!"
Prediction: Positive

Input: "I am not satisfied with the product quality."
Prediction: Negative

yaml
Copy code

---

# 🤖 Task 2: GPT-2 / LLaMA — Pseudo-code to Python Code Generation

## Objective
Train a decoder-only language model to translate **structured pseudo-code** into fully working, valid Python code.

### Dataset
SPOC Pseudocode → Code dataset  
https://github.com/sumith1896/spoc  
Paper: https://arxiv.org/pdf/1906.04908

### Preprocessing
- Loaded pseudo-code → Python code pairs
- Stripped formatting inconsistencies
- Added special tokens: `<|pseudo|>` `<|code|>`
- Prepared text for causal LM training

### Fine-Tuning on GPT-2
- Model: **GPT-2 / GPT-2-medium**
- Objective: Causal Language Modeling (CLM)
- Optimizer: AdamW
- Sequence length: 512
- Epochs: 5–10
- Trained to generate clean Python code

### Evaluation Metrics
- BLEU
- CodeBLEU
- Human Evaluation

Example:
BLEU Score: 0.78
CodeBLEU: 0.81

css
Copy code

### Live Interface
A **Streamlit/Gradio app** is included for real-time code generation.

streamlit run app.py

makefile
Copy code

Example:
Pseudo-code:
READ N
IF N is even PRINT "YES" ELSE PRINT "NO"

Generated Python:
n = int(input())
if n % 2 == 0:
print("YES")
else:
print("NO")
