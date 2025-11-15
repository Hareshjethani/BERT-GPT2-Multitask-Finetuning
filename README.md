🌟 NLP Dual Fine-Tuning Suite

BERT-based Sentiment Classification + GPT-2/LLaMA Pseudo-code → Python Code Generation

Yeh repository do advanced NLP tasks rakhti hai:

Task 1 — Encoder-only (BERT): Customer Feedback Sentiment Classification

Task 2 — Decoder-only (GPT-2/LLaMA): Pseudo-code se Python Code Generation

Project end-to-end workflow: preprocessing, training, evaluation, example predictions.

Project Structure (suggested folders and files)

task1_bert_sentiment/

preprocessing.py

train.py

evaluate.py

samples/

task2_gpt2_code_generation/

preprocess_pairs.py

train_gpt2.py

evaluate_code.py

app.py (Streamlit/Gradio Interface)

examples/

README.md

requirements.txt

🧠 Task 1: BERT — Customer Feedback Sentiment Classification
Objective

Fine-tune ek BERT model jo customer feedback ko teen classes mein classify kare: Positive, Negative, Neutral.

Dataset

Kaggle: Customer Feedback Dataset — https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset

Preprocessing & Tokenization

Missing/duplicate text remove karna

Text cleaning (lowercase, punctuation handling)

BERT tokenizer use karna (bert-base-uncased)

Max length: 128 tokens

Train/Validation split: 80/20

Training Pipeline (summary)

Model: bert-base-uncased

Loss: CrossEntropyLoss

Batch size: 16

Optimizer: AdamW

Scheduler: warmup steps + linear LR decay

Epochs: 3–5

Evaluation Metrics

Accuracy

F1-score (macro & weighted)

Confusion matrix

Example metrics (illustrative):
Accuracy: 0.91
F1-Score: 0.90

Example Predictions (illustrative)

Input: "The service was amazing!" → Prediction: Positive
Input: "I am not satisfied with the product quality." → Prediction: Negative

🤖 Task 2: GPT-2 / LLaMA — Pseudo-code to Python Code Generation
Objective

Decoder-only model ko train karna taake structured pseudo-code ko syntactically aur semantically valid Python code mein translate kare.

Dataset

SPOC (pseudo-code → code): https://github.com/sumith1896/spoc

Research paper: https://arxiv.org/pdf/1906.04908

Preprocessing

Load pseudo-code / code pairs

Normalize formatting; strip inconsistent whitespace/indentation

Add special delimiters/tokens (example: <|pseudo|> and <|code|>) to separate input from target

Prepare sequences for causal LM training

Fine-Tuning (summary)

Model: GPT-2 ya GPT-2-medium (ya LLaMA if available)

Objective: Causal Language Modeling

Optimizer: AdamW

Sequence length: e.g., 512

Epochs: 5–10 (dataset-dependent)

Evaluation Metrics

BLEU

CodeBLEU

Human evaluation (manual checks for correctness and runtime)

Example scores (illustrative):
BLEU Score: 0.78
CodeBLEU: 0.81

Live Interface

Include a Streamlit or Gradio app for real-time pseudo-code → code generation.
Example pseudo-code input: READ N; IF N is even PRINT "YES" ELSE PRINT "NO"
Generated Python (example):
n = int(input())
if n % 2 == 0:
 print("YES")
else:
 print("NO")

Installation (instructions as plain text)

Clone the repo: git clone https://github.com/yourusername/NLP-Dual-Finetuning-Suite.git

Change directory: cd NLP-Dual-Finetuning-Suite

Install dependencies: pip install -r requirements.txt

(Adjust model names / GPU settings according to your environment.)

Technologies Used

PyTorch

HuggingFace Transformers

Scikit-learn

Pandas / NumPy

Streamlit or Gradio

Matplotlib

Contributions

Contributions, issues, and pull requests are welcome. Please open an issue to discuss major changes.
