========================================================================
SpamAdapt: SMS Spam Detection using Pretrained BERT, LoRA, and DoRA
========================================================================

Project Overview:
-----------------
This project implements and evaluates parameter-efficient fine-tuning (PEFT)
techniques for SMS spam and fraud detection using BERT (google-bert/bert-base-uncased).
We compare:
1. Full Fine-Tuning (all 109.5M parameters)
2. LoRA (Low-Rank Adaptation with rank r=8, alpha=16)
3. DoRA (Weight-Decomposed Low-Rank Adaptation with rank r=8, alpha=16)

Key Results:
------------
- Full Fine-Tuning: 98.45% Test Accuracy, 96.40% Macro F1
- LoRA (PEFT):      98.32% Test Accuracy, 96.12% Macro F1 (99.7% performance of Full FT)
- DoRA (PEFT):      98.19% Test Accuracy, 95.84% Macro F1
- Parameter Savings: LoRA trains only 296,450 parameters (0.27% of full model),
  reducing adapter disk storage from ~418 MB to 1.14 MB (~366x savings).

Directory Structure:
--------------------
SpamAdapt/
  app/            - Streamlit web application and inference module
  src/            - Model definitions, tokenizer, data loader, evaluation, and benchmark
  data/           - Processed datasets and external real-world benchmark datasets
  experiments/    - Fine-tuning checkpoints, adapter configurations, and thresholds
  results/        - Evaluation metric JSON files and confusion matrix plots
  tests/          - Automated unit tests and regression tests
  test_app.py     - Application and inference verification suite
  requirements.txt- Project Python dependencies

Requirements & Setup:
---------------------
1. Python 3.10+ (tested on Python 3.11)
2. Install dependencies:
     pip install -r SpamAdapt/requirements.txt

How to Run:
-----------
1. Run the Streamlit web application:
     python -m streamlit run SpamAdapt/app/main.py

2. Run the application verification tests:
     python SpamAdapt/test_app.py

3. Run benchmark evaluation:
     python SpamAdapt/experiments/run_benchmark.py

