# Finetune (Qwen2.5-1.5B-Instruct)
**CPU-only LoRA fine-tuning of Qwen2.5-1.5B-Instruct on a Medical Q&A dataset**

<!-- Optional: add a banner/diagram here -->
<!-- 
<img width="900" alt="banner" src="https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/blob/main/assets/banner.png">
<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/finetune-Qwen2.5-1.5B-Instruct/blob/main/assets/model.svg">
-->

This repository contains one of my end-to-end fine-tuning projects: adapting the LLM named *Qwen2.5-1.5B-Instruct* to better answer medical-style questions using a CPU-only LoRA approach. It includes a training script that is robust to common environment issues (especially Jupyter/IPykernel arguments and Transformers version differences), plus a lightweight inference/evaluation notebook-style workflow for testing the adapter after training.

> ⚠️ Medical disclaimer: This project is for research/education. The finetuned model is not a substitute for clinician judgment or emergency services.




## Motivation
Medical Q&A is a high-impact domain where models must be **careful, consistent, and precise**. However, full fine-tuning can be expensive and GPU-dependent. The goal of this project was to build a practical, reproducible pipeline that:
- runs **without a GPU** (CPU-only),
- uses **parameter-efficient fine-tuning** (LoRA) rather than updating all model weights,
- preserves **chat/instruction formatting** via the model’s chat template, and
- includes quick checks to ensure the model is behaving as intended (sanity tests, base-vs-finetuned comparison, and a simple overfitting indicator).




## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Repository Structure](#repository-structure)  
4. [Dataset Format](#dataset-format)  
5. [Requirements](Requirements)  
6. [Training](#training)  
7. [Inference & Testing](#inference--testing)  
8. [Evaluation & Sanity Checks](#evaluation--sanity-checks)  
9. [Configuration](assets/configurations.md)  
10. [Author’s Note](#authors-note)



## Project Overview
This repo fine-tunes:

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
- **Method:** LoRA (PEFT adapters)  
- **Hardware target:** CPU-only with 8GB of RAM  
- **Dataset:** `Medical_QA_Dataset.csv` (publickly available on [Kaggle](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset)) with columns:  
  - `qtype`  
  - `Question`  
  - `Answer`

During training, the script:
1. Loads and cleans the CSV (drops empty rows, trims strings).
2. Converts each example into a chat-style conversation:
   - `system`: a safety-oriented instruction prompt  
   - `user`: the question (optionally with “Question type”)  
   - `assistant`: the answer  
3. Masks the prompt tokens so the training loss is computed mainly on the assistant answer (a standard supervised instruction-tuning pattern).
4. Applies LoRA to common attention/MLP projection layers (auto-detected).
5. Saves the trained adapter and optionally merges it into the base model for a standalone checkpoint.




## Key Features
✅ CPU-only training (float32, no fp16/bf16)  
✅ Jupyter/IPykernel compatible (`parse_known_args` ignores `-f <kernel.json>`)  
✅ Robust across Transformers versions (filters unsupported `TrainingArguments` kwargs)  
✅ Auto-detects LoRA target modules (common projection layer names)  
✅ Chat-template training using `tokenizer.apply_chat_template`  
✅ Prompt masking so loss focuses on assistant responses  
✅ Saves adapter + tokenizer for easy reuse  
✅ Optional merge adapter into base model to create a standalone model  
✅ Optional quick generation test from the script  
✅ Includes a simple base vs fine-tuned comparison and a small ROUGE-L overfitting sanity check




## Repository Structure

```text
.
├── codes/
│   ├── finetune_qwen25_medqa_cpu.ipynb       # Main CPU-only LoRA fine-tuning script
│   └── inference_and_eval.ipynb              # testing + comparison + quick ROUGE-L checks
├── assets/
│   └── configurations.md                     # Experiment notes / hyperparameters
├── requirements.txt                          # Pinned dependencies
└── README.md
```

Quick links:
- Training: [`codes/finetune_qwen25_medqa_cpu.ipynb`](codes/finetune_qwen25_medqa_cpu.ipynb)  
- Inference/Eval: [`codes/inference_and_eval.ipynb`](codes/inference_and_eval.ipynb)  
- Config tables: [`assets/configurations.md`](assets/configurations.md)  
- Requirements: [`assets/requirements.txt`](assets/requirements.txt)  




## Dataset Format
The CSV file, publickly available on [Kaggle](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset), contains:
- `qtype`
- `Question`
- `Answer`



## Requirements

```txt
torch>=2.1.0
transformers>=4.37.0
datasets>=2.18.0
peft>=0.10.0
accelerate>=0.26.0
pandas>=2.0.0
packaging>=23.0
```



## Training
**Notebook:** [`codes/finetune_qwen25_medqa_cpu.ipynb`](codes/finetune_qwen25_medqa_cpu.ipynb)

### Steps
1. Download the dataset CSV from Kaggle (or use your own CSV with the same columns).
2. Open `codes/finetune_qwen25_medqa_cpu.ipynb`.
3. Update the notebook configuration variables (paths + output directory), for example:
   - `DATA_PATH` (path to `Medical_QA_Dataset.csv`)
   - `MODEL_NAME` (default: `Qwen/Qwen2.5-1.5B-Instruct`)
   - `OUTPUT_DIR` (where results will be saved)
4. Run all cells to:
   - load/clean/split data,
   - tokenize using the model chat template,
   - apply LoRA adapters,
   - fine-tune on CPU, and
   - save the resulting adapter.

> Tip: CPU fine-tuning is slow by nature. Smoke tests are the best way to confirm everything works before running longer training.



## Inference & Testing
**Notebook:** [`codes/inference_and_eval.ipynb`](codes/inference_and_eval.ipynb)

This notebook loads:
- the base model (`Qwen2.5-1.5B-Instruct`), plus  
- your saved adapter from `adapter/`,  

then runs:
- quick manual Q&A tests,
- optional base-vs-fine-tuned comparisons (adapter OFF vs ON), and
- a small CPU-friendly overfitting sanity check.

Typical steps:
1. Set `ADAPTER_DIR` to your saved adapter path, e.g.:
   - `./qwen25_1p5b_medqa_lora_cpu/adapter`
2. Run the notebook cells to generate sample answers and comparisons.


---

## Author’s Note
Thanks for checking out this project. My goal was to create a clean, reproducible, and practical pipeline demonstrating how to adapt an instruction-tuned LLM to a medical Q&A style using *LoRA*, even when limited to CPU-only resources.
