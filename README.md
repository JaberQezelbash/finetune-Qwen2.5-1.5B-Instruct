# Finetune (Qwen2.5-1.5B-Instruct)
**CPU-only LoRA fine-tuning of Qwen2.5-1.5B-Instruct on a Medical Q&A dataset**

<!-- Optional: add a banner/diagram here -->
<!-- 
<img width="900" alt="banner" src="https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/blob/main/assets/banner.png">
-->

This repository contains my first end-to-end fine-tuning project: adapting Qwen/Qwen2.5-1.5B-Instruct to better answer medical-style questions using a CPU-only LoRA approach.  
It includes a training script that is robust to common environment issues (especially Jupyter/IPykernel arguments and Transformers version differences), plus a lightweight inference/evaluation notebook-style workflow for testing the adapter after training.

> ⚠️ Medical disclaimer: This project is for research/education. The finetuned model is not a substitute for clinician judgment or emergency services.




## Motivation
Medical Q&A is a high-impact domain where models must be **careful, consistent, and safety-aware**. However, full fine-tuning can be expensive and GPU-dependent. The goal of this project was to build a practical, reproducible pipeline that:
- runs **without a GPU** (CPU-only),
- uses **parameter-efficient fine-tuning** (LoRA) rather than updating all model weights,
- preserves **chat/instruction formatting** via the model’s chat template, and
- includes quick checks to ensure the model is behaving as intended (sanity tests, base-vs-finetuned comparison, and a simple overfitting indicator).




## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Repository Structure](#repository-structure)  
4. [Dataset Format](#dataset-format)  
5. [Installation & Requirements](#installation--requirements)  
6. [Training](#training)  
7. [Outputs](#outputs)  
8. [Inference & Testing](#inference--testing)  
9. [Evaluation & Sanity Checks](#evaluation--sanity-checks)  
10. [Configuration & CLI Arguments](#configuration--cli-arguments)  
% 11. [Troubleshooting](#troubleshooting)  
% 12. [Safety Notes](#safety-notes)  
% 13. [Citation](#citation)  
% 14. [Contact](#contact)  
15. [Author’s Note](#authors-note)



## Project Overview
This repo fine-tunes:

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
- **Method:** LoRA (PEFT adapters)  
- **Hardware target:** CPU-only  
- **Dataset:** `Medical_QA_Dataset.csv` (publickly available on [Kaggle](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset)) with columns:  
  - `qtype`  
  - `Question`  
  - `Answer`

During training, the script:
1. Loads and cleans the CSV (drops empty rows, trims strings).
2. Converts each example into a **chat-style conversation**:
   - `system`: a safety-oriented instruction prompt  
   - `user`: the question (optionally with “Question type”)  
   - `assistant`: the answer  
3. Masks the prompt tokens so the training loss is computed mainly on the assistant answer (a standard supervised instruction-tuning pattern).
4. Applies LoRA to common attention/MLP projection layers (auto-detected).
5. Saves the trained adapter and optionally merges it into the base model for a standalone checkpoint.




## Key Features
✅ **CPU-only training** (float32, no fp16/bf16)  
✅ **Jupyter/IPykernel compatible** (`parse_known_args` ignores `-f <kernel.json>`)  
✅ **Robust across Transformers versions** (filters unsupported `TrainingArguments` kwargs)  
✅ **Auto-detects LoRA target modules** (common projection layer names)  
✅ **Chat-template training** using `tokenizer.apply_chat_template`  
✅ **Prompt masking** so loss focuses on assistant responses  
✅ **Saves adapter + tokenizer** for easy reuse  
✅ Optional **merge adapter into base model** to create a standalone model  
✅ Optional **quick generation test** from the script  
✅ Includes a simple **base vs fine-tuned comparison** and a small **ROUGE-L overfitting sanity check** (CPU-friendly)




## Repository Structure
A suggested structure (adapt as needed):

```text
.
├── finetune_qwen25_medqa_cpu.py          # Main CPU-only LoRA fine-tuning script
├── notebooks/
│   └── inference_and_eval.ipynb          # Optional: testing + comparison + quick ROUGE-L checks
├── assets/
│   ├── configurations.md                 # Optional: experiment notes / hyperparameters
│   └── banner.png                        # Optional: repo banner / diagram
├── requirements.txt                      # Pinned dependencies (recommended)
└── README.md
