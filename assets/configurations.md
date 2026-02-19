# Configurations (assets/configurations.md)

This document summarizes the **core configurations, defaults, and practical tuning knobs** used in this repository for **CPU-only LoRA fine-tuning** of `Qwen/Qwen2.5-1.5B-Instruct` on a **Medical Q&A** CSV dataset.

> Note: “Suggested” ranges below are **recommended experimentation ranges** for this setup. Your best values depend on dataset size/quality and your CPU/RAM.

---

## Hyperparameters & Implementation Details

### A) Data, formatting, and training objective

| **Setting** | **Default** | **Suggested** | **How to tune** | **Notes** |
|---|---:|---:|---|---|
| Dataset file | `Medical_QA_Dataset.csv` | — | Data curation | CSV must contain `qtype`, `Question`, `Answer`. |
| Train/eval split | `test_size = 0.05` | `0.05–0.15` | Manual | Larger eval split gives a more reliable sanity check, but reduces training data. |
| Random seed | `42` | Any fixed integer | Fixed seed | Helps reproducibility across runs. |
| Chat formatting | `apply_chat_template()` | — | Keep consistent | Uses the model’s native chat template to match inference-time formatting. |
| System prompt | Safety-oriented medical assistant | — | Prompt iteration | Encourages educational content + clinician referral + emergency guidance. |
| Loss focus | Answer-only loss (prompt masked) | — | Keep consistent | Prompt tokens are masked to `-100` so loss emphasizes assistant response. |

---

### B) Tokenization & sequence settings

| **Hyperparameter** | **Default** | **Suggested** | **How to tune** | **Notes** |
|---|---:|---:|---|---|
| Max sequence length (`max_seq_len`) | `512` | `384, 512, 768, 1024` | Manual | Longer helps long answers but increases CPU time and RAM. Truncation may cut off long outputs. |
| Truncation | `True` | — | Keep enabled | Prevents out-of-memory and runaway sequences. |
| Padding side | `right` | — | Keep consistent | Stable batching behavior for causal LM. |
| Pad token | `eos_token` (if missing) | — | Keep consistent | Ensures tokenizer can pad even if pad token isn’t defined. |

---

### C) LoRA (PEFT) configuration

| **Hyperparameter** | **Default** | **Suggested** | **How to tune** | **Notes** |
|---|---:|---:|---|---|
| LoRA rank (`lora_r`) | `8` | `4, 8, 16` | Manual | Higher rank = more capacity + more compute/memory. |
| LoRA alpha (`lora_alpha`) | `16` | `8, 16, 32` | Manual | Often scaled with `r` (e.g., `alpha = 2*r` is a common heuristic). |
| LoRA dropout (`lora_dropout`) | `0.05` | `0.0–0.1` | Manual | Higher dropout can reduce overfitting on small/noisy datasets. |
| Bias | `"none"` | — | Keep | Standard LoRA choice. |
| Task type | `"CAUSAL_LM"` | — | Keep | Matches instruction-tuned causal modeling. |
| Target modules | Auto-detected (`q/k/v/o`, `up/down/gate`) | Manual override if needed | Inspect model | Script scans `torch.nn.Linear` layers for common projection names. |

---

### D) Optimization & training schedule (CPU-friendly defaults)

| **Hyperparameter** | **Default** | **Suggested** | **How to tune** | **Notes** |
|---|---:|---:|---|---|
| Train batch size | `1` | `1–2` | Hardware-limited | CPU RAM is usually the limiter. |
| Grad accumulation (`gradient_accumulation_steps`) | `16` | `8, 16, 32` | Manual | Effective batch ≈ `batch_size × grad_accum`. Increases stability without increasing per-step RAM much. |
| Learning rate | `2e-4` | `1e-4–3e-4` | Manual + monitor loss | LoRA often tolerates higher LR than full fine-tuning. If unstable, reduce. |
| Weight decay | `0.0` | `0.0–0.01` | Manual | If you see overfitting, a small value may help. |
| Epochs (`num_train_epochs`) | `1.0` | `1–3` | Manual | CPU training is slow; start with 1 epoch and increase only if needed. |
| Step cap (`max_steps`) | `-1` (disabled) | `100–500` (smoke tests) | Manual | Great for verifying pipeline correctness quickly. |
| Warmup ratio | `0.03` | `0.01–0.1` | Manual | More warmup can improve stability in small-data regimes. |
| LR scheduler | `"cosine"` | `"linear", "cosine"` | Manual | Cosine often works well with LoRA; linear is a solid baseline. |
| Optimizer | `"adamw_torch"` | `"adamw_torch"` | Keep | Reliable default in Transformers. |

---

### E) CPU execution & memory controls

| **Detail** | **Default** | **Suggested** | **How to tune** | **Notes** |
|---|---|---|---|---|
| Device | CPU | — | Keep | Script attempts to enforce CPU usage through `TrainingArguments` flags when supported. |
| Precision | float32 (`fp16=False`, `bf16=False`) | — | Keep | CPU mixed precision is generally not beneficial/available. |
| Gradient checkpointing | `True` | `True` | Keep on | Reduces RAM usage at the cost of slower training—usually worth it on CPU. |
| DataLoader workers | `0` | `0–2` | Manual | `0` is most stable (especially on Windows). |
| Pin memory | `False` | `False` | Keep | Primarily helpful for GPU pipelines. |

---

## TrainingArguments compatibility (Transformers-version robust)
This repo is intentionally defensive against version drift:

| **Compatibility Feature** | **What it does** | **Why it matters** |
|---|---|---|
| Filter unsupported `TrainingArguments` kwargs | Inspects the installed signature and only passes supported args | Prevents errors like `unexpected keyword argument ...` across Transformers versions. |
| Output dir collision-safe saving | If output dir exists and is non-empty, a timestamped directory is created | Prevents accidental overwrites without requiring `overwrite_output_dir`. |
| Jupyter-safe CLI parsing | Uses `parse_known_args()` and ignores unknown `-f kernel.json` args | Prevents crashes when running inside notebooks. |

---

## Inference / Generation Settings (Recommended)

| **Mode** | **Parameters** | **Use case** | **Notes** |
|---|---|---|---|
| Deterministic (greedy) | `do_sample=False`, `max_new_tokens=200–300` | Evaluation, comparisons | Most stable for “base vs fine-tuned” checks. |
| Sampling (demo) | `do_sample=True`, `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.05` | Interactive Q&A demos | More natural outputs, but less reproducible. |
| Safety emphasis | Keep system prompt enabled | Medical domain | Encourages cautious, educational tone and referral to clinicians. |

---

## Output Artifacts & What They Contain

| **Artifact** | **Location** | **Contents** | **When created** |
|---|---|---|---|
| LoRA adapter | `<output_dir>/adapter/` | Adapter weights + tokenizer files | Always (after training). |
| Trainer logs/metrics | `<output_dir>/` | `trainer_state`, metrics JSON (depending on version) | During/after training. |
| Merged checkpoint (optional) | `<output_dir>/merged_model/` | Full model weights with LoRA merged into base | Only with `--merge_model`. |

---

## Suggested Experiment Matrix (Quick Guide)

| **Goal** | **Try** | **Why** |
|---|---|---|
| Faster smoke test | `--max_steps 100` or `--max_train_samples 500` | Validates pipeline end-to-end quickly. |
| Better domain adaptation | `--num_train_epochs 2` | More passes can improve alignment if not overfitting. |
| More adapter capacity | `--lora_r 16 --lora_alpha 32` | Increases learnable capacity for nuanced medical phrasing. |
| Reduce overfitting | `--lora_dropout 0.1` and/or `--weight_decay 0.01` | Helps when train metrics improve but eval quality degrades. |
| Handle longer answers | `--max_seq_len 768` (if RAM allows) | Reduces truncation of assistant responses. |

---

## Quick Overfitting / Sanity Check (Lightweight)
A small-sample ROUGE-L F1 check is included (train vs eval) to catch obvious overfitting signals.

| **Check** | **What it indicates** | **How to interpret** |
|---|---|---|
| Train ROUGE-L ≫ Eval ROUGE-L | Potential overfitting | Consider more dropout, weight decay, or fewer epochs. |
| Both low | Underfitting or metric mismatch | ROUGE is rough for QA; also inspect outputs manually. |

> ROUGE is not a medical correctness metric—use it only as a quick signal and always review qualitative outputs.

---

## Computational Effort (Practical Notes)
- **CPU-only LoRA** is significantly more accessible than full fine-tuning, but it can still be slow depending on CPU/RAM and dataset size.
- The defaults are designed to be stable on CPU: small per-step batches + gradient accumulation + gradient checkpointing.
- For rapid iteration, use smoke-test flags like `--max_steps` or `--max_train_samples`.
- If you want to maximize CPU throughput, you can set CPU threads (example shown in the inference/eval workflow):  
  `torch.set_num_threads(os.cpu_count())` (behavior varies by system).

---

## Reproducibility Notes
| **Aspect** | **What’s controlled** | **What may still vary** |
|---|---|---|
| Seeds | Python `random`, PyTorch, Transformers seed | Minor differences due to library/OS/thread scheduling. |
| Version drift | Script filters unsupported kwargs | Outputs may still differ across Transformers/PEFT versions. |
| Data order | Fixed seed split | Different CSV ordering/cleaning changes results. |

---
