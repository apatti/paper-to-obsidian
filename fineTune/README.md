# Fine-Tuning Pipeline

This directory contains the complete pipeline for creating LoRA adapters that power the multi-perspective paper analysis.

## Overview

The pipeline generates training data from academic papers and fine-tunes Llama 3.2 with three distinct "personas":

| Adapter | Purpose | Style |
|---------|---------|-------|
| **ELI5** | Beginner-friendly explanations | Analogies, simple language |
| **Intuitive** | First-principles understanding | Mechanism-focused, Feynman-style |
| **Executive** | Quick business insights | Utility, cost, value proposition |

## Pipeline Steps

### 1. Download Papers

```bash
python paperExtractor.py
```

Downloads research papers from arXiv across multiple CS categories (cs.LG, cs.AI, cs.DS, etc.) and extracts:
- Markdown text via `pymupdf4llm`
- Images and figures

**Output**: `papers.json`

### 2. Generate Training Data

```bash
export GEMINI_API_KEY="your-key-here"
python sdg.py
```

Uses Google Gemini 2.5 Flash to generate training examples. For each paper section, it creates three style-specific outputs with structured schemas.

**Output**:
- `data/eli5.jsonl`
- `data/intuitive.jsonl`
- `data/executive.jsonl`

### 3. Clean Data

```bash
python trim_jsonl.py
```

Ensures all training examples fit within the 2048 token limit required by the model.

**Output**: `data/{style}_2048.jsonl`

### 4. Split Train/Validation

```bash
python split_data.py
```

Creates 85/15 train/validation splits.

**Output**: `data/{style}_2048/train.jsonl`, `data/{style}_2048/valid.jsonl`

### 5. Train Adapters

```bash
mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --data data/eli5_2048 \
  --train \
  --adapter-path adapters/eli5 \
  --config adapter_config.json
```

Repeat for each style (eli5, intuitive, executive).

## Configuration

See `adapter_config.json` for training hyperparameters:

```json
{
  "fine_tune_type": "lora",
  "batch_size": 2,
  "iters": 500,
  "learning_rate": 1e-05,
  "lora_parameters": {
    "rank": 8,
    "dropout": 0.0,
    "scale": 20.0
  },
  "max_seq_length": 2048,
  "report_to": "wandb"
}
```

## Training Data Format

Each JSONL file contains entries like:

```json
{
  "messages": [
    {"role": "user", "content": "Explain this like I'm 5:\n{section_text}"},
    {"role": "assistant", "content": "**Analogy:** ...\n\n**Explanation:** ..."}
  ]
}
```

## File Structure

```
fineTune/
├── paperExtractor.py   # Step 1: Download papers
├── sdg.py              # Step 2: Generate training data
├── trim_jsonl.py       # Step 3: Enforce token limits
├── split_data.py       # Step 4: Train/val split
├── adapter_config.json # Training configuration
├── papers.json         # Downloaded paper data
├── papers/             # Raw PDF files
└── data/               # Generated datasets
    ├── eli5.jsonl
    ├── eli5_2048.jsonl
    ├── eli5_2048/
    │   ├── train.jsonl
    │   └── valid.jsonl
    └── ... (same for intuitive, executive)
```

## Requirements

- `google-genai`: For training data generation
- `mlx-lm`: For LoRA training on Apple Silicon
- `wandb`: For training metrics (optional)
- `arxiv`: For paper downloads
