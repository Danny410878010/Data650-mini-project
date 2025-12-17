# DATA650 Mini Project — Benchmarking vLLM vs SGLang

A UMD **DATA650 final mini project** focused on comparing two LLM inference frameworks: **vLLM** and **SGLang**.

This repository contains experiments, benchmarks, and analysis comparing the performance characteristics of these two popular serving frameworks for large language models.

---

## Repository Structure

```
Data650-mini-project/
├── README.md
├── vLLM/                    # vLLM experiments and notes
├── SGLang/                  # SGLang experiments and notes
└── vLLM vs SGLang/          # Side-by-side benchmarks and comparisons
```

Each framework folder follows this convention:
- **notebooks/**: Exploratory analysis and quick plots
- **scripts/**: Repeatable benchmark runners
- **results/**: CSV logs, figures, and tables
- **notes/**: Setup notes and troubleshooting

---

## Project Goals

The goal is to **deploy, run, and compare** vLLM and SGLang across controlled settings and answer key performance questions:

- How does **throughput** (tokens/sec) scale with **batch size**?
- How does **latency** change with **generation length** (`max_new_tokens`)?
- What tradeoffs appear across different **prompt workload sizes** (short/medium/long)?
- What happens under **GPU memory pressure** (KV cache growth, longer contexts)?

---

## Metrics

### Core Metrics (per run)
- **Latency per batch** (seconds)
- **Latency per request** (seconds) = latency per batch / batch size
- **Total output tokens** generated
- **Throughput** = output tokens / total time (tokens/sec)

### GPU Metrics (if available)
- GPU utilization (%)
- GPU memory used (MiB / GB)
- Power draw (optional)
- Temperature (optional)

---

## Environment Setup

### Option A: Google Colab (Recommended for GPU access)

1. Open the notebooks inside each folder in Colab
2. Install dependencies:

```bash
# Core dependencies
pip install -q -U pandas matplotlib psutil requests

# Framework installs
pip install -q -U vllm sglang
```

**Note:** If SGLang crashes on certain attention backends in Colab, try switching the backend (e.g., Triton) and disabling CUDA graphs.

### Option B: Local Setup (GPU strongly recommended)

**Requirements:**
- Python 3.10+ (3.11 often works; 3.12 compatibility varies)
- NVIDIA GPU + driver + CUDA-compatible PyTorch

**Installation:**
```bash
# Core dependencies
pip install -U pandas matplotlib psutil requests

# Frameworks
pip install -U vllm sglang
```

**Note:** CPU-only runs are possible with tiny models but won't reflect real serving performance.

---

## Benchmark Design

### Controlled Variables (keep identical across frameworks)
- **Model**: Same exact checkpoint
- **Prompt set**: Same prompts, same ordering
- **Decoding parameters**: temperature, top_p, etc.
- **Batch sizes**: e.g., 1, 2, 4, 8, 16
- **Generation length**: max_new_tokens (e.g., 32, 64, 128, 256)
- **Warmup**: Exclude first run from results

### Workload Types
- **Short prompts**: 1–2 sentences
- **Medium prompts**: A paragraph (~50-100 tokens)
- **Long prompts**: Multi-paragraph or multi-turn context (200+ tokens)

### Experiment Grid

```
framework ∈ {vllm, sglang}
workload ∈ {short, medium, long}
batch_size ∈ {1, 2, 4, 8, 16}
max_new_tokens ∈ {32, 64, 128, 256}
```

---

## Running Benchmarks

### Suggested Workflow

1. **Pick a model**: Start small (e.g., 0.5B–1B parameters) to validate the pipeline
2. **Warm up once**: Avoid counting first-run compilation/downloads
3. **Run the full grid**: Execute all combinations of parameters
4. **Save results**: Store one CSV with one row per run/config
5. **Generate plots**: Create visualizations for analysis

### CSV Schema (suggested)

```
framework, workload, batch_size, max_new_tokens, latency_s_per_batch, 
latency_s_per_request, out_tokens_total, tokens_per_s, gpu_util_avg, 
gpu_mem_used_mb_avg, ...
```

---

## Visualization & Analysis

### Recommended Plots

1. **Throughput Comparison** (conditioned on `max_new_tokens`)
   - X-axis: Batch size
   - Y-axis: Tokens/sec
   - Separate figure per `max_new_tokens` value

2. **Latency Comparison** (conditioned on `max_new_tokens`)
   - X-axis: Batch size
   - Y-axis: Latency per request (or per batch)
   - Separate figure per `max_new_tokens` value

3. **GPU Memory Usage** (optional)
   - X-axis: Batch size
   - Y-axis: GPU memory (MB/GB)
   - Helps identify memory pressure points

---

## Interpretation Hints

- **Throughput drops with longer sequences**: KV-cache pressure and/or memory bandwidth may be the bottleneck
- **Latency improves then worsens with batching**: You may be hitting GPU memory limits or scheduling overhead
- **Framework differences**: Look for where one framework significantly outperforms the other and consider why (batching strategy, kernel optimization, memory management)

---

## Contributing

This is a course project, but feedback and suggestions are welcome! Please open an issue or submit a pull request.

---

**Course**: DATA650 — Advanced Data Science  
**Institution**: University of Maryland, College Park  
**Semester**: Fall 2025
