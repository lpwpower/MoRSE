<div align="center">

# MoRSE: Task-Oriented Multi-Agent System with<br>Mixture of Role-Subtask Experts

**NeurIPS 2026**

Peiwen Li · Shiyang Zhang · Yangtian Zhang · Sizhuang He · David van Dijk · Rex Ying

**Yale University**

<p>
  <a href="https://arxiv.org/abs/2608.09251"><img src="https://img.shields.io/badge/arXiv-2608.09251-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://neurips.cc/Conferences/2026"><img src="https://img.shields.io/badge/NeurIPS-2026-4b44ce" alt="NeurIPS 2026"></a>
</p>

</div>

<p align="center">
  <img src="assets/framework.png" width="100%" alt="MoRSE framework: ToMAS task decomposition, MoLE role-subtask LoRA experts, and HGRPO two-layer credit assignment">
</p>

> [!NOTE]
> 🚧 This is the **initial code release** accompanying the paper: the `morse` core library and the full training / inference pipelines for both benchmarks are included. We are continuing to sort out a cleaner, better-documented version and will update the repository as soon as possible.

## 🔥 News

- **[2026-09]** 🚀 Initial code release: `morse` core library + SRDD / SciCode training & inference pipelines.
- **[2026-09]** 🎉 MoRSE is accepted to **NeurIPS 2026**!
- **[2026-08]** 📄 The paper is available on [arXiv](https://arxiv.org/abs/2608.09251).

## ✨ Overview

Existing LLM-based multi-agent systems mainly rely on coarse prompt-level differentiation without parameter adaptation for diverse subtasks, resulting in insufficient inter-agent heterogeneity and limited specialized capability. **MoRSE** distinguishes agents with **(role, subtask)-conditional specialization** at both the task-structure and parameter levels:

- 🧭 **Task-Oriented Multi-Agent System (ToMAS)** decomposes each task into a dependency-aware DAG of subtasks and assigns each agent a specific (role, subtask), introducing task-level specialization across collaborating agents.
- 🧩 **Mixture of Role-Subtask LoRA Experts (MoLE)** uses a dynamic mixture of (role, subtask) LoRA experts with a prototype-based semantic router, augmenting agents with parameter-level specialization on a shared LLM substrate cost-effectively.
- 🎯 **Hierarchical GRPO (HGRPO)** co-optimizes experts and router stably under sparse task rewards via two-layer credit assignment, disentangling expert quality from routing quality.

Experiments on code-generation benchmarks across three backbones show improvements in both whole-task and step-wise performance, and the gains generalize across held-out task categories and domains.

## 🗂️ Repository Layout

```
morse/        # shared core
  taskgraph/  #   (role, subtask) DAG: data structure + LLM generator (prompt-injectable)
  mole/       #   LoRA experts + prototype router + MoLE generator
  llm/        #   HF model wrapper + taskgraph->execution converter
  hgrpo/      #   two-layer credit (within-route / cross-route advantage)
srdd/         # SRDD benchmark: ToMAS execution · eval (reward) · data · train/infer · run scripts
scicode/      # SciCode benchmark: pipeline · eval · data · train/infer · run scripts
envs/         # conda environments (Qwen/Llama vs Gemma-4)
```

## 🛠️ Installation

The two environments differ **only** in the `transformers` version:

| Backbone | Environment | transformers |
|---|---|---|
| Qwen3-4B-Instruct / Llama-3.1-8B-Instruct | `envs/environment_qwen_llama.yml` | 4.57 |
| Gemma-4-31B-IT | `envs/environment_gemma.yml` | 5.6 |

```bash
conda env create -f envs/environment_qwen_llama.yml   # or envs/environment_gemma.yml
conda activate morse-qwen-llama
```

Install the `torch==2.9.1` build matching your CUDA from <https://pytorch.org>.

## 📊 Data

- **SRDD** — shipped under `srdd/data/` (full set + IID split + OOD split).
  Splits are reproducible via `srdd/data/make_splits.py`; rules in `srdd/data/SPLIT_README.md`.
- **SciCode** — problems under `scicode/data/` (full + IID split + OOD split).
  The ~1 GB numerical ground truth `test_data.h5` is fetched separately:
  `bash scicode/eval/download_test_data.sh`.

## 🚀 Quick Start

Run everything from the repo root so the `morse`, `srdd`, and `scicode` packages
import. The `run_*.sh` scripts already export `PYTHONPATH` (they resolve the repo
root themselves); set it manually only if you invoke `train.py` / `infer.py`
directly:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"   # repo root; only needed for direct python calls
```

Each run script reads its model / data / GPU settings from environment-overridable
variables at the top of the file (e.g. `MODEL_NAME`, `DATASET`, `NPROC`,
`MOLE_CHECKPOINT`) — edit those or override inline; see the script header for the
full list. SciCode evaluation also needs the numerical ground truth:

```bash
# SciCode: fetch the ~1 GB ground-truth tests once (-> scicode/eval/data/test_data.h5)
bash scicode/eval/download_test_data.sh
```

The pipelines consume per-instance *(role, subtask)* task graphs. Generate them once
(uses `morse.taskgraph` + the decomposition prompt; written to `<bench>/data/taskgraphs/`):

```bash
# SRDD: one task graph per sample — point --csv at the split you train/eval on
python srdd/data/make_taskgraphs.py    --csv srdd/data/iid_train.csv --output-root srdd/data/taskgraphs
# SciCode: defaults to scicode/data/mydev.jsonl -> scicode/data/taskgraphs
python scicode/data/make_taskgraphs.py --dataset scicode/data/mydev.jsonl --output-dir scicode/data/taskgraphs
```

```bash
# SRDD
bash srdd/run_train.sh        # HGRPO training (MoLE + two-layer credit)
bash srdd/run_infer.sh        # evaluate a trained checkpoint (set CHECKPOINT_DIR)

# SciCode
bash scicode/run_train.sh     # HGRPO training (no-anchor, hierarchical credit)
bash scicode/run_infer.sh     # evaluate a trained checkpoint (set MOLE_CHECKPOINT)
```

## 🗓️ Release Plan

- [x] Paper on arXiv
- [x] Initial code release: `morse` core library (ToMAS · MoLE · HGRPO) + SRDD / SciCode training & inference pipelines
- [ ] Camera-ready version
- [ ] Cleaned-up full release with improved documentation

## 📖 Citation

If you find MoRSE useful for your research, please cite:

```bibtex
@inproceedings{li2026morse,
  title     = {MoRSE: Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts},
  author    = {Li, Peiwen and Zhang, Shiyang and Zhang, Yangtian and He, Sizhuang and van Dijk, David and Ying, Rex},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## 📄 License

Apache-2.0 (see [LICENSE](LICENSE)). Third-party dataset attributions in [NOTICE](NOTICE).
