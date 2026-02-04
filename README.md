# [ICLR 2026] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval

![Nissan skyline](assets/skyline.png)

> "The GT-R is not a supercar for a select few; it is a supercar for everyone, built to be enjoyed anywhere, anytime, by anyone." --**Nissan skyline**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the **(GTR)** — a lightweight, plug-and-play module designed to empower any multivariate time series forecasting (MTSF) model with the ability to capture global periodic patterns far beyond the fixed look-back window.

---

## 📣 News
* `2026/2/4` 🚀 Code has be released.
* `2026/1/26` 💥💥 GTR is honored to be accepted by ICLR 2026!

---

## 🌟 Key Innovation: The Global Temporal Retriever (GTR)

Existing MTSF models are fundamentally limited by their reliance on a fixed-length historical window, making them unable to capture crucial global periodic patterns (e.g., weekly, monthly, seasonal trends) that span cycles much longer than the input.

**GTR solves this by**:
1.  **Maintaining a Learnable Global Representation**: A parameter matrix `Q ∈ R^(L×N)` encodes the entire global cycle pattern for all `N` variables.
2.  **Dynamic Retrieval & Alignment**: For any input sequence, GTR identifies its position within the global cycle and retrieves the corresponding segment.
3.  **Joint Local-Global Modeling**: The retrieved global segment is stacked with the local input and processed by a 2D convolution to model dependencies across both scales.
4.  **Seamless Integration**: The enriched representation is fused back via a residual connection, making GTR compatible with *any* existing forecasting backbone (MLP, Transformer, Mamba, etc.) without architectural changes.

---

## 📈 Performance Highlights

*   **State-of-the-Art Results**: GTR+MLP achieves SOTA performance on 6 real-world datasets for both short-term and long-term forecasting.
*   **Significant Gains**: On the challenging Solar-Energy dataset, GTR outperforms the second-best model by **8.2% in MSE** and **6.5% in MAE**.
*   **Plug-and-Play Enhancement**: GTR consistently improves diverse SOTA models (iTransformer, PatchTST, DLinear) by up to **91.9% MSE reduction** (DLinear on PEMS04).
*   **Extreme Efficiency**: The GTR module itself adds only **40.1K parameters** and **4.50M MACs**. The full GTR+MLP model uses just **0.98M parameters**, which is only 19% of iTransformer's size.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch 1.10+
*   Other dependencies (see `requirements.txt`)

### Installation

```bash
conda create -n GTR python=3.8
conda activate GTR
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Run
You can use the following script to obtain the prediction results (Recommended). For example, to reproduce all the experiment in the paper, you can run the following script:

```bash
bash run_main.sh
```

To reproduce all the ablation experiment in the paper, run the following script:

```bash
bash run_ablation.sh
```

## 📜 Citation
If you find GTR useful, please consider citing our paper:
```bibtex
@inproceedings{GTR,
title={Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval},
author={Anonymous},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=QUJBPSfyui}
}
```