# TinySSL-Distill

A lightweight, open-source toolkit for self-supervised learning and knowledge distillation, with robust small-sample experiments and zero-shot/robustness benchmarks.

## Overview

**TinySSL-Distill** implements a practical and reproducible pipeline for evaluating the effectiveness of knowledge distillation and self-supervised pretraining on small datasets.  
It supports evaluation on custom datasets (e.g., 65-class small sample), Caltech101 (zero-shot), and CIFAR-10-C (robustness to corruption), using a simple, modular PyTorch codebase.

**Key Features:**

- 🧠 **Self-supervised + Distillation:** Easily train, distill, and compare methods.
- 🔬 **Small-sample Evaluation:** Robust linear probing on few-shot datasets.
- 🏆 **Zero-Shot Transfer:** Zero-shot accuracy on Caltech101 using CLIP text prompts.
- 🛡 **Robustness Benchmarks:** Automated evaluation on CIFAR-10-C with auto-adapted heads.
- ⚡ **Simple & Modular:** Plug in your own backbone, dataset, or evaluation logic.

## Main Experiments

- **Linear Probe** on a custom 65-class dataset (small sample scenario).
- **Zero-shot CLIP Transfer** on Caltech101.
- **Robustness Evaluation** on CIFAR-10-C (with automatic head adaptation and feature extraction).
- **Model Statistics:** Inference speed and parameter count.
