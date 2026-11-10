---
title: Five Letters, 270M Parameters
emoji: 🔤
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
short_description: A 270M model plays the five-letter word game, live
models:
  - aafimalek2032/wordle-rl-gemma3-270m
datasets:
  - aafimalek2032/five-letters-results
---

# Five Letters, 270M Parameters

Watch Gemma 3 270M, trained on free GPUs, play the five-letter word game, with the paper's own code and full-precision model on ZeroGPU (model alone or word-level decoding), or replay the exact games of the paper's final test.

- Code and paper: https://github.com/Aafimalek/wordle-rl · DOI: https://doi.org/10.5281/zenodo.24114970
- Model: https://huggingface.co/aafimalek2032/wordle-rl-gemma3-270m
