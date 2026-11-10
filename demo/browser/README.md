---
title: Five Letters, 270M Parameters
emoji: 🔤
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: A 270M model plays the five-letter word game, live
models:
  - aafimalek2032/wordle-rl-gemma3-270m
  - aafimalek2032/wordle-rl-gemma3-270m-onnx
datasets:
  - aafimalek2032/five-letters-results
---

# Five Letters, 270M Parameters

Gemma 3 270M, trained on free GPUs, plays the five-letter word game **live in your browser** (transformers.js; nothing runs on a server), and every recorded game behind the paper's numbers can be replayed.

- Code and paper: https://github.com/Aafimalek/wordle-rl
- Model: https://huggingface.co/aafimalek2032/wordle-rl-gemma3-270m
- Browser copy (ONNX): https://huggingface.co/aafimalek2032/wordle-rl-gemma3-270m-onnx

`game.js` is a line-for-line port of the project's game rules and prompt format; `games.json` holds the recorded games of the final model (alone and with word-level decoding) on every word set of the paper.
