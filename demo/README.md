# Demos

| Folder | Hugging Face Space type | What it runs |
|---|---|---|
| `browser/` | Static (free for every account) | The final model in the visitor's browser with transformers.js (ONNX copy: `aafimalek2032/wordle-rl-gemma3-270m-onnx`), plus every recorded game of the paper |
| `zerogpu/` | Gradio on ZeroGPU | The paper's own code and full-precision PyTorch model, model alone or with word-level decoding, plus the recorded final-test games |

Build the data files from the fetched Kaggle outputs (`kaggle_runs/`) with `python demo/<folder>/build_data.py <output folder>`; the browser demo also needs `words.txt` (the 12,972 dictionary words, one per line).

`browser/game.js` is a line-for-line port of `src/fiveletters/game.py`, `feedback.py` and `prompts.py`. `browser/check_against_records.mjs` replays the 100 WordleBench words with the browser code path (Node, transformers.js) and compares each game with the paper's record: the full-precision ONNX copy reproduced all 100 recorded games exactly (won 87, as in the paper), and the 8-bit copy reproduced 83 of 100 exactly (won 83; paper 87).
