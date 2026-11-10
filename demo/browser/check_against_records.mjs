// Play the 100 WordleBench test words with the browser code path (transformers.js + game.js) and compare every
// game with the paper's record (Python, PyTorch, T4). Usage: node test_node.mjs <dtype: q8|fp32> [limit]
import { readFileSync } from "node:fs";
import { AutoModelForCausalLM, AutoTokenizer, env } from "@huggingface/transformers";
import { Game, modelTurn, parseGuess } from "../static/game.js";

const dtype = process.argv[2] || "q8";
const limit = Number(process.argv[3] || 100);
const folder = process.argv[4] || "onnx_repo";
env.localModelPath = "C:/Users/WIN10~1/";
env.allowRemoteModels = false;
const valid = new Set(readFileSync("../static/words.txt", "utf8").split("\n").filter(Boolean));
const data = JSON.parse(readFileSync("../static/games.json", "utf8"));
const words = Object.keys(data.sets.wordlebench.alone).sort().slice(0, limit);

const tok = await AutoTokenizer.from_pretrained(folder);
const model = await AutoModelForCausalLM.from_pretrained(folder, { dtype, device: "cpu" });
let same = 0, won = 0, recWon = 0, turns = 0;
const t0 = Date.now();
const diffs = [];
for (const w of words) {
  const g = new Game(w);
  while (!g.done) {
    const raw = await modelTurn(model, tok, g);
    g.step(raw, parseGuess(raw), valid);
    turns++;
  }
  const rec = data.sets.wordlebench.alone[w];
  const mine = g.turns.map((t) => t.guess || "?").join(" ");
  const theirs = rec.t.map((t) => t[0] || "?").join(" ");
  if (mine === theirs) same++;
  else diffs.push(`${w}: browser ${mine} | record ${theirs}`);
  won += g.won ? 1 : 0;
  recWon += rec.w;
}
const s = (Date.now() - t0) / 1000;
console.log(`dtype ${dtype}: ${words.length} games, identical to the record ${same}, won ${won} (record ${recWon}), ` +
            `${(s / turns).toFixed(2)} s per turn`);
diffs.slice(0, 15).forEach((d) => console.log("  " + d));
