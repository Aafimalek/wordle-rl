// Five Letters game engine for the browser: a line-for-line port of fiveletters/game.py, feedback.py and
// prompts.py (strict rule, spaced-letter prompt format with a state line). Shared by the page and the Node test.

export const MAX_TURNS = 6;
const SYMBOLS = "XYG";
const ERROR_TEXT = {
  format: "INVALID (no <guess> found)",
  not_in_word_list: "INVALID (not in word list)",
};

// Greens first; then, left to right, a letter is yellow only while the answer has an unmatched copy of it.
export function feedback(guess, answer) {
  const result = [0, 0, 0, 0, 0];
  const unmatched = {};
  for (let i = 0; i < 5; i++) {
    if (guess[i] === answer[i]) result[i] = 2;
    else unmatched[answer[i]] = (unmatched[answer[i]] || 0) + 1;
  }
  for (let i = 0; i < 5; i++) {
    if (result[i] === 0 && (unmatched[guess[i]] || 0) > 0) {
      result[i] = 1;
      unmatched[guess[i]] -= 1;
    }
  }
  return result;
}

export const feedbackString = (fb) => fb.map((v) => SYMBOLS[v]).join("");
const spaced = (s) => s.toUpperCase().split("").join(" ");

export function instructions(maxTurns = MAX_TURNS) {
  return [
    `Guess the hidden 5-letter English word in at most ${maxTurns} turns.`,
    "Feedback per letter: G = right letter, right spot. Y = in the word, wrong spot. X = not in the word.",
    `Reply with <state>what you know</state> then <guess>${spaced("crane")}</guess>.`,
  ].join("\n");
}

export function parseGuess(text) {
  const found = [...(text || "").matchAll(/<guess>([\s\S]*?)<\/guess>/gi)];
  if (!found.length) return null;
  const letters = found[found.length - 1][1].replace(/\s+/g, "");
  if (letters.length !== 5 || !/^[A-Za-z]{5}$/.test(letters)) return null;
  return letters.toLowerCase();
}

export function parseState(text) {
  const found = [...(text || "").matchAll(/<state>([\s\S]*?)<\/state>/gi)];
  return found.length ? found[found.length - 1][1].split(/\s+/).filter(Boolean).join(" ") : null;
}

export class Game {
  constructor(answer, maxTurns = MAX_TURNS) {
    this.answer = answer;
    this.maxTurns = maxTurns;
    this.turns = []; // {raw, guess, valid, feedback, error}
  }
  get won() {
    return this.turns.some((t) => t.valid && t.feedback.every((v) => v === 2));
  }
  get done() {
    return this.won || this.turns.length >= this.maxTurns;
  }
  get turnsLeft() {
    return this.maxTurns - this.turns.length;
  }
  step(raw, guess, validWords) {
    if (this.done) throw new Error("game is already over");
    let turn;
    if (guess === null) turn = { raw, guess: null, valid: false, feedback: null, error: "format" };
    else if (!validWords.has(guess)) turn = { raw, guess, valid: false, feedback: null, error: "not_in_word_list" };
    else turn = { raw, guess, valid: true, feedback: feedback(guess, this.answer), error: null };
    this.turns.push(turn);
    return turn;
  }
  prompt() {
    const lines = [instructions(this.maxTurns)];
    this.turns.forEach((t, i) => {
      if (t.valid) lines.push(`Turn ${i + 1}: ${spaced(t.guess)} -> ${feedbackString(t.feedback).split("").join(" ")}`);
      else if (t.guess !== null) lines.push(`Turn ${i + 1}: ${spaced(t.guess)} -> ${ERROR_TEXT[t.error]}`);
      else lines.push(`Turn ${i + 1}: (none) -> ${ERROR_TEXT[t.error]}`);
    });
    lines.push(`Turns left: ${this.turnsLeft}`);
    return lines.join("\n");
  }
}

// One model turn: the chat template, greedy decoding (the model's generation_config samples by default, so
// do_sample is forced off), at most 96 new tokens, stop at <eos> or <end_of_turn>.
export async function modelTurn(model, tokenizer, game) {
  const inputs = tokenizer.apply_chat_template([{ role: "user", content: game.prompt() }], {
    add_generation_prompt: true,
    return_dict: true,
  });
  const out = await model.generate({ ...inputs, max_new_tokens: 96, do_sample: false });
  const n = inputs.input_ids.dims[1];
  const completion = out.slice(null, [n, null]);
  return tokenizer.batch_decode(completion, { skip_special_tokens: true })[0];
}
