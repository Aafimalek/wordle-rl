# I Taught a 270M-Parameter Model to Play Wordle on Free GPUs. Its Problem Was Never Strategy.

### How reading lost games, not a bigger model, took a tiny language model to 97% on WordleBench, and what I found inside two public leaderboards

![The same word before and after the last training round. Left: the model writes the rejected non-word SATON four times and loses. Right: the final model wins in four turns.](https://raw.githubusercontent.com/Aafimalek/wordle-rl/main/docs/blog/img/board.png)

Frontier models win almost every game of Wordle. On the public WordleBench leaderboard, GPT-5 wins 99 of 100 games, and many other large models win 94 to 98.

I wanted to know how far a very small model could go if I trained it for this one game and paid nothing for compute. So I took Gemma 3 270M, a language model with 270 million parameters, small enough to run on a laptop CPU, and trained it on free Kaggle notebook GPUs.

The final model wins 97 of 100 WordleBench games when its guesses are limited to dictionary words, and 87 with no help at all. That alone wasn't the most interesting part. The interesting part is *why* it lost the games it lost, and what I found when I read the scoring code of the leaderboards I was comparing against.

This is the full story: what worked, what didn't, and the mistakes along the way.

**Links:** [live demo](https://huggingface.co/spaces/aafimalek2032/five-letters) (runs in your browser) · [paper](https://doi.org/10.5281/zenodo.24116756) · [code and data](https://github.com/Aafimalek/wordle-rl) · [model](https://huggingface.co/aafimalek2032/wordle-rl-gemma3-270m)

---

## Why Wordle is a good test

Wordle has complete, simple rules. You get six guesses to find a hidden five-letter word. After each guess, every letter is marked green (right letter, right place), yellow (in the word, somewhere else) or grey (not in the word).

The optimal strategy is solved: 3.4212 guesses on average over the 2,315 classic answers, winning every game. And you don't even need strategy to win most games. In my measurement, a player that simply guesses *any* dictionary word that fits every clue so far wins 98.5% of the time.

So when a language model loses at Wordle, the interesting question is which part it fails at. For a language model there are two parts:

1. **Reasoning about the clues:** which letters are where, which are out.
2. **Writing a real word letter by letter.** This is not trivial for a model that reads and writes *tokens*, not letters. "CRANE" is one or two tokens to the model, not five characters.

The second part turned out to matter far more than I expected.

## The rules I set for myself

Before training anything, I froze an evaluation protocol, because small models are easy to fool yourself with.

- **A strict rule.** Every model output uses a turn. If the guess isn't one of the 12,972 words in the Wordle dictionary, it gets no feedback and still uses the turn.
- **Separate word sets.** A 534-word *selection* set chose checkpoints and recipes. A 463-word *heldout* set checked the selection set. The test sets (WordleBench, the Jeffers benchmark, 500 rare answers and TextArena's Wordle environment) were scored **once**, for a model declared in advance.
- **Decisions written before results.** Before each training round, I wrote down what the round tested and what I would do for each possible result. That stops you from picking whatever number looks best afterwards.
- **An independent audit.** A separate script replays every saved game with its own implementation of the Wordle rules and checks every win. Every number reported here passed it.

Everything ran on Kaggle's free tier: two T4 GPUs per session and a weekly quota of about 30 GPU hours. Total cost: $0.

## The method: read the lost games, then fix one thing

The approach turned into a loop:

1. Train a round.
2. Read the games the model lost.
3. Name the most common failure.
4. Write training data for that failure only.

![The method: each training round answers a diagnosis of the lost games of the round before it.](https://raw.githubusercontent.com/Aafimalek/wordle-rl/main/docs/blog/img/overview.png)

The loop found three failures, one after the other. Each became visible only after the previous one was fixed.

![Win rate on the 534-word selection set after each training round, model alone and with word-level decoding.](https://raw.githubusercontent.com/Aafimalek/wordle-rl/main/docs/blog/img/progress.png)

## Failure 1: it memorised the answers instead of reading the clues

I started with imitation learning. A solver (a lookahead search that averages 3.4216 guesses, within 0.0004 of the proven optimum) played thousands of games, and the model learned to copy its moves. Each prompt shows the board so far in spaced letters, and the model answers with a short note on what it knows, then a guess:

```
Turn 3: C R O N K -> X G G G X
Turns left: 3
-> <state>fixed: _ R O N _ | has: - | out: C E K L M S W</state>
   <guess>F R O N D</guess>
```

The first model won 20 of 20 games on words it had trained on, but only 13 of 20 on new words. It had memorised the answer list. Adding games on 1,610 extra common words fixed that part.

A subtler problem remained. The optimal solver often plays "probe" words that deliberately break the clues to gather information. A small model can't learn when that's a good idea, so 49% of its guesses contradicted clues it already had. I switched to a teacher that only plays words consistent with every clue. That cut contradictions to 7.5%.

## Failure 2: it followed the clues but couldn't spell

After that fix, the model reasoned well. Of the games where it never wrote an invalid word, it won 269 of 272. Its problem was elsewhere:

- Every invalid turn was a well-formed five-letter **non-word**.
- 78% of those non-words fit every clue, and 38% were one letter away from the answer.
- 97.6% of the real words it played were words it had seen as training targets, and those covered only 6,871 of the 12,972 dictionary words.

The model knew the *pattern* the answer had to match. It couldn't recall a *real word* with that pattern. Its win rate rose with how often the answer had appeared in training: 12.5% for words it had never seen as a target, 77% for words seen 100 times or more.

So the fix wasn't more games. It was a **spelling drill**: thousands of boards, each paired with one real, common word that fits it. Every one of 5,557 common words was a target the same number of times, so the model practised the whole vocabulary evenly.

Two rounds of the drill raised the model-alone win rate from **62.0% to 83.1%** (mean of three training seeds). That was a bigger jump than any change to the teacher or the training method.

## Failure 3: when the game said "not a word", it wrote the same word again

After the drill, the model rarely wrote a non-word. But when it did, something strange happened. Here is a real game, with the hidden word BATON:

```
CRATE  -> X X Y Y X
TALON  -> Y G X G G
SATON  -> INVALID (not in word list)
SATON  -> INVALID (not in word list)
SATON  -> INVALID (not in word list)
SATON  -> INVALID (not in word list)
```

The prompt shows the rejection, but the model writes SATON again and again until the game ends. This wasn't a one-off. After a first non-word, one model's next turn was **the same word 76 times and a different real word only 8 times**. Most of its lost games were one rejected word, repeated.

The reason is a classic problem in imitation learning, called covariate shift. The model had only ever trained on states that a good player creates. It had never seen a board on which *its own* first choice had just been rejected, so it didn't know what to do there.

The fix, in the spirit of an imitation-learning method called DAgger, was to train on the model's own mistakes:

1. Let the model play about 8,500 words that are never used for evaluation.
2. Every time the game rejects one of *its own* non-words, keep that exact board, rejection included.
3. Label it with a real word that fits the clues, and train on those boards.

That gave about 3,000 new training samples per run, trained for half an hour on one free T4.

![Lost games by cause. Before the recovery round, most losses are one rejected word written again and again; after it, few are.](https://raw.githubusercontent.com/Aafimalek/wordle-rl/main/docs/blog/img/losses.png)

The result:

- After a first non-word, the model now writes a real word 51 times and repeats the rejected one 32 times (before: 8 against 76).
- Losses that consist of one repeated non-word fell from **71 to 15**.
- Model-alone wins on the selection set rose from **83.9% to 88.0%**. That is a statistically significant gain (paired test, p = 0.013), and the heldout set, which played no part in choosing the model, rose by about the same amount (84.7% to 88.3%).

## What I found inside two public leaderboards

To compare with frontier models, I read the scoring code of the two public Wordle leaderboards whose word lists are published: WordleBench and the Jeffers Wordle-AI-Benchmark.

**Neither checks guesses against a dictionary.**

- WordleBench scores whatever text is inside the guess tag. There is no word list.
- Jeffers accepts any five lowercase letters. Its code *imports* a dictionary check but never calls it.

So on both leaderboards a misspelled guess like SATON still gets colours, and the game moves on. Under my strict rule it gets nothing.

That matters a lot, because misspelling was exactly my model's weakness. The same model, on the same words, scores very differently depending on that one rule:

![The same model on the same 534 words under three rules: strict, retry-3, and the leaderboards' rule where non-words get colours.](https://raw.githubusercontent.com/Aafimalek/wordle-rl/main/docs/blog/img/rules.png)

Before the recovery round, the gap between the two rules was 10.5 points. After it, the gap shrank to 7.5, which agrees with the diagnosis: the leaderboards' rule hides exactly the failure the recovery round fixed.

So I report three labelled rules everywhere: **strict** (my headline), **retry-3** (a rejected non-word doesn't use a turn, at most three times in a row) and **open-vocab** (the leaderboards' rule). I think any game benchmark for language models should state this rule, because it can move a score by 10 points.

## The final test, scored once

The final model was chosen on the selection set by a rule I wrote before the last training round, and declared before any test set was scored.

![Final test on WordleBench and Jeffers: the model alone under two rules, with word-level decoding, and the cited GPT-5 score.](https://raw.githubusercontent.com/Aafimalek/wordle-rl/main/docs/blog/img/results.png)

| Test set | Alone, strict rule | Alone, leaderboard rule | Word-level decoding | GPT-5 (cited) |
|---|---|---|---|---|
| WordleBench (100 words) | 87.0% | 94.0% | **97.0%** | 99% |
| Jeffers (50 words) | 74.0% | 88.0% | **96.0%** | 100% |
| TextArena (500 games) | 85.8% | – | **98.6%** | – |
| Rare answers (500 words) | 42.2% | 51.4% | 76.6% | – |

**Word-level decoding** limits each guess to dictionary words that the model ranks itself. The code never looks at the clues; the model still decides which real word fits. It's an aid, so I always report it next to the model alone.

A few things stand out:

- **Under the leaderboard's own rule, the model alone wins 94% of WordleBench games.** That's the score listed for gpt-5-mini, and in the range of many frontier models. With very different harnesses and prompts, it isn't a perfectly fair comparison, but it's the same 100 words.
- **Jeffers is harder alone** (74%), but it has only 50 games, so one game is 2 points.
- **Rare words are the real weakness.** The model's word knowledge is concentrated on common words.
- Compared with an earlier final model from the start of the project, the model alone gained **27 points on WordleBench** (60% to 87%).

The paper's appendix shows every WordleBench game the final model lost. Of its 13 losses, 10 still contain non-words, and in 9 of those a rejected word is written again. So the recovery round reduced that failure but didn't remove it. The other 3 losses are traps: sets of real words that differ in one letter, like BONDS, BONGS and BONKS.

## What didn't work

Negative results are part of the story, so here they are:

- **Reinforcement learning (GRPO).** It reduced invalid turns a little, but never raised the win rate on held-out words. The likely reason: rollouts on mostly memorised answers gave a quarter of the training groups no signal at all.
- **A stronger teacher** that prefers common words and plays probe words on traps: no better for the model alone.
- **Writing whole words** instead of spaced letters: fewer invalid words, but half the guesses ignored the clues, and wins dropped to 44.9%.
- **Self-distillation** of the model's own best games: +3 points alone, but it hurt the decoding-aided score, so I dropped it.

The biggest gains came from small, targeted data changes, not from new algorithms.

## Engineering on free GPUs

Free tiers have sharp edges. A few that cost me time:

- **T4 GPUs have no fast bf16, and Gemma 3 can overflow in fp16.** I kept weights in fp32 and checked fp16 against fp32 on real batches before trusting it.
- **Kaggle's 20 GB output disk filled up with checkpoints.** Large files moved to scratch disk, and every save now makes room first.
- **Hugging Face's free private storage (100 GB) filled up.** Uploads then failed *without an error* and left a stale checkpoint. I evaluated that stale model by accident and had to retract those results. Uploads now skip optimiser state, retry, and log the real error.
- **Free sessions end without warning,** so every job is resumable and every evaluation saves results game by game.

Over the whole project, the recorded Kaggle sessions add up to about 31 hours.

## Try it yourself

The [live demo](https://huggingface.co/spaces/aafimalek2032/five-letters) runs the final model **in your browser**, with no server. Pick a hidden word and watch it play, notes and all. You can choose an 8-bit copy (440 MB, faster) or full precision (1.7 GB). I checked both against the paper on all 100 WordleBench words: the full-precision copy reproduces every recorded game exactly, and the 8-bit copy matches 83 of them. The demo also lets you replay every game from the paper's evaluations.

Everything is open:

- **Paper:** [doi.org/10.5281/zenodo.24116756](https://doi.org/10.5281/zenodo.24116756)
- **Code, word lists and every evaluation summary:** [github.com/Aafimalek/wordle-rl](https://github.com/Aafimalek/wordle-rl)
- **Model:** [huggingface.co/aafimalek2032/wordle-rl-gemma3-270m](https://huggingface.co/aafimalek2032/wordle-rl-gemma3-270m)
- **Every game the model played**, with its raw output, plus the training data of each round: [Hugging Face collection](https://huggingface.co/collections/aafimalek2032/five-letters-wordle-rl-6af332879e666b4001ede7ed)

## What I'd tell someone starting a project like this

1. **Read your failures, not just your scores.** The aggregate win rate never told me "spelling" or "repeats a rejected word". Counting one simple event, *what does the model do on the turn after its first non-word?*, did.
2. **Imitation learning needs the student's own mistakes.** My spelling drill already contained rejected words, but they were random ones, not the model's own. Only training on states the model itself creates fixed the recovery problem.
3. **Check the rules behind every number you compare against.** Two leaderboards I trusted were measuring something easier than I assumed. Reading their code took an hour and changed how I report every result.
4. **Write your decisions down before the result exists.** It's the cheapest protection against fooling yourself, and it makes negative results easy to report honestly.

A 270M-parameter model now plays this game almost as well as frontier models, on the same word lists, for $0. What held it back wasn't strategy. It was spelling, and knowing what to do after a mistake.

One caveat in the spirit of honesty: the spelling drill used common English words as targets, and some of those are also answers in the test sets. No training game ever used a test word as its *answer*, and the paper reports a strict variant without those targets, but it's a real limitation and I disclose it there.

*I'm Aafi Khan Malek, an independent researcher. If you have questions or ideas, the [GitHub repo](https://github.com/Aafimalek/wordle-rl) is the best place to reach me.*
