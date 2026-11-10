import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory):
    """Tests build their own pattern table and never touch the user's solver cache."""
    os.environ["FL_CACHE_DIR"] = str(tmp_path_factory.mktemp("cache"))
    os.environ.pop("FL_TIME_BUDGET_HOURS", None)
    os.environ.pop("FL_RUNS_DIR", None)
    yield


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A 2-layer random Llama with a byte-level BPE tokenizer and a chat template. CPU-fast."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from fiveletters.game import Game
    from fiveletters.prompts import render_completion, render_prompt

    g = Game("ghost")
    corpus = [render_prompt(g), render_completion(g, "salet"), "<guess>G H O S T</guess> fixed has out"] * 20
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<pad>", "<bos>", "<eos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>",
                                               pad_token="<pad>")
    tok.chat_template = (
        "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<eos>\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
                                   bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                                   pad_token_id=tok.pad_token_id)
    model = transformers.LlamaForCausalLM(cfg)
    d = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)
