"""Golden phonemes for the fallback network, from the PyTorch weights it was trained as.

Run from the package root:
    uv run --python 3.12 --with torch==2.8.0 --with transformers==4.51.2 --with safetensors \
        Tools/reference.py

Reads Tools/fallback-words.txt and writes Tests/MisakiSwiftTests/Fixtures/fallback-golden.json.
The decoding loop mirrors the Swift network exactly: greedy, start from BOS, stop at EOS,
at most 49 generated tokens, and phoneme ids above 3 map into phoneme_chars.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import BartConfig, BartForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
RESOURCES = ROOT / "MisakiData"
WORDS = ROOT / "Tools" / "fallback-words.txt"
OUT = ROOT / "Tests" / "MisakiSwiftTests" / "Fixtures" / "fallback-golden.json"


def build(prefix):
    cfg = BartConfig.from_json_file(str(RESOURCES / f"{prefix}_bart_config.json"))
    model = BartForConditionalGeneration(cfg)
    missing, unexpected = model.load_state_dict(
        load_file(str(RESOURCES / f"{prefix}_bart.safetensors")), strict=False
    )
    assert not unexpected, unexpected
    tied = {"lm_head.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"}
    assert set(missing) <= tied, missing
    model.tie_weights()
    shared = model.model.shared.weight
    for emb in (model.model.encoder.embed_tokens, model.model.decoder.embed_tokens, model.lm_head):
        assert emb.weight.data_ptr() == shared.data_ptr(), "weights are not tied"
    model.eval()
    raw = json.loads((RESOURCES / f"{prefix}_bart_config.json").read_text())
    return model, raw["grapheme_chars"], raw["phoneme_chars"], cfg


def phonemize(model, graphemes, phonemes, cfg, word):
    g2t = {c: i for i, c in enumerate(graphemes)}
    ids = [cfg.bos_token_id] + [g2t.get(c, 3) for c in word] + [cfg.eos_token_id]
    with torch.no_grad():
        enc = model.get_encoder()(input_ids=torch.tensor([ids]))
        dec = [cfg.bos_token_id]
        for i in range(50):
            if i == 49:
                break
            logits = model(encoder_outputs=enc, decoder_input_ids=torch.tensor([dec])).logits[0, -1]
            nxt = int(torch.argmax(logits))
            if nxt == cfg.eos_token_id:
                break
            dec.append(nxt)
    return "".join(phonemes[t] for t in dec[1:] if t > 3)


words = [w for w in WORDS.read_text().splitlines() if w.strip()]
out = {}
for prefix in ("us", "gb"):
    model, g, p, cfg = build(prefix)
    out[prefix] = {w: phonemize(model, g, p, cfg, w) for w in words}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {OUT.relative_to(ROOT)} with {len(words)} words per dialect")
