# Third-party notices

`victoriana_training/muon.py` is adapted from the Muon/AdamW optimizer in
[nanochat](https://github.com/karpathy/nanochat) (Andrej Karpathy), which in turn derives from
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (Keller Jordan), with
contributions from @chrisjmccormick. Both upstream projects are MIT-licensed; their license text
applies to that file.

The training loop's optimizer grouping and learning-rate conventions (e.g. AdamW LR scaled by
(d_model/768)^-0.5) follow nanochat.

Model architecture: Hugging Face Transformers `LlamaForCausalLM` (Apache-2.0).
Tokenizer training: Hugging Face `tokenizers` (Apache-2.0).
