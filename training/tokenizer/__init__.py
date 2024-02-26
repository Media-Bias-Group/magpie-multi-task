"""Initialize the tokenizer."""

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("training/tokenizer/mb-mtl-tokenizer", add_prefix_space=True)
