"""The tokenizer used by the subtasks."""

from transformers import RobertaTokenizerFast

from config import NEW_TOKENS

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
added_tokens_count = tokenizer.add_tokens(NEW_TOKENS)
tokenizer.save_pretrained("training/tokenizer/mb-mtl-tokenizer")
