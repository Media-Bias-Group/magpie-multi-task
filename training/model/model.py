"""Model module."""
from typing import List

import torch
from torch import nn
from transformers import AutoModel

from enums.model_checkpoints import ModelCheckpoint
from training.model.optimization.grads_wrapper import GradsWrapper
from training.model.head import HeadFactory
from training.tokenizer import tokenizer



class BackboneLM(GradsWrapper):
    """Language encoder model which is shared across all tasks."""

    def __init__(self, LM):
        """Fetch Language model from huggingface."""
        super(BackboneLM, self).__init__()
        if LM.value == "roberta-dummy":
            from transformers import RobertaConfig, RobertaModel

            configuration = RobertaConfig()
            configuration.num_attention_heads = 2
            configuration.num_hidden_layers = 1
            configuration.vocab_size = 60000

            model = RobertaModel(configuration)
            self.backbone = model
        else:
            self.backbone = AutoModel.from_pretrained(LM.value)


class Model(nn.Module):
    """Torch-based module."""

    def __init__(self, stl: List, LM: ModelCheckpoint, *args, **kwargs):
        """Inititialize model and create heads."""
        super().__init__()
        self.stl = stl
        self.subtask_id_to_subtask = {int(f"{st.id}"): st for st in stl}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.language_model = BackboneLM(LM=LM)
        self.language_model.backbone.pooler = None
        self.language_model.backbone.resize_token_embeddings(len(tokenizer))
        self.heads = nn.ModuleDict({str(st.id): HeadFactory(st, *args, **kwargs) for st in stl})

    def forward(self, X, attention_masks, Y, st_id):
        """Pass the data through the model and according head decided from heads dict."""
        # pass through the model
        x_enc = self.language_model.backbone(input_ids=X, attention_mask=attention_masks).last_hidden_state
        head = self.heads[str(st_id.item())]
        logits, loss, metric_values = head(x_enc, Y)

        return loss, metric_values
