from transformers import  AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
tqdm.pandas()
from transformers import AutoModel
from torch import nn
from training.data import st_1_babe_10 as babe


def HeadFactory(num_classes):
    """Decide which head to use for the specific SubTask (st)."""
    # return RegressionHead(input_dimension=768, hidden_dimension=768, dropout_prob=0.1)
    return CLSHead(num_classes=num_classes, input_dimension=768, hidden_dimension=768, dropout_prob=0.1)

class CLSHead(nn.Module):
    """Classifier inspired by one used in RoBERTa."""

    def __init__(self, num_classes, input_dimension, hidden_dimension, dropout_prob):
        """Initialize the head."""
        super().__init__()
        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dimension, num_classes)

    def forward(self, X):
        """Feed the data through head accordingly to RoBERTa approach and compute loss."""
        x = X[:, 0, :]  # take <s> token (equiv. to [CLS])

        # pass CLS through classifier
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits


class BackboneLM(nn.Module):
    """Language encoder model which is shared across all tasks."""

    def __init__(self):
        """Fetch Language model from huggingface."""
        super(BackboneLM, self).__init__()
        model = AutoModel.from_pretrained('roberta-base')
        self.backbone = model


class Model(nn.Module):
    """Torch-based module."""

    def __init__(self, weight_file,st_id,num_classes):
        """Inititialize model and create heads."""
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.language_model = BackboneLM()
        self.language_model.backbone.pooler = None
        self.st_id = st_id
        self.heads = nn.ModuleDict({st_id: HeadFactory(num_classes)})
        self.language_model.backbone.resize_token_embeddings(50268)
        self.load_weights(weight_file=weight_file)
        self.to(self.device)
        self.eval()

    def load_weights(self, weight_file):
        """Load pre-trained weights. The weight-dict architecture must match architecture in this module (strict=True)."""
        pretrained_weights = torch.load(weight_file, map_location=self.device)
        weight_dict = {k: value for k, value in pretrained_weights.items() if "loss" not in k}
        self.load_state_dict(weight_dict, strict=True)

    def forward(self, input_ids, attention_mask):
        """Pass the data through the model and according head decided from heads dict."""
        x_latent = self.language_model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.heads[self.st_id](x_latent)


class ModelInference:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = Model("./model_files/babe_mtl_en.pth",'10001',num_classes=2)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def classify_sentence(self,sent:str):
        toksentence = self.tokenizer(sent,truncation=True,return_tensors="pt")
        self.model.eval()
        with torch.no_grad():
            toksentence.to(self.device)
            output = self.model(**toksentence)
        # return output
        logits = F.softmax(output,dim=1)
        return {'unbiased': logits[0][0].item(),'biased':logits[0][1].item()}
    
    def classify_body(self,body):
        sents = sent_tokenize(body)
        avg = list(map(lambda x: self.classify_sentence(x),sents))
        return avg


print("Loading the model...")
mi = ModelInference()
while True:
    print("Enter a sentence:\n")
    sent = input()
    if not sent:
        break
    print(mi.classify_sentence(sent))