import os
import re
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from transformers import BertTokenizer

#device = torch.device("cpu")
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#PATH_SAVE_MODEL_1 = "Twitter_model1.pth"

#max_len       = 300          # limitaciones porque google collab tiene un ram limitado
#batch_size    = 16           # Paquetes de 16 elementos
#nclases       = 2            # Comentarios positivos y negativos
#num_epochs    = 2
#learning_rate = 5e-5 


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False, nclases=2):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, nclases

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-cased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits