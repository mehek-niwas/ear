import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

from dataset import get_dataset_by_name
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from train_bert import compute_negative_entropy, LMForSequenceClassification
from collections import defaultdict
from typing import Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel
import ear

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Ensure inputs are tensors
item = tokenizer("Today it's a good day!", return_tensors="pt")
outputs = model(**item, output_attentions=True)

# Check if attentions exist (BERT base model might not output attentions unless explicitly specified)
if outputs.attentions is None:
    raise ValueError("The model did not output attentions. Make sure you're using a model that supports attention outputs.")

reg_strength = 0.01
neg_entropy, neg_entropies = ear.compute_negative_entropy(
    inputs=outputs.attentions,
    attention_mask=item["attention_mask"], return_values=True
)
reg_loss = reg_strength * neg_entropy

# BERT base model doesn’t return loss unless wrapped in a classification head — ensure you handle loss properly
loss = reg_loss  # No outputs.loss from base BERT; use your own loss calculation if needed

print("Negative Entropy:", neg_entropy)
for i, entropy in enumerate(neg_entropies):
    print(f"Sample {i} Negative Entropy Shape: {entropy.shape}")

print("This is an inference example so there is only 1 sample in the batch")
print("Shape is Layers x Nonpadded Tokens")
for i in range(len(neg_entropies)):
    print(f"Sample {i} Negative Entropy:", neg_entropies[i])
    
print("Regularization Loss:", reg_loss)
print("Total Loss:", loss)
