import os
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import jieba as jb
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
model_path = './results/weight/pytorch_model.bin'
import torch
state_dict = torch.load(model_path, map_location="cpu")
torch.save(state_dict, model_path, _use_new_zipfile_serialization=False)