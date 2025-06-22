import torch
from transformers import AutoTokenizer
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

tokenizer_path = "/scratch/jts75596/llama/models/1B_model/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

file_path = "/home/jts75596/comp_ling/babylm_data/train_100M/processed/*.txt"
file_list = glob.glob(file_path)
print(file_list)

all_sentences = []
for file_path in tqdm(file_list, total=len(file_list)):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]  # skip empty lines
        
        all_sentences.extend(sentences)

all_sentences = random.sample(all_sentences, 2000000) # just do a subset, running out of time and memory to train on all sents
        
train, val = train_test_split(all_sentences, test_size=0.2, random_state=42)
        
max_seq_len = 512
    
print("Making training data")
tokenized_train = tokenizer(
    train,
    padding=False,
    truncation=True,
    max_length=max_seq_len,
    return_tensors=None  # We want Python lists, not PyTorch tensors yet
)

print("Done making training data")
print("Making validation data")
tokenized_val = tokenizer(
    val,
    padding=False,
    truncation=True,
    max_length=max_seq_len,
    return_tensors=None  # We want Python lists, not PyTorch tensors yet
)
print("Done making validation data")

# tokenized is a dict with 'input_ids' and possibly 'attention_mask'
train_input_ids = tokenized_train['input_ids']
val_input_ids = tokenized_val['input_ids']

print("Saving datasets!")
torch.save(train_input_ids, "/scratch/jts75596/llama/train_data.pt")
torch.save(val_input_ids, "/scratch/jts75596/llama/val_data.pt")
