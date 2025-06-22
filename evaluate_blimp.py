from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama
import sys
sys.path.insert(0, '/home/jts75596/mlsys/LRP-eXplains-Transformers')
from prune_llama import patched_forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import pandas as pd
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="Path to model to run the BLiMP eval on.")
parser.add_argument("--tokenizer", type=str, help="Path to tokenizer for that model.")
parser.add_argument("--seed", type=int, default=3455, help="Seed for setting randomness for reproducibility.")
parser.add_argument("--blimp_path", type=str, default="./blimp_data/*.jsonl", help="Path to BLiMP data.")
parser.add_argument("--save_filename", type=str, help="What to name the saved .csv file of scores")
parser.add_argument("--pruned_model", type=int, default=0, help="1 if the model to evaluate is a pruned model, 0 if original")

if torch.cuda.is_available():
    device = "cuda"
    print("Cuda is available. Using GPU.")
else:
    device = "cpu"
    print("Cuda is not available. Using CPU.")


def load_model_and_tokenizer(args):
    tok_path = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    
    model_path = args.model
    if args.pruned_model == 0:
        model = modeling_llama.LlamaForCausalLM.from_pretrained(model_path, 
                                                            device_map='cuda', 
                                                            torch_dtype=torch.bfloat16)
    elif args.pruned_model == 1:
        model = torch.load(model_path, map_location=torch.device("cuda"), weights_only=False)
        if isinstance(model, dict):
            model = model['MODEL']
        
        # Force the eager attention implementation:
        model.config._attn_implementation = "eager"
        model.config._attn_implementation_autoset = False
        print("Using patched forward method")
        modeling_llama.eager_attention_forward = patched_forward
        
    model.eval()
    return model, tokenizer

def get_probs(sentence, model, tokenizer):
    with torch.no_grad():
        input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        input_embeds = model.get_input_embeddings()(input_ids)
             
        # After getting embeddings for each token in the sentence, inputs is everything
        # except for the last embedding. The second to last embedding will be used
        # to predict the token-ID for the last token in the sequence
        inputs = input_embeds[:, :-1]
        labels = input_ids[:, 1:]

        logits = model(inputs_embeds=inputs, use_cache=False).logits
        log_probs_word = F.log_softmax(logits, dim=-1)

        # Add dimension to labels
        # Then gather lob probs in log_probs_word based on values in labels
        # Then ditch the last dimension and sum total log_probs
        gathered_log_probs = torch.gather(log_probs_word, 2, labels.unsqueeze(2)).squeeze(2).sum(1)
        return gathered_log_probs


def run_test_suite(model, files, tokenizer, seed=42):
    torch.manual_seed(seed)
    score_dict = {}
    i = 0
    for file in files:
        with open(file, "r") as f:
            data = [json.loads(line) for line in f]
        total_sents = len(data)
        total_correct = 0
        for test in data:
            good_sentence = test['sentence_good']
            bad_sentence = test['sentence_bad']

            good_probs = get_probs(good_sentence, model, tokenizer)
            bad_probs = get_probs(bad_sentence, model, tokenizer)

            if good_probs > bad_probs:
                total_correct += 1
        score = (total_correct / total_sents)*100
        score = round(score, 2)
        score_dict[file] = score
        i += 1
        print(f'{i}:\t{file}\t- Score: {score}')
    return score_dict


def main(args):
    model, tokenizer = load_model_and_tokenizer(args)
    print("Loaded model and tokenizer!")
    
    # Gather all the BLiMP test files
    pattern = args.blimp_path
    blimp_files = glob.glob(pattern, recursive=True)
    
    model_results = run_test_suite(model, blimp_files, tokenizer, seed=args.seed)
    
    res_df = pd.DataFrame([model_results], index=[0]).T
    
    save_dir = "blimp_results"
        
    res_df.to_csv(f"./{save_dir}/{args.save_filename}.csv", index=True)
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
