import torch
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import argparse
import os
import glob
import pickle

from lxt.efficient import monkey_patch
monkey_patch(modeling_llama, verbose=True)

parser = argparse.ArgumentParser()

parser.add_argument("--output_maps_dir", type=str, default="output_maps", help="Directory name for storing relevance heatmaps.")
parser.add_argument("--save_map", type=int, default=1, help="1 if you want to save the final heatmap, 0 otherwise")
parser.add_argument("--corpus", type=str, help="Path to list of .txt file of sentences to use as data to collect relevances")
parser.add_argument("--save_filename", type=str, help="Filename you want to save the output map as, along with the resulting tensor.")
parser.add_argument("--model", type=str, default="1B", choices=['1B', '8B'], help="Which Llama model to use. Choices are '1B' and '8B'")
parser.add_argument("--save_model", type=int, default=0, help="1 if you want to save a copy of the LLM locally, 0 otherwise.")
parser.add_argument("--test_subset", type=int, default=0, help="1 if you want to test on a subset of sentences, 0 if full data")
    
    
def load_model_and_tokenizer(model=None, load_local=False):
    """
    This funciton loads the model and tokenizer. Model can be from the transformers library, or one you saved
    locally, perhaps after pruning it.
    
    Args:
        model (str): Can be either '1B' or '8B' for the respective Llama model
        load_local (bool): If False, then load the model from huggingface transformers. If True,
            then load the model from a location on your device.
            
    Returns:
        model: the torch.nn.module that is a version of a Llama LLM.
        tokenizer: the pretrained tokenizer for the respective model.
    """
    if model == '1B':
        model_path = 'meta-llama/Llama-3.2-1B-Instruct'
        tokenizer_path = model_path
    elif model == '8B':
        model_path = 'meta-llama/Llama-3.1-8B'
        tokenizer_path = model_path
        
    if load_local:
        model_path = f"../../../../scratch/jts75596/llama/models/{model}_model/model"
        tokenizer_path = f"../../../../scratch/jts75596/llama/models/{model}_model/tokenizer"
        
    model = modeling_llama.LlamaForCausalLM.from_pretrained(model_path,
                                                            device_map='cuda', 
                                                            torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    

    print(f"Model Device: {model.device}")

    # Force the eager attention implementation:
    model.config._attn_implementation = "eager"
    model.config._attn_implementation_autoset = False

    # optional gradient checkpointing to save memory (2x forward pass)
    model.train()
    return model, tokenizer

original_eager_attention_forward = modeling_llama.eager_attention_forward

def hooked_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """
    This function just overwrites the attention forward method inside the model. This is so we can 
    hook and store the gradients of the attn_weights. These grads and weights will be stored to compute
    relevance propagation later.
    """
    # Compute attention outputs as usual
    attn_output, attn_weights = original_eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
    )
    
    # Save forward activations: post-softmax attention weights
    module.saved_attn_weights = attn_weights.detach().clone()
    
    # Register a hook to capture the gradients of attn_weights during backpropagation
    def save_grad(grad):
        module.attn_weights_grad = grad.detach().clone()

    attn_weights.register_hook(save_grad)
    
    return attn_output, attn_weights

modeling_llama.eager_attention_forward = hooked_eager_attention_forward


def masked_mean_aggregate(relevance_trace: torch.Tensor=None, device=None):
    """
    This function aggregates the relevance scores of an attention matrix.
    Since autoregressive modeling applies an upper right triangular mask, we don't want to
    average the keys based on the total sequence length, because a token halfway through
    the sequence has only seen 1/2 sequence length queries via attention. So average is based
    on the number of tokens they attended to.
    """
    B, H, Q, K = relevance_trace.shape
    
    mask = torch.tril(torch.ones(Q,K))
    relevance_sum = relevance_trace.sum(dim=-1) # shape: [1, 1, Q]
    
    valid_counts = mask.sum(dim=-1)  # shape: [Q]
    valid_counts = valid_counts.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, Q]
    valid_counts = valid_counts.to(device)

    masked_mean = relevance_sum / valid_counts  # could add epsilon to avoid division by zero, but probably not necessary

    return masked_mean


def get_attn_relevance(model: torch.nn.Module):
    """
    This function gets called after processing a sequence and calling backward() on the logits tensor.
    It iterates over each layer in the model to grab the stored attention weights and gradients.
    These were stored using the hook function inside the forward method that we overwrote the original forward method with.
    """
    relevance_trace = []
    for i, layer in enumerate(model.model.layers):
        
        output = layer.self_attn.saved_attn_weights # [B, H, S, S], where S == sequence length
        grad = layer.self_attn.attn_weights_grad # [B, H, S, S]
        
        relevance_heads = (output * grad).float()  
        relevance_heads = masked_mean_aggregate(relevance_trace=relevance_heads, device=model.device) 
        # now shape: [batch, num_heads, query_tokens]
        
        relevance_trace.append(relevance_heads.detach().cpu())
        
    relevance_trace = torch.stack(relevance_trace, dim=0)  # shape: [layers, batch, num_heads, query_tokens]
    return relevance_trace


def display_relevance(relevance_trace: torch.Tensor = None,
                      save_path: str=None,
                      filename: str=None,
                      agg_over_sequence=False):
    """
    This function stores the attention relevance as a heatmap.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if agg_over_sequence:
        num_tokens = relevance_trace.size(-1)
        for i in range(num_tokens):
            tok_relevance_trace = relevance_trace[:, :, i].detach().cpu().numpy()

            # Create a heatmap: x-axis is layers, y-axis is attention heads.
            plt.figure(figsize=(8, 6))
            plt.imshow(tok_relevance_trace.T, aspect='auto', cmap='seismic')#, vmin=-1, vmax=1)
            plt.colorbar(label='Aggregated Relevance')
            plt.xlabel("Layer")
            plt.ylabel("Attention Head")
            plt.title("Aggregated Relevance per Attention Head per Layer")
            plt.savefig(f"{save_path}/{i}.png", dpi=150, bbox_inches='tight')
    else:
        vmin_val = relevance_trace.min().item()
        vmax_val = relevance_trace.max().item()
        
        if vmin_val < 0 and vmax_val > 0:
            # Create, say, 3 ticks for the negative side (including vmin and 0)
            neg_ticks = np.linspace(vmin_val, 0, num=3, endpoint=True)
            # Create 3 ticks for the positive side (including 0 and vmax)
            pos_ticks = np.linspace(0, vmax_val, num=3, endpoint=True)
            # Concatenate, but remove the duplicate 0 from pos_ticks
            ticks = np.concatenate((neg_ticks, pos_ticks[1:]))
        else:
            ticks = np.linspace(vmin_val, vmax_val, num=7)
        norm = colors.TwoSlopeNorm(vmin=vmin_val, 
                           vcenter=0, 
                           vmax=vmax_val)
        # Create a heatmap: x-axis is layers, y-axis is attention heads.
        plt.figure(figsize=(8, 6))
        plt.imshow(relevance_trace.T, aspect='auto', cmap='seismic', norm=norm)
        

        cb = plt.colorbar(label='Aggregated Relevance Over Entire Corpus', ticks=ticks)
        cb.ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])

        plt.xlabel("Layer")
        plt.ylabel("Attention Head")
        plt.title("Aggregated Relevance per Attention Head per Layer")
        plt.savefig(f"{save_path}/{filename}.png", dpi=150, bbox_inches='tight')


def process_sentence(sentence, model, tokenizer, num_layers):
    """
    This function processes a single sentence by passing in all of the possible sub-sequences of that sentence.
    This way, the model has to predict the next token at every position in the sentence, so we can call backward()
    for each of those predictions and accumulate an average relevance for each attention head.
    """
    input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)
    seq_length = input_ids.size(1)
    
    sentence_accum = torch.zeros(num_layers, 32, device="cpu")
    count = 0
    for i in range(1, seq_length+1):
        sub_input = input_embeds[:, :i, :].requires_grad_()
        output_logits = model(inputs_embeds=sub_input, use_cache=False).logits
        max_logits, _ = torch.max(output_logits[:, -1, :], dim=-1)
        max_logits.backward(max_logits)

        attn_relevance = get_attn_relevance(model=model) # shape: [layers, 1, num_heads, query_tokens] (assuming batch=1)

        attn_relevance = attn_relevance.squeeze(1).mean(dim=-1) # shape: [layers, num_heads]

        sentence_accum += attn_relevance
        count += 1
        model.zero_grad()
        del output_logits, max_logits, attn_relevance, sub_input
        torch.cuda.empty_cache()
    return sentence_accum, count
        
def collect_relevances(model: torch.nn.Module=None, tokenizer=None, files: list[str]=None, test_subset: int=0):
    """
    This function iterates over all .txt files containing sentences to process. This will aggregate a relevance
    score for each attention head in each layer across all sequences.
    """
    num_layers = model.config.num_hidden_layers
    accumulator = torch.zeros(num_layers, 32, device="cpu")
    count = 0
    
    for file in files:
        with open(file, "r") as f:
            if test_subset == 1:
                lines = f.readlines()[:20]
            else:
                lines = f.readlines()
        
        for j, line in enumerate(lines):
            print(f"{j}.")
            try:
                sent_results, sent_count = process_sentence(line, model, tokenizer, num_layers)
                accumulator += sent_results
                count += sent_count
            except Exception as e:
                print(e)
                continue
    average_relevance = accumulator / count
    print(f"Total number of sequences: {count}")
    return average_relevance
    

def main(args):
    model, tokenizer = load_model_and_tokenizer(model=args.model, load_local=True)
    
    if args.save_model == 1:
        model.save_pretrained(f"../../../../scratch/jts75596/llama/models/{args.model}_model/model")
        tokenizer.save_pretrained(f"../../../../scratch/jts75596/llama/models/{args.model}_model/tokenizer")
    
    print(model.config._attn_implementation)         # Should print "eager"
    print(model.config._attn_implementation_autoset)   # Should print False
    
    for param in model.parameters():
        param.requires_grad = False    
    
    corpus = args.corpus
    file_pattern = f"{corpus}/*.txt"
    files = glob.glob(file_pattern, recursive=True)
    filename = args.save_filename
    
    avg_corpus_relevances = collect_relevances(model=model, tokenizer=tokenizer, files=files, test_subset=args.test_subset)
    print(f"Avg corpus relevances shape: {avg_corpus_relevances.shape}")
    
    output_data_dir = "saved_averages"
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
        
    with open(f"{output_data_dir}/{filename}.pkl", "wb") as f:
        pickle.dump(avg_corpus_relevances, f)
    
    if args.save_map == 1:
        display_relevance(relevance_trace=avg_corpus_relevances, save_path=args.output_maps_dir, filename=filename)
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
