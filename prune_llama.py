import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaAttention

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from thop import profile, clever_format

from lxt.efficient import monkey_patch
monkey_patch(modeling_llama, verbose=True)

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="1B", choices=['1B', '8B'], help="Which Llama model to use. Choices are '1B' and '8B'")
parser.add_argument("--head_dim", type=int, default=64, help="Dimension for attention heads. 64 is 1B model, 128 is 8B model.")
parser.add_argument("--save_name", type=str, help="Save name for pruning dict and model")
parser.add_argument("--save_model", type=int, default=1, help="1 if you want to save a copy of the LLM and pruning dictionary locally, 0 otherwise.")

class PruneAttentionHeads(prune.BasePruningMethod):
    """
    Class for pruning entire attention heads for a transformer model.
    """
    PRUNING_TYPE = "structured"

    def __init__(self, heads_to_prune, projection_type, head_dim, num_groups):
        """
        Args:
            heads_to_prune (set or list): Indices of q heads (0 to 31) to prune.
            projection_type (str): One of 'q', 'k', or 'v' indicating which projection
                                   this pruning is being applied to.
        """
        super().__init__()
        self.heads_to_prune = set(heads_to_prune)
        self.projection_type = projection_type
        self.head_dim = head_dim
        self.num_groups = num_groups

    def compute_mask(self, t, default_mask):
        out_feat, in_feat = t.shape
        # print(t.shape)
        if default_mask is None:
            default_mask = torch.ones_like(t)

        if self.projection_type == 'q':# and t.shape[0] == 2048:
            # q_proj: reshape mask to [32, 64, in_features]
            mask = default_mask.view(32, self.head_dim, -1)
            # For each head in the list of heads to prune, zero out its entire slice.
            for head in self.heads_to_prune:
                if 0 <= head < 32:
                    mask[head, :, :] = 0
            # Flatten back to original shape
            return mask.view(out_feat, -1)
        elif self.projection_type in ['k', 'v']:# and t.shape[0] == 512:
            # k_proj/v_proj: reshape mask to [8, 64, in_features]
            mask = default_mask.view(self.num_groups, self.head_dim, -1)
            # Every 4 q heads share a single k/v projection.
            # Create groups: group 0 -> q heads [0,1,2,3], group 1 -> [4,5,6,7], etc.
            for group in range(8):
                group_q_heads = set(range(group * 4, group * 4 + 4))
                # If all q heads in this group are marked for pruning, zero out the entire group.
                if group_q_heads.issubset(self.heads_to_prune):
                    mask[group, :, :] = 0
            return mask.view(out_feat, -1)
        else:
            return default_mask

def extract_kept_weight_rows_q(linear_layer, kept_heads, head_dim=64):
    # Get the original weight matrix; shape should be [out_features, in_features]
    original_weight = linear_layer.weight.data
    new_weight_rows = []
    for head in kept_heads:
        start = head * head_dim
        end = (head + 1) * head_dim
        new_weight_rows.append(original_weight[start:end, :])
    # Concatenate along the 0th dimension to get the new weight matrix.
    new_weight = torch.cat(new_weight_rows, dim=0)
    return new_weight


def extract_kept_weight_rows_kv(linear_layer, kept_groups, head_dim=64):
    # Get the original weight matrix; shape should be [out_features, in_features]
    original_weight = linear_layer.weight.data
    new_weight_rows = []
    for group in kept_groups:
        start = group * head_dim
        end = (group + 1) * head_dim
        new_weight_rows.append(original_weight[start:end, :])
    # Concatenate along the 0th dimension to get the new weight matrix.
    new_weight = torch.cat(new_weight_rows, dim=0)
    return new_weight
        
    
def extract_kept_weight_cols_o(linear_layer, kept_heads, head_dim=64):
    original_weight = linear_layer.weight.data
    new_weight_cols = []
    for head in kept_heads:
        start = head * head_dim
        end = (head + 1) * head_dim
        new_weight_cols.append(original_weight[:, start:end])
    new_weight = torch.cat(new_weight_cols, dim=1)
    return new_weight
    
    
def rebuild_linear_layer(linear_layer, new_weight):
    in_features = linear_layer.in_features
    new_out_features = new_weight.shape[0]
    new_layer = nn.Linear(in_features, new_out_features, bias=False)
    # Copy over the new weight matrix.
    new_layer.weight.data = new_weight.clone()
    return new_layer


def rebuild_o_proj_layer(linear_layer, new_weight):
    out_features = linear_layer.out_features
    new_in_features = new_weight.shape[1]
    new_layer = nn.Linear(new_in_features, out_features, bias=False)
    new_layer.weight.data = new_weight.clone()
    return new_layer


def prune_model_heads(model, pruning_dict: dict, num_heads=32, num_groups=8, head_dim=64):
    for i, layer in enumerate(model.model.layers):
        print(f"Layer: {i}")
        attn = layer.self_attn
        # Get the heads to prune for this layer, defaulting to an empty set if not specified.
        heads_to_prune = pruning_dict.get(i, set())
        kept_heads = sorted(set(range(num_heads)) - heads_to_prune)

        heads_per_group = []
        for i in range(num_groups):
            group_range = range(i*4, (i+1)*4)
            count = sum(1 for head in group_range if head in kept_heads)
            if count > 0:
                heads_per_group.append(count)

        attn.heads_per_group = heads_per_group

        kept_groups = set()
        for i in range(num_groups):
            group_head_subset = set(range(i * 4, i * 4 + 4))
            if group_head_subset.issubset(heads_to_prune):
                continue
            else:
                kept_groups.add(i)

        attn.effective_heads = len(kept_heads)

        if len(heads_to_prune) == 0:
            continue

        for proj, proj_type in [(attn.q_proj, 'q'), (attn.k_proj, 'k'), (attn.v_proj, 'v')]:
            PruneAttentionHeads.apply(proj, "weight", heads_to_prune=heads_to_prune, projection_type=proj_type, head_dim=head_dim, num_groups=num_groups)
            prune.remove(proj, "weight")
            if proj_type == 'q':
                new_weights = extract_kept_weight_rows_q(proj, kept_heads=kept_heads, head_dim=head_dim)
                new_layer = rebuild_linear_layer(proj, new_weight=new_weights)
                attn.q_proj = new_layer
            elif proj_type == 'k':
                new_weights = extract_kept_weight_rows_kv(proj, kept_groups=kept_groups, head_dim=head_dim)
                new_layer = rebuild_linear_layer(proj, new_weight=new_weights)
                attn.k_proj = new_layer
            elif proj_type == 'v':
                new_weights = extract_kept_weight_rows_kv(proj, kept_groups=kept_groups, head_dim=head_dim)
                new_layer = rebuild_linear_layer(proj, new_weight=new_weights)
                attn.v_proj = new_layer
        new_weights_o = extract_kept_weight_cols_o(attn.o_proj, kept_heads=kept_heads, head_dim=head_dim)
        new_layer = rebuild_o_proj_layer(attn.o_proj, new_weight=new_weights_o)
        attn.o_proj = new_layer
    return model


def custom_repeat_kv(hidden_states: torch.Tensor, repeat_factors: list) -> torch.Tensor:
    batch, num_groups, slen, head_dim = hidden_states.shape

    if len(repeat_factors) != num_groups:
        msg = f"Expected repeat_factors to have the same length as num_groups. Got {len(repeat_factors)}, needed {num_groups}."
        raise ValueError(msg)
    
    repeated_groups = []
    for group_idx, r in enumerate(repeat_factors):
        if r > 0:
            group_tensor = hidden_states[:, group_idx:group_idx+1, :, :] # shape: [batch, 1, seq_len, head_dim]
            expanded = group_tensor.expand(batch, r, slen, head_dim)
            repeated_groups.append(expanded)
        # If r == 0, skip that group entirely
        # Shouldn't actually be an issue, r == 0 means no heads for that group, so we wouldn't have added anything to the list
    return torch.cat(repeated_groups, dim=1)


def optimized_repeat_kv(hidden_states: torch.Tensor, repeat_factors: list) -> torch.Tensor:
    """
    hidden_states: [batch, num_groups, seqlen, head_dim]
    repeat_factors: length=num_groups, how many times to repeat each group
    returns: [batch, sum(repeat_factors), seqlen, head_dim]
    """
    repeats = torch.tensor(
        repeat_factors, 
        device=hidden_states.device, 
        dtype=torch.long
    )

    return hidden_states.repeat_interleave(repeats, dim=1)


def patched_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    # Use the effective_heads attribute if set; fallback to the config value otherwise.
    num_heads = getattr(module, "effective_heads", module.config.num_attention_heads)
    
    batch_size, current_heads, seq_len, head_dim = query.size()
    
#     print(f"Key shape: {key.shape}\tValue shape: {value.shape}")
    key = custom_repeat_kv(key, module.heads_per_group)
    value = custom_repeat_kv(value, module.heads_per_group)
#     key = optimized_repeat_kv(key, module.heads_per_group)
#     value = optimized_repeat_kv(value, module.heads_per_group)
#     print(f"Key shape: {key.shape}\tValue shape: {value.shape}")

    # Generate a causal mask of shape [seq_len, seq_len]
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=query.device))
    # Expand the mask to shape [batch_size, effective_heads, seq_len, seq_len]
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

        
def main(args):
    model_path = f"/scratch/jts75596/llama/models/{args.model}_model/model"
    tokenizer_path = f"/scratch/jts75596/llama/models/{args.model}_model/tokenizer"
    model = modeling_llama.LlamaForCausalLM.from_pretrained(model_path, 
                                                        device_map='cuda', 
                                                        torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Force the eager attention implementation:
    model.config._attn_implementation = "eager"
    model.config._attn_implementation_autoset = False
    
    prompt_response = f"George Washington was the first"
    input_ids = tokenizer(prompt_response, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)

    macs, params = profile(model, inputs=(input_ids, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"BEFORE pruning -- MACs: {macs}, Parameters: {params}")
    
    if args.model == "1B":
        layer_prune_dict = {
            0: {4, 5, 8, 23, 26},
            1: {2, 25, 31},
            2: {7, 9, 11, 17, 18, 21, 23},
            3: {6, 19, 20, 21, 23, 29},
            4: {5, 9, 11, 25, 26},
            5: {0, 2, 15, 19, 22, 26, 28},
            6: {2, 5, 6, 9, 11, 13, 14, 23, 28, 29, 30},
            7: {3, 5, 6, 7, 8, 9, 11, 12, 13, 19, 20, 21, 23, 25, 27, 28}, 
            8: {2, 4, 6, 8, 9, 13, 15, 16, 19, 20, 31},
            9: {1, 3, 6, 12, 22, 27},
            10: {0, 2, 4, 14, 17, 19, 22, 27},
            11: {0, 2, 4, 10, 11, 15, 23, 26, 29},
            12: {1, 4, 5, 13, 15, 20, 22},
            13: {3, 8, 10, 18, 19, 20, 21, 23, 25, 26, 29, 31},
            14: {8, 9, 16, 17, 20, 21, 22, 23},
            15: {0, 2, 14, 23, 25, 27}
        }
    elif args.model == "8B":
        layer_prune_dict = {
            0: {10, 19, 20, 22, 25, 29},
            1: {9, 25, 31},
            2: set(),
            3: {19},
            4: {5, 19, 26},
            5: {7, 16},
            6: {5, 15, 17, 18, 21, 24},
            7: {0, 3, 6, 8, 26},
            8: {5, 9, 23, 30},
            9: {0, 5, 10, 12, 17, 28},
            10: {0, 6, 10, 17},
            11: {3, 7, 11, 12, 13, 19, 23, 24, 25},
            12: {11, 12, 31},
            13: {0, 10, 13, 16, 17, 28},
            14: {9, 14, 16, 20},
            15: {6, 10},
            16: {15, 17, 30},
            17: {6, 12},
            18: set(),
            19: {11},
            20: set(),
            21: {3, 8},
            22: {18},
            23: set(),
            24: {29},
            25: {8, 20, 21},
            26: {0, 22},
            27: {11, 25, 30},
            28: {6, 11, 17, 24},
            29: {10, 17, 26},
            30: {8, 25},
            31: {2, 25, 31}
        }
        
    num_heads = 32
    num_groups = 8
    head_dim = args.head_dim
    
    pruned_model = prune_model_heads(model, layer_prune_dict, num_heads=num_heads, num_groups=num_groups, head_dim=head_dim)
    
    # Patch the forward method.
    modeling_llama.eager_attention_forward = patched_forward
    
    macs, params = profile(pruned_model, inputs=(input_ids, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"AFTER pruning -- MACs: {macs}, Parameters: {params}")
    
    if args.save_model == 1:
        dict_save_path = f"saved_pruning_dicts/{args.model}"
        if not os.path.exists(dict_save_path):
            os.makedirs(dict_save_path)
        
        dict_filename = f"{dict_save_path}/{args.save_name}.pkl"
        with open(dict_filename, "wb") as f:
            pickle.dump(layer_prune_dict, f)
        print(f"Pruning dictionary saved to {dict_filename}")

        model_save_path = f"/scratch/jts75596/llama/models/{args.model}_model/pruned_models"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            
        torch.save(pruned_model, f"{model_save_path}/{args.save_name}.pt")
        print(f"Pruned model saved to {model_save_path}")
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
