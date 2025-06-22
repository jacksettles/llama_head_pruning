import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse

from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama
import sys
sys.path.insert(0, '/home/jts75596/mlsys/LRP-eXplains-Transformers')
from prune_llama import patched_forward

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--model_path", default="models/1B_model/model", type=str, help="Path to where the model is stored")
parser.add_argument("--tokenizer_path", default="models/1B_model/tokenizer", type=str, help="Path to where tokenizer is stored")
parser.add_argument("--pruned_model_path", default="models/1B_model/pruned_models/targeted_blue_head_pruning.pt", type=str, help="Path to where the pruned model is stored")

def load_model_and_tokenizer(model_path=None, tok_path=None, pruned=False, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    
    if not pruned:
        model = modeling_llama.LlamaForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16)
        # Force the eager attention implementation for fair comparison with pruned model:
        model.config._attn_implementation = "eager"
        model.config._attn_implementation_autoset = False
        print(f"Original Llama model is using the {model.config._attn_implementation} attention implementation!\n")
    else:
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        # Force the eager attention implementation and then patch over it:
        model.config._attn_implementation = "eager"
        model.config._attn_implementation_autoset = False
        modeling_llama.eager_attention_forward = patched_forward
        print(f"Pruned Llama model is using the {model.config._attn_implementation} attention implementation!\n")
        
    model.eval()
    return model, tokenizer


def get_dummy_input(model, batch_size=1, seq_len=64):
    """
    This function is used to create a random input tensor to test inference speed.
    """
    vocab_size = model.config.vocab_size
    dummy_input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=model.device
    )
    return dummy_input_ids
    
    
def end_to_end_test(model, tokenizer, input_sentence=None, num_steps=100, warmup=True, device='cpu'):
    """
    This function is used for testing "end to end" speed for an LLM - end to end here is really just 
    initial tokenization and decoding one output token, so it is the same for single token generation.
    """
    if warmup:
        with torch.no_grad():
            for i in range(5):
                print(f"Warmup {i}.")
                input_ids = tokenizer(input_sentence, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
                logits = model(input_ids=input_ids, use_cache=False).logits
                _, max_indices = torch.max(logits[:, -1, :], dim=-1)
                next_token = tokenizer.convert_ids_to_tokens(max_indices)
    
    if device == 'cpu':
        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(num_steps):
                print(f"CPU e2e test {i}.")
                input_ids = tokenizer(input_sentence, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
                logits = model(input_ids=input_ids, use_cache=False).logits
                _, max_indices = torch.max(logits[:, -1, :], dim=-1)
                next_token = tokenizer.convert_ids_to_tokens(max_indices)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time # seconds
        total_ms = elapsed_time * 1000
        avg_latency = total_ms / num_steps # Average ms per inference
        return avg_latency
    elif device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            for i in range(num_steps):
                print(f"CUDA e2e test {i+1}.")
                input_ids = tokenizer(input_sentence, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
                logits = model(input_ids=input_ids, use_cache=False).logits
                _, max_indices = torch.max(logits[:, -1, :], dim=-1)
                next_token = tokenizer.convert_ids_to_tokens(max_indices)
        
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        avg_latency = total_ms / num_steps # Average ms per inference
        return avg_latency
    else:
        print("Please specify if these tests are on the 'cpu' or 'cuda'.")
        return


def inference_test(model, num_steps=100, warmup=True, device='cpu'):
    input_ids = get_dummy_input(model)
    print(f"Input shape: {input_ids.shape}")
        
    if warmup:
        with torch.no_grad():
            for i in range(10):
                print(f"Warmup inference {i+1}.")
                _ = model(input_ids=input_ids, use_cache=False)
    
    if device == 'cpu':
        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(num_steps):
                print(f"CPU inference test {i}.")
                _ = model(input_ids=input_ids, use_cache=False)
        end_time = time.perf_counter()
        print("Done with CPU inference test")
        
        elapsed_time = end_time - start_time # seconds
        total_ms = elapsed_time * 1000
        avg_latency = total_ms / num_steps # Average ms per inference
        return avg_latency
    elif device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            for i in range(num_steps):
                print(f"CUDA inference test {i+1}.")
                _ = model(input_ids=input_ids, use_cache=False)
        end.record()
        torch.cuda.synchronize()
        
        total_ms = start.elapsed_time(end)
        avg_latency = total_ms / num_steps # Average ms per inference
        return avg_latency
    else:
        print("Please specify if these tests are on the 'cpu' or 'cuda'.")
        return
    
    
def generate_tokens(model, tokenizer, input_sentence=None, max_new=64, use_cache=True, device='cpu'):
    input_ids = tokenizer(input_sentence, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    num_input_tokens = len(input_ids[0])
    
    if use_cache:
        generated = input_ids
        with torch.no_grad():
            outputs = model(input_ids=generated, use_cache=True)
            logits, past = outputs.logits, outputs.past_key_values
            next_token = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            for _ in range(max_new - 1): # minus 1 because we already generated 1 token above
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )
                logits, past = outputs.logits, outputs.past_key_values
                next_token = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
        return generated, num_input_tokens
    else:
        generated = input_ids
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(input_ids=generated, use_cache=False).logits
                next_token = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
        return generated, num_input_tokens

def throughput_test(model, tokenizer, input_sentence=None, max_new=64, use_cache=True, num_steps=100, warmup=True, device='cpu'):
    if warmup:
        for _ in range(5):
            _, _ = generate_tokens(model,
                                tokenizer,
                                input_sentence=input_sentence,
                                max_new=max_new,
                                use_cache=use_cache,
                                device=device)
    
    if device == 'cpu':
        start_time = time.perf_counter()
        for i in range(num_steps):
            print(f"CPU throughput test {i}.")
            _, num_input_tokens = generate_tokens(model,
                                tokenizer,
                                input_sentence=input_sentence,
                                max_new=max_new,
                                use_cache=use_cache,
                                device=device)
        
        end_time = time.perf_counter()
        print("Done with CPU throughput test.")
        elapsed_time = end_time - start_time # seconds
        total_tokens_generated = max_new * num_steps
        tok_per_sec = total_tokens_generated / elapsed_time
        return tok_per_sec, num_input_tokens
    elif device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for i in range(num_steps):
            print(f"CUDA throughput test {i+1}.")
            _, num_input_tokens = generate_tokens(model,
                                tokenizer,
                                input_sentence=input_sentence,
                                max_new=max_new,
                                use_cache=use_cache,
                                device=device)
        
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        total_s = total_ms / 1000
        total_tokens_generated = max_new * num_steps
        tok_per_sec = total_tokens_generated / total_s
        return tok_per_sec, num_input_tokens
    else:
        print("Please specify if these tests are on the 'cpu' or 'cuda'.")
        return
    

def run_tests(model, tokenizer, device=None):
    """
    This function runs all 10 tests on a given model.
    10 tests are:
        1. End to end latency test (tokenization, inference, and then next token decoding) - same as single token latency
        2. Inference latency - forward pass only
        3.-10. Throughput tests (tok/s) with a short and long input sentence, 64 and 256 output tokens, with and without cache.
    """
    short_sent = "George Washington was the"
    long_sent = "This is a much longer sentence than the first one because I am just going to fill this sentence up with grammatical constructions and other random words that makes this sentence much longer than, once again, the first sentence that I gave to the model, and I can do this because language is recursive where I can embed a sentence within a sentence iteratively like I just did with the 'and' conjunction before this, but I will now stop this sentence here."
    
    if device == 'cpu':
        num_steps = 5
        warmup = False
    else:
        num_steps = 100
        warmup = False
    
    results = {
        "end_to_end_latency_ms": end_to_end_test(model, tokenizer, input_sentence=short_sent, num_steps=num_steps,
                                                 warmup=True, device=device),
        "inference_latency_ms": inference_test(model, num_steps=num_steps, warmup=warmup, device=device),
        "throughput_short_64_cache": throughput_test(model, tokenizer, input_sentence=short_sent, max_new=64, use_cache=True,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
        "throughput_short_256_cache": throughput_test(model, tokenizer, input_sentence=short_sent, max_new=256, use_cache=True,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
        "throughput_long_64_cache": throughput_test(model, tokenizer, input_sentence=long_sent, max_new=64, use_cache=True,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
        "throughput_long_256_cache": throughput_test(model, tokenizer, input_sentence=long_sent, max_new=256, use_cache=True,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
        "throughput_short_64_no_cache": throughput_test(model, tokenizer, input_sentence=short_sent, max_new=64, use_cache=False,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
        "throughput_short_256_no_cache":throughput_test(model, tokenizer,input_sentence=short_sent, max_new=256, use_cache=False,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
        "throughput_long_64_no_cache": throughput_test(model, tokenizer, input_sentence=long_sent, max_new=64, use_cache=False,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
        "throughput_long_256_no_cache": throughput_test(model, tokenizer, input_sentence=long_sent, max_new=256, use_cache=False,
                                                          num_steps=num_steps,
                                                          warmup=warmup,
                                                          device=device)[0],
    }
    return results
    

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    og_model_path = args.model_path
    pruned_model_path = args.pruned_model_path
    tokenizer_path = args.tokenizer_path
    
#     device = 'cpu'
#     model, tokenizer = load_model_and_tokenizer(model_path=og_model_path, tok_path=tokenizer_path, pruned=False, device=device)
#     print(f"Testing original model on CPU!")
#     original_model_cpu = run_tests(model, tokenizer, device=device)
#     del model, tokenizer
    
    device = 'cuda'
    model, tokenizer = load_model_and_tokenizer(model_path=og_model_path, tok_path=tokenizer_path, pruned=False, device=device)
    print(f"Testing original model on GPU!")
    original_model_cuda = run_tests(model, tokenizer, device=device)
    del model, tokenizer
    
#     device = 'cpu'
#     model, tokenizer = load_model_and_tokenizer(model_path=pruned_model_path,
#                                                 tok_path=tokenizer_path, pruned=True, device=device)
#     print(f"Testing pruned model on CPU!")
#     pruned_model_cpu = run_tests(model, tokenizer, device=device)
#     del model, tokenizer
    
    device = 'cuda'
    model, tokenizer = load_model_and_tokenizer(model_path=pruned_model_path,
                                                tok_path=tokenizer_path, pruned=True, device=device)
    print(f"Testing pruned model on GPU!")
    pruned_model_cuda = run_tests(model, tokenizer, device=device)
    del model, tokenizer
    
    data = {
        "test": list(original_model_cuda.keys()),
#         "Original Model-CPU": list(original_model_cpu.values()),
#         "Pruned Model-CPU": list(pruned_model_cpu.values()),
        "Original Model-GPU": list(original_model_cuda.values()),
        "Pruned Model-GPU": list(pruned_model_cuda.values())
    }
    df = pd.DataFrame(data)
    df.to_csv("./1B_optim_eager_attn_latency_tests.csv")
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
