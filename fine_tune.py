import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama

import sys
sys.path.insert(0, '/home/jts75596/mlsys/LRP-eXplains-Transformers')
from prune_llama import patched_forward
from llama_data import LlamaDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import argparse
import time
import os
import numpy as np

parser = argparse.ArgumentParser(description='Multi-GPU language model training')

parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--trainfile", type=str, help="path to train data file")
parser.add_argument("--valfile", type=str, help="path to validation data file")
parser.add_argument("--model_path", type=str, help="Path to pruned model you want to fine tune.")
parser.add_argument("--model_type", type=str, default="1B", help="Options are 1B or 8B")
parser.add_argument("--tokenizer_path", type=str, help="Path to model tokenizer")
parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate for the model's optimizer")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of epochs to train for")
parser.add_argument("--min_epochs", default=25, type=int, help="Minimum number of epochs to train for before tracking # of bad epochs")
parser.add_argument("--break_value", default=15, type=int, help="Break if this many epochs returned a worse Val PPL")
parser.add_argument("--distill", default=0, type=int, help="Whether or not you want to train this model using knowledge distillation")
parser.add_argument("--teacher_model", default=None, type=str, help="The path to the trained model for knowledge distillation")
parser.add_argument("--temperature", default=2.0, type=float, help="If doing KD, scale the output logits by this temperature variable")
parser.add_argument("--alpha", default=0.5, type=float, help="Weight applied to combined loss during KD. alpha*NLLLoss + (1-alpha)*KLDiv")
parser.add_argument("--savepath", default=None, type=str, required=True, help="Where to save the models to")


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        snapshot_path: str,
        distill: int,
        min_epochs: int,
        break_value: int,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.distill = distill
        if self.distill == 1:
            self.temp = args.temperature
            self.alpha = args.alpha
            print(f"Rank {self.gpu_id}: Loading teacher model from {args.teacher_model}")
            teacher_model = modeling_llama.LlamaForCausalLM.from_pretrained(args.teacher_model,
                                                            device_map=self.device, 
                                                            torch_dtype=torch.bfloat16)
            self.teacher_model = DDP(teacher_model, device_ids=[self.gpu_id])
            self.criterion_kl = nn.KLDivLoss(reduction='batchmean') # If batch size = 1, reduction='batchmean' is same as 'sum'
            
        if os.path.exists(f"/scratch/jts75596/llama/models/1B_model/finetuned_models/{snapshot_path}"):
            print(f"Rank {self.gpu_id} loading snapshot")
            self._load_snapshot(f"/scratch/jts75596/llama/models/1B_model/finetuned_models/{snapshot_path}")

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.criterion = nn.NLLLoss(reduction='sum')
        self.min_epochs = min_epochs
        self.break_value = break_value

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model = snapshot["MODEL"].to(self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, sents):
        self.optimizer.zero_grad()

        sents = sents.to(self.device)
        labels = sents[:, 1:]
        input_ids = sents[:, :-1]
        batch, seq_length = input_ids.shape
        num_tokens = torch.tensor(batch*seq_length, device=self.device)
        
        logits = self.model(input_ids=input_ids, use_cache=False).logits
        log_probs_word = F.log_softmax(logits, dim=-1)
        log_probs_word = log_probs_word.view(batch*seq_length, -1)
        labels = labels.reshape(-1)
        nll_loss = self.criterion(log_probs_word, labels)
        
        if self.distill == 1:
            # Don't need gradients for the teacher model
            with torch.no_grad():
                teacher_logits = self.teacher_model(input_ids=input_ids, use_cache=False).logits
            teacher_probs = F.softmax(teacher_logits / self.temp, dim=-1)

            temp_scaled_log_probs_word = F.log_softmax(logits / self.temp, dim=-1) # From student (pruned) model
            kl_loss = self.criterion_kl(temp_scaled_log_probs_word, teacher_probs)
            combined_loss = (self.alpha*nll_loss) + ((1-self.alpha)*kl_loss)
            combined_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            return nll_loss, kl_loss, num_tokens
        else:
            nll_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            return nll_loss, num_tokens


    def _run_epoch(self, epoch):
        b_sz = next(iter(self.train_data)).size(0)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        acc_train_nll_loss = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0, device=self.device)
        
        if self.distill == 1:
            acc_kl_div_loss = torch.tensor(0.0, device=self.device)
        
        for sents in self.train_data:
            try:
                train_loss = self._run_batch(sents)
                if self.distill == 1:
                    train_nll_loss, train_kl_div_loss, num_tokens = train_loss
                    acc_kl_div_loss += train_kl_div_loss
                else:
                    train_nll_loss, num_tokens = train_loss
                
                acc_train_nll_loss += train_nll_loss
                total_tokens += num_tokens
                del train_nll_loss, num_tokens
                torch.cuda.empty_cache()
            except Exception as e:
                print(repr(e))
        dist.all_reduce(acc_train_nll_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        if self.distill == 1:
            dist.all_reduce(acc_kl_div_loss, op=dist.ReduceOp.SUM)
            avg_kl_div_loss = acc_kl_div_loss / total_tokens
        
        avg_train_nll_loss = acc_train_nll_loss / total_tokens
        train_ppl = torch.exp(avg_train_nll_loss)
        
        if dist.get_rank() == 0:
            if self.distill == 1:
                return avg_train_nll_loss.item(), train_ppl.item(), avg_kl_div_loss.item()
            else:
                return avg_train_nll_loss.item(), train_ppl.item()
        else:
            return None
            
    def _run_eval(self):
        self.model.eval()
        print(f"Rank {self.gpu_id} running eval!")
        val_nll_loss = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0, device=self.device)
        with torch.no_grad():
            for sents in self.val_data:

                sents = sents.to(self.device)

                labels = sents[:, 1:]
                input_ids = sents[:, :-1]
                batch, seq_length = input_ids.shape
                
                logits = self.model(input_ids=input_ids, use_cache=False).logits
                log_probs_word = F.log_softmax(logits, dim=-1)
                
                log_probs_word = log_probs_word.view(batch*seq_length, -1)
                labels = labels.reshape(-1)
                nll_loss = self.criterion(log_probs_word, labels)
                val_nll_loss += nll_loss
                n_tokens = torch.tensor(batch*seq_length, device=self.device)
                total_tokens += n_tokens
                del sents, labels, input_ids, logits, nll_loss, n_tokens, log_probs_word
                torch.cuda.empty_cache()
            
            print(f"Rank {self.gpu_id} done with eval")
            dist.barrier()
            dist.all_reduce(val_nll_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            avg_val_nll_loss = val_nll_loss / total_tokens
            val_ppl = torch.exp(avg_val_nll_loss)

        if dist.get_rank() == 0:
            return avg_val_nll_loss.item(), val_ppl.item()
        else:
            return None, None
        
        
    def _save_snapshot(self, epoch, args):
        snapshot = {
            "ARGS": args.__dict__,
            "MODEL": self.model.module,
            "EPOCHS_RUN": epoch,
        }
        save_path = f"/scratch/jts75596/llama/models/{args.model_type}_model/finetuned_models/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = save_path + self.snapshot_path
        torch.save(snapshot, save_file)
        print(f"Epoch {epoch} | Training snapshot saved at {save_file}")

        
    def _log_progress(self, log_string=None, file_name=None):
        # Ensure the directory exists
        directory = "../../../home/jts75596/mlsys/LRP-eXplains-Transformers/progress_outputs/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{file_name}_progress_log.txt")

        with open(file_path, "a") as f:
            f.write(log_string + "\n")
        
        
    def train(self, max_epochs: int, args):
        print("Running initial eval to get baseline metrics")
        initial_val_nll_loss, init_val_ppl = self._run_eval()
        best_loss = initial_val_nll_loss
        best_ppl = init_val_ppl
        
        print(f"Initial Val-NLL Loss: {initial_val_nll_loss}, Initial Val-PPL: {init_val_ppl}\n")
        num_bad_epochs = 0
        print("Training!\n")

        for epoch in range(self.epochs_run, max_epochs):
            dist.barrier()
            self.model.train()
            epoch_results = self._run_epoch(epoch)
            if dist.get_rank() == 0:
                if self.distill == 1:
                    avg_train_nll_loss, train_ppl, avg_kl_div_loss = epoch_results
                else:
                    avg_train_nll_loss, train_ppl = epoch_results
            
            try:
                avg_val_nll_loss, val_ppl = self._run_eval()
            except Exception as e:
                print(repr(e))
            current_lr = self.scheduler.get_last_lr()[0]
            if self.gpu_id == 0:
                log_string = f"Epoch: {epoch+1}/{max_epochs}, LR: {current_lr}, T-Loss: {avg_train_nll_loss:.2f}, T-PPL: {train_ppl:.2f}, V-Loss: {avg_val_nll_loss:.2f}, V-PPL: {val_ppl:.2f}"
                save_path = self.snapshot_path.split('.')[0]
                self._log_progress(log_string=log_string, file_name=save_path)
           
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    self._save_snapshot(epoch, args)
                else:
                    if epoch > self.min_epochs:
                        num_bad_epochs += 1
                    if num_bad_epochs == self.break_value:
                        print(f"Breaking after {epoch} epochs because PPL failed to improve")
                        break
        dist.barrier()


def lr_lambda(current_step: int, warmup_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

                
def make_scheduler(optimizer, args, epoch_length):
    warmup_steps = int(epoch_length*0.05)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch_length*15, T_mult=2, eta_min=1e-9)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                schedulers=[warmup_scheduler, cosine_scheduler],
                                                milestones=[warmup_steps])
    return scheduler


def load_train_objs(args):
    train_data = LlamaDataset(args.trainfile)  # load your dataset
    num_batches = 100000 #len(train_data)
    val_data = LlamaDataset(args.valfile)
    print("Loaded training and validation data")
    
    model = torch.load(args.model_path, map_location=torch.device("cuda"), weights_only=False)
    model.config._attn_implementation = "eager"
    model.config._attn_implementation_autoset = False
    modeling_llama.eager_attention_forward = patched_forward
    
    # Only track grad for attn params
    for name, param in model.named_parameters():
        if "attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    
    attn_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(attn_params, lr=args.lr)
    scheduler = make_scheduler(optimizer, args, num_batches)
    return train_data, val_data, model, optimizer, scheduler

def collate_tokenized_batch(batch, pad_token_id=128001):
    input_ids = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    return input_ids

def prepare_dataloader(dataset: Dataset, batch_size: int=16):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        collate_fn=collate_tokenized_batch,
        num_workers=0,
        sampler=DistributedSampler(dataset)
    )

def main(args, total_epochs: int):
    ddp_setup()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Loading training objects!")
    train_dataset, val_dataset, model, optimizer, scheduler = load_train_objs(args)
    
    print("Making DataLoaders!")
    train_data = prepare_dataloader(train_dataset, batch_size=16)
    val_data = prepare_dataloader(val_dataset, batch_size=16)
    
    snapshot_path = args.savepath + ".pt"
    
    print("Making Trainer object...")

    trainer = Trainer(model, train_data, val_data, optimizer, scheduler, snapshot_path, args.distill, args.min_epochs, args.break_value)
    trainer.train(total_epochs, args)
    destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()

    start_time = time.perf_counter()
    main(args, args.num_epochs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total training time (min:sec): {minutes}:{seconds}")