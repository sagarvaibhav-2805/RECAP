

import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "/home/scai/msr/aiy237528/flash/final-climb-adivaani/datasets/data_for_inference/Legal_Hindi_20_40_Words_Proper (1).csv"
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "predictions.csv")
WEIGHTS_DIR = "/home/scai/msr/aiy237528/flash/final-climb-adivaani/GARuD-Phase-2/code/translation/bert-frozen-enc-tx-dec-mlm-only/hi2bh/weights"
SRC_MODEL_PATH = "/home/scai/msr/aiy257590/flash/GRPO_RESEARCH/BERT-frozen-MLM_only/hindi-bert-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEAM = 4
ALPHA = 0.7
INFERENCE_BATCH_SIZE = 512

def setup_ddp():
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False

def cleanup_ddp(use_ddp):
    if use_ddp:
        dist.destroy_process_group()

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.head_dim = d_model // nhead
        self.nhead = nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, use_cache=False, cache=None):
        b, tq, _ = q.shape
        q = self.q(q).view(b, tq, self.nhead, self.head_dim).transpose(1, 2)
        if k is not None:
            k_proj = self.k(k).view(b, k.size(1), self.nhead, self.head_dim).transpose(1, 2)
            v_proj = self.v(v).view(b, v.size(1), self.nhead, self.head_dim).transpose(1, 2)
            if use_cache and cache is not None:
                k_proj = torch.cat([cache[0], k_proj], dim=2)
                v_proj = torch.cat([cache[1], v_proj], dim=2)
        else:
            k_proj, v_proj = cache
        
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v_proj).transpose(1, 2).contiguous().view(b, tq, -1)
        out = self.resid_drop(self.o(out))
        return (out, (k_proj, v_proj)) if use_cache else out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ffn, dropout):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.n2 = RMSNorm(d_model)
        self.n3 = RMSNorm(d_model)
        self.self_attn = MHA(d_model, nhead, dropout)
        self.cross_attn = MHA(d_model, nhead, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d_model), nn.Dropout(dropout))

    def forward(self, x, enc, tgt_mask, src_mask, use_cache=False, cache=None):
        nx = self.n1(x)
        self_c, cross_c = cache if cache is not None else (None, None)
        if use_cache:
            self_out, self_c = self.self_attn(nx, nx, nx, tgt_mask, use_cache=True, cache=self_c)
            x = x + self_out
            cx = self.n2(x)
            if cross_c is None:
                cross_out, cross_c = self.cross_attn(cx, enc, enc, src_mask, use_cache=True, cache=None)
            else:
                cross_out, cross_c = self.cross_attn(cx, None, None, src_mask, use_cache=True, cache=cross_c)
            x = x + cross_out
        else:
            x = x + self.self_attn(nx, nx, nx, tgt_mask)
            x = x + self.cross_attn(self.n2(x), enc, enc, src_mask)
        x = x + self.ff(self.n3(x))
        return (x, (self_c, cross_c)) if use_cache else x

class BertEncTxDec(nn.Module):
    def __init__(self, enc_model, tgt_vocab_size, src_pad_id, d_model, nhead, nlayers, ffn, dropout, max_len):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.encoder = enc_model
        self.src_proj = nn.Linear(self.encoder.config.hidden_size, d_model)
        self.tok = nn.Embedding(tgt_vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, ffn, dropout) for _ in range(nlayers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.max_len = max_len

    def encode(self, src_ids):
        with torch.no_grad():
            enc = self.encoder(input_ids=src_ids, attention_mask=(src_ids != self.src_pad_id).long()).last_hidden_state
        return self.src_proj(enc)

    def decode(self, tgt_ids, enc, src_mask, use_cache=False, cache=None):
        _, t = tgt_ids.shape
        offset = cache[0][0][0].size(2) if (use_cache and cache is not None and cache[0] is not None and cache[0][0] is not None) else 0
        p = torch.arange(offset, offset + t, device=tgt_ids.device).unsqueeze(0)
        x = self.drop(self.tok(tgt_ids) + self.pos(p))
        if use_cache:
            tgt_mask = None
            new_cache = []
            cache = cache if cache is not None else [None] * len(self.layers)
            for i, blk in enumerate(self.layers):
                x, c = blk(x, enc, tgt_mask, src_mask, use_cache=True, cache=cache[i])
                new_cache.append(c)
            return self.lm_head(self.norm(x)), new_cache
        else:
            tgt_mask = torch.tril(torch.ones((t, t), device=tgt_ids.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
            for blk in self.layers:
                x = blk(x, enc, tgt_mask, src_mask)
            return self.lm_head(self.norm(x))

    def forward(self, src_ids, tgt_ids):
        enc = self.encode(src_ids)
        src_mask = (src_ids != self.src_pad_id).unsqueeze(1).unsqueeze(2)
        return self.decode(tgt_ids, enc, src_mask)

@torch.no_grad()
def beam_generate(model, src_tok, tgt_tok, src_input, max_len, beam=1, alpha=0.7, device="cuda"):
    is_tensor = isinstance(src_input, torch.Tensor)
    if is_tensor:
        src = src_input
        if src.dim() == 1:
            src = src.unsqueeze(0)
        src = src.to(device)
        batch_size = src.size(0)
    else:
        if isinstance(src_input, str):
            src_input = [src_input]
        batch_size = len(src_input)
        tokenized = src_tok(src_input, add_special_tokens=True, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        src = tokenized["input_ids"].to(device)
        
    bos = tgt_tok.cls_token_id or tgt_tok.bos_token_id or tgt_tok.pad_token_id
    eos = tgt_tok.sep_token_id or tgt_tok.eos_token_id or tgt_tok.pad_token_id
    pad = tgt_tok.pad_token_id if tgt_tok.pad_token_id is not None else eos
    
    enc = model.encode(src)
    src_mask = (src != model.src_pad_id).unsqueeze(1).unsqueeze(2)
    
    seqs = torch.full((batch_size * beam, 1), bos, device=device, dtype=torch.long)
    scores = torch.full((batch_size, beam), -1e9, device=device)
    scores[:, 0] = 0.0
    finished = torch.zeros((batch_size, beam), dtype=torch.bool, device=device)
    
    enc = enc.unsqueeze(1).expand(-1, beam, -1, -1).reshape(batch_size * beam, -1, enc.size(-1))
    src_mask = src_mask.unsqueeze(1).expand(-1, beam, -1, -1, -1).reshape(batch_size * beam, 1, 1, -1)
    
    cache = None
    for step in range(max_len - 1):
        if finished.all():
            break
            
        cur = seqs[:, -1:] if cache is not None else seqs
        logits, cache = model.decode(cur, enc, src_mask, use_cache=True, cache=cache)
        
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        log_probs[:, pad] = -1e9
        
        flat_finished = finished.view(-1)
        log_probs[flat_finished] = -1e9
        log_probs[flat_finished, eos] = 0.0
        
        total_scores = scores.view(-1, 1) + log_probs
        total_scores = total_scores.view(batch_size, -1)
        
        k = min(beam, total_scores.size(1))
        top_scores, top_idx = torch.topk(total_scores, k, dim=-1)
        
        parent_beam = top_idx // log_probs.size(-1)
        token = top_idx % log_probs.size(-1)
        
        batch_offsets = torch.arange(0, batch_size, device=device).unsqueeze(1) * beam
        flat_parent_idx = (batch_offsets + parent_beam).view(-1)
        
        seqs = torch.cat([seqs[flat_parent_idx], token.view(-1, 1)], dim=1)
        scores = top_scores
        finished = finished.gather(1, parent_beam) | (token == eos)
        
        reordered_cache = []
        for layer_c in cache:
            self_c, cross_c = layer_c
            new_self_c = (self_c[0][flat_parent_idx], self_c[1][flat_parent_idx])
            new_cross_c = (cross_c[0][flat_parent_idx], cross_c[1][flat_parent_idx]) if cross_c is not None else None
            reordered_cache.append((new_self_c, new_cross_c))
        cache = reordered_cache
        
    lengths = ((seqs != eos) & (seqs != bos)).sum(dim=-1).view(batch_size, beam).clamp(min=1).float()
    best_beam_idx = (scores / (lengths ** alpha)).argmax(dim=-1)
    
    flat_best_idx = torch.arange(0, batch_size, device=device) * beam + best_beam_idx
    best_seqs = seqs[flat_best_idx]
    
    decoded_outputs = []
    for i in range(batch_size):
        tokens = best_seqs[i, 1:].tolist()
        if eos in tokens:
            tokens = tokens[:tokens.index(eos)]
        decoded_outputs.append(tgt_tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip())
        
    is_single_input = isinstance(src_input, str) or (is_tensor and (src_input.dim() == 1 or src_input.size(0) == 1))
    if is_single_input:
        return decoded_outputs[0]
    return decoded_outputs

class InferDataset(Dataset):
    def __init__(self, df, src_col):
        self.samples = [str(x).strip() for x in df[src_col].tolist() if str(x).strip()]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def infer_collate(batch):
    return list(batch)

def infer():
    # Setup DDP
    rank, world_size, local_rank, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if use_ddp and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load model and tokenizers
    weights_dir = WEIGHTS_DIR
    model_path = os.path.join(weights_dir, "model.safetensors")
    config_path = os.path.join(weights_dir, "config.json")
    
    if not (os.path.isfile(model_path) and os.path.isfile(config_path)):
        if rank == 0:
            print(f"Missing checkpoint/config in {weights_dir}")
        return
            
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    src_tok = AutoTokenizer.from_pretrained(os.path.join(weights_dir, "src_tokenizer"))
    tgt_tok = AutoTokenizer.from_pretrained(os.path.join(weights_dir, "tgt_tokenizer"), strip_accents=False)
    
    src_model_path = SRC_MODEL_PATH
            
    enc_model = AutoModel.from_pretrained(src_model_path)
    enc_model.resize_token_embeddings(len(src_tok))
    model = BertEncTxDec(
        enc_model, 
        int(cfg["vocab_size"]), 
        int(cfg["src_pad_id"]), 
        int(cfg["d_model"]), 
        int(cfg["nhead"]), 
        int(cfg["n_layers"]), 
        int(cfg["ffn_dim"]), 
        float(cfg["dropout"]), 
        int(cfg["max_len"])
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
    model.load_state_dict(load_file(model_path, device=str(device)), strict=True)
    model.eval()
    
    df = pd.read_csv(CSV_PATH)
    if "Hindi" not in df.columns:
        raise ValueError("Input CSV must contain a 'Hindi' column.")
        
    ds = InferDataset(df, "Hindi")
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if use_ddp else None
    dl = DataLoader(ds, batch_size=INFERENCE_BATCH_SIZE, sampler=sampler, shuffle=False, collate_fn=infer_collate)
    
    srcs, hyps = [], []
    for src_batch in tqdm(dl, total=len(dl), disable=rank!=0, desc="Inference hi2bh"):
        preds = beam_generate(model, src_tok, tgt_tok, src_batch, int(cfg["max_len"]), beam=BEAM, alpha=ALPHA, device=device)
        srcs.extend(src_batch)
        hyps.extend(preds)
        
    if use_ddp:
        all_s = [None for _ in range(world_size)]
        all_h = [None for _ in range(world_size)]
        dist.all_gather_object(all_s, srcs)
        dist.all_gather_object(all_h, hyps)
        if rank == 0:
            srcs = [s for sl in all_s for s in sl]
            hyps = [h for sl in all_h for h in sl]
            
    if rank == 0:
        out_df = pd.DataFrame({"Hindi": srcs, "Predicted Bhili": hyps})
        out_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Predictions saved to {OUTPUT_CSV_PATH}")
        
    cleanup_ddp(use_ddp)

if __name__ == "__main__":
    infer()


