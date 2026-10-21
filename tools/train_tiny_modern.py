"""Train a tiny *modern* (Llama-style) transformer on the nano-gpt sort task and export it for the visualizer.

Architecture (all parts named in the visualizer):
  RMSNorm (no mean, weight only) · RoPE (rotate-half, theta 10000) · grouped-query attention (n_head query heads
  share n_kv_head key/value heads) · SwiGLU MLP (silu(gate) * up -> down) · optional Mixture-of-Experts
  (softmax router, top-k experts per token, weights renormalised over the chosen k) · no biases · final RMSNorm.

Output: public/tiny-<variant>-model.json  (config + weights)
        public/tiny-<variant>-check.json  (one input + every intermediate tensor, used to verify the JS forward pass)

Usage:  tools/.venv/bin/python tools/train_tiny_modern.py --variant llama   (or --variant moe)
"""
import argparse
import base64
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIGITS = 6           # letters to sort
VOCAB = 3              # A, B, C
T = 2 * N_DIGITS - 1   # 11, same as nano-gpt


# ----------------------------------------------------------------------------- model

class RMSNorm(nn.Module):
    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.eps = eps

    def forward(self, x, rec=None, name=None):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        y = x / rms * self.weight
        if rec is not None:
            rec[f'{name}.rms'] = rms.squeeze(-1)
            rec[f'{name}.out'] = y
        return y


def rope_tables(t, a, theta=10000.0):
    """cos/sin of shape [t, a/2]; pair j rotates channels (j, j + a/2)."""
    inv = 1.0 / theta ** (torch.arange(0, a // 2, dtype=torch.float32) * 2 / a)
    ang = torch.outer(torch.arange(t, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: [..., t, a]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c, nh, nkv = cfg['n_embd'], cfg['n_head'], cfg['n_kv_head']
        self.nh, self.nkv, self.a = nh, nkv, c // nh
        self.wq = nn.Linear(c, nh * self.a, bias=False)
        self.wk = nn.Linear(c, nkv * self.a, bias=False)
        self.wv = nn.Linear(c, nkv * self.a, bias=False)
        self.wo = nn.Linear(nh * self.a, c, bias=False)

    def forward(self, x, cos, sin, rec=None, p=''):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.nh, self.a).transpose(1, 2)    # [b, nh, t, a]
        k = self.wk(x).view(b, t, self.nkv, self.a).transpose(1, 2)   # [b, nkv, t, a]
        v = self.wv(x).view(b, t, self.nkv, self.a).transpose(1, 2)
        qr, kr = apply_rope(q, cos[:t], sin[:t]), apply_rope(k, cos[:t], sin[:t])
        rep = self.nh // self.nkv                                      # query heads per kv head
        kr_full, v_full = kr.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        scores = qr @ kr_full.transpose(-1, -2) / math.sqrt(self.a)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float('-inf'))
        probs = scores.softmax(-1)
        heads = probs @ v_full                                         # [b, nh, t, a]
        cat = heads.transpose(1, 2).reshape(b, t, self.nh * self.a)
        out = self.wo(cat)
        if rec is not None:
            rec.update({f'{p}.q': q, f'{p}.k': k, f'{p}.v': v, f'{p}.q_rope': qr, f'{p}.k_rope': kr,
                        f'{p}.scores': scores.masked_fill(mask, 0.0), f'{p}.probs': probs,
                        f'{p}.heads': cat, f'{p}.out': out})
        return out


class SwiGLU(nn.Module):
    def __init__(self, c, h):
        super().__init__()
        self.gate = nn.Linear(c, h, bias=False)
        self.up = nn.Linear(c, h, bias=False)
        self.down = nn.Linear(h, c, bias=False)

    def forward(self, x, rec=None, p=''):
        g, u = self.gate(x), self.up(x)
        act = F.silu(g) * u
        out = self.down(act)
        if rec is not None:
            rec.update({f'{p}.gate': g, f'{p}.up': u, f'{p}.act': act, f'{p}.out': out})
        return out


class MoE(nn.Module):
    def __init__(self, c, h, n_exp, top_k):
        super().__init__()
        self.router = nn.Linear(c, n_exp, bias=False)
        self.experts = nn.ModuleList([SwiGLU(c, h) for _ in range(n_exp)])
        self.k = top_k

    def forward(self, x, rec=None, p=''):
        logits = self.router(x)                                        # [b, t, E]
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, -1)
        w = torch.zeros_like(probs).scatter(-1, topi, topv / topv.sum(-1, keepdim=True))  # renormalised over top-k
        out = torch.zeros_like(x)
        per_exp = []
        for e, exp in enumerate(self.experts):
            sub = {} if rec is not None else None
            y = exp(x, sub, f'{p}.e{e}')
            out = out + w[..., e:e + 1] * y
            per_exp.append(y)
            if rec is not None:
                rec.update(sub)
        if rec is not None:
            rec.update({f'{p}.router_logits': logits, f'{p}.router_probs': probs, f'{p}.weights': w, f'{p}.out': out})
        return out


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg['n_embd']
        self.norm1 = RMSNorm(c)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(c)
        if cfg.get('n_expert', 0) > 0:
            self.mlp = MoE(c, cfg['n_hidden'], cfg['n_expert'], cfg['top_k'])
        else:
            self.mlp = SwiGLU(c, cfg['n_hidden'])

    def forward(self, x, cos, sin, rec=None, p=''):
        x = x + self.attn(self.norm1(x, rec, f'{p}.norm1'), cos, sin, rec, f'{p}.attn')
        if rec is not None:
            rec[f'{p}.resid1'] = x
        x = x + self.mlp(self.norm2(x, rec, f'{p}.norm2'), rec, f'{p}.mlp')
        if rec is not None:
            rec[f'{p}.resid2'] = x
        return x


class TinyModern(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['n_embd'])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['n_layer'])])
        self.norm_f = RMSNorm(cfg['n_embd'])
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)
        cos, sin = rope_tables(cfg['block_size'], cfg['n_embd'] // cfg['n_head'], cfg['rope_theta'])
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, idx, rec=None):
        x = self.tok_emb(idx)
        if rec is not None:
            rec['embed'] = x
            rec['rope.cos'], rec['rope.sin'] = self.cos, self.sin
        for i, b in enumerate(self.blocks):
            x = b(x, self.cos, self.sin, rec, f'l{i}')
        x = self.norm_f(x, rec, 'norm_f')
        logits = self.lm_head(x)
        if rec is not None:
            rec['logits'] = logits
            rec['probs'] = logits.softmax(-1)
        return logits


# ----------------------------------------------------------------------------- data (same task as nano-gpt)

def batch(n, device):
    x = torch.randint(0, VOCAB, (n, N_DIGITS), device=device)
    s = x.sort(-1).values
    cat = torch.cat([x, s], 1)
    inp, tgt = cat[:, :-1].clone(), cat[:, 1:].clone()
    tgt[:, :N_DIGITS - 1] = -1  # only score the sorted half
    return inp, tgt


@torch.no_grad()
def accuracy(model, device, n=2000):
    x = torch.randint(0, VOCAB, (n, N_DIGITS), device=device)
    seq = x
    for _ in range(N_DIGITS):
        nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
    return (seq[:, N_DIGITS:] == x.sort(-1).values).all(-1).float().mean().item()


# ----------------------------------------------------------------------------- export

def enc(t):
    t = t.detach().to('cpu', torch.float32).contiguous()
    return {'shape': list(t.shape), 'dtype': 'torch.float32', 'data': base64.b64encode(t.numpy().tobytes()).decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['llama', 'moe'], default='llama')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'public'))
    args = ap.parse_args()

    torch.manual_seed(3407)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = dict(vocab_size=VOCAB, block_size=T, n_embd=48, n_layer=3, n_head=4, n_kv_head=2,
               n_hidden=128, rope_theta=10000.0, norm_eps=1e-5, arch='llama')
    if args.variant == 'moe':
        cfg.update(n_hidden=64, n_expert=4, top_k=2, arch='moe')

    model = TinyModern(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for step in range(args.steps):
        inp, tgt = batch(512, device)
        loss = F.cross_entropy(model(inp).reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f'step {step:5d}  loss {loss.item():.4f}')
    model.eval()
    acc = accuracy(model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'sort accuracy {acc:.4f}   params {n_params}')

    model = model.to('cpu').float()
    sd = {k: enc(v) for k, v in model.state_dict().items()}
    os.makedirs(args.out, exist_ok=True)
    name = f'tiny-{args.variant}'
    with open(os.path.join(args.out, f'{name}-model.json'), 'w') as f:
        json.dump({'config': {**cfg, 'n_params': n_params, 'sort_accuracy': acc}, **sd}, f)

    # one full-length input (6 letters + 5 sorted outputs) with every intermediate, for the JS parity test
    x = torch.tensor([[2, 1, 0, 1, 1, 2]])
    with torch.no_grad():
        seq = x
        for _ in range(N_DIGITS - 1):
            seq = torch.cat([seq, model(seq)[:, -1].argmax(-1, keepdim=True)], 1)
        rec = {}
        model(seq, rec)
    check = {'tokens': seq[0].tolist(), 'tensors': {k: enc(v if k.startswith('rope.') else v[0]) for k, v in rec.items()}}
    with open(os.path.join(args.out, f'{name}-check.json'), 'w') as f:
        json.dump(check, f)
    print(f'wrote {name}-model.json and {name}-check.json  ({len(rec)} intermediate tensors)  tokens {seq[0].tolist()}')


if __name__ == '__main__':
    main()
