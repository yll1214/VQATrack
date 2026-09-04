import torch
import math
from typing import List, Dict


def register_vit_attention_hooks(vit, fusion_layer_idxs: List[int]):
    handles = []
    attn_store = {}

    def make_hook(i):
        def hook(module, input, output):
            # try to reconstruct q,k from module
            try:
                x = input[0]  # (B,N,C)
            except Exception:
                return
            mask = input[1] if len(input) > 1 else None
            if hasattr(module, 'qkv'):
                qkv = module.qkv(x)
                B, N, _ = qkv.shape
                num_heads = module.num_heads
                C = qkv.shape[-1] // 3
                qkv = qkv.view(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
            elif hasattr(module, 'q') and hasattr(module, 'kv'):
                q = module.q(x)
                kv = module.kv(x)
                B, N, _ = q.shape
                num_heads = module.num_heads
                q = q.view(B, N, num_heads, -1).permute(0, 2, 1, 3)
                kv = kv.view(B, N, 2, num_heads, -1).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
            else:
                return

            head_dim = q.shape[-1]
            scale = getattr(module, 'scale', head_dim ** -0.5)
            attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
            if mask is not None:
                try:
                    attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e10)
                except Exception:
                    pass
            attn = torch.softmax(attn, dim=-1)
            attn_store[i] = attn.detach().cpu()

        return hook

    for i in fusion_layer_idxs:
        try:
            module = vit.blocks[i].attn
            h = module.register_forward_hook(make_hook(i))
            handles.append(h)
        except Exception:
            continue

    return handles, attn_store


def remove_handles(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def text_to_search_maps_from_attn(attn_store: Dict[int, torch.Tensor], vit, layer_idx: int = None, merge_heads=True, topk: int = None):
    if len(attn_store) == 0:
        raise ValueError('No attention collected')
    keys = sorted(attn_store.keys())
    if layer_idx is None:
        layer_idx = keys[-1]
    if layer_idx not in attn_store:
        raise ValueError(f'layer {layer_idx} not in attn_store')
    att = attn_store[layer_idx]  # (B, heads, N, N)
    B, heads, Nq, Nk = att.shape
    if merge_heads:
        A = att.mean(axis=1)  # (B, N, N)
    else:
        A = att.mean(axis=1)

    num_patches_z = vit.num_patches_z
    num_patches_x = vit.num_patches_x
    img_len = 1 + num_patches_z + num_patches_x
    # assume text tokens appended after image tokens
    txt_start = img_len
    txt_len = att.shape[2] - img_len
    x_start = 1 + num_patches_z
    x_end = x_start + num_patches_x

    A_text_to_x = A[:, txt_start:txt_start+txt_len, x_start:x_end]
    H = W = int(math.sqrt(num_patches_x))
    maps = A_text_to_x.reshape(B, txt_len, H, W)
    maps_sum = maps.reshape(B, txt_len, -1).sum(-1, keepdims=True)
    maps_norm = maps / (maps_sum.reshape(B, txt_len, 1, 1) + 1e-8)

    flat = maps_norm.reshape(B, txt_len, -1)
    if topk is None:
        grounding = flat.mean(-1)
    else:
        k = max(1, min(topk, flat.shape[-1]))
        grounding = flat.topk(k, dim=2)[0].mean(dim=2)

    return maps_norm, grounding


def prune_tokens_by_grounding(grounding: torch.Tensor, threshold: float = None, pct: float = None):
    if threshold is None and pct is None:
        raise ValueError('Provide threshold or pct')
    B, N = grounding.shape
    mask = torch.zeros_like(grounding, dtype=torch.bool)
    if threshold is not None:
        mask = grounding < threshold
        return mask
    k = max(1, int(N * pct))
    for b in range(B):
        vals = grounding[b]
        if k >= N:
            inds = torch.arange(N)
        else:
            inds = torch.topk(vals, k, largest=False).indices
        mask[b, inds] = True
    return mask
