# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
from functools import partial
from itertools import repeat
import collections.abc as container_abcs
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from .block import Block,Block_Token


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        img_height, img_width = img_size
        patch_height, patch_width = patch_size
        # 计算 H 和 W
        self.H = img_height // patch_height
        self.W = img_width // patch_width

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(128, 256), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 learnable_pos=False, drop_path_rate=0.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches_z = (img_size[0] // patch_size) ** 2
        self.num_patches_x = (img_size[1] // patch_size) ** 2
        self.feat_sz_z = img_size[0] // patch_size
        self.feat_sz_x = img_size[1] // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_z, embed_dim),
                                        requires_grad=learnable_pos)  # fixed sin-cos embedding
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_x, embed_dim),
                                        requires_grad=learnable_pos)  # fixed sin-cos embedding

        self.modal_embed = nn.Parameter(torch.zeros(2, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  drop_path=dpr[i])
            for i in range(depth)])
        self.blocks_token = nn.ModuleList([
            Block_Token(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_z = get_2d_sincos_pos_embed(self.pos_embed_z.shape[-1], int(self.num_patches_z ** .5),
                                              cls_token=False)
        self.pos_embed_z.data.copy_(torch.from_numpy(pos_embed_z).float().unsqueeze(0))
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], int(self.num_patches_x ** .5),
                                              cls_token=False)
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.modal_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, z, x):
        # embed patches
        z = self.patch_embed(z)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        embedding = torch.cat((z, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            embedding = blk(embedding)
        embedding = self.norm(embedding)

        z, x = torch.split(embedding, [self.num_patches_z, self.num_patches_x], dim=1)
        return x  # , mask, ids_restore

    def forward(self, z, x, **kwargs):
        x = self.forward_encoder(z, x)
        out_dict = {
            "search": x  # .transpose(-2, -1).reshape(x.shape[0], -1, self.feat_sz_x, self.feat_sz_x).contiguous()
        }
        return out_dict

    def forward_joint(self, img_feat, txt_feat, mask, idx, flag=None):
        ime_len, txt_len = img_feat.shape[1], txt_feat.shape[1]
        # TODO
        embedding = torch.cat([img_feat + self.modal_embed[0], txt_feat + self.modal_embed[1]], dim=1)
        embedding = self.blocks[idx](embedding, None, flag=flag)
        img_feat, txt_feat = embedding.split([ime_len, txt_len], dim=1)
        return img_feat, txt_feat

#語言掩
    # def forward_joint(self, img_feat, txt_feat, mask, idx, flag=None):
    #     ime_len, txt_len = img_feat.shape[1], txt_feat.shape[1]

    #     if idx == 6:
    #                 # 视觉全局特征
    #                 img_global = img_feat.mean(dim=1, keepdim=True)  # [B, 1, C]
                    
    #                 # 计算视觉-语言交互置信度
    #                 confidence = img_global * txt_feat  # [B, N_t, C]
    #                 confidence = torch.abs(confidence)  # 取绝对值作为置信度

    #                 k = int(txt_feat.shape[-1] * 0.75)


    #                 _, top_k_indices = torch.topk(confidence, k, dim=-1)
                    
    #                 # 保存通道掩码（供后续层使用）
    #                 self.channel_mask = torch.zeros_like(txt_feat, dtype=torch.bool)
    #                 self.channel_mask.scatter_(-1, top_k_indices, True)
                    
    #                 # 第6层应用掩码
    #                 txt_feat = txt_feat.clone()
    #                 txt_feat[~self.channel_mask] = 0
                
    #             # ----- 3. 第7-11层：直接使用第6层保存的掩码 -----
    #     if idx > 6:
    #                 txt_feat = txt_feat.clone()
    #                 txt_feat[~self.channel_mask] = 0
                
    #             # 日志
    #     if idx == 6 or idx == 11:
    #                 #keep_img = (~self.accum_unimportant).float().mean().item() * 100
    #                 keep_txt_channel = self.channel_mask.float().mean().item() * 100
    #                 print(f"语言通道保留 {keep_txt_channel:.1f}%")


    #     # TODO
    #     embedding = torch.cat([img_feat + self.modal_embed[0], txt_feat + self.modal_embed[1]], dim=1)
    #     embedding = self.blocks[idx](embedding, None, flag=flag)
    #     img_feat, txt_feat = embedding.split([ime_len, txt_len], dim=1)
    #     return img_feat, txt_feat


    # def forward_joint(self, img_feat, txt_feat, mask, idx, flag=None):
    #     ime_len, txt_len = img_feat.shape[1], txt_feat.shape[1]
        
    #     # 模板和搜索图像的分界点
    #     cls_len, z_len, x_len = 1, 64, 256
    #     if ime_len != cls_len + z_len + x_len:
    #         x_len = ime_len - cls_len - z_len if ime_len > cls_len + z_len else 0
        
    #     # 分离并重组特征
    #     cls_feat = img_feat[:, :cls_len, :]
    #     z_feat = img_feat[:, cls_len:cls_len+z_len, :]
    #     x_feat = img_feat[:, cls_len+z_len:cls_len+z_len+x_len, :]
    #     img_feat = torch.cat([cls_feat, z_feat, x_feat], dim=1)
        
    #     # 语言通道掩码（第6层计算，后续层复用）
    #     if idx == 6:
    #         confidence = torch.abs(img_feat.mean(dim=1, keepdim=True) * txt_feat)
    #         k = int(txt_feat.shape[-1] * 0.75)
    #         _, top_k_indices = torch.topk(confidence, k, dim=-1)
    #         self.channel_mask = torch.zeros_like(txt_feat, dtype=torch.bool)
    #         self.channel_mask.scatter_(-1, top_k_indices, True)
    #         txt_feat = txt_feat.clone()
    #         txt_feat[~self.channel_mask] = 0
    #     elif idx > 6:
    #         txt_feat = txt_feat.clone()
    #         txt_feat[~self.channel_mask] = 0
        
    #     # 合并特征
    #     embedding = torch.cat([img_feat + self.modal_embed[0], txt_feat + self.modal_embed[1]], dim=1)
        
    #     # 视觉token掩码（从第6层开始）
    #     if idx >= 8 and x_len > 0:
    #         updated_embedding, attn_scores = self.blocks[idx](embedding, None, flag=flag, return_attn=True)
            
    #         # 提取cls token对搜索图像的注意力
    #         search_start = cls_len + z_len
    #         cls_attn = attn_scores[:, :, 0, :].mean(dim=1)
    #         search_attn = cls_attn[:, search_start:search_start+x_len]
            
    #         # 逐层增加掩码比例：第6层10%，第7层20%，...第11层60%
    #         mask_ratio = (idx - 7) * 0.05
    #         k = int(x_len * (1 - mask_ratio))
    #         _, indices = torch.topk(search_attn, k, dim=1)
            
    #         mask_binary = torch.zeros_like(search_attn)
    #         mask_binary.scatter_(1, indices, 1.0)
            
    #         updated_img_feat, txt_feat = updated_embedding.split([ime_len, txt_len], dim=1)
    #         masked_img_feat = updated_img_feat.clone()
    #         masked_img_feat[:, search_start:search_start+x_len, :] *= mask_binary.unsqueeze(-1)
    #         updated_img_feat = masked_img_feat
    #     else:
    #         updated_embedding = self.blocks[idx](embedding, None, flag=flag)
    #         updated_img_feat, txt_feat = updated_embedding.split([ime_len, txt_len], dim=1)
        
    #     # 日志
    #     if idx == 6 or idx == 11:
    #         keep_txt = self.channel_mask.float().mean().item() * 100
    #         print(f"语言通道保留 {keep_txt:.1f}%, 视觉层{idx}: 掩码{(idx-7)*5:.0f}%")
        
    #     return updated_img_feat, txt_feat





    def patchify(self, z, x, image_ids=None):
        B = x.shape[0]
        
        #img_feat = self.vit.patchify(template, search)

        z_patches = self.patch_embed(z) 
        x_patches = self.patch_embed(x) 
        if not hasattr(self.__class__, '_global_background_mask_cache'):
            self.__class__._global_background_mask_cache = {}
            self.__class__._cache_device = x.device
        batch_masks = []
        for b in range(B):
            img_id = image_ids[b] if image_ids else f'{x.device}_{b}'  # 备用键
            if img_id not in self._global_background_mask_cache:
                patch_mask = torch.zeros(x_patches.shape[1], dtype=torch.bool, device=x.device)
                for i in range(1, x_patches.shape[1]):
                    sim = torch.cosine_similarity(
                        x_patches[b, i:i + 1].unsqueeze(1),  # [1,1,dim]
                        x_patches[b, :i].unsqueeze(0),  # [1,i,dim]
                        dim=-1
                    ).max()
                    patch_mask[i] = sim > 0.8
                self._global_background_mask_cache[img_id] = patch_mask
            batch_masks.append(self._global_background_mask_cache[img_id])
        background_mask = torch.stack(batch_masks).unsqueeze(2).expand(-1, -1, x_patches.shape[-1])
        x_patches = x_patches * (~background_mask).float() + x_patches.detach() * background_mask.float()

        z_patches = z_patches + self.pos_embed_z
        x_patches = x_patches + self.pos_embed_x
        #print("z_patches 的 patch 数量 (num_z):", z_patches.shape[1])
        #print("x_patches 的 patch 数量 (num_x):", x_patches.shape[1])
        cls_token = self.cls_token.expand(B, -1, -1)
        return torch.cat((cls_token, z_patches, x_patches), dim=1)


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

