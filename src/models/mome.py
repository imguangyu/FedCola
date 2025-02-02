""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Hacked together by / Copyright 2020 Ross Wightman
"""
from copy import copy, deepcopy
from ctypes import Union
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from torch.jit import Final
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType

from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models import create_model
# from pytorch_lightning.utilities.distributed import rank_zero_info
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

# From https://github.com/AILab-CVC/M2PT to add complementary modality weights
class CrossModalReparamLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True,
                 origin_layer=None,
                 aux_weight=None,
                 is_aux_trainable=True):
        super().__init__(in_features, out_features, bias)
        self.cross_modal_scale = nn.Parameter(torch.zeros(1))
        assert self.weight.size() == aux_weight.size(), 'Target weight and aux weight must have the same shape'
        self.aux_weight = aux_weight
        self.aux_weight.requires_grad_(is_aux_trainable)
        if origin_layer is not None:
            with torch.no_grad():
                self.weight.copy_(origin_layer.weight)
                self.bias.copy_(origin_layer.bias)

    def forward(self, input):
        weight = self.weight + self.cross_modal_scale * self.aux_weight
        return F.linear(input, weight, self.bias)


def build_cross_modal_reparam_linear(origin_layer, aux_layer, is_aux_trainable=True):
    assert origin_layer.weight.size() == aux_layer.weight.size()
    return CrossModalReparamLinear(in_features=origin_layer.in_features, out_features=origin_layer.out_features, origin_layer=origin_layer,
                                   bias=origin_layer.bias is not None,
                                   aux_weight=aux_layer.weight, is_aux_trainable=is_aux_trainable)


def _get_attr_by_name(obj, attr_name):
    attrs = attr_name.split('.')
    for a in attrs:
        obj = obj.__getattr__(a)
    return obj

def _set_attr_by_name(obj, attr_name, attr_value):
    owner = obj
    attr_names = attr_name.split('.')
    if len(attr_names) > 1:
        for a in attr_names[:-1]:
            owner = owner.__getattr__(a)
    owner.__setattr__(attr_names[-1], attr_value)

def change_original_linear_to_reparam(target_module, aux_module, layer_name, is_aux_trainable=True):
    origin_linear_layer = _get_attr_by_name(target_module, layer_name)
    aux_linear_layer = _get_attr_by_name(aux_module, layer_name)
    reparam_layer = build_cross_modal_reparam_linear(origin_linear_layer, aux_linear_layer, is_aux_trainable=is_aux_trainable)
    _set_attr_by_name(target_module, layer_name, reparam_layer)


def reparameterize_aux_into_target_model(target_model, aux_model,
                               layer_names=('attn.qkv', 'attn.proj', 'mlp.fc1','mlp.fc2'), is_aux_trainable=True):
    target_transformer_blocks = target_model
    aux_transformer_blocks = aux_model
    for target_block, aux_block in zip(target_transformer_blocks, aux_transformer_blocks):
        for layer_name in layer_names:
            change_original_linear_to_reparam(target_block, aux_block, layer_name, is_aux_trainable)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
    
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_scale = None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x

def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            modality: Literal['img', 'txt', 'video'] = 'img',
            max_len: int = 196,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x




class Embedding(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed = None
        self.modality = None
    
    def forward(self):
        raise NotImplementedError

class ImageEmbedding(Embedding):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, drop_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.embed.num_patches

        self.modality = 'img'

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, _x):
        x = self.embed(_x)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # x_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)

        return x

class TextEmbedding(Embedding):

        def __init__(self, vocab_size, num_features, max_text_len, drop_path_rate, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=num_features,
            max_position_embeddings=max_text_len,
            hidden_dropout_prob=drop_path_rate,
            position_embedding_type="absolute", 
            )

            self.text_embeddings = BertEmbeddings(bert_config)
            
            self.modality = 'txt'

            # self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features)) # Not needed since the tokenizer will give u the cls token.
        
        def forward(self, x):
            x = self.text_embeddings(x)
            B, L, _ = x.shape

            # cls_tokens = self.cls_token.expand(B, -1, -1)
            # x = torch.cat((cls_tokens, x), dim=1)
        
            return x

class ClassificationHead(nn.Module):
    def __init__(self, num_features, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = x[:, 0]  # class token
        return self.head(x)
    
class RetrievalHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.head = nn.Identity()

    def forward(self, x):
        x = x[:, 0]  # class token
        return self.head( x / x.norm(dim=-1, keepdim=True))

SCOPE_RANK = {
    'dataset': 0,
    'modality': 1,
    'all': 2,
    'none': 0,
    'attn': 1,
    'blocks': 2
}


class ModalityAgnosticTransformer(nn.Module):
    def __init__(self, 
                 modalities, 
                 num_classes,
                 tasks,
                 shared_param='none',
                 share_scope='dataset',
                 colearn_param='none',
                 img_size=224, patch_size=16, in_chans=3, embed_dim=768, drop_rate=0.0, 
                 num_heads=12,
                 vocab_size=30522, max_text_len=40,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 depth=12,
                 shared_start_index=-1,
                 layer_scale_init_values=None,
                 block_fn=Block,
                 *args, **kwargs) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.with_aux = kwargs.get('with_aux', False)
        self.aux_trained = kwargs.get('aux_trained', False)
        self.aux_attn_only = kwargs.get('aux_attn_only', False)
        self.aux_mlp_only = kwargs.get('aux_mlp_only', False)

        if shared_start_index == -1:
            shared_start_index = depth
        
        self.shared_start_index = shared_start_index
        self.shared_param = shared_param
        self.scope = share_scope
        self.colearn_param = colearn_param

        # Embedding
        self.embeddings = []
        # self.modalities = []

        self.modalities = modalities
        for modality in modalities:
            if modality == 'img':
                self.embeddings.append(ImageEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, drop_rate=drop_rate))
            elif modality == 'txt':
                self.embeddings.append(TextEmbedding(vocab_size=vocab_size, num_features=embed_dim, max_text_len=max_text_len, drop_path_rate=drop_rate))
            elif modality is None:
                self.embeddings.append(None)
            else:
                raise NotImplementedError
            
        self.embeddings = nn.ModuleList(self.embeddings)
        # Blocks
        self.num_heads = num_heads
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ] 
        self.blockses = []
        for i, modality in enumerate(modalities):
            if modality is None:
                self.blockses.append(None)
                continue
            self.blockses.append(nn.Sequential(*
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        init_values=layer_scale_init_values,
                    )
                    for i in range(depth)
                ]
            ))
        
        self.blockses = nn.ModuleList(self.blockses)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        # Heads
        self.heads = []
        for i, task in enumerate(tasks):
            if task == 'cls':
                self.heads.append(ClassificationHead(num_features=embed_dim, num_classes=num_classes[i]))
            elif task == 'rtv':
                self.heads.append(RetrievalHead())
            elif task is None:
                self.heads.append(None)
            else:
                raise NotImplementedError
            
        self.heads = nn.ModuleList(self.heads)

        # Build aux part
        if self.with_aux and None in self.modalities:
            self.build_aux()
    
    def build_aux(self):
        for i, blocks in enumerate(self.blockses):
            if blocks is not None:
                main_idx = i
                break

        if self.aux_attn_only:
            if self.aux_mlp_only:
                raise ValueError('Both aux_attn_only and aux_mlp_only cannot be True.')
            layer_name = ('attn.qkv', 'attn.proj')
        elif self.aux_mlp_only:
            layer_name = ('mlp.fc1','mlp.fc2')
        else:
            layer_name = ('attn.qkv', 'attn.proj', 'mlp.fc1','mlp.fc2')
        
        reparameterize_aux_into_target_model(self.blockses[main_idx], self.blockses[main_idx], layer_names=layer_name, is_aux_trainable=self.aux_trained)
    
    def pretrain_vit(self, model_strs):
        none_idx = [i for i, model_str in enumerate(model_strs) if model_str is None]
        for i, model_str in enumerate(model_strs):
            if model_str is None:
                continue

            if 'ours' in model_str:
                    sd = torch.load("pretrain.pt")
                    old_sd = copy(sd)
                    for k,v in old_sd.items():
                        if 'head' in k:
                            sd.update({k.replace('head', 'heads.head'): v})
            else:
                sd = create_model(model_str, pretrained=True).cpu().state_dict()
            old_sd = copy(sd)
            for k,v in old_sd.items():
                if 'patch_embed' in k:
                    sd.update({k.replace('patch_embed', f'embeddings.{i}.embed'): v})
                elif 'blocks.' in k:
                    sd.update({k.replace('blocks', f'blockses.{i}'): v})
                    # for j in none_idx:
                    #     sd.update({k.replace('blocks', f'blockses.{j}'): v})
            sd.update({f'embeddings.{i}.cls_token': sd['cls_token']})
            sd.update({f'embeddings.{i}.pos_embed': sd['pos_embed']})

            self.load_state_dict(sd, strict=False)

        
        self.sync_shared_weights()
    
    def sync_shared_weights(self):
        for i, blocks in enumerate(self.blockses):
            if blocks is not None:
                main_idx = i
                break
        # link for uni-modal aggregation
        if self.scope == 'all':
            for i, blocks in enumerate(self.blockses):
                if blocks is None:
                    self.blockses[i] = self.blockses[main_idx]
        
        # link for mm colearn
        if self.colearn_param == 'none':
            return

        if self.colearn_param == 'blocks':
            for i, blocks in enumerate(self.blockses):
                if blocks is not None and i != main_idx:
                    blocks = self.blockses[main_idx]
        elif self.colearn_param == 'attn':
            for i, blocks in enumerate(self.blockses):
                if blocks is not None and i != main_idx:
                    for j, block in enumerate(blocks):
                        block.attn = self.blockses[main_idx][j].attn   
        gc.collect()

    def required_params(self):
        sd = self.state_dict()
        new_sd = copy(sd)
        for i, modality in enumerate(self.modalities):
            if modality is None:
                for k,_ in sd.items():
                    if f'blockses.{i}' in k:
                        new_sd.pop(k)
        if self.with_aux:
            new_new_sd = copy(new_sd)
            for k,_ in new_sd.items():
                if 'aux' in k or 'cross_modal_scale' in k:
                    new_new_sd.pop(k)
        else:
            new_new_sd = new_sd
            
        return new_new_sd
    
    def aux_params(self):
        if not self.with_aux:
            raise ValueError('No aux params.')
        sd = self.state_dict()
        new_sd = {}
        none_idx = [i for i, m in enumerate(self.modalities) if m is None]
        
        for k,v in sd.items():
            required_param = True
            for idx in none_idx:
                if f'blockses.{idx}' in k:
                    required_param=False
                    break
            if ('aux' in k ) and required_param:
                    new_sd.update({k:v})

        return new_sd


    def forward(self, x, feat_out=False):
        # _, C, _, _ = x.shape

        outs = []
        # embeds = torch.zeros(x[0].shape[0], 0, self.embed_dim).to(x[0].device)

        embeds = []
        for i, modality in enumerate(self.modalities):
            if modality is None:
                assert x[i] is None, 'None modality should have None input.'
                embeds.append(None)
                continue
            if len(x[i].shape)==4 and x[i].shape[1]==1:
                x[i] = x[i].repeat(1,3,1,1)

            embeds.append(self.embeddings[i](x[i]))
        
        # print(embeds[0].shape, embeds[1].shape)

        feats = [None for _ in range(len(self.modalities))]

        for i, modality in enumerate(self.modalities):
            if modality is None:
                continue
            features = self.blockses[i](embeds[i])
            features = self.norm(features)
            feats[i] = features

        outs = [None for _ in range(len(self.modalities))]

        if feat_out:
            for i, modality in enumerate(self.modalities):
                if modality is None:
                    continue
                outs[i] = feats[i][:, 0] / feats[i][:, 0].norm(dim=-1, keepdim=True)
        else:
            for i, modality in enumerate(self.modalities):
                if modality is None:
                    continue
                outs[i] = self.heads[i](feats[i])

        return outs

@register_model
def mome_small_patch16(pretrained, args, **kwargs):
    '''
    We unify the img and text encoders into one model 
    shared_param: Shared parameters between same modality in different type of client 
                  (i.e., img encoder in img client and img encoder in img-txt client) 
    share_scope: Shared scope during aggregation
                 dataset: share parameters only to encoders with the same dataset
                 modality: share parameters only to encoders with the same modality
                 all: share parameters among all encoders
    colearn_param: Shared parameters between img and txt encoders
    '''

    model = ModalityAgnosticTransformer(img_size=224,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                vocab_size=args.vocab_size, 
                max_text_len=args.seq_len,
                drop_path_rate=args.dropout,
                shared_param=args.shared_param,
                share_scope=args.share_scope,
                colearn_param=args.colearn_param,
                **kwargs
                )
    model.sync_shared_weights()
    if pretrained:
        model.pretrain_vit(['vit_small_patch16_224', None])
    return model 

@register_model
def mome_tiny_patch16(pretrained, args, **kwargs):

        model = ModalityAgnosticTransformer(img_size=224,
                    patch_size=16,
                    embed_dim=192,
                    depth=12,
                    num_heads=3,
                    vocab_size=args.vocab_size, 
                    max_text_len=args.seq_len,
                    drop_path_rate=args.dropout,
                    shared_param=args.shared_param,
                    share_scope=args.share_scope,
                    colearn_param=args.colearn_param,
                    **kwargs
                    )
        model.sync_shared_weights()
        if pretrained:
            model.pretrain_vit(['vit_tiny_patch16_224', None])
        return model

@register_model
def mome_small_patch16_224_in21k(pretrained, args, **kwargs):
    
    model = ModalityAgnosticTransformer(img_size=224,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                vocab_size=args.vocab_size, 
                max_text_len=args.seq_len,
                drop_path_rate=args.dropout,
                shared_param=args.shared_param,
                share_scope=args.share_scope,
                colearn_param=args.colearn_param,
                **kwargs
                )
    model.sync_shared_weights()
    if pretrained:
        model.pretrain_vit(['vit_small_patch16_224_in21k', None])
    return model 


@register_model
def mome_base_patch16_224_ours(pretrained, args, **kwargs):
    
    model = ModalityAgnosticTransformer(img_size=224,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                vocab_size=args.vocab_size, 
                max_text_len=args.seq_len,
                drop_path_rate=args.dropout,
                share_strategy=args.strategy,
                colearn_param=args.colearn_param,
                **kwargs)
    if pretrained:
        model.pretrain_vit(['vit_small_patch16_224_ours', None])
    return model 

@register_model
def mome_toy_patch16_224(pretrained, args, **kwargs):
    
    model = ModalityAgnosticTransformer(img_size=224,
                patch_size=16,
                embed_dim=4,
                depth=1,
                num_heads=2,
                vocab_size=args.vocab_size, 
                max_text_len=args.seq_len,
                drop_path_rate=args.dropout,
                shared_param=args.shared_param,
                share_scope=args.share_scope,
                colearn_param=args.colearn_param,
                **kwargs)
    model.sync_shared_weights()

    return model 









