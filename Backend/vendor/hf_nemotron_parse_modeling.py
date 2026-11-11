# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native Transformers inference for Nemotron 3.5 Parse.

The complete C-RADIOv4-SO400M encoder, content-normalized NaViT geometry,
causal mBART visual prefix, and trained one-step MTP tensors are loaded from
this checkpoint. No second encoder model or remote encoder code is fetched.
"""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Block
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .hf_nemotron_parse_config import NemotronParseConfig
from .hf_prefix_geometry import _normalized_2d_position_pages


class _ClassTokens(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.empty(10, 1152))


class _PatchGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = _ClassTokens()
        self.embedder = nn.Linear(768, 1152, bias=False)
        self.pos_embed = nn.Parameter(torch.empty(1, 16384, 1152))


class _VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_generator = _PatchGenerator()
        self.blocks = nn.ModuleList([
            Block(1152, 16, mlp_ratio=4304 / 1152, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(27)
        ])

    def forward(self, patches, image_size, content_size):
        pg = self.patch_generator
        patches = pg.embedder(patches).unsqueeze(0)
        patches = _normalized_2d_position_pages(
            torch, patches=patches, pos_embed=pg.pos_embed,
            source_rows=128, source_cols=128, image_sizes=[image_size],
            patch_size=16, content_sizes=[content_size], content_normalized=True,
        )
        hidden = torch.cat((pg.cls_token.token.unsqueeze(0), patches), dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class _InputConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(3, 1, 1))
        self.register_buffer("norm_std", torch.ones(3, 1, 1))


class _RadioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("summary_idxs", torch.tensor([0, 1]))
        self.input_conditioner = _InputConditioner()
        self.model = _VisionTransformer()


class _RadioWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.radio_model = _RadioModel()


class RadioWithNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_encoder = _RadioWrapper()
        self.conv1 = nn.Conv1d(1152, 1024, 1)
        self.layer_norm1 = nn.LayerNorm(1024, eps=1e-6)
        self.conv2 = nn.Conv2d(1024, 1024, (1, 4), stride=(1, 4), bias=False)
        self.layer_norm2 = nn.LayerNorm(1024, eps=1e-6)
        self.sum_proj = nn.Linear(2304, 1024)
        self.layer_norm3 = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, pixel_values, image_sizes, content_sizes):
        # The shipped processor has already applied CLIP normalization and
        # flattened each 16x16 patch in channel/row/column order.
        if image_sizes is None or content_sizes is None:
            raise ValueError("Packed inference requires image_sizes and content_sizes from AutoProcessor")
        if pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        if pixel_values.ndim != 2 or pixel_values.shape[1] != 768:
            raise ValueError(f"Expected packed raw patches [patches,768], got {tuple(pixel_values.shape)}")
        pixel_values = pixel_values.to(dtype=self.conv1.weight.dtype)
        pages, cursor = [], 0
        radio = self.model_encoder.radio_model
        for size, content in zip(image_sizes.tolist(), content_sizes.tolist()):
            height, width = map(int, size)
            count = (height // 16) * (width // 16)
            hidden = radio.model(pixel_values[cursor:cursor + count], size, content)
            cursor += count
            summary = hidden[:, radio.summary_idxs].flatten(1)
            features = hidden[:, 10:]
            features = self.layer_norm1(self.conv1(features.transpose(1, 2)).transpose(1, 2))
            features = features.transpose(1, 2).reshape(1, 1024, height // 16, width // 16)
            features = self.conv2(features).flatten(2).transpose(1, 2)
            features = self.layer_norm2(features)
            summary = self.layer_norm3(self.sum_proj(summary)).unsqueeze(1)
            pages.append(torch.cat((features, summary), dim=1)[0])
        if cursor != len(pixel_values):
            raise ValueError("Image metadata did not consume all packed patches")
        max_length = max(len(page) for page in pages)
        output = pages[0].new_zeros(len(pages), max_length, 1024)
        mask = torch.zeros(len(pages), max_length, device=output.device, dtype=torch.bool)
        for i, page in enumerate(pages):
            output[i, :len(page)] = page
            mask[i, :len(page)] = True
        return output, mask


class _ScaledEmbedding(nn.Embedding):
    def __init__(self, config):
        super().__init__(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

    def forward(self, ids):
        return super().forward(ids) * self.embed_scale


class _GQAAttention(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.index = index
        self.num_heads = config.decoder_attention_heads
        self.num_kv_heads = config.self_attention_kv_heads
        self.head_dim = config.d_model // self.num_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden, cache, key_mask):
        batch, length, _ = hidden.shape
        q = self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(k, v, self.index)
        past = k.shape[-2] - length
        allowed = torch.arange(k.shape[-2], device=hidden.device)[None, :] <= (
            torch.arange(length, device=hidden.device)[:, None] + past
        )
        allowed = allowed[None, None] & key_mask[:, None, None, :]
        repeat = self.num_heads // self.num_kv_heads
        output = F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1),
            attn_mask=allowed, dropout_p=0.0,
        )
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))


class _DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = _GQAAttention(config, index)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)

    def forward(self, hidden, cache, mask):
        hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden), cache, mask)
        return hidden + self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden))))


class NemotronParseDecoder(nn.Module):
    def __init__(self, config, num_extra_heads):
        super().__init__()
        self.embed_tokens = _ScaledEmbedding(config)
        self.layers = nn.ModuleList([_DecoderLayer(config, i) for i in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.extra_heads = nn.ModuleList([nn.Linear(config.d_model, config.d_model) for _ in range(num_extra_heads)])
        self.extra_proj = nn.ModuleList([nn.Linear(config.d_model, config.d_model) for _ in range(num_extra_heads)])

    def forward(self, embeddings, cache, mask):
        hidden = self.layernorm_embedding(embeddings)
        for layer in self.layers:
            hidden = layer(hidden, cache, mask)
        return self.layer_norm(hidden)


class NemotronParseForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = NemotronParseConfig
    main_input_name = "input_ids"
    _tied_weights_keys = {"lm_head.weight": "decoder.embed_tokens.weight"}
    _no_split_modules = ["RadioWithNeck", "Block", "_DecoderLayer"]
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        if not getattr(config, "legacy_causal_vision_prefix", False):
            raise ValueError("This release implements the trained causal visual-prefix architecture")
        self.encoder = RadioWithNeck(config.encoder)
        self.decoder = NemotronParseDecoder(config.decoder, getattr(config, "num_extra_heads", 0))
        self.lm_head = nn.Linear(config.decoder.d_model, config.decoder.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def _tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.decoder.embed_tokens)

    def forward(self, input_ids=None, decoder_input_ids=None, pixel_values=None,
                image_sizes=None, content_sizes=None, attention_mask=None,
                past_key_values=None, use_cache=True, return_dict=True, **kwargs):
        del kwargs
        if input_ids is None:
            input_ids = decoder_input_ids
        if input_ids is None:
            raise ValueError("input_ids are required")
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache()
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        text_embeddings = self.decoder.embed_tokens(input_ids)
        text_offset = 0
        if past_length == 0:
            if pixel_values is None:
                raise ValueError("pixel_values are required for the first generation step")
            vision, visual_mask = self.encoder(pixel_values, image_sizes, content_sizes)
            embeddings = torch.cat((vision, text_embeddings), dim=1)
            text_offset = vision.shape[1]
            if past_key_values is not None:
                past_key_values._nemotron_visual_mask = visual_mask
        else:
            embeddings = text_embeddings
            visual_mask = past_key_values._nemotron_visual_mask
        if attention_mask is None:
            text_length = past_length + input_ids.shape[1] - visual_mask.shape[1]
            if past_length == 0:
                text_length = input_ids.shape[1]
            attention_mask = torch.ones(input_ids.shape[0], text_length, device=input_ids.device, dtype=torch.bool)
        mask = torch.cat((visual_mask, attention_mask.bool()), dim=1)
        if mask.shape[1] > self.config.max_sequence_length:
            raise ValueError("Visual prefix plus text exceeds the model context limit")
        hidden = self.decoder(embeddings, past_key_values if use_cache else None, mask)
        logits = self.lm_head(hidden[:, text_offset:])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values if use_cache else None)
        return output if return_dict else (output.logits, output.past_key_values)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, pixel_values=None,
                                      image_sizes=None, content_sizes=None, **kwargs):
        del kwargs
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            pixel_values = None
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "pixel_values": pixel_values,
                "image_sizes": image_sizes, "content_sizes": content_sizes, "use_cache": True}
