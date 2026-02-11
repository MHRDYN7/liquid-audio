from __future__ import annotations

import json
import math
from collections.abc import Generator
from dataclasses import asdict, dataclass
from enum import IntEnum, auto
from pathlib import Path
from typing import ClassVar, Literal, Self, TypedDict

import torch
from einops import rearrange
from torch import nn
from transformers import Lfm2Config, Lfm2Model
from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache


class LFMModality(IntEnum):
    TEXT = auto()
    AUDIO_IN = auto()
    AUDIO_OUT = auto()


def mel2emb_len(l):
    return -(l // -8)


@dataclass(kw_only=True)
class DepthformerConfig:
    layers: int
    dim: int
    tie: bool


@dataclass(kw_only=True)
class ConformerEncoderConfig:
    feat_in: int
    feat_out: int
    n_layers: int
    d_model: int
    subsampling: str
    subsampling_factor: int
    subsampling_conv_channels: int
    causal_downsampling: bool
    reduction: str | None
    reduction_position: int | None
    reduction_factor: int
    ff_expansion_factor: int
    self_attention_model: str
    n_heads: int
    att_context_size: list[list[int]]
    xscaling: bool
    untie_biases: bool
    pos_emb_max_len: int
    conv_kernel_size: int
    conv_norm_type: str
    conv_context_size: list[int] | None
    dropout: float
    dropout_pre_encoder: float
    dropout_emb: float
    dropout_att: float


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: list[int],
        bias: bool = True,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [in_channels, *hidden_dim, out_channels]
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(channels[0]))
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            if i != (len(channels) - 2):
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SharedEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int = 65_536,
        embed_init_scale: float = 1.0,
        norm_eps: float = 0.00001,
        *,
        tie_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        std = embed_init_scale / math.sqrt(dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        self.embedding_norm = RMSNorm(dim, eps=norm_eps)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        if tie_embedding:
            self.to_logits.weight = self.embedding.weight
        else:
            std = embed_init_scale / math.sqrt(dim)
            nn.init.normal_(self.to_logits.weight, mean=0.0, std=std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.to_logits(self.embedding_norm(embeddings))


class MHA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = True,
        out_bias: bool = True,
        out_init_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim, bias=out_bias)
        std = out_init_scale / math.sqrt(dim)
        nn.init.normal_(self.o.weight, std=std)
        if out_bias:
            nn.init.zeros_(self.o.bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = [y.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2) for y in (q, k, v)]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = attn.softmax(-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.o(x)


class StandardBlock(nn.Module):
    def __init__(self, mha: MHA, out_init_scale: float = 1.0):
        super().__init__()
        self.mha = mha
        self.norm1 = RMSNorm(mha.dim)
        self.norm2 = RMSNorm(mha.dim)
        self.ff = nn.Sequential(
            nn.Linear(mha.dim, mha.dim * 4),
            nn.GELU(),
            nn.Linear(mha.dim * 4, mha.dim),
        )
        std = out_init_scale / math.sqrt(mha.dim)
        nn.init.normal_(self.ff[-1].weight, std=std)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.mha(self.norm1(x), attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


class RawLMBackbone(nn.Module):
    def __init__(self, layers: list[StandardBlock], has_embedding: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = self.layers[0].mha.dim
        self.has_embedding = has_embedding

    def forward(self, x: torch.Tensor, cache: list | None = None) -> torch.Tensor:
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, None)
        return x

    def forward_cached(self, x: torch.Tensor, cache: list | None = None) -> tuple[torch.Tensor, list]:
        if cache is None:
            cache = [None] * len(self.layers)
        cache_out = []
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, None)
            cache_out.append(None)
        return x, cache_out


class ConformerEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = ConformerEncoderConfig(**kwargs)
        self.cfg = cfg
        self.d_model = cfg.d_model
        self._feat_in = cfg.feat_in
        self._feat_out = cfg.feat_out if cfg.feat_out > 0 else cfg.d_model

        d_ff = cfg.d_model * cfg.ff_expansion_factor
        self.xscale = math.sqrt(cfg.d_model) if cfg.xscaling else None

        if cfg.subsampling and cfg.subsampling_factor > 1:
            self.pre_encode = nn.Linear(cfg.feat_in, cfg.d_model)
        else:
            self.pre_encode = nn.Linear(cfg.feat_in, cfg.d_model)

        self.pos_enc = None
        if cfg.self_attention_model == "rel_pos":
            self.pos_enc = nn.Parameter(torch.randn(1, cfg.pos_emb_max_len, cfg.d_model) * 0.02)
        elif cfg.self_attention_model == "abs_pos":
            self.pos_enc = nn.Embedding(cfg.pos_emb_max_len, cfg.d_model)
            nn.init.normal_(self.pos_enc.weight, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            layer = ConformerLayer(
                d_model=cfg.d_model,
                d_ff=d_ff,
                n_heads=cfg.n_heads,
                conv_kernel_size=cfg.conv_kernel_size,
                conv_norm_type=cfg.conv_norm_type,
                conv_context_size=cfg.conv_context_size,
                dropout=cfg.dropout,
                dropout_att=cfg.dropout_att,
            )
            self.layers.append(layer)

        if cfg.feat_out > 0 and cfg.feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, cfg.feat_out)
            self._feat_out = cfg.feat_out
        else:
            self.out_proj = None
            self._feat_out = cfg.d_model

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = audio_signal.shape
        if self.xscale is not None:
            audio_signal = audio_signal * self.xscale

        if self.cfg.subsampling and self.cfg.subsampling_factor > 1:
            audio_signal = self.pre_encode(audio_signal)
            length = length // self.cfg.subsampling_factor
        else:
            audio_signal = self.pre_encode(audio_signal)

        if self.pos_enc is not None:
            if self.cfg.self_attention_model == "abs_pos":
                pos_idx = torch.arange(T, device=audio_signal.device)
                pos_emb = self.pos_enc(pos_idx)
                audio_signal = audio_signal + pos_emb
            else:
                pos_emb = self.pos_enc[:, :T, :]
                audio_signal = audio_signal + pos_emb

        pad_mask = torch.arange(T, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)
        pad_mask = ~pad_mask

        for layer in self.layers:
            audio_signal = layer(audio_signal, pad_mask)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        return audio_signal, length


class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        conv_kernel_size: int,
        conv_norm_type: str,
        conv_context_size: list[int],
        dropout: float,
        dropout_att: float,
    ):
        super().__init__()
        self.norm_mha = RMSNorm(d_model)
        self.mha = MHA(d_model, n_heads, out_init_scale=1.0 / math.sqrt(2))
        self.dropout_mha = nn.Dropout(dropout_att)

        self.norm_conv = RMSNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=conv_context_size[0], bias=False),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.dropout_conv = nn.Dropout(dropout)

        self.norm_ff = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        residual = x
        x = self.norm_mha(x)
        x = self.mha(x, pad_mask)
        x = self.dropout_mha(x)
        x = residual + x

        residual = x
        x = self.norm_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x = self.dropout_conv(x)
        x = residual + x

        residual = x
        x = self.norm_ff(x)
        x = self.ff(x)
        x = residual + x

        return x


@dataclass(kw_only=True)
class LFM2AudioConfig:
    architectures: list[str]
    codebooks: int
    tie_audio_embeddings: bool
    semantic_codebook_factor: float
    codebook_weight: Literal["log", "linear"]
    interleaved_n_text: int
    interleaved_n_audio: int
    preprocessor: dict
    encoder: ConformerEncoderConfig
    lfm: Lfm2Config
    depthformer: DepthformerConfig


class LFM2AudioModel(nn.Module):
    audio_vocab_size: ClassVar[int] = 2048 + 1

    def __init__(self, conf: LFM2AudioConfig):
        super().__init__()
        self.conf = conf
        self.codebooks = conf.codebooks

        self.lfm = Lfm2Model(conf.lfm)

        self.conformer = ConformerEncoder(**asdict(conf.encoder))
        self.audio_adapter = MLP(self.conformer._feat_out, self.lfm.config.hidden_size, [self.lfm.config.hidden_size])

        self.depthformer_layers = conf.depthformer.layers
        self.depthformer_dim = conf.depthformer.dim
        self.depthformer_tie = conf.depthformer.tie
        self.audio_embedding = SharedEmbedding(
            dim=self.lfm.config.hidden_size,
            vocab_size=self.audio_vocab_size * self.conf.codebooks,
            embed_init_scale=1.0,
            norm_eps=0.00001,
            tie_embedding=conf.tie_audio_embeddings,
        )

        self.register_buffer("codebook_offsets", torch.arange(self.conf.codebooks) * self.audio_vocab_size)

        if conf.codebook_weight == "log":
            weights = (torch.linspace(1, 0, self.codebooks) * math.log(conf.semantic_codebook_factor)).exp()
        else:
            weights = torch.ones((self.codebooks,))
            weights[0] *= conf.semantic_codebook_factor
        self.register_buffer("audio_loss_weights", weights)

        scale = 1 / math.sqrt(2 * self.depthformer_layers)

        layers = [
            StandardBlock(MHA(self.depthformer_dim, out_init_scale=scale), out_init_scale=scale)
            for _ in range(self.depthformer_layers)
        ]
        self.depthformer = RawLMBackbone(layers, has_embedding=False)

        self.depth_linear = nn.Linear(self.lfm.config.hidden_size, self.depthformer_dim * self.codebooks)
        self.depth_embeddings = nn.ModuleList(
            [
                SharedEmbedding(
                    dim=self.depthformer_dim,
                    vocab_size=self.audio_vocab_size,
                    tie_embedding=self.depthformer_tie,
                )
                for _ in range(self.codebooks)
            ]
        )

    @torch.no_grad()
    def generate_sequential(
        self,
        *,
        text: torch.Tensor,
        audio_in: torch.Tensor,
        audio_in_lens: torch.Tensor,
        audio_out: torch.Tensor,
        modality_flag: torch.Tensor,
        max_new_tokens: int = 20,
        text_temperature: float | None = None,
        text_top_k: int | None = None,
        audio_temperature: float | None = None,
        audio_top_k: int | None = None,
    ) -> Generator[torch.Tensor, None, None]:
        in_emb = self._prefill(
            text=text,
            audio_in=audio_in,
            audio_in_lens=audio_in_lens,
            audio_out=audio_out,
            modality_flag=modality_flag,
        )

        current_modality: LFMModality = LFMModality.TEXT
        cache: Lfm2HybridConvCache | None = None

        for _ in range(max_new_tokens):
            lfm_out = self.lfm(
                inputs_embeds=in_emb,
                past_key_values=cache,
                use_cache=True,
            )
            output_embeddings = lfm_out.last_hidden_state
            cache = lfm_out.past_key_values

            if current_modality == LFMModality.TEXT:
                text_logits = nn.functional.linear(output_embeddings[0, -1], self.lfm.embed_tokens.weight)
                next_token = self._sample_text_token(text_logits, temperature=text_temperature, top_k=text_top_k)
                yield next_token

                if next_token == 128:
                    current_modality = LFMModality.AUDIO_OUT
                if next_token == 7:
                    break

                in_emb = self.lfm.embed_tokens(next_token)[None, :]

            elif current_modality == LFMModality.AUDIO_OUT:
                next_token = self._sample_audio_frame(
                    output_embeddings[0, -1],
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                )

                if next_token[0] == 2048:
                    next_token[:] = 2048
                    current_modality = LFMModality.TEXT

                yield next_token
                in_emb = self.audio_embedding(next_token + self.codebook_offsets).sum(0)[None, None, :]

    @torch.no_grad()
    def generate_interleaved(
        self,
        *,
        text: torch.Tensor,
        audio_in: torch.Tensor,
        audio_in_lens: torch.Tensor,
        audio_out: torch.Tensor,
        modality_flag: torch.Tensor,
        max_new_tokens: int = 20,
        text_temperature: float | None = None,
        text_top_k: int | None = None,
        audio_temperature: float | None = None,
        audio_top_k: int | None = None,
    ) -> Generator[torch.Tensor, None, None]:
        in_emb = self._prefill(
            text=text,
            audio_in=audio_in,
            audio_in_lens=audio_in_lens,
            audio_out=audio_out,
            modality_flag=modality_flag,
        )

        current_modality: LFMModality = LFMModality.TEXT
        modality_left: int = self.conf.interleaved_n_text
        cache: Lfm2HybridConvCache | None = None

        text_done: bool = False

        for _ in range(max_new_tokens):
            modality_left -= 1
            lfm_out = self.lfm(
                inputs_embeds=in_emb,
                past_key_values=cache,
                use_cache=True,
            )
            output_embeddings = lfm_out.last_hidden_state
            cache = lfm_out.past_key_values

            if current_modality == LFMModality.TEXT:
                text_logits = nn.functional.linear(output_embeddings[0, -1], self.lfm.embed_tokens.weight)
                next_token = self._sample_text_token(text_logits, temperature=text_temperature, top_k=text_top_k)

                if next_token == 7:
                    break

                yield next_token

                if next_token == 130:
                    text_done = True
                if not modality_left or text_done:
                    current_modality = LFMModality.AUDIO_OUT
                    modality_left = self.conf.interleaved_n_audio

                in_emb = self.lfm.embed_tokens(next_token)[None, :]

            elif current_modality == LFMModality.AUDIO_OUT:
                next_token = self._sample_audio_frame(
                    output_embeddings[0, -1],
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                )

                if not modality_left and not text_done:
                    current_modality = LFMModality.TEXT
                    modality_left = self.conf.interleaved_n_text

                if next_token[0] == 2048:
                    next_token[:] = 2048
                    current_modality = LFMModality.TEXT

                yield next_token
                in_emb = self.audio_embedding(next_token + self.codebook_offsets).sum(0)[None, None, :]

    def _prefill(
        self,
        *,
        text: torch.Tensor,
        audio_in: torch.Tensor,
        audio_in_lens: torch.Tensor,
        audio_out: torch.Tensor,
        modality_flag: torch.Tensor,
    ) -> torch.Tensor:
        assert len(text.shape) == 2
        assert len(audio_in.shape) == 2
        assert len(audio_in_lens.shape) == 1
        assert len(audio_out.shape) == 2
        assert len(modality_flag.shape) == 2
        assert text.shape[0] == 1
        assert audio_in.shape[0] == 128
        assert audio_out.shape[0] >= self.codebooks
        assert modality_flag.shape[0] == 1

        text_emb = self.lfm.embed_tokens(text[0])
        text_mask = modality_flag == LFMModality.TEXT

        audio_in_list = audio_in.mT.split(audio_in_lens.tolist())
        if audio_in_list:
            padded_audio_in = nn.utils.rnn.pad_sequence(audio_in_list, batch_first=True)
        else:
            padded_audio_in = text_emb.new_empty((0, 8 + 1, 128))

        audio_enc, audio_in_len = self.conformer(padded_audio_in.mT, audio_in_lens)

        len_mask = torch.arange(audio_enc.shape[-1], device=audio_enc.device).unsqueeze(0) < audio_in_len.unsqueeze(1)
        audio_enc_concatenated = audio_enc.mT[len_mask]

        audio_in_emb = self.audio_adapter(audio_enc_concatenated)
        audio_in_mask = modality_flag == LFMModality.AUDIO_IN

        offset_audio_tokens = audio_out[: self.codebooks] + self.codebook_offsets.unsqueeze(1)
        audio_out_emb = self.audio_embedding(offset_audio_tokens).sum(0)
        audio_out_mask = modality_flag == LFMModality.AUDIO_OUT

        B, L, D = *modality_flag.shape, self.lfm.config.hidden_size
        in_emb = text_emb.new_empty((B, L, D))

        in_emb[text_mask] = text_emb
        in_emb[audio_in_mask] = audio_in_emb
        in_emb[audio_out_mask] = audio_out_emb

        return in_emb

    def _sample_text_token(
        self, logits: torch.Tensor, *, temperature: float | None = None, top_k: int | None = None
    ) -> torch.Tensor:
        greedy = temperature is None or temperature <= 0 or top_k == 1
        if greedy:
            next_token = logits.argmax(keepdim=True)
        else:
            assert isinstance(temperature, float) and temperature > 0
            logits /= temperature
            if top_k is not None:
                min_score = torch.topk(logits, top_k).values[-1]
                to_remove = logits < min_score
                logits = torch.masked_fill(logits, to_remove, -float("inf"))
            probs = logits.softmax(0)
            next_token = torch.multinomial(probs, 1)
        return next_token

    def _sample_audio_frame(
        self,
        embedding: torch.Tensor,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        greedy = temperature is None or temperature <= 0 or top_k == 1
        depthformer_in = rearrange(self.depth_linear(embedding), "(C D) -> C D", C=self.codebooks, D=self.depthformer_dim)
        depthformer_token = torch.zeros_like(depthformer_in[0])
        cache = None

        out_tokens: list[torch.Tensor] = []
        for i in range(self.codebooks):
            cur_depthformer_input = depthformer_in[i] + depthformer_token
            depthformer_out, cache = self.depthformer.forward_cached(cur_depthformer_input[None, None, :], cache)
            depthformer_logits = self.depth_embeddings[i].get_logits(depthformer_out.squeeze())

            if greedy:
                next_token = depthformer_logits.argmax(keepdim=True)
            else:
                assert isinstance(temperature, float) and temperature > 0
                depthformer_logits /= temperature
                if top_k is not None:
                    min_score = torch.topk(depthformer_logits, top_k).values[-1]
                    to_remove = depthformer_logits < min_score
                    depthformer_logits = torch.masked_fill(depthformer_logits, to_remove, -float("inf"))
                probs = depthformer_logits.softmax(0)
                next_token = torch.multinomial(probs, 1)

            out_tokens.append(next_token)
            depthformer_token = self.depth_embeddings[i](next_token).squeeze()

        return torch.cat(out_tokens)
