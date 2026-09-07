import re
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, computed_field
from typing_extensions import Self

from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig as HFGlm4MoeLiteConfig
from xtuner.v1.model.moe.moe import BalancingLossConfig, ZLossConfig
from xtuner.v1.module.attention import MLAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .deepseek_v3 import DeepSeekV3, DeepSeekV3Config


class Glm47Flash(DeepSeekV3):
    """GLM-4.7-Flash model using XTuner's native MLA/MoE implementation."""

    def to_hf_key_list(self, key: str) -> list[str]:
        if key.startswith("mtp_block."):
            match = re.match(r"mtp_block\.layers\.(\d+)\.(.+)", key)
            assert match is not None, f"Unexpected GLM-4.7-Flash MTP key: {key}"
            mtp_layer_idx = self.config.num_hidden_layers + int(match.group(1))
            key = f"layers.{mtp_layer_idx}.{match.group(2)}"
            key = key.replace(".decoder_layer.", ".")
            key = re.sub(r"layers\.(\d+)\.final_layernorm\.", r"layers.\1.shared_head.norm.", key)
        return super().to_hf_key_list(key)


class Glm47FlashConfig(DeepSeekV3Config):
    model_type: str = "glm4_moe_lite"
    vocab_size: int = 154880
    max_position_embeddings: int = 202752
    pad_token_id: int | None = 154820
    eos_token_id: int = 154820
    hf_eos_token_id: int | list[int] = Field(default_factory=lambda: [154820, 154827, 154829])
    num_hidden_layers: int = 47
    first_k_dense_replace: int = 1
    hidden_size: int = 2048
    intermediate_size: int = 10240
    rms_norm_eps: float = 1e-5
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(rope_theta=1000000.0)
    )
    hidden_act: str = "silu"
    attention: MLAConfig = MLAConfig(
        kv_lora_rank=512,
        q_lora_rank=768,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        head_dim=64,
        num_attention_heads=20,
        qkv_bias=False,
        o_bias=False,
        rms_norm_eps=1e-5,
    )
    tie_word_embeddings: bool = False
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 1536
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=1.8,
    )
    balancing_loss_cfg: BalancingLossConfig | None = None
    z_loss_cfg: ZLossConfig | None = None
    mlp_layer_types: list[Literal["dense", "sparse"]] | None = None
    rope_interleave: bool = True
    mtp_config: MTPConfig | None = None

    @computed_field
    def num_key_value_heads(self) -> int:
        return self.attention.num_attention_heads

    def build(self) -> Glm47Flash:
        return Glm47Flash(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        cfg = HFGlm4MoeLiteConfig.from_pretrained(hf_path)
        assert isinstance(cfg, HFGlm4MoeLiteConfig)

        rope_interleave = getattr(cfg, "rope_interleave", True)
        if not rope_interleave:
            raise ValueError("GLM-4.7-Flash requires interleaved rotary embeddings.")

        mlp_layer_types = getattr(cfg, "mlp_layer_types", None)
        first_k_dense_replace = getattr(cfg, "first_k_dense_replace", None)
        if first_k_dense_replace is None:
            first_k_dense_replace = 0
            for layer_type in mlp_layer_types or []:
                if layer_type != "dense":
                    break
                first_k_dense_replace += 1

        hf_eos_token_id = cfg.eos_token_id
        eos_token_id = hf_eos_token_id[0] if isinstance(hf_eos_token_id, list) else hf_eos_token_id
        num_mtp_layers = int(getattr(cfg, "num_nextn_predict_layers", 0) or 0)
        return cls(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=getattr(cfg, "pad_token_id", None),
            eos_token_id=eos_token_id,
            hf_eos_token_id=hf_eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            first_k_dense_replace=first_k_dense_replace,
            max_window_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            model_type=cfg.model_type,
            rope_parameters_cfg=RopeParametersConfig.from_hf_config(cfg),
            hidden_act=cfg.hidden_act,
            attention=MLAConfig(
                kv_lora_rank=cfg.kv_lora_rank,
                q_lora_rank=cfg.q_lora_rank,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                head_dim=cfg.qk_rope_head_dim,
                num_attention_heads=cfg.num_attention_heads,
                qkv_bias=cfg.attention_bias,
                o_bias=cfg.attention_bias,
                dropout=cfg.attention_dropout,
                rms_norm_eps=cfg.rms_norm_eps,
            ),
            tie_word_embeddings=cfg.tie_word_embeddings,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            hidden_factor=1.0,
            moe_intermediate_size=cfg.moe_intermediate_size,
            router=NoAuxRouterConfig(
                n_group=cfg.n_group,
                topk_group=cfg.topk_group,
                scoring_func="sigmoid",
                norm_topk_prob=cfg.norm_topk_prob,
                router_scaling_factor=cfg.routed_scaling_factor,
            ),
            balancing_loss_cfg=None,
            z_loss_cfg=None,
            mlp_layer_types=mlp_layer_types,
            rope_interleave=rope_interleave,
            mtp_config=MTPConfig(num_layers=num_mtp_layers, share_weights=True) if num_mtp_layers else None,
        )

    @property
    def hf_config(self) -> HFGlm4MoeLiteConfig:
        """HuggingFace configuration."""
        assert isinstance(self.router, NoAuxRouterConfig), (
            "Only support saving NoAuxRouter to HF GLM-4.7-Flash format."
        )
        attention = self.attention
        mlp_layer_types = self.mlp_layer_types or [
            "dense" if layer_idx < self.first_k_dense_replace else "sparse"
            for layer_idx in range(self.num_hidden_layers)
        ]
        cfg = HFGlm4MoeLiteConfig(
            architectures=["Glm4MoeLiteForCausalLM"],
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.hf_eos_token_id,
            num_hidden_layers=self.num_hidden_layers,
            mlp_layer_types=mlp_layer_types,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_parameters=self.rope_parameters,
            hidden_act=self.hidden_act,
            num_attention_heads=attention.num_attention_heads,
            num_key_value_heads=attention.num_attention_heads,
            kv_lora_rank=attention.kv_lora_rank,
            q_lora_rank=attention.q_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            v_head_dim=attention.v_head_dim,
            attention_bias=attention.qkv_bias or attention.o_bias,
            attention_dropout=attention.dropout,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_group=self.router.n_group,
            topk_group=self.router.topk_group,
            norm_topk_prob=self.router.norm_topk_prob,
            routed_scaling_factor=self.router.router_scaling_factor,
            rope_interleave=self.rope_interleave,
            tie_word_embeddings=self.tie_word_embeddings,
            dtype=torch.bfloat16,
        )
        cfg.first_k_dense_replace = self.first_k_dense_replace
        cfg.num_nextn_predict_layers = self.mtp_config.num_layers if self.mtp_config is not None else 0
        return cfg
