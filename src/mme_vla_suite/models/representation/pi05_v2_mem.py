from __future__ import annotations

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

from mme_vla_suite.models.representation.utils import kernel_init


def _get_cfg(cfg: Any, key: str, default: Any) -> Any:
    return getattr(cfg, key, default) if cfg is not None else default


class Pi05V2MemoryEncoder(nnx.Module):
    """Encode recurrent history features into one action-expert-width latent."""

    def __init__(
        self,
        config,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike = jnp.float32,
    ):
        self.cfg = config
        mem_cfg = _get_cfg(config, "pi05_v2_memory", None)
        self.history_deltas = tuple(_get_cfg(mem_cfg, "history_deltas", [-12, -4, -1]))
        self.expert_dim = _get_cfg(mem_cfg, "expert_dim", config.memory_token_dim)
        self.ff_dim = _get_cfg(mem_cfg, "ff_dim", self.expert_dim * 2)
        self.use_pos_emb = _get_cfg(config, "use_pos_emb", True)
        self.use_state_emb = _get_cfg(config, "use_state_emb", True)

        img_dim = config.memory_feature.img.input_dim
        pos_dim = config.memory_feature.pos.input_dim
        state_dim = config.memory_feature.state.input_dim
        num_slots = max(len(self.history_deltas), 1)

        self.image_proj = nnx.Linear(
            img_dim,
            self.expert_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        if self.use_pos_emb:
            self.pos_proj = nnx.Linear(
                pos_dim,
                self.expert_dim,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
        if self.use_state_emb:
            self.state_proj = nnx.Linear(
                state_dim,
                self.expert_dim,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )

        self.slot_emb = nnx.Param(
            jax.random.normal(rngs.params(), (num_slots, self.expert_dim), dtype=dtype) * 0.02
        )
        self.delta_proj = nnx.Linear(
            1,
            self.expert_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.pre_norm = nnx.LayerNorm(self.expert_dim, rngs=rngs, dtype=dtype)
        self.ff_in = nnx.Linear(
            self.expert_dim,
            self.ff_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.ff_out = nnx.Linear(
            self.ff_dim,
            self.expert_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.out_norm = nnx.LayerNorm(self.expert_dim, rngs=rngs, dtype=dtype)

    def _select_delta_steps(self, values, recur_mask):
        batch, max_steps = recur_mask.shape
        deltas = jnp.asarray(self.history_deltas, dtype=jnp.int32)
        valid_count = jnp.sum(recur_mask.astype(jnp.int32), axis=1)
        raw_idx = valid_count[:, None] + deltas[None, :]
        select_mask = raw_idx >= 0
        select_idx = jnp.clip(raw_idx, 0, max_steps - 1)
        gather_idx = select_idx.reshape((batch,) + (len(self.history_deltas),) + (1,) * (values.ndim - 2))
        gather_idx = jnp.broadcast_to(gather_idx, (batch, len(self.history_deltas)) + values.shape[2:])
        selected = jnp.take_along_axis(values, gather_idx, axis=1)
        selected_mask = select_mask & jnp.take_along_axis(recur_mask, select_idx, axis=1)
        return selected, selected_mask

    def __call__(
        self,
        recur_image_emb: at.Float[at.Array, "b t v p d1"],
        recur_mask: at.Bool[at.Array, "b t"],
        recur_pos_emb: at.Float[at.Array, "b t v p d2"] | None = None,
        recur_state_emb: at.Float[at.Array, "b t d3"] | None = None,
    ):
        image_steps, step_mask = self._select_delta_steps(recur_image_emb, recur_mask)
        image_steps = jnp.mean(image_steps, axis=(2, 3))
        hidden = self.image_proj(image_steps)

        if self.use_pos_emb and recur_pos_emb is not None:
            pos_steps, _ = self._select_delta_steps(recur_pos_emb, recur_mask)
            hidden = hidden + self.pos_proj(jnp.mean(pos_steps, axis=(2, 3)))

        if self.use_state_emb and recur_state_emb is not None:
            state_steps, _ = self._select_delta_steps(recur_state_emb, recur_mask)
            hidden = hidden + self.state_proj(state_steps)

        slot_emb = self.slot_emb.value[None, :, :]
        deltas = jnp.asarray(self.history_deltas, dtype=hidden.dtype)[:, None]
        delta_emb = self.delta_proj(deltas / jnp.maximum(jnp.abs(deltas).max(), 1.0))[None, :, :]
        hidden = self.pre_norm(hidden + slot_emb + delta_emb)
        hidden = hidden + self.ff_out(nnx.silu(self.ff_in(hidden)))

        weights = step_mask.astype(hidden.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1.0)
        latent = jnp.sum(hidden * weights[:, :, None], axis=1) / denom
        latent = self.out_norm(latent)
        stats = {
            "pi05_v2/history_valid_slots": jnp.mean(jnp.sum(step_mask.astype(jnp.float32), axis=1)),
        }
        return latent, stats


class DynamicLoRABasisHyperNet(nnx.Module):
    """Generate per-layer, per-sample LoRA deltas for Gemma action stream q/v."""

    def __init__(
        self,
        config,
        action_expert_config,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike = jnp.float32,
    ):
        dyn_cfg = config.dynamic_lora
        self.depth = action_expert_config.depth
        self.width = action_expert_config.width
        self.num_heads = action_expert_config.num_heads
        self.num_kv_heads = action_expert_config.num_kv_heads
        self.head_dim = action_expert_config.head_dim
        self.rank = dyn_cfg.rank
        self.basis_count = dyn_cfg.basis_count
        self.hidden_dim = dyn_cfg.hidden_dim
        self.target_layers = tuple(dyn_cfg.target_layers)
        self.target_modules = tuple(dyn_cfg.target_modules)
        self.scale_init = dyn_cfg.scale_init

        self.q_out_dim = self.num_heads * self.head_dim
        self.v_out_dim = self.num_kv_heads * self.head_dim
        coeff_dim = self.basis_count * max(len(self.target_modules), 1)

        self.norm = nnx.LayerNorm(self.width, rngs=rngs, dtype=dtype)
        self.fc1 = nnx.Linear(self.width, self.hidden_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.fc2 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.coeff_head = nnx.Linear(
            self.hidden_dim,
            coeff_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
        )
        self.q_A_basis = nnx.Param(
            jax.random.normal(rngs.params(), (self.basis_count, self.rank, self.width), dtype=dtype) * 0.02
        )
        self.q_B_basis = nnx.Param(
            jax.random.normal(rngs.params(), (self.basis_count, self.q_out_dim, self.rank), dtype=dtype) * 0.02
        )
        self.v_A_basis = nnx.Param(
            jax.random.normal(rngs.params(), (self.basis_count, self.rank, self.width), dtype=dtype) * 0.02
        )
        self.v_B_basis = nnx.Param(
            jax.random.normal(rngs.params(), (self.basis_count, self.v_out_dim, self.rank), dtype=dtype) * 0.02
        )
        self.scale_head = nnx.Linear(
            self.hidden_dim,
            max(len(self.target_modules), 1),
            rngs=rngs,
            dtype=dtype,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.constant(self.scale_init),
        )

    def _empty_params(self, batch_size, dtype):
        return {
            "q_A": jnp.zeros((self.depth, batch_size, self.rank, self.width), dtype=dtype),
            "q_B": jnp.zeros((self.depth, batch_size, self.q_out_dim, self.rank), dtype=dtype),
            "q_scale": jnp.zeros((self.depth, batch_size), dtype=dtype),
            "v_A": jnp.zeros((self.depth, batch_size, self.rank, self.width), dtype=dtype),
            "v_B": jnp.zeros((self.depth, batch_size, self.v_out_dim, self.rank), dtype=dtype),
            "v_scale": jnp.zeros((self.depth, batch_size), dtype=dtype),
        }

    def __call__(self, memory_latent: at.Float[at.Array, "b d"]):
        batch_size = memory_latent.shape[0]
        h = self.norm(memory_latent)
        h = nnx.silu(self.fc1(h))
        h = nnx.silu(self.fc2(h))
        coeffs = self.coeff_head(h).reshape(batch_size, max(len(self.target_modules), 1), self.basis_count)
        scales = self.scale_head(h)
        params = self._empty_params(batch_size, memory_latent.dtype)

        for module_idx, module_name in enumerate(self.target_modules):
            coeff = coeffs[:, module_idx]
            scale = scales[:, module_idx]
            if module_name == "q":
                A = jnp.einsum("bk,krd->brd", coeff, self.q_A_basis.value)
                B = jnp.einsum("bk,kor->bor", coeff, self.q_B_basis.value)
                for layer_idx in self.target_layers:
                    params["q_A"] = params["q_A"].at[layer_idx].set(A)
                    params["q_B"] = params["q_B"].at[layer_idx].set(B)
                    params["q_scale"] = params["q_scale"].at[layer_idx].set(scale)
            elif module_name == "v":
                A = jnp.einsum("bk,krd->brd", coeff, self.v_A_basis.value)
                B = jnp.einsum("bk,kor->bor", coeff, self.v_B_basis.value)
                for layer_idx in self.target_layers:
                    params["v_A"] = params["v_A"].at[layer_idx].set(A)
                    params["v_B"] = params["v_B"].at[layer_idx].set(B)
                    params["v_scale"] = params["v_scale"].at[layer_idx].set(scale)
            else:
                raise ValueError(f"Unsupported dynamic LoRA module: {module_name}")
        return params
