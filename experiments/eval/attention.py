import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaAttention

def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,

) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    ### PAI's modification
    if hasattr(self, "use_attn"):
        use_attn = self.use_attn
        img_start_idx = self.img_start_idx
        img_end_idx = self.img_end_idx
        question_start_idx = img_end_idx
        question_end_idx = question_start_idx + self.question_len
        context_start_idx = question_end_idx + self.prompt_len
        context_end_idx = context_start_idx + self.context_len
    else:
        use_attn = False

    if hasattr(self, "use_cfg"):
        use_cfg = self.use_cfg
    else:
        use_cfg = False

    if use_attn:
        alpha_img = attn_weights[:, :, -1, img_start_idx:img_end_idx].mean(dim=-1, keepdim=False)
        alpha_context = attn_weights[:, :, -1, context_start_idx:context_end_idx].mean(dim=-1, keepdim=False)
        # print('attention', alpha_img.mean(), alpha_context.mean())
        alpha_ori = abs(alpha_img / alpha_context)
        # print(alpha_ori)
        # alpha = torch.clamp(alpha, min=0.9, max=1.3)
        alpha_img = torch.clamp(alpha_ori, min=1.1, max=1.3)
        alpha_context = torch.clamp(alpha_ori, min=0.95, max=1.0)

        attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
            attn_weights[:, :, -1, img_start_idx:img_end_idx] / alpha_img[:,:,None])
        attn_weights[:, :, -1, context_start_idx:context_end_idx] = (
            attn_weights[:, :, -1, context_start_idx:context_end_idx] / alpha_context[:,:,None])

    

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_new_mlp(self, x):
    if self.config.pretraining_tp > 1:
        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        if self.use_mlp:
            temp = self.act_fn(self.gate_proj(x))
            self.zero_indices = torch.randperm(temp.shape[-1])[:int(temp.shape[-1]*0.3)]
            temp[-1][-1][self.zero_indices] = 0
            down_proj = self.down_proj(temp * self.up_proj(x))
        else:
            temp = self.act_fn(self.gate_proj(x))
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


def llama_modify(model, start_layer, end_layer, use_attn, alpha, use_mlp,
                 img_start_idx, img_end_idx, question_len, prompt_len, context_len, ret_sim):
    modify_layers = list(range(start_layer, end_layer))
    for i in modify_layers:
        model.layers[i].self_attn.use_attn = use_attn
        model.layers[i].self_attn.alpha = alpha
        model.layers[i].self_attn.img_start_idx = img_start_idx
        model.layers[i].self_attn.img_end_idx = img_end_idx
        model.layers[i].self_attn.question_len = question_len
        model.layers[i].self_attn.prompt_len = prompt_len
        model.layers[i].self_attn.context_len = context_len
        model.layers[i].self_attn.ret_sim = ret_sim
        model.layers[i].self_attn.forward = types.MethodType(llama_new_forward, model.layers[i].self_attn)
        if i == 31:
            model.layers[i].mlp.forward = types.MethodType(llama_new_mlp, model.layers[i].mlp)
            model.layers[i].mlp.use_mlp = use_mlp
            model.layers[i].mlp.alpha = alpha
            