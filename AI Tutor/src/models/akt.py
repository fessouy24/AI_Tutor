from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AKTModelConfig:
    n_questions: int
    n_skills: int
    max_seq_len: int = 200
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.2
    ff_multiplier: int = 4


class AKTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_multiplier: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        causal_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x, attn_weights if return_attention else None


class AKTModel(nn.Module):
    """Readable AKT-style attention-based Knowledge Tracing model.

    The model consumes past question/skill/correctness interactions and predicts
    whether the next target question/skill will be answered correctly.
    """

    def __init__(self, cfg: AKTModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.question_emb = nn.Embedding(cfg.n_questions + 1, cfg.d_model, padding_idx=0)
        self.skill_emb = nn.Embedding(cfg.n_skills + 1, cfg.d_model, padding_idx=0)
        self.correct_emb = nn.Embedding(3, cfg.d_model, padding_idx=0)
        self.position_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.interaction_proj = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.blocks = nn.ModuleList(
            [AKTBlock(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.ff_multiplier) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        question_ids: torch.Tensor,
        skill_ids: torch.Tensor,
        correct_ids: torch.Tensor,
        target_question_ids: torch.Tensor,
        target_skill_ids: torch.Tensor,
        input_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if input_mask is None:
            input_mask = question_ids.ne(0)

        batch_size, seq_len = question_ids.shape
        position_ids = torch.arange(seq_len, device=question_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        correct_tokens = torch.where(input_mask, correct_ids.long() + 1, torch.zeros_like(correct_ids.long()))

        interaction = torch.cat(
            [
                self.question_emb(question_ids),
                self.skill_emb(skill_ids),
                self.correct_emb(correct_tokens),
            ],
            dim=-1,
        )
        x = self.interaction_proj(interaction) + self.position_emb(position_ids)
        x = x * math.sqrt(self.cfg.d_model)

        key_padding_mask = ~input_mask.bool()
        causal_mask = self._causal_mask(seq_len, question_ids.device)
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, key_padding_mask=key_padding_mask, causal_mask=causal_mask, return_attention=return_attention)
            if attn is not None:
                attentions.append(attn.detach())

        history_state = self.final_norm(x)
        target = self.target_proj(torch.cat([self.question_emb(target_question_ids), self.skill_emb(target_skill_ids)], dim=-1))
        logits = self.prediction_head(torch.cat([history_state, target], dim=-1)).squeeze(-1)
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {"logits": logits}
        if return_attention:
            out["attentions"] = attentions
        return out


class AKTONNXWrapper(nn.Module):
    def __init__(self, model: AKTModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        question_ids: torch.Tensor,
        skill_ids: torch.Tensor,
        correct_ids: torch.Tensor,
        target_question_ids: torch.Tensor,
        target_skill_ids: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            question_ids,
            skill_ids,
            correct_ids,
            target_question_ids,
            target_skill_ids,
            input_mask=input_mask.bool(),
            return_attention=False,
        )["logits"]

