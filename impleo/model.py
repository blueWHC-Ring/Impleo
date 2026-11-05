from __future__ import annotations

import torch
import torch.nn as nn


class ExprValueEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        return self.net(x.unsqueeze(-1))


class TargetGeneCrossAttentionDecoder(nn.Module):
    """
    Target-gene query cross-attention decoder.

    Each output gene embedding acts as a query over encoded panel tokens.
    Chunked decoding keeps memory manageable for large output vocabularies.
    """
    def __init__(
        self,
        num_output_genes: int,
        d_model: int,
        nhead: int = 8,
        dropout: float = 0.1,
        query_chunk_size: int = 1024,
        use_concat_query: bool = True,
    ):
        super().__init__()
        self.out_gene_emb = nn.Embedding(num_output_genes, d_model)
        self.query_ln = nn.LayerNorm(d_model)
        self.memory_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_ln = nn.LayerNorm(d_model)
        self.use_concat_query = bool(use_concat_query)
        self.query_chunk_size = max(1, int(query_chunk_size))

        pred_in_dim = d_model * 2 if self.use_concat_query else d_model
        self.mlp = nn.Sequential(
            nn.Linear(pred_in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _decode_chunk(
        self,
        query_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        # query_ids: [B, Gc], memory: [B, L, D], memory_mask: [B, L]
        q = self.out_gene_emb(query_ids)                                   # [B, Gc, D]
        q_in = self.query_ln(q)
        mem = self.memory_ln(memory)
        z, _ = self.cross_attn(
            query=q_in,
            key=mem,
            value=mem,
            key_padding_mask=(memory_mask == 0),
            need_weights=False,
        )
        z = self.post_attn_ln(z + q)                                       # residual keeps gene identity
        pred_in = torch.cat([z, q], dim=-1) if self.use_concat_query else z
        return self.mlp(pred_in).squeeze(-1)                               # [B, Gc]

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        target_gene_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # memory: [B, L, D], memory_mask: [B, L]
        bsz = memory.shape[0]
        device = memory.device

        if target_gene_ids is None:
            target_gene_ids = torch.arange(
                self.out_gene_emb.num_embeddings,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(bsz, -1)                                 # [B, G]
        elif target_gene_ids.dim() == 1:
            target_gene_ids = target_gene_ids.unsqueeze(0).expand(bsz, -1)

        chunks = torch.split(target_gene_ids, self.query_chunk_size, dim=1)
        preds = [self._decode_chunk(chunk, memory, memory_mask) for chunk in chunks]
        return torch.cat(preds, dim=1)


class PanelCompletionModel(nn.Module):
    def __init__(
        self,
        num_input_genes: int,
        num_output_genes: int,
        num_datasets: int,
        num_modalities: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_dataset_emb: bool = False,
        use_modality_emb: bool = True,
        decoder_type: str = "cross_attn",
        decoder_query_chunk_size: int = 1024,
        decoder_use_concat_query: bool = True,
    ):
        super().__init__()
        self.use_dataset_emb = bool(use_dataset_emb)
        self.use_modality_emb = bool(use_modality_emb)
        self.decoder_type = str(decoder_type)

        self.gene_emb = nn.Embedding(num_input_genes + 1, d_model, padding_idx=0)
        self.dataset_emb = nn.Embedding(max(1, num_datasets), d_model) if self.use_dataset_emb else None
        self.modality_emb = nn.Embedding(max(1, num_modalities), d_model) if self.use_modality_emb else None
        self.value_encoder = ExprValueEncoder(d_model)
        self.token_ln = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        if self.decoder_type == "cross_attn":
            self.decoder = TargetGeneCrossAttentionDecoder(
                num_output_genes=num_output_genes,
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                query_chunk_size=decoder_query_chunk_size,
                use_concat_query=decoder_use_concat_query,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def encode_panel(
        self,
        panel_gene_ids: torch.Tensor,
        panel_expr: torch.Tensor,
        panel_mask: torch.Tensor,
        dataset_ids: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gene_tok = self.gene_emb(panel_gene_ids)
        value_tok = self.value_encoder(panel_expr)
        x = gene_tok + value_tok

        if self.modality_emb is not None and modality_ids is not None:
            x = x + self.modality_emb(modality_ids).unsqueeze(1)
        if self.dataset_emb is not None and dataset_ids is not None:
            x = x + self.dataset_emb(dataset_ids).unsqueeze(1)

        x = self.token_ln(x)
        key_padding_mask = panel_mask == 0
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

    def forward(
        self,
        panel_gene_ids: torch.Tensor,
        panel_expr: torch.Tensor,
        panel_mask: torch.Tensor,
        dataset_ids: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        target_gene_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode_panel(
            panel_gene_ids=panel_gene_ids,
            panel_expr=panel_expr,
            panel_mask=panel_mask,
            dataset_ids=dataset_ids,
            modality_ids=modality_ids,
        )

        return self.decoder(memory=memory, memory_mask=panel_mask, target_gene_ids=target_gene_ids)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom
