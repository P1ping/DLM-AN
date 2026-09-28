from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from einops import repeat

from dlm_an.utils.common import IGNORE_ID
from dlm_an.utils.file_utils import logging


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Add Gumbel noise to logits for sampling.
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.

    Args:
        logits: (N, V) logits tensor
        temperature: temperature for noise scaling

    Returns:
        Noisy logits for sampling
    """
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        lm_input_dim: int,
        lm_output_dim: int,
        speech_vocab_size: int,
        speech_embed_dim: int,
        token_encoder: torch.nn.Module,
        lm: torch.nn.Module,
        lcs_encoder: Optional[torch.nn.Module] = None,
        duration_predictor: Optional[torch.nn.Module] = None,
        ctc_vocab_size: int = 0,
        ctc_target_key: str = "phone_token",
        ctc_loss_weight: float = 1.0,
        share_speech_embedding: bool = True,
        lcs_loss_weight: float = 0.5,
        lcs_pos_weight: float = 2.0,
        dp_loss_weight: float = 1.0,
        cfg_ratio: float = 0.2,
    ):
        super().__init__()
        self.lm_input_dim = lm_input_dim
        self.speech_vocab_size = speech_vocab_size
        self.ctc_loss_weight = ctc_loss_weight
        self.ctc_vocab_size = ctc_vocab_size
        self.ctc_target_key = ctc_target_key
        self.lcs_loss_weight = lcs_loss_weight
        self.lcs_pos_weight = lcs_pos_weight

        # 1. build speech token inputs related modules
        self.source_speech_embedding = torch.nn.Embedding(
            speech_vocab_size + 2, speech_embed_dim
        )
        self.contam_mask_id = speech_vocab_size  # For BART-style contamination
        self.diff_mask_id = speech_vocab_size + 1  # For diffusion masking
        self.token_encoder = token_encoder
        self.token_encoder_output_proj = nn.Linear(
            token_encoder.output_size(), lm_input_dim
        )
        if ctc_vocab_size > 0:
            self.ctc_proj = nn.Linear(token_encoder.output_size(), ctc_vocab_size)
        else:
            self.ctc_proj = None

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.special_embedding = torch.nn.Embedding(2, lm_input_dim)
        self.lm = lm
        # NOTE (fix): Remove the unused <eos>
        # An extra entry for <eos>
        self.lm_output_proj = nn.Linear(lm_output_dim, speech_vocab_size + 1)

        # 3. Build optional speech token language model related modules
        if share_speech_embedding:
            assert (
                speech_embed_dim == lm_input_dim
            ), "For shared speech embedding, dimensions must match."
            self.target_speech_embedding = self.source_speech_embedding
        else:
            self.target_speech_embedding = torch.nn.Embedding(
                speech_vocab_size, lm_input_dim
            )
        self.lcs_encoder = lcs_encoder
        if self.lcs_encoder is not None:
            self.lcs_proj = nn.Linear(self.lcs_encoder.output_size(), 1)
            self.lcs_criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.lcs_pos_weight), reduction="none"
            )
        else:
            self.lcs_proj = None
            self.lcs_criterion = None

        self.duration_predictor = duration_predictor
        self.dp_loss_weight = dp_loss_weight

        self.cfg_ratio = cfg_ratio
        self.frozen_modules = []

        encoder_params = sum(
            p.numel() for p in self.token_encoder.parameters() if p.requires_grad
        )
        lm_params = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            f"TransformerLM total parameters: {total_params} "
            f"= encoder: {encoder_params} + lm: {lm_params} + others: {total_params - encoder_params - lm_params}"
        )

    def encode(
        self,
        speech_tokens: torch.Tensor,
        speech_token_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.token_encoder(
            speech_tokens,
            speech_token_lengths,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_logits = self.ctc_proj(encoder_out) if self.ctc_proj is not None else None
        encoder_out = self.token_encoder_output_proj(encoder_out)
        return encoder_out, encoder_out_lens, ctc_logits

    def encode_lcs(self, src_tokens_embedded, encoded_source, src_token_lengths):
        """
        Predict LCS mask between source tokens and encoded source representations.

        Args:
            src_tokens_embedded: Embedded source tokens (B, T, D)
            encoded_source: Encoded source representations (B, T, D')
        Returns:
            lcs_logits: Logits for LCS prediction (B, T)
        """
        lcs_input = torch.cat(
            [src_tokens_embedded, encoded_source], dim=-1
        )  # (B, T, D + D')

        lcs_encoded, lcs_mask = self.lcs_encoder(
            lcs_input,
            src_token_lengths,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
        )  # (B, T, D_lcs)
        lcs_mask = lcs_mask.squeeze(1)  # (B, T)

        lcs_logits = self.lcs_proj(lcs_encoded).squeeze(-1)  # (B, T)

        return lcs_logits, lcs_mask

    def pad_unpad_sequence(
        self,
        sos_eos_emb,
        src_tokens,
        src_token_len,
        task_id_emb,
        tgt_tokens,
        tgt_token_len,
    ):
        src_tokens = unpad_sequence(src_tokens, src_token_len.cpu(), batch_first=True)
        tgt_tokens = unpad_sequence(tgt_tokens, tgt_token_len.cpu(), batch_first=True)

        lm_input = [
            torch.concat(
                [
                    sos_eos_emb.squeeze(dim=0),
                    src_tokens[i],
                    task_id_emb.squeeze(dim=0),
                    tgt_tokens[i],
                    sos_eos_emb.squeeze(dim=0),
                ],
                dim=0,
            )
            for i in range(len(src_tokens))
        ]

        lm_input_len = torch.tensor(
            [i.size(0) for i in lm_input],
            dtype=torch.int32,
            device=src_token_len.device,
        )
        lm_cond_len = src_token_len + 1
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len, lm_cond_len

    def make_position_and_mask(
        self, total_lengths: torch.Tensor, cond_lengths: torch.Tensor
    ) -> torch.Tensor:
        max_len = total_lengths.max().item()
        _p = torch.arange(
            0, max_len, 1, dtype=torch.long, device=total_lengths.device
        )  # [T]
        p = repeat(_p, "t -> b t", b=total_lengths.shape[0])  # [B, T]

        B, T = p.shape
        cond_mask = p < cond_lengths.unsqueeze(1)  # [B, T]
        total_mask = p < total_lengths.unsqueeze(1)  # [B, T]
        cond_attn_mask = cond_mask.unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
        output_attn_mask = ~cond_mask.unsqueeze(2) & total_mask.unsqueeze(
            1
        )  # [B, T, T]
        attn_mask = cond_attn_mask | output_attn_mask  # [B, T, T]

        return p, attn_mask

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            batch: Dictionary containing:
                src_tokens: (B, L) - Input speech tokens
                src_token_len: (B,) - Length of input tokens
                tgt_tokens: (B, T) - Target speech tokens
                tgt_token_len: (B,) - Length of target tokens
        """
        src_tokens = batch["src_token"].to(device)
        src_token_len = batch["src_token_len"].to(device)
        tgt_tokens = batch["tgt_token"].to(device)
        tgt_token_len = batch["tgt_token_len"].to(device)
        if self.ctc_loss_weight > 0:
            aux_tokens = batch[self.ctc_target_key].to(device)
            aux_len = batch[f"{self.ctc_target_key}_len"].to(device)
        else:
            aux_tokens = None
            aux_len = None
        if self.lcs_loss_weight > 0:
            lcs_weights = batch["src_lcs_weight"].to(device)
        else:
            lcs_weights = None
        # Absorbing (mask-only) diffusion process.
        noisy_tgt_tokens, p_mask_mask, mask_mask = self.forward_process(
            tgt_tokens, tgt_token_len
        )

        # Prepare input source tokens
        src_tokens_embedded = self.source_speech_embedding(src_tokens)
        encoded_source, encoded_source_len, ctc_logits = self.encode(
            src_tokens_embedded, src_token_len
        )

        # CFG masking
        cfg_mask = torch.rand(encoded_source.size(0), device=device) < self.cfg_ratio
        source_cond = torch.where(
            cfg_mask.view(-1, 1, 1), torch.zeros_like(encoded_source), encoded_source
        )

        # Prepare input special tokens
        sos_eos_emb = self.special_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.special_embedding.weight[self.task_id].reshape(1, 1, -1)
        # Prepare input noised target tokens
        tgt_tokens_embedded = self.target_speech_embedding(noisy_tgt_tokens)

        # Prepare LM input
        lm_input, lm_input_len, lm_cond_len, src_write_indices, tgt_write_indices = (
            self.prepare_lm_input(
                sos_eos_emb,
                source_cond,
                encoded_source_len,
                task_id_emb,
                tgt_tokens_embedded,
                tgt_token_len,
            )
        )

        # Run language model forward
        position, attn_mask = self.make_position_and_mask(lm_input_len, lm_cond_len)
        lm_output = self.lm(lm_input, position, attn_mask)
        logits = self.lm_output_proj(lm_output)

        gathered_logits = torch.gather(
            logits, 1, tgt_write_indices.unsqueeze(-1).expand(-1, -1, logits.size(-1))
        )  # (B, T_tgt, V+1)

        mask_token_ce = F.cross_entropy(
            gathered_logits[mask_mask], tgt_tokens[mask_mask], reduction="none"
        )
        mask_ce_weight = 1 / p_mask_mask[:, None].repeat(
            1, tgt_tokens.size(1)
        )  # (B, T_tgt)
        mask_token_ce = mask_token_ce * mask_ce_weight[mask_mask]
        mask_ce = mask_token_ce.sum() / tgt_token_len.sum()
        ce_loss = mask_ce
        acc = (
            (gathered_logits[mask_mask].argmax(dim=-1) == tgt_tokens[mask_mask])
            .float()
            .mean()
            if mask_mask.any()
            else gathered_logits.new_zeros(())
        )

        # =============================================================
        # CTC Projection
        # =============================================================
        # The tokenizer reserves its first token for CTC <blank>.
        if self.ctc_loss_weight > 0:
            ctc_logp = ctc_logits.log_softmax(dim=-1)
            ctc_loss = F.ctc_loss(
                ctc_logp.transpose(0, 1),  # (T, B, V_ctc)
                aux_tokens,  # (B, N_phn)
                encoded_source_len,  # (B,)
                aux_len,  # (B,)
                blank=0,
                zero_infinity=True,
            )
        else:
            ctc_loss = torch.tensor(0.0, device=device)

        # =============================================================
        # Longest Common Subsequence Prediction
        # =============================================================
        if self.lcs_encoder is not None and self.lcs_loss_weight > 0:
            lcs_logits, lcs_mask = self.encode_lcs(
                src_tokens_embedded, encoded_source, src_token_len
            )  # (B, T)
            lcs_loss = self.lcs_criterion(lcs_logits, lcs_weights) * lcs_mask  # (B, T)
            lcs_loss = torch.sum(lcs_loss.sum(dim=1) / src_token_len) / lcs_mask.size(0)
            # Compute accuracy
            lcs_predictions = (lcs_logits >= 0.5).long()
            lcs_hard_labels = (lcs_weights > 0.0).long()
            lcs_acc = torch.sum(
                (lcs_predictions == lcs_hard_labels) * lcs_mask
            ) / torch.sum(lcs_mask)
        else:
            lcs_loss = torch.tensor(0.0, device=device)
            lcs_acc = torch.tensor(0.0, device=device)

        # =============================================================
        # Duration Ratio Prediction
        # =============================================================
        if self.duration_predictor is not None and self.dp_loss_weight > 0:
            dp_input = torch.cat(
                [src_tokens_embedded, encoded_source], dim=-1
            )  # (B, T, D + D')
            duration_ratio = tgt_token_len.float() / src_token_len.float()  # (B,)
            dp_loss = self.duration_predictor.compute_loss(
                dp_input, encoded_source_len, duration_ratio
            )
        else:
            dp_loss = torch.tensor(0.0, device=device)

        loss = (
            ce_loss
            + ctc_loss * self.ctc_loss_weight
            + lcs_loss * self.lcs_loss_weight
            + dp_loss * self.dp_loss_weight
        )

        return {
            "loss": loss,
            "acc": acc,
            "ctc_loss": ctc_loss,
            "ce_loss": ce_loss,
            "lcs_loss": lcs_loss,
            "lcs_acc": lcs_acc,
            "dp_loss": dp_loss,
        }

    def forward_process(self, tokens, token_lens, eps=1e-3):
        B, T = tokens.shape
        t = torch.rand(B, device=tokens.device)

        # Absorbing process
        p_mask_mask = (1 - eps) * t + eps  # (B,)
        p_mask_ = p_mask_mask[:, None].repeat(1, T)  # (B, T)

        mask_mask = torch.rand((B, T), device=tokens.device) < p_mask_  # (B, T)
        trunc_mask = (
            torch.arange(T, device=tokens.device)[None, :].repeat(B, 1)
            < token_lens[:, None]
        )
        mask_mask = mask_mask & trunc_mask  # (B, T)

        noisy_tokens = torch.where(mask_mask, self.diff_mask_id, tokens)  # (B, T)

        return noisy_tokens, p_mask_mask, mask_mask

    def prepare_lm_target(self, src_token_len, tgt_tokens, tgt_token_len):
        """
        Output: IGNORE_ID..., [tgt_tokens], IGNORE_ID
        """
        device = tgt_tokens.device

        # Compute indices for different parts
        prefix_len = 1
        task_id_idx = prefix_len + src_token_len  # (B,)
        tgt_start_idx = task_id_idx + 1  # (B,)
        total_lengths = tgt_start_idx + tgt_token_len + 1  # (B,)

        B, T_tgt = tgt_tokens.size()
        T_tot = total_lengths.max().item()
        lm_target = torch.full((B, T_tot), IGNORE_ID, dtype=torch.long, device=device)

        # Create a grid of indices for the target sequence: [0, 1, 2, ... T_tgt-1]
        tgt_grid = torch.arange(T_tgt, device=device).unsqueeze(0)  # (1, T_tgt)

        # Calculate write positions: Start Index + Grid
        write_indices = tgt_start_idx.unsqueeze(1) + tgt_grid  # (B, T_tgt)
        safe_write_indices = write_indices.clamp(max=T_tot - 1)

        # Use scatter to place targets
        lm_target.scatter_(1, safe_write_indices, tgt_tokens)

        # Create a mask to ignore positions beyond the target tokens
        mask_scatter = torch.zeros_like(lm_target, dtype=torch.bool)
        valid_tgt_mask = tgt_grid < tgt_token_len.unsqueeze(1)  # (B, T_tgt)
        mask_scatter.scatter_(1, safe_write_indices, valid_tgt_mask)

        # Apply the mask to mask the padded part
        lm_target = lm_target.masked_fill(~mask_scatter, IGNORE_ID)

        return lm_target, valid_tgt_mask, safe_write_indices

    def prepare_lm_input(
        self,
        sos_eos_emb,
        src_tokens,
        src_token_len,
        task_id_emb,
        tgt_tokens,
        tgt_token_len,
    ):
        """
        Input: <sos/eos>, [src_tokens], <task_embed>, [tgt_tokens], <sos/eos>
        """
        B, T_src, D = src_tokens.size()
        device = src_tokens.device
        dtype = task_id_emb.dtype

        # Compute indices for different parts
        prefix_len = 1
        task_id_idx = prefix_len + src_token_len  # (B,)
        tgt_start_idx = task_id_idx + 1  # (B,)
        total_lengths = tgt_start_idx + tgt_token_len + 1  # (B,)

        # Initialize lm_input with zeros
        T_tot = total_lengths.max().item()
        lm_input = torch.zeros((B, T_tot, D), device=device, dtype=dtype)

        # Place <sos>
        lm_input[:, 0:1] = sos_eos_emb
        current_fixed_pos = 1

        # Place source tokens
        _, T_src, _ = src_tokens.shape
        lm_input[:, current_fixed_pos : current_fixed_pos + T_src] = src_tokens
        src_write_indices = (
            torch.arange(T_src, device=device).unsqueeze(0).expand(B, -1)
            + current_fixed_pos
        )

        # Place <task>
        task_idx_expanded = task_id_idx.view(-1, 1, 1).expand(-1, 1, D)  # (B, 1, 1)
        lm_input.scatter_(1, task_idx_expanded, task_id_emb.expand(B, -1, -1))

        # Calculate target token positions: Start Index + Grid
        T_tgt = tgt_tokens.size(1)
        tgt_grid = torch.arange(T_tgt, device=device).unsqueeze(0)  # (1, T_tgt)
        tgt_write_indices = tgt_start_idx.unsqueeze(1) + tgt_grid  # (B, T_tgt)
        tgt_write_indices = tgt_write_indices.clamp(max=T_tot - 1)

        # Place target tokens
        tgt_indices_expanded = tgt_write_indices.unsqueeze(-1).expand(-1, -1, D)
        lm_input.scatter_(1, tgt_indices_expanded, tgt_tokens)

        # Place final <sos/eos>
        eos_idx = total_lengths - 1  # (B,)
        eos_idx_expanded = eos_idx.view(-1, 1, 1).expand(-1, 1, D)  # (B, 1, 1)
        lm_input.scatter_(1, eos_idx_expanded, sos_eos_emb.expand(B, -1, -1))

        # Mask the padded part
        tot_grid = (
            torch.arange(T_tot, device=device).unsqueeze(0).repeat(B, 1)
        )  # (B, T_tot)
        tot_mask = tot_grid < total_lengths.unsqueeze(1)  # (B, T_tot)
        lm_input = lm_input.masked_fill(~tot_mask.unsqueeze(-1), IGNORE_ID)

        lm_input_len = total_lengths
        lm_cond_len = task_id_idx

        return lm_input, lm_input_len, lm_cond_len, src_write_indices, tgt_write_indices

    @torch.inference_mode()
    def inference(
        self,
        src_tokens: torch.Tensor,
        src_token_len: torch.Tensor,
        length_ratio: float = None,
        n_timesteps: int = 32,
        cfg_scale: float = 2.0,
        temperature: float = 0.0,
        alg: str = "greedy",
        eb_gamma: float = 8.0,
        eps: float = 1e-5,
        reuse_proportion: float = 0.0,
        reuse_threshold: float = 1.0,
        **kwargs,
    ) -> Generator[int, None, None]:
        """
        Discrete diffusion inference for speech token generation.
        Reference: https://github.com/ML-GSAI/SMDM/blob/583aa4716d17728dbb825aec6c24a121164d616a/eval/gen_model_answer.py

        Args:
            src_tokens: Source speech tokens (B, L)
            src_token_len: Length of source tokens (B,)
            length_ratio: Target/source token length ratio; predicted if omitted
            n_timesteps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale (0 = no CFG)
            temperature: Temperature for Gumbel noise
            alg: Sampling algorithm ('origin', 'greedy', or 'eb_greedy')
                For 'eb_greedy' (entropy-bounded greedy), see reference: https://arxiv.org/abs/2505.24857
            eps: Small epsilon for numerical stability
            eb_gamma: float = 5.0,
                Gamma value for entropy-bounded greedy sampling
            reuse_proportion: Proportion of tokens to reuse based on LCS
            reuse_threshold: Threshold for token reuse based on LCS

        Yields:
            Generated token IDs one by one
        """
        device = src_tokens.device
        batch_size = src_tokens.size(0)
        assert batch_size == 1, "Batch size must be 1 for inference"
        assert alg in [
            "origin",
            "greedy",
            "eb_greedy",
        ], f"Invalid sampling algorithm: {alg}"

        # 1. Embed and encode source tokens
        src_tokens_embedded = self.source_speech_embedding(src_tokens)
        encoded_source, encoded_source_len, ctc_logits = self.encode(
            src_tokens_embedded, src_token_len
        )
        reuse_mask = None
        if self.lcs_encoder is not None:
            lcs_logits, lcs_mask = self.encode_lcs(
                src_tokens_embedded, encoded_source, src_token_len
            )  # (B, T)
            lcs_probs = torch.sigmoid(lcs_logits) * lcs_mask  # (B, T)
            if reuse_proportion > 0.0:
                # Sort the probs and select top-k for reuse
                k = int(reuse_proportion * lcs_mask.sum().item())
                if k > 0:
                    _, topk_indices = torch.topk(lcs_probs, k, dim=1)
                    reuse_mask = torch.zeros_like(lcs_mask).bool()
                    reuse_mask[0, topk_indices[0]] = True  # (B, T)
            elif reuse_threshold < 1.0:
                reuse_mask = (lcs_probs >= reuse_threshold) & lcs_mask  # (B, T)
                if reuse_mask.sum().item() == 0:
                    reuse_mask = None
            num_reused = reuse_mask.sum().item() if reuse_mask is not None else 0
            logging.info(
                f"[Reusing {num_reused}/{src_token_len.item()} tokens based on LCS prediction."
            )
        elif reuse_proportion > 0.0:
            # Randomly reuse k tokens from source based on proportion
            target_len = src_tokens.size(1)
            num_reused = int(reuse_proportion * src_token_len.item())
            indices = torch.randperm(src_token_len.item(), device=device)[:num_reused]
            reuse_mask = torch.zeros(1, target_len, device=device).bool()
            reuse_mask[0, indices] = True
            logging.info(
                f"[Reusing {num_reused}/{src_token_len.item()} tokens via random selection."
            )

        # Prepare special token embeddings
        sos_eos_emb = self.special_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.special_embedding.weight[self.task_id].reshape(1, 1, -1)

        # Calculate target sequence length
        input_len = src_token_len.item()
        if length_ratio is not None:
            target_len = int(input_len * length_ratio)
        else:
            assert (
                self.duration_predictor is not None
            ), "Duration predictor must be provided if length_ratio is not specified"
            dp_input = torch.cat(
                [src_tokens_embedded, encoded_source], dim=-1
            )  # (B, T, D + D')
            length_ratio = self.duration_predictor(dp_input, encoded_source_len)  # (B,)
            length_ratio = length_ratio.item()
            target_len = int(input_len * length_ratio)
        if target_len < 1:
            raise ValueError(
                "Target length must be positive; increase length_ratio or check the duration predictor"
            )

        if alg == "eb_greedy":
            n_timesteps = target_len
            logging.info(
                f"EB-Greedy selected, setting n_timesteps to target length: {n_timesteps}"
            )

        # 5. Initialize target tokens with mask tokens
        tgt_tokens = torch.full(
            (batch_size, target_len), self.diff_mask_id, dtype=torch.long, device=device
        )

        # 6. Create timestep schedule
        # Reduce the number of timesteps if target length is short.
        n_timesteps = min(target_len, n_timesteps)
        timesteps = torch.linspace(1, eps, n_timesteps + 1, device=device)

        if reuse_mask is not None:
            reused_tgt_tokens = torch.where(
                reuse_mask,
                src_tokens,
                torch.full(
                    (batch_size, input_len),
                    self.diff_mask_id,
                    dtype=torch.long,
                    device=device,
                ),
            )  # (B, T_tgt)
            if length_ratio == 1.0:
                tgt_tokens = reused_tgt_tokens
            else:
                reuse_mask = F.interpolate(
                    reuse_mask.float().unsqueeze(0), size=target_len, mode="nearest"
                )  # (1, B, T_tgt)
                reuse_mask = reuse_mask.squeeze(0).bool()  # (B, T_tgt)
                tgt_tokens = F.interpolate(
                    reused_tgt_tokens.float().unsqueeze(0),
                    size=target_len,
                    mode="nearest",
                )  # (1, B, T_tgt)
                tgt_tokens = tgt_tokens.squeeze(0).long()  # (B, T_tgt)
                actual_num_reused = reuse_mask.sum().item()
                logging.info(
                    f"After interpolation, actually reused {actual_num_reused}/{target_len} tokens."
                )
            t_start = (reuse_mask.sum(dim=1).float() / target_len).item()
            n_timesteps = int(n_timesteps * (1 - t_start)) + 1
            timesteps = torch.linspace(t_start, eps, n_timesteps + 1, device=device)

        # 7. Discrete diffusion denoising loop
        for i in range(n_timesteps):
            # Find masked positions
            mask_index = tgt_tokens == self.diff_mask_id
            num_mask_tokens = mask_index.sum().item()

            if num_mask_tokens == 0:
                break

            # Prepare LM input
            tgt_tokens_embedded = self.target_speech_embedding(tgt_tokens)
            # TODO: Optimize by caching unchanging parts: <sos>, encoded_source, <task_id>
            lm_input, lm_input_len, lm_cond_len = self.pad_unpad_sequence(
                sos_eos_emb,
                encoded_source,
                encoded_source_len,
                task_id_emb,
                tgt_tokens_embedded,
                torch.tensor([target_len], device=device),
            )

            # Forward pass with optional CFG
            position, attn_mask = self.make_position_and_mask(lm_input_len, lm_cond_len)

            if cfg_scale > 0:
                # Classifier-free guidance: run with and without conditioning
                # Create unconditional input by zeroing out the encoded source
                lm_input_uncond = lm_input.clone()
                cond_start = 1
                cond_end = cond_start + encoded_source_len.item()
                lm_input_uncond[:, cond_start:cond_end] = 0

                # Concatenate conditional and unconditional inputs
                lm_input_combined = torch.cat([lm_input, lm_input_uncond], dim=0)
                position_combined = torch.cat([position, position], dim=0)
                attn_mask_combined = torch.cat([attn_mask, attn_mask], dim=0)

                # Forward pass
                lm_output = self.lm(
                    lm_input_combined, position_combined, attn_mask_combined
                )
                logits = self.lm_output_proj(lm_output)

                # Split conditional and unconditional outputs
                target_start = cond_end + 1  # +1 for task_id
                target_end = target_start + target_len
                logits_cond, logits_uncond = torch.chunk(
                    logits[:, target_start:target_end], 2, dim=0
                )

                # Apply CFG
                logits = logits_cond + cfg_scale * (logits_cond - logits_uncond)
            else:
                # No CFG: standard forward pass
                lm_output = self.lm(lm_input, position, attn_mask)
                logits = self.lm_output_proj(lm_output)

                cond_start = 1
                cond_end = cond_start + encoded_source_len.item()
                target_start = cond_end + 1  # +1 for task_id
                target_end = target_start + target_len
                logits = logits[:, target_start:target_end]

            # Slice logits to exclude special tokens (mask/contam)
            logits = logits[..., : self.speech_vocab_size]

            # Current and next timesteps
            t = timesteps[i]
            s = timesteps[i + 1]

            # Sample based on algorithm
            if alg == "origin":
                # Extract logits for masked positions only
                logits = logits[mask_index]

                # Original algorithm: probabilistic transition
                p_transfer = 1 - s / t if i < n_timesteps - 1 else 1
                x0 = torch.full(
                    (num_mask_tokens,),
                    self.diff_mask_id,
                    dtype=torch.long,
                    device=device,
                )
                transfer_index_t_s = (
                    torch.rand(num_mask_tokens, device=device) <= p_transfer
                )

                if transfer_index_t_s.any():
                    logits_with_noise = add_gumbel_noise(
                        logits[transfer_index_t_s], temperature=temperature
                    )
                    x0[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)

                tgt_tokens[mask_index] = x0

            elif alg == "greedy":
                # Extract logits for masked positions only
                logits = logits[mask_index]

                # Greedy algorithm: unmask highest confidence tokens
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Calculate confidence scores
                p = F.softmax(logits, dim=-1)
                confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

                # Determine how many tokens to unmask
                number_transfer_tokens = (
                    int(num_mask_tokens * (1 - s / t))
                    if i < n_timesteps - 1
                    else num_mask_tokens
                )

                if number_transfer_tokens > 0:
                    # Unmask top-k confident tokens
                    _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    x0_partial = torch.full(
                        (num_mask_tokens,),
                        self.diff_mask_id,
                        dtype=torch.long,
                        device=device,
                    )
                    x0_partial[transfer_index] = x0[transfer_index]
                    tgt_tokens[mask_index] = x0_partial

            elif alg == "eb_greedy":
                # Extract logits for masked positions only
                logits = logits[mask_index]

                # Greedy algorithm: unmask highest confidence tokens
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if num_mask_tokens < 1:
                    break

                # Calculate confidence scores
                p = F.softmax(logits, dim=-1)  # (T_masked, V)
                confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                    -1
                )  # (T_masked,)
                _, ids = torch.sort(confidence, dim=-1)  # _, (T_masked,)

                # Calculdate number of tokens to unmask based on entropy bound
                entropy = torch.distributions.Categorical(probs=p).entropy()[
                    ids
                ]  # (T_masked,)
                acc_entropy = torch.cumsum(entropy, dim=-1)  # (T_masked,)
                cummax_entropy = torch.cummax(entropy, dim=0).values  # (T_masked,)
                k = (acc_entropy - cummax_entropy <= eb_gamma).sum()
                k = torch.clamp(k, min=1, max=num_mask_tokens)

                # Unmask top-k confident tokens
                _, transfer_index = torch.topk(confidence, k)
                x0_partial = torch.full(
                    (num_mask_tokens,),
                    self.diff_mask_id,
                    dtype=torch.long,
                    device=device,
                )
                x0_partial[transfer_index] = x0[transfer_index]
                tgt_tokens[mask_index] = x0_partial

            else:
                raise NotImplementedError(f"Algorithm '{alg}' not implemented")

        # 8. Yield the final tokens
        for token_id in tgt_tokens[0].tolist():
            yield token_id
