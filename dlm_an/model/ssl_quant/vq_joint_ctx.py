import logging

import torch
import torch.nn.functional as F
from einops import repeat
from vector_quantize_pytorch import VectorQuantize

from dlm_an.model.conv.conv_layers import ResBlock1d
from dlm_an.model.ssl_quant.utils import align_features, deduplicate
from dlm_an.model.flow.length_regulator import LengthRegulator


class SslVectorQuantizerJointCtx(torch.nn.Module):
    def __init__(
        self,
        ssl_extractor,
        ctc_encoder,
        flow,
        ssl_dim,
        ctc_vocab_size,
        codebook_size,
        duration_predictor=None,
        codebook_dim: int = None,
        num_resblocks: int = 1,
        resblock_dim: int = None,
        resblock_kernel_size: int = 3,
        reconstruction_weight: float = 5.0,
        commitment_weight: float = 1.0,
        decay: float = 0.99,
        ctc_target_key: str = "phone_token",
        ctc_deduplicate: bool = True,
        ctc_weight: float = 1.0,
        flow_weight: float = 1.0,
        spk_embed_dim: int = 256,
        normalize_spk_embed: bool = True,
        input_key="speech",
        input_length_key="speech_len",
    ):
        super().__init__()
        self.ssl_extractor = ssl_extractor

        resblock_dim = resblock_dim if resblock_dim is not None else ssl_dim
        self.use_resblock = num_resblocks > 0
        self.pre_resblock = self.build_resblock(
            num_resblocks, ssl_dim, resblock_dim, resblock_dim, resblock_kernel_size
        )

        hidden_dim = resblock_dim if self.use_resblock else ssl_dim
        codebook_dim = codebook_dim if codebook_dim is not None else hidden_dim

        self.pre_quant_proj = (
            torch.nn.Linear(hidden_dim, codebook_dim)
            if codebook_dim != hidden_dim
            else torch.nn.Identity()
        )
        self.vq = VectorQuantize(
            dim=codebook_dim,
            codebook_size=codebook_size,
            ema_update=True,
            learnable_codebook=False,
            commitment_weight=commitment_weight,
            use_cosine_sim=False,
            rotation_trick=False,
            codebook_diversity_loss_weight=0.0,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
            decay=decay,
        )
        self.post_quant_proj = (
            torch.nn.Linear(codebook_dim, hidden_dim)
            if codebook_dim != hidden_dim
            else torch.nn.Identity()
        )

        self.post_resblock = self.build_resblock(
            num_resblocks, resblock_dim, resblock_dim, ssl_dim, resblock_kernel_size
        )

        self.reconstruction_weight = reconstruction_weight

        # CTC component
        self.ctc_prenet = ResBlock1d(
            hidden_dim, ctc_encoder.output_size(), kernel_size=resblock_kernel_size
        )
        self.ctc_encoder = ctc_encoder
        self.ctc_proj = torch.nn.Linear(ctc_encoder.output_size(), ctc_vocab_size)
        self.ctc_target_key = ctc_target_key
        self.ctc_vocab_size = ctc_vocab_size
        self.ctc_deduplicate = ctc_deduplicate
        self.ctc_weight = ctc_weight

        self.spk_embed_dim = spk_embed_dim
        self.normalize_spk_embed = normalize_spk_embed

        if duration_predictor is not None:
            self.duration_prenet = ResBlock1d(hidden_dim, ssl_dim, kernel_size=3)
            if spk_embed_dim > 0:
                self.spk_duration_proj = torch.nn.Linear(spk_embed_dim, hidden_dim)
            self.duration_predictor = duration_predictor
            self.length_regulator = LengthRegulator()
        else:
            self.duration_prenet = None
            self.spk_duration_proj = None
            self.duration_predictor = None
            self.length_regulator = None

        # Flow component
        # flow_prenet maps hidden_dim → ssl_dim; omitted (Identity) when dimensions already match.
        # NOTE: the kernel size must be 1 so that the prefix tokens and main tokens can be processed separately
        self.flow_prenet = (
            ResBlock1d(hidden_dim, ssl_dim, kernel_size=1)
            if hidden_dim != ssl_dim
            else torch.nn.Identity()
        )
        self.flow = flow
        self.flow_weight = flow_weight

        self.input_key = input_key
        self.input_length_key = input_length_key

        preresblock_params = (
            sum(p.numel() for p in self.pre_resblock.parameters() if p.requires_grad)
            if self.use_resblock
            else 0
        )
        postresblock_params = (
            sum(p.numel() for p in self.post_resblock.parameters() if p.requires_grad)
            if self.use_resblock
            else 0
        )
        ctc_encoder_params = sum(
            p.numel() for p in self.ctc_encoder.parameters() if p.requires_grad
        )
        flow_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
        dp_params = (
            sum(
                p.numel()
                for p in self.duration_predictor.parameters()
                if p.requires_grad
            )
            if self.duration_predictor is not None
            else 0
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            f"Total trainable parameters: {total_params/1e6:.2f}M"
            f" = pre_resblock: {preresblock_params/1e6:.2f}M + post_resblock: {postresblock_params/1e6:.2f}M"
            f" + ctc_encoder: {ctc_encoder_params/1e6:.2f}M"
            f" + duration_predictor: {dp_params/1e6:.2f}M"
            f" + flow: {flow_params/1e6:.2f}M"
            f" + others: {(total_params - ctc_encoder_params - preresblock_params - postresblock_params - flow_params - dp_params)/1e6:.2f}M"
        )

    def forward(self, batch, device):
        waveforms_or_specs = batch[self.input_key].to(device)  # (B, T_wav)
        lengths = batch[self.input_length_key].to(device)  # (B,)
        speech_feat = batch["speech_feat"].to(device)
        speech_feat_len = batch["speech_feat_len"].to(device)
        spk_embed = batch["spk_embedding"].to(device)
        aux_tokens = batch[self.ctc_target_key].to(device)
        aux_len = batch[f"{self.ctc_target_key}_len"].to(device)

        B = waveforms_or_specs.size(0)

        if self.normalize_spk_embed:
            spk_embed = F.normalize(spk_embed, p=2, dim=-1)

        with torch.inference_mode():
            features, feature_lengths = self.ssl_extractor(waveforms_or_specs, lengths)

        B, T_feat, D = features.size()
        feat_mask = torch.arange(T_feat, device=device).unsqueeze(
            0
        ) < feature_lengths.unsqueeze(1)

        # --- Shared VQ pipeline ---
        if self.use_resblock:
            features_before_quant = self.forward_resblock(
                self.pre_resblock, features, feat_mask
            )
        else:
            features_before_quant = features

        features_before_quant = self.pre_quant_proj(features_before_quant)
        feat_quantized, quant_indices, vq_loss, vq_losses = self.vq(
            features_before_quant, mask=feat_mask, return_loss_breakdown=True
        )
        feat_quantized = self.post_quant_proj(feat_quantized)
        num_activated_codes = torch.sum(self.vq._codebook.cluster_size >= 1)

        # Reconstruction loss
        if self.use_resblock:
            feat_recon = self.forward_resblock(
                self.post_resblock, feat_quantized, feat_mask
            )
            recon_loss = F.l1_loss(feat_recon[feat_mask], features[feat_mask])
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # --- CTC path (operates on original feature lengths, no mel alignment needed) ---
        if self.ctc_deduplicate:
            ctc_embed, ctc_embed_len, ctc_durations = deduplicate(
                feat_quantized, quant_indices, feat_mask
            )
            ctc_mask = torch.arange(ctc_embed.size(1), device=device).unsqueeze(
                0
            ) < ctc_embed_len.unsqueeze(1)
        else:
            ctc_embed, ctc_embed_len, ctc_durations, ctc_mask = (
                feat_quantized,
                feature_lengths,
                None,
                feat_mask,
            )

        ctc_loss = torch.tensor(0.0, device=device)
        if self.ctc_weight > 0.0:
            ctc_logits = self.encode_token_for_ctc(ctc_embed, ctc_mask, ctc_embed_len)
            ctc_logp = ctc_logits.log_softmax(dim=-1)
            ctc_loss = F.ctc_loss(
                ctc_logp.transpose(0, 1),
                aux_tokens,
                ctc_embed_len,
                aux_len,
                blank=0,
                zero_infinity=True,
            )

        # -- Duration path ---
        dp_loss = torch.tensor(0.0, device=device)
        if self.duration_predictor is not None:
            # Reuse the CTC dedup result if possible
            if self.ctc_deduplicate:
                dp_embed, dp_embed_len, token_durations = (
                    ctc_embed,
                    ctc_embed_len,
                    ctc_durations,
                )
                dp_mask = ctc_mask
            else:
                dp_embed, dp_embed_len, token_durations = deduplicate(
                    feat_quantized, quant_indices, feat_mask
                )
                dp_mask = torch.arange(dp_embed.size(1), device=device).unsqueeze(
                    0
                ) < dp_embed_len.unsqueeze(1)
            spk_embed_dp = self.spk_duration_proj(spk_embed)
            dp_input = dp_embed.detach() + repeat(
                spk_embed_dp, "b d -> b t d", t=dp_embed.size(1)
            )
            dp_input = self.duration_prenet(dp_input, dp_mask)
            dp_loss = self.duration_predictor.compute_loss(
                dp_input, dp_mask, token_durations
            )

        # --- Flow path (requires alignment with mel features first) ---
        feat_quantized_flow, quant_indices_flow, speech_feat, flow_lengths = (
            align_features(
                feat_quantized,
                quant_indices,
                feature_lengths,
                speech_feat,
                speech_feat_len,
            )
        )
        T_feat_flow = feat_quantized_flow.size(1)
        feat_mask_flow = torch.arange(T_feat_flow, device=device).unsqueeze(
            0
        ) < flow_lengths.unsqueeze(1)

        fm_loss = torch.tensor(0.0, device=device)
        if self.flow_weight > 0.0:
            flow_embed = self.flow_prenet(feat_quantized_flow, feat_mask_flow)
            fm_loss = self.flow.compute_loss(
                speech_feat, feat_mask_flow, flow_embed, spk_embed
            )

        loss = (
            self.reconstruction_weight * recon_loss
            + vq_loss
            + self.ctc_weight * ctc_loss
            + self.flow_weight * fm_loss
            + dp_loss
        )

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "ctc_loss": ctc_loss,
            "quant_commit_loss": vq_losses.commitment,
            "fm_loss": fm_loss,
            "dp_loss": dp_loss,
            "batch_size": torch.tensor(B, device=device),
            "num_activated_codes": num_activated_codes,
        }

    def inference(
        self,
        waveforms,
        lengths,
        mode="quantize",
        spk_embed=None,
        ctx_indices=None,
        ctx_lengths=None,
        ctx_speech_feat=None,
    ):
        """
        waveforms: (B, T_wav)
        lengths: (B,)
        mode: one of "quantize", "reconstruct", "recognize", "synthesize"
        """
        assert mode in ["quantize", "reconstruct", "recognize", "synthesize"]

        feat_quantized, quant_indices, feature_lengths = self.quantize(
            waveforms, lengths
        )

        if mode == "quantize":
            return feat_quantized, quant_indices, feature_lengths
        elif mode == "reconstruct":
            return self.reconstruct(quant_indices, feature_lengths)
        elif mode == "recognize":
            return self.recognize(quant_indices, feature_lengths)
        else:  # mode == "synthesize"
            return self.synthesize(
                quant_indices,
                feature_lengths,
                spk_embed=spk_embed,
                ctx_indices=ctx_indices,
                ctx_lengths=ctx_lengths,
                ctx_speech_feat=ctx_speech_feat,
            )

    def quantize(self, waveforms, lengths):
        """
        waveforms: (B, T_wav)
        lengths: (B,)
        """
        features, feature_lengths = self.ssl_extractor.inference(waveforms, lengths)

        B, T_feat, D = features.size()
        feat_mask = torch.arange(T_feat, device=lengths.device).unsqueeze(
            0
        ) < feature_lengths.unsqueeze(1)

        if self.use_resblock:
            features = self.forward_resblock(self.pre_resblock, features, feat_mask)

        features = self.pre_quant_proj(features)
        feat_quantized, quant_indices, _ = self.vq(features)

        return feat_quantized, quant_indices, feature_lengths

    def reconstruct(self, quant_indices, feature_lengths):
        """
        quant_indices: (B, T_feat)
        feature_lengths: (B,)
        """
        B, T_feat = quant_indices.size()

        feat_quantized = self.vq.get_codes_from_indices(quant_indices)
        feat_quantized = self.post_quant_proj(feat_quantized)
        feat_mask = torch.arange(T_feat, device=feature_lengths.device).unsqueeze(
            0
        ) < feature_lengths.unsqueeze(1)

        if self.use_resblock:
            return self.forward_resblock(self.post_resblock, feat_quantized, feat_mask)
        return feat_quantized

    def recognize(self, quant_indices, feature_lengths):
        """
        quant_indices: (B, T_feat)
        feature_lengths: (B,)
        """
        B, T_feat = quant_indices.size()
        device = quant_indices.device

        feat_quantized = self.vq.get_codes_from_indices(quant_indices)
        feat_quantized = self.post_quant_proj(feat_quantized)
        feat_mask = torch.arange(T_feat, device=feature_lengths.device).unsqueeze(
            0
        ) < feature_lengths.unsqueeze(1)

        if self.ctc_deduplicate:
            token_embed, token_embed_len, _ = deduplicate(
                feat_quantized, quant_indices, feat_mask
            )
            feat_mask = torch.arange(token_embed.size(1), device=device).unsqueeze(
                0
            ) < token_embed_len.unsqueeze(1)
        else:
            token_embed, token_embed_len = feat_quantized, feature_lengths

        ctc_logits = self.encode_token_for_ctc(token_embed, feat_mask, token_embed_len)
        ctc_logp = ctc_logits.log_softmax(dim=-1)
        ctc_pred = ctc_logp.argmax(dim=-1)

        return ctc_pred, token_embed_len

    def synthesize(
        self,
        quant_indices,
        feature_lengths,
        spk_embed,
        ctx_indices,
        ctx_lengths,
        ctx_speech_feat,
        full_cfg=0.5,
        cond_cfg=0.0,
        spk_cfg=0.0,
        n_timesteps=32,
        total_duration=None,
        use_source_duration=False,
    ):
        """
        quant_indices: (B, T_feat)
        feature_lengths: (B,)
        spk_embed: (B, D_spk)
        full_cfg: float, CFG scale for the full model (applied on top of cond_cfg and spk_cfg)
        cond_cfg: float, CFG scale for conditioning information (token embeddings)
        spk_cfg: float, CFG scale for speaker embedding
        n_timesteps: int, number of diffusion steps for synthesis
        use_source_duration: bool, whether to use source duration values
        preserve_total_duration: bool, whether to preserve the total duration of the input
        """
        B, T_feat = quant_indices.size()
        device = quant_indices.device

        feat_quantized = self.vq.get_codes_from_indices(quant_indices)
        feat_quantized = self.post_quant_proj(feat_quantized)
        feat_mask = torch.arange(T_feat, device=feature_lengths.device).unsqueeze(
            0
        ) < feature_lengths.unsqueeze(1)

        ctx_token_embed = self.vq.get_codes_from_indices(ctx_indices)
        ctx_token_embed = self.post_quant_proj(ctx_token_embed)
        ctx_token_mask = torch.arange(ctx_token_embed.size(1), device=device).unsqueeze(
            0
        ) < ctx_lengths.unsqueeze(1)

        assert use_source_duration or (
            self.duration_predictor is not None
        ), "Duration information is required for synthesis but no duration predictor is defined and use_source_duration is False."
        if use_source_duration:
            token_embed, token_embed_len = feat_quantized, feature_lengths
        else:
            dedup_embed, dedup_embed_len, token_durations = deduplicate(
                feat_quantized, quant_indices, feat_mask
            )
            dedup_mask = torch.arange(dedup_embed.size(1), device=device).unsqueeze(
                0
            ) < dedup_embed_len.unsqueeze(1)
            spk_embed_dp = self.spk_duration_proj(spk_embed)
            dp_input = dedup_embed + repeat(
                spk_embed_dp, "b d -> b t d", t=dedup_embed.size(1)
            )
            dp_input = self.duration_prenet(dp_input, dedup_mask)
            token_durations = self.duration_predictor(
                dp_input, dedup_mask, total_duration=total_duration
            )  # [B, T]
            tgt_total_duration = (
                token_durations.sum(dim=1) if total_duration is None else total_duration
            )
            token_embed, token_embed_len = self.length_regulator(
                dedup_embed,
                token_durations,
                dedup_mask,
                total_duration=tgt_total_duration,
            )

        token_mask = torch.arange(token_embed.size(1), device=device).unsqueeze(
            0
        ) < token_embed_len.unsqueeze(1)
        ctx_token_mask = torch.arange(ctx_token_embed.size(1), device=device).unsqueeze(
            0
        ) < ctx_lengths.unsqueeze(1)

        token_embed = self.flow_prenet(token_embed, token_mask)
        ctx_token_embed = self.flow_prenet(ctx_token_embed, ctx_token_mask)
        speech_feat = self.flow.inference(
            token_mask,
            token_embed,
            ctx_speech_feat,
            ctx_token_embed,
            ctx_token_mask,
            spk_emb=spk_embed,
            t_scheduler="cosine",
            n_timesteps=n_timesteps,
            full_cfg=full_cfg,
            cond_cfg=cond_cfg,
            spk_cfg=spk_cfg,
        )

        return speech_feat, token_embed_len

    def encode_token_for_ctc(self, token_embed, token_mask, token_lengths):
        token_embed = self.ctc_prenet(token_embed, token_mask)
        ctc_embed, encoder_mask = self.ctc_encoder(
            token_embed,
            token_lengths,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
        )
        ctc_logits = self.ctc_proj(ctc_embed)
        return ctc_logits

    def build_resblock(
        self, num_layers, input_dim, hidden_dim, output_dim, kernel_size=3
    ):
        if num_layers <= 0:
            return None
        layers = []
        for idx in range(num_layers):
            dim_in = input_dim if idx == 0 else hidden_dim
            dim_out = output_dim if idx == num_layers - 1 else hidden_dim
            layers.append(ResBlock1d(dim_in, dim_out, kernel_size=kernel_size))
        return torch.nn.ModuleList(layers)

    def forward_resblock(self, resblock, x, x_mask):
        for layer in resblock:
            x = layer(x, x_mask)
        return x

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        ssl_keys = [
            key for key in state_dict.keys() if key.startswith("ssl_extractor.")
        ]
        for key in ssl_keys:
            del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        ssl_state_dict = {
            f"ssl_extractor.{k}": v for k, v in self.ssl_extractor.state_dict().items()
        }
        state_dict.update(ssl_state_dict)
        super().load_state_dict(state_dict, *args, **kwargs)
