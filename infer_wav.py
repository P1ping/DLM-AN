"""Single-utterance DLM-AN inference with separate or contextual joint decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from checkpoint_presets import DEFAULT_REPO, resolve_checkpoints
from dlm_an.utils.audio import mel_spectrogram

ROOT = Path(__file__).resolve().parent


def load_audio(
    path: str | Path, sample_rate: int, device: torch.device
) -> torch.Tensor:
    """Return a mono waveform [1, samples] at the requested sampling rate."""
    # torchaudio.load() uses TorchCodec on recent torchaudio releases and
    # requires native FFmpeg libraries even for ordinary PCM WAV files.
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(samples.mean(axis=1).copy()).unsqueeze(0)
    if rate != sample_rate:
        audio = torchaudio.functional.resample(audio, rate, sample_rate)
    if audio.shape[1] < 400:
        raise ValueError(f"Audio is too short for WavLM: {path}")
    return audio.to(device)


def speaker_embedding(path: str | Path, device: torch.device) -> torch.Tensor:
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(device="cuda" if device.type == "cuda" else "cpu")
    embedding = encoder.embed_utterance(preprocess_wav(str(path)))
    return F.normalize(
        torch.from_numpy(np.asarray(embedding, dtype=np.float32)).unsqueeze(0), dim=-1
    ).to(device)


def load_checkpoint(
    module: torch.nn.Module, path: str | Path, *, metadata: bool = True
) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if metadata:
        state.pop("epoch", None)
        state.pop("step", None)
    module.load_state_dict(state, strict=True)
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return load_hyperpyyaml(stream)


class DLMAN:
    """Load a DLM and one of two *non-interchangeable* quantizer/decoder bundles."""

    def __init__(
        self,
        backend: str,
        lm_checkpoint: str | Path,
        quantizer_checkpoint: str | Path,
        hift_checkpoint: str | Path,
        flow_checkpoint: str | Path | None = None,
        lm_config: str | Path = ROOT / "configs/dlm.model-only.yaml",
        backend_config: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        if backend not in ("km-flow", "joint-ctx"):
            raise ValueError("backend must be 'km-flow' or 'joint-ctx'")
        if backend == "km-flow" and flow_checkpoint is None:
            raise ValueError("km-flow requires a separate --flow_checkpoint")
        if backend == "joint-ctx" and flow_checkpoint is not None:
            raise ValueError(
                "joint-ctx flow weights are part of the quantizer checkpoint"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.backend = backend
        backend_config = backend_config or ROOT / f"configs/{backend}.model-only.yaml"
        lm_cfg = load_config(lm_config)
        config = load_config(backend_config)
        self.lm = load_checkpoint(lm_cfg["lm"].to(self.device), lm_checkpoint)
        self.quantizer = load_checkpoint(
            config["quantizer"].to(self.device), quantizer_checkpoint
        )
        self.flow = (
            load_checkpoint(config["flow"].to(self.device), flow_checkpoint)
            if backend == "km-flow"
            else None
        )
        self.hift = load_checkpoint(
            config["hift"].to(self.device), hift_checkpoint, metadata=False
        )
        self.sample_rate = int(config["sample_rate"])
        self.hop_size = int(config.get("hop_size", 480))

    def tokenize(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([audio.shape[1]], dtype=torch.long, device=self.device)
        if self.backend == "km-flow":
            _, indices, token_lengths = self.quantizer.inference(audio, lengths)
        else:
            _, indices, token_lengths = self.quantizer.quantize(audio, lengths)
        return indices, token_lengths

    def prompt_mel(self, path: str | Path) -> torch.Tensor:
        prompt_audio = load_audio(path, self.sample_rate, self.device)
        return mel_spectrogram(
            prompt_audio,
            n_fft=1920,
            num_mels=80,
            sampling_rate=self.sample_rate,
            hop_size=self.hop_size,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        ).transpose(1, 2)

    @torch.inference_mode()
    def convert(
        self,
        source_wav: str | Path,
        reference_wav: str | Path | None = None,
        *,
        n_timesteps: int = 32,
        flow_steps: int = 32,
        alg: str = "greedy",
        cfg_scale: float = 2.0,
        temperature: float = 0.0,
        length_ratio: float | None = None,
        reuse_proportion: float = 0.0,
        reuse_threshold: float = 1.0,
        full_cfg: float | None = None,
        cond_cfg: float | None = None,
        spk_cfg: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Return (24-kHz audio [1,T], mel [1,T,80], target token IDs)."""
        if n_timesteps < 1 or flow_steps < 1:
            raise ValueError("step counts must be positive")
        if length_ratio is not None and (
            not np.isfinite(length_ratio) or length_ratio <= 0
        ):
            raise ValueError("length_ratio must be finite and positive")
        if not np.isfinite(reuse_proportion) or not 0.0 <= reuse_proportion <= 1.0:
            raise ValueError("reuse_proportion must be between 0 and 1")
        if not np.isfinite(reuse_threshold) or not 0.0 <= reuse_threshold <= 1.0:
            raise ValueError("reuse_threshold must be between 0 and 1")
        source = load_audio(source_wav, 16000, self.device)
        src_tokens, src_len = self.tokenize(source)
        if src_len.item() == 0:
            raise RuntimeError("Source audio produced no speech tokens")
        ref_path = reference_wav or source_wav
        spk = speaker_embedding(ref_path, self.device)
        if spk.shape[1] != 256:
            raise ValueError(
                f"Expected a 256-dimensional speaker embedding, got {spk.shape[1]}"
            )

        tokens = list(
            self.lm.inference(
                src_tokens=src_tokens,
                src_token_len=src_len,
                n_timesteps=n_timesteps,
                alg=alg,
                cfg_scale=cfg_scale,
                temperature=temperature,
                length_ratio=length_ratio,
                reuse_proportion=reuse_proportion,
                reuse_threshold=reuse_threshold,
            )
        )
        if not tokens:
            raise RuntimeError("DLM produced no target tokens")
        target = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        target_len = torch.tensor(
            [target.shape[1]], dtype=torch.long, device=self.device
        )

        if self.backend == "km-flow":
            # flow_wo_dp.yaml has no duration predictor; every full-rate DLM
            # token corresponds to exactly one Mel frame. Never deduplicate.
            mel, _ = self.flow.inference(
                speech_token=target,
                speech_token_len=target_len,
                spk_embed=spk,
                duration=torch.ones_like(target),
                n_timesteps=flow_steps,
                full_cfg=0.0 if full_cfg is None else full_cfg,
                cond_cfg=1.0 if cond_cfg is None else cond_cfg,
                spk_cfg=1.0 if spk_cfg is None else spk_cfg,
            )
        else:
            # Use one and the same reference waveform to extract prompt tokens
            # (WavLM at 50 Hz) and prompt Mel (24 kHz / 480 = 50 Hz).
            reference = load_audio(ref_path, 16000, self.device)
            ctx_indices, ctx_lengths = self.tokenize(reference)
            ctx_feat = self.prompt_mel(ref_path)
            ctx_len = min(ctx_lengths.item(), ctx_feat.shape[1])
            if ctx_len < 1:
                raise RuntimeError("Reference audio produced no aligned prompt frames")
            ctx_lengths = torch.tensor([ctx_len], dtype=torch.long, device=self.device)
            mel, _ = self.quantizer.synthesize(
                quant_indices=target,
                feature_lengths=target_len,
                spk_embed=spk,
                ctx_indices=ctx_indices[:, :ctx_len],
                ctx_lengths=ctx_lengths,
                ctx_speech_feat=ctx_feat[:, :ctx_len],
                use_source_duration=True,
                n_timesteps=flow_steps,
                full_cfg=0.5 if full_cfg is None else full_cfg,
                cond_cfg=0.0 if cond_cfg is None else cond_cfg,
                spk_cfg=0.0 if spk_cfg is None else spk_cfg,
            )

        waveform, _ = self.hift.inference(
            speech_feat=mel.transpose(1, 2),
            cache_source=torch.zeros(1, 1, 0, device=self.device),
        )
        return waveform, mel, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("km-flow", "joint-ctx"), default="km-flow"
    )
    parser.add_argument("--source_wav", required=True)
    parser.add_argument(
        "--reference_wav",
        help="Speaker reference (and joint-ctx acoustic prompt); defaults to source",
    )
    parser.add_argument("--output_wav", required=True)
    parser.add_argument(
        "--lm_checkpoint", help="Local override; otherwise download the selected preset"
    )
    parser.add_argument(
        "--quantizer_checkpoint",
        help="Local override; otherwise download the selected preset",
    )
    parser.add_argument("--flow_checkpoint", help="Required only for km-flow")
    parser.add_argument(
        "--hift_checkpoint",
        help="Local override; otherwise download the shared checkpoint",
    )
    parser.add_argument(
        "--lm_config", help="Override the selected backend's model-only LM config"
    )
    parser.add_argument(
        "--checkpoint_repo",
        default=DEFAULT_REPO,
        help="Hugging Face model repo for missing checkpoints",
    )
    parser.add_argument(
        "--backend_config", help="Override the matching model-only backend config"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--n_timesteps", type=int, default=32, help="DLM denoising steps"
    )
    parser.add_argument(
        "--flow_steps", type=int, default=32, help="Flow-matching steps"
    )
    parser.add_argument(
        "--alg",
        choices=("origin", "greedy", "eb_greedy"),
        default="greedy",
    )
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--length_ratio",
        type=float,
        help="Target/source token duration ratio; if omitted, predict the ratio from source tokens",
    )
    parser.add_argument(
        "--reuse_proportion",
        type=float,
        default=0.0,
        help="Fraction of source tokens to retain (top LCS scores); >0 overrides reuse_threshold",
    )
    parser.add_argument(
        "--reuse_threshold",
        type=float,
        default=1.0,
        help="Retain source tokens whose predicted LCS probability meets this threshold; 1 disables reuse",
    )
    parser.add_argument("--full_cfg", type=float)
    parser.add_argument("--cond_cfg", type=float)
    parser.add_argument("--spk_cfg", type=float)
    parser.add_argument(
        "--mel_path", help="Optional destination for generated Mel tensor"
    )
    args = parser.parse_args()
    bundle = resolve_checkpoints(
        args.backend,
        lm_checkpoint=args.lm_checkpoint,
        quantizer_checkpoint=args.quantizer_checkpoint,
        flow_checkpoint=args.flow_checkpoint,
        hift_checkpoint=args.hift_checkpoint,
        lm_config=args.lm_config,
        checkpoint_repo=args.checkpoint_repo,
    )
    engine = DLMAN(
        args.backend,
        bundle.lm_checkpoint,
        bundle.quantizer_checkpoint,
        bundle.hift_checkpoint,
        bundle.flow_checkpoint,
        bundle.lm_config,
        args.backend_config,
        args.device,
    )
    audio, mel, _ = engine.convert(
        args.source_wav,
        args.reference_wav,
        n_timesteps=args.n_timesteps,
        flow_steps=args.flow_steps,
        alg=args.alg,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        length_ratio=args.length_ratio,
        reuse_proportion=args.reuse_proportion,
        reuse_threshold=args.reuse_threshold,
        full_cfg=args.full_cfg,
        cond_cfg=args.cond_cfg,
        spk_cfg=args.spk_cfg,
    )
    output = Path(args.output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio.squeeze(0).cpu().numpy(), engine.sample_rate)
    if args.mel_path:
        mel_path = Path(args.mel_path)
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel.cpu(), mel_path)


if __name__ == "__main__":
    main()
