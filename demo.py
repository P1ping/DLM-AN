"""Local Gradio demo for both DLM-AN inference backends."""

from __future__ import annotations

import argparse
import math

import gradio as gr

from checkpoint_presets import DEFAULT_REPO, resolve_checkpoints
from infer_wav import DLMAN


def parse_length_ratio(value: str) -> float | None:
    if not value.strip():
        return None
    try:
        ratio = float(value)
    except ValueError as exc:
        raise gr.Error("Duration ratio must be a positive number or blank") from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise gr.Error("Duration ratio must be a positive finite number or blank")
    return ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("km-flow", "joint-ctx"), default="km-flow"
    )
    parser.add_argument("--lm_checkpoint")
    parser.add_argument("--quantizer_checkpoint")
    parser.add_argument("--flow_checkpoint")
    parser.add_argument("--hift_checkpoint")
    parser.add_argument("--lm_config")
    parser.add_argument("--backend_config")
    parser.add_argument("--checkpoint_repo", default=DEFAULT_REPO)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
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
        lm_config=bundle.lm_config,
        backend_config=args.backend_config,
        device=args.device,
    )

    def convert(
        source: str,
        reference: str | None,
        steps: int,
        flow_steps: int,
        alg: str,
        cfg_scale: float,
        length_ratio: str,
        reuse_proportion: float,
        reuse_threshold: float,
    ):
        if not source:
            raise gr.Error("Upload a source recording")
        audio, _, _ = engine.convert(
            source,
            reference,
            n_timesteps=int(steps),
            flow_steps=int(flow_steps),
            alg=alg,
            cfg_scale=cfg_scale,
            length_ratio=parse_length_ratio(length_ratio),
            reuse_proportion=float(reuse_proportion),
            reuse_threshold=float(reuse_threshold),
        )
        return engine.sample_rate, audio.squeeze(0).cpu().numpy()

    prompt_note = (
        "The reference supplies both the speaker identity and an acoustic prompt."
        if args.backend == "joint-ctx"
        else "The reference supplies the speaker identity only."
    )
    with gr.Blocks(title="DLM-AN") as app:
        gr.Markdown(
            f"# DLM-AN · {args.backend}\n\n{prompt_note} Leave reference blank to use source audio."
        )
        with gr.Row():
            source = gr.Audio(label="Source speech", type="filepath")
            reference = gr.Audio(label="Reference speech (optional)", type="filepath")
        with gr.Row():
            steps = gr.Slider(1, 128, value=32, step=1, label="DLM steps")
            flow_steps = gr.Slider(1, 128, value=32, step=1, label="Flow steps")
            cfg = gr.Slider(0, 10, value=2, step=0.1, label="DLM CFG scale")
            alg = gr.Dropdown(
                ("greedy", "origin", "eb_greedy"),
                value="greedy",
                label="Sampling algorithm",
            )
        with gr.Row():
            reuse_proportion = gr.Slider(
                0,
                1,
                value=0,
                step=0.01,
                label="Reuse proportion",
                info="Retain this fraction of source tokens with highest LCS scores; overrides threshold when > 0.",
            )
            reuse_threshold = gr.Slider(
                0,
                1,
                value=1,
                step=0.01,
                label="Reuse threshold",
                info="Retain source tokens with LCS probability at least this value; 1 disables reuse.",
            )
            length_ratio = gr.Textbox(
                value="",
                label="Target/source duration ratio (optional)",
                placeholder="Automatic (predicted duration)",
                info="Leave blank to use the trained duration predictor; 1 preserves token duration.",
            )
        submit = gr.Button("Convert", variant="primary")
        result = gr.Audio(label="Converted speech")
        submit.click(
            convert,
            (
                source,
                reference,
                steps,
                flow_steps,
                alg,
                cfg,
                length_ratio,
                reuse_proportion,
                reuse_threshold,
            ),
            result,
        )
    app.queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
