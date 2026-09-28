# DLM-AN

**DLM-AN** stands for **Diffusion Language Model for Accent Normalization**. This repository is the inference release of [*Controllable Accent Normalization via Discrete Diffusion*](https://arxiv.org/abs/2603.14275), accepted to **Interspeech 2026 as a long paper**.

A source waveform is quantized with WavLM-Large (layer 22), converted into target tokens with a DLM, decoded into an 80-bin Mel spectrogram, and rendered by HiFT at 24 kHz. Two **different, checkpoint-incompatible** quantizer/decoder stacks are supported:

| Backend     | Source quantizer              | Mel decoder                                                 | Reference                                                    |
| ----------- | ----------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| `km-flow`   | WavLM + 1024-centroid k-means | Separate `DiTFlow` without duration predictor               | Resemblyzer 256-D speaker embedding                          |
| `joint-ctx` | WavLM + joint VQ              | Context-conditioned flow in the *same quantizer checkpoint* | Speaker embedding **and** aligned reference token/Mel prompt |

`--backend` selects both a matching LM architecture and its checkpoints: `km-flow` uses [configs/dlm.paper.model-only.yaml](configs/dlm.paper.model-only.yaml), while `joint-ctx` uses [configs/dlm.model-only.yaml](configs/dlm.model-only.yaml). If using another LM checkpoint, provide its exact model-only config with `--lm_config`. The inference implementation lives in [dlm_an/model/dlm/dlm.py](dlm_an/model/dlm/dlm.py); training pipelines are not included.

## Setup

Use a Python environment with a compatible PyTorch/torchaudio pair, then install [requirements.txt](requirements.txt). Run from this repository root. `microsoft/wavlm-large` is fetched through Transformers on first use; Resemblyzer downloads its pretrained voice encoder on first use. The preset downloader fetches a shared HiFT weight and backend-specific LM/quantizer/flow weights from [Piping/DLM-AN](https://huggingface.co/Piping/DLM-AN). Supply matching files from the *same training runs*; 1024 entries alone does not make the k-means and joint-VQ token IDs interchangeable. Standard torch-DDP checkpoints are flat state dicts with optional `epoch`/`step` entries; convert DeepSpeed checkpoints to a flat model state dict first. Only load checkpoints you trust.

`km-flow` is the default and is selected with no checkpoint flags:

```sh
python infer_wav.py --source_wav input.wav --output_wav output.wav
```

For the optional joint VQ/flow preset, **only** `--backend joint-ctx` changes the checkpoint and config selection:

```sh
python infer_wav.py --backend joint-ctx --source_wav input.wav --output_wav output.wav
```

Missing weights are fetched to the Hugging Face cache on first run; subsequent runs reuse cached copies. Use `--checkpoint_repo owner/model` to select a different model repository. A provided `--lm_checkpoint`, `--quantizer_checkpoint`, `--flow_checkpoint` (for `km-flow`), or `--hift_checkpoint` replaces just that download; `--lm_config` and `--backend_config` can override the matching local configs. This works in both the CLI and [demo.py](demo.py) (`python demo.py` or `python demo.py --backend joint-ctx`). Joint flow weights are contained in the joint quantizer; a separate flow checkpoint is not accepted for that backend. The Hub filenames are `km-flow/lm.pt` (epoch 8), `km-flow/quantizer.pt` (epoch 1, step 1522), `km-flow/flow.pt` (epoch 299), `joint-ctx/lm.pt` (epoch 28), `joint-ctx/quantizer.pt` (epoch 8, step 620000), and `shared/hift.pt`.

### Separate k-means + flow

```sh
python infer_wav.py --backend km-flow \
  --source_wav input.wav --reference_wav voice.wav --output_wav output.wav \
  --lm_checkpoint /path/to/lm.pt --quantizer_checkpoint /path/to/km.pt \
  --flow_checkpoint /path/to/flow.pt --hift_checkpoint /path/to/hift.pt
```

If `--reference_wav` is omitted, the source supplies the speaker embedding. The generated full-rate tokens are passed directly to the flow with **one Mel frame per token**; they are not deduplicated because this flow has no duration predictor.

### Joint context-conditioned VQ + flow

```sh
python infer_wav.py --backend joint-ctx \
  --source_wav input.wav --reference_wav prompt.wav --output_wav output.wav \
  --lm_checkpoint /path/to/lm.pt --quantizer_checkpoint /path/to/joint.pt \
  --hift_checkpoint /path/to/hift.pt
```

The same reference recording is used for the speaker embedding and as the **acoustic flow prompt** (50 Hz quantizer IDs and 50 Hz Mel features). Prompt streams are trimmed to the shorter frame count. If omitted, the source recording is the prompt. `--reference_wav` does **not** prompt the DLM, only the joint synthesizer. The joint flow is run with `use_source_duration=True`, treating every DLM token as one feature frame.

Add `--mel_path output_mel.pt` to save the generated Mel tensor, `--n_timesteps` for DLM steps, `--flow_steps` for flow-matching steps, `--alg` for DLM sampling and `--device cpu|cuda|auto` for device selection. Both flows support `--full_cfg`, `--cond_cfg`, and `--spk_cfg`; defaults follow each source implementation (`km-flow`: 0/1/1; `joint-ctx`: 0.5/0/0). Inference is single-sample and may be memory-intensive.

Set `--length_ratio 1.0` to request one output token/Mel frame per input token (roughly preserving total duration), or another positive target/source token-length ratio to control the total duration. When omitted, the DLM's trained duration predictor determines the target length. The demo offers the same optional control; leave it blank for prediction. Mask-only sampling supports `origin`, `greedy`, and `eb_greedy`; corrective algorithms requiring uniform corruption are not included.

### Accent-strength controls

Both backends expose `--reuse_proportion` and `--reuse_threshold` (each in `[0, 1]`) in the CLI and the demo. Reused source speech tokens are retained during DLM denoising, generally preserving more of the source accent. For example, add `--reuse_proportion 0.3` to keep the top 30% of source tokens by LCS score, or `--reuse_threshold 0.6` to keep those with predicted LCS probability at least 0.6. A positive proportion takes precedence over the threshold; the defaults (`0` and `1`) disable reuse. Higher reuse typically weakens conversion, but is not a calibrated accent-strength score. An LCS encoder is required for score-based selection; without it a positive proportion selects random source positions.

For a local web UI, use the same checkpoint options with `python demo.py` or `python demo.py --backend joint-ctx`. Models are loaded once when the demo starts. To use alternate checkpoint architectures, point `--backend_config` and `--lm_config` at matching *model-only* HyperPyYAML definitions; checkpoint loading is strict, not partial.

The mask-only diffusion implementation removes uniform noise, its corrective-loss path and its incompatible corrective samplers. Architecture parameters and checkpoint loading remain strict; use checkpoints that match the chosen config.

## Citation

If you use DLM-AN, please cite the [paper](https://arxiv.org/abs/2603.14275):

```bibtex
@inproceedings{bai2026controllable,
  title={Controllable Accent Normalization via Discrete Diffusion},
  author={Qibing Bai and Yuhan Du and Tom Ko and Shuai Wang and Yannan Wang and Haizhou Li},
  year={2026},
  booktitle = {Proc. Interspeech 2026},
}
```