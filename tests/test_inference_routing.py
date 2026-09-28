"""Checkpoint-free tests for the two public waveform inference routes."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch
import tempfile

import gradio as gr

from demo import parse_length_ratio
from infer_wav import DLMAN, load_audio
from dlm_an.model.dlm.dlm import TransformerLM


class InferenceRoutingTests(unittest.TestCase):
    def test_demo_optional_length_ratio(self):
        field = gr.Textbox(value="")
        self.assertEqual(field.value, "")
        self.assertIsNone(parse_length_ratio(field.preprocess(field.value)))
        self.assertIsNone(parse_length_ratio("  "))
        self.assertEqual(parse_length_ratio("1.25"), 1.25)
        for value in ("0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(gr.Error):
                parse_length_ratio(value)

    def test_audio_loader_avoids_torchcodec_and_resamples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stereo.wav"
            samples = np.stack([np.ones(2400), np.zeros(2400)], axis=1)
            sf.write(path, samples, 24000)
            with patch(
                "infer_wav.torchaudio.load",
                side_effect=RuntimeError("TorchCodec unavailable"),
            ):
                audio = load_audio(path, 16000, torch.device("cpu"))
            self.assertEqual(tuple(audio.shape), (1, 1600))
            self.assertAlmostEqual(float(audio[:, 200:1400].mean()), 0.5, places=2)

    def make_engine(self, backend):
        engine = DLMAN.__new__(DLMAN)
        engine.backend = backend
        engine.device = torch.device("cpu")
        engine.sample_rate = 24000
        engine.hop_size = 480
        engine.lm = Mock()
        engine.lm.inference.return_value = iter([3, 3, 5])
        engine.quantizer = Mock()
        engine.flow = Mock()
        engine.hift = Mock()
        engine.hift.inference.return_value = (torch.zeros(1, 480 * 3), None)
        return engine

    @patch("infer_wav.speaker_embedding", return_value=torch.ones(1, 256))
    @patch("infer_wav.load_audio", return_value=torch.zeros(1, 16000))
    def test_separate_flow_uses_full_rate_tokens(self, _audio, _speaker):
        engine = self.make_engine("km-flow")
        engine.quantizer.inference.return_value = (
            None,
            torch.tensor([[1, 2, 3]]),
            torch.tensor([3]),
        )
        engine.flow.inference.return_value = (torch.zeros(1, 3, 80), torch.tensor([3]))

        audio, mel, tokens = engine.convert("source.wav")

        self.assertEqual(tokens, [3, 3, 5])
        self.assertEqual(tuple(audio.shape), (1, 1440))
        self.assertEqual(tuple(mel.shape), (1, 3, 80))
        call = engine.flow.inference.call_args.kwargs
        self.assertEqual(call["speech_token"].tolist(), [[3, 3, 5]])
        self.assertEqual(call["duration"].tolist(), [[1, 1, 1]])
        engine.quantizer.synthesize.assert_not_called()

    @patch("infer_wav.speaker_embedding", return_value=torch.ones(1, 256))
    @patch("infer_wav.load_audio", return_value=torch.zeros(1, 16000))
    def test_joint_flow_uses_aligned_reference_prompt(self, _audio, _speaker):
        engine = self.make_engine("joint-ctx")
        engine.quantizer.quantize.side_effect = [
            (None, torch.tensor([[1, 2, 3]]), torch.tensor([3])),
            (None, torch.tensor([[4, 5, 6, 7]]), torch.tensor([4])),
        ]
        engine.prompt_mel = Mock(return_value=torch.zeros(1, 3, 80))
        engine.quantizer.synthesize.return_value = (
            torch.zeros(1, 3, 80),
            torch.tensor([3]),
        )

        _, _, tokens = engine.convert("source.wav", "prompt.wav")

        self.assertEqual(tokens, [3, 3, 5])
        call = engine.quantizer.synthesize.call_args.kwargs
        self.assertEqual(call["quant_indices"].tolist(), [[3, 3, 5]])
        self.assertEqual(call["ctx_indices"].tolist(), [[4, 5, 6]])
        self.assertEqual(call["ctx_lengths"].tolist(), [3])
        self.assertEqual(tuple(call["ctx_speech_feat"].shape), (1, 3, 80))
        self.assertTrue(call["use_source_duration"])
        engine.flow.inference.assert_not_called()

    @patch("infer_wav.speaker_embedding", return_value=torch.ones(1, 256))
    @patch("infer_wav.load_audio", return_value=torch.zeros(1, 16000))
    def test_reuse_controls_reach_both_backends(self, _audio, _speaker):
        for backend in ("km-flow", "joint-ctx"):
            with self.subTest(backend=backend):
                engine = self.make_engine(backend)
                if backend == "km-flow":
                    engine.quantizer.inference.return_value = (
                        None,
                        torch.tensor([[1, 2, 3]]),
                        torch.tensor([3]),
                    )
                    engine.flow.inference.return_value = (
                        torch.zeros(1, 3, 80),
                        torch.tensor([3]),
                    )
                else:
                    engine.quantizer.quantize.return_value = (
                        None,
                        torch.tensor([[1, 2, 3]]),
                        torch.tensor([3]),
                    )
                    engine.prompt_mel = Mock(return_value=torch.zeros(1, 3, 80))
                    engine.quantizer.synthesize.return_value = (
                        torch.zeros(1, 3, 80),
                        torch.tensor([3]),
                    )
                engine.convert("source.wav", reuse_proportion=0.4, reuse_threshold=0.8)
                self.assertEqual(
                    engine.lm.inference.call_args.kwargs["reuse_proportion"], 0.4
                )
                self.assertEqual(
                    engine.lm.inference.call_args.kwargs["reuse_threshold"], 0.8
                )

    @patch("infer_wav.speaker_embedding", return_value=torch.ones(1, 256))
    @patch("infer_wav.load_audio", return_value=torch.zeros(1, 16000))
    def test_invalid_reuse_values_rejected(self, _audio, _speaker):
        engine = self.make_engine("km-flow")
        for options in (
            {"reuse_proportion": -0.1},
            {"reuse_proportion": 1.1},
            {"reuse_threshold": float("nan")},
            {"reuse_threshold": 1.1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                engine.convert("source.wav", **options)
        engine.lm.inference.assert_not_called()

    def test_dlm_reuses_lcs_tokens_and_accepts_length_ratio(self):
        # Minimal model double: exercise real inference logic without loading weights.
        model = TransformerLM.__new__(TransformerLM)
        torch.nn.Module.__init__(model)
        model.diff_mask_id = 10
        model.source_speech_embedding = torch.nn.Embedding(11, 2)
        model.target_speech_embedding = model.source_speech_embedding
        model.special_embedding = torch.nn.Embedding(2, 2)
        model.sos_eos = 0
        model.task_id = 1
        model.lcs_encoder = torch.nn.Identity()
        model.duration_predictor = Mock(return_value=torch.tensor([1.0]))
        model.encode = Mock(
            return_value=(torch.zeros(1, 4, 2), torch.tensor([4]), None)
        )
        model.encode_lcs = Mock(
            return_value=(
                torch.tensor([[10.0, -10.0, 10.0, -10.0]]),
                torch.ones(1, 4, dtype=torch.bool),
            )
        )
        model.lm = Mock(return_value=torch.zeros(1, 10, 2))
        model.lm_output_proj = Mock(return_value=torch.tensor([[[0.0, 1.0]] * 10]))
        model.speech_vocab_size = 2
        model.pad_unpad_sequence = Mock(
            return_value=(
                torch.zeros(1, 11, 2),
                torch.tensor([11]),
                torch.tensor([5]),
            )
        )
        model.make_position_and_mask = Mock(
            return_value=(
                torch.zeros(1, 10, dtype=torch.long),
                torch.ones(1, 10, 10, dtype=torch.bool),
            )
        )

        source = torch.tensor([[2, 3, 4, 5]])
        generated = list(
            model.inference(
                source,
                torch.tensor([4]),
                length_ratio=1.0,
                n_timesteps=1,
                cfg_scale=0.0,
                reuse_proportion=0.5,
            )
        )
        self.assertEqual(generated[0], 2)
        self.assertEqual(generated[2], 4)
        model.duration_predictor.assert_not_called()

        model.duration_predictor = Mock(return_value=torch.tensor([0.75]))
        predicted = list(
            model.inference(source, torch.tensor([4]), n_timesteps=1, cfg_scale=0.0)
        )
        self.assertEqual(len(predicted), 3)
        model.duration_predictor.assert_called_once()
        with self.assertRaisesRegex(ValueError, "Target length must be positive"):
            list(model.inference(source, torch.tensor([4]), length_ratio=0.01))
        with self.assertRaisesRegex(AssertionError, "Invalid sampling algorithm"):
            list(model.inference(source, torch.tensor([4]), alg="corrective"))

    def test_dlm_forward_process_uses_only_absorbing_masks(self):
        model = TransformerLM.__new__(TransformerLM)
        torch.nn.Module.__init__(model)
        model.diff_mask_id = 11
        source = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
        noisy, probabilities, masked = model.forward_process(
            source, torch.tensor([4, 3])
        )
        self.assertEqual(noisy.shape, source.shape)
        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(torch.equal(noisy, source.masked_fill(masked, 11)))
        self.assertFalse(masked[0, 4])
        self.assertFalse(masked[1, 3:].any())

    @patch("infer_wav.speaker_embedding", return_value=torch.ones(1, 256))
    @patch("infer_wav.load_audio", return_value=torch.zeros(1, 16000))
    def test_length_ratio_forwarded_and_validated(self, _audio, _speaker):
        engine = self.make_engine("km-flow")
        engine.quantizer.inference.return_value = (
            None,
            torch.tensor([[1, 2, 3]]),
            torch.tensor([3]),
        )
        engine.flow.inference.return_value = (torch.zeros(1, 3, 80), torch.tensor([3]))
        engine.lm.inference.side_effect = lambda **kwargs: iter([3, 3, 5])
        engine.convert("source.wav", length_ratio=1.25)
        self.assertEqual(engine.lm.inference.call_args.kwargs["length_ratio"], 1.25)
        engine.convert("source.wav")
        self.assertIsNone(engine.lm.inference.call_args.kwargs["length_ratio"])
        for ratio in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                engine.convert("source.wav", length_ratio=ratio)


if __name__ == "__main__":
    unittest.main()
