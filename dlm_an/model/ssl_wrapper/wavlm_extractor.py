import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers import WavLMModel
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from dlm_an.utils.mask import make_pad_mask

SAMPLING_RATE = 16000
HOP_SIZE = 320
CONV_PAD_LEN = 80


def get_wavlm_output_length(input_length):
    """
    Calculates the output feature length for standard Wav2Vec2/HuBERT/WavLM
    given the input audio length (number of samples).
    """

    # Configuration for microsoft/wavlm-base-plus
    # (kernel_size, stride) for the 7 layers
    conv_layers = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]

    current_length = input_length

    for kernel, stride in conv_layers:
        # Formula: floor((L_in - kernel) / stride) + 1
        current_length = torch.floor((current_length - kernel) / stride) + 1

    return current_length.long()


class WavLMModelModified(WavLMModel):
    """
    Modified WavLMModel to allow directly passing attention_mask to encoder.
    By contrast, the original WavLMModel assumes attention_mask is waveform-level
    and computes feature-level mask internally.
    """

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Wav2Vec2BaseModelOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = self.feature_projection(extract_features)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class WavLMExtractor(torch.nn.Module):
    def __init__(self, model_name, layer, normalize_wav=True, padding=False):
        super().__init__()
        try:
            self.wavlm_model = WavLMModelModified.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            self.wavlm_model = WavLMModelModified.from_pretrained(model_name)

        self.wavlm_model.eval()
        self.layer = layer

        self.normalize_wav = normalize_wav
        self.padding = padding

    @torch.inference_mode()
    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        if self.normalize_wav:
            max_vals = waveforms.abs().max(dim=1, keepdim=True)[0]
            waveforms = waveforms / torch.clamp(max_vals, min=1e-9)  # (B, T')

        if self.padding:
            # Pad waveforms to the nearest multiple of 320,
            # while considering CONV_PAD_LEN caused by conv subsampling
            max_len = waveforms.size(1)
            pad_len = (HOP_SIZE - (max_len % HOP_SIZE)) % HOP_SIZE + CONV_PAD_LEN
            waveforms = F.pad(waveforms, (0, pad_len), mode="constant", value=0)
            feature_lengths = torch.ceil(lengths / HOP_SIZE).long()  # (B,)
        else:
            feature_lengths = get_wavlm_output_length(lengths)  # (B,)

        attn_mask = ~make_pad_mask(feature_lengths)

        output = self.wavlm_model(
            waveforms, attention_mask=attn_mask, output_hidden_states=True
        )
        features = output.hidden_states[self.layer]  # (B, T, D)

        return features, feature_lengths

    @torch.inference_mode()
    def inference(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        return self.forward(waveforms, lengths)


class WavLMTokenExtractor(torch.nn.Module):
    def __init__(
        self, model_name, km_weight_path, layer, device="cuda", normalize_wav=True
    ):
        super().__init__()
        self.wavlm_model = WavLMModel.from_pretrained(model_name).to(device)
        self.wavlm_model.eval()

        self.km_weights = torch.load(km_weight_path, map_location=device)  # (N, D)
        self.km_weights_norm = F.normalize(self.km_weights, dim=1)  # (N, D)
        self.layer = layer

        self.device = device
        self.normalize_wav = normalize_wav

    @torch.inference_mode()
    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        if self.normalize_wav:
            max_vals = waveforms.abs().max(dim=1, keepdim=True)[0]
            waveforms = waveforms / torch.clamp(max_vals, min=1e-9)  # (B, T')

        # TODO: Add padding and correct length calculation
        token_lengths = torch.clamp(lengths // HOP_SIZE, min=1)  # (B,)

        output = self.wavlm_model(waveforms.to(self.device), output_hidden_states=True)
        features = output.hidden_states[self.layer]  # (B, T, D)

        features_norm = F.normalize(features, dim=2)  # (B, T, D)
        logits = torch.matmul(features_norm, self.km_weights_norm.t())  # (B, T, N)
        tokens = torch.argmax(logits, dim=-1)  # (B, T)

        return tokens, token_lengths
