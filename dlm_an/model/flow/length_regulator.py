import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def sequence_mask(lengths, max_len):
    """Create a binary mask tensor from sequence lengths.

    Args:
        lengths (torch.Tensor): 1D tensor of sequence lengths
            shape: (B)
        max_len (int): Maximum length to consider

    Returns:
        Binary mask tensor of shape [B, max_len], where mask[i,:lengths[i]] = 1
    """
    batch_size = lengths.shape[0]
    device = lengths.device

    # Create position indices
    indices = torch.arange(max_len, device=device).expand(batch_size, max_len)

    # Create mask where positions < length are 1, else 0
    mask = indices < lengths.unsqueeze(1)
    return mask


def convert_pad_shape(pad_shape):
    """Convert pad shape to format expected by F.pad"""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


class LengthRegulator(nn.Module):
    """Length Regulator that expands text representations to match speech features."""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        text_repr: torch.Tensor,
        duration: torch.Tensor,
        text_mask: torch.Tensor,
        total_duration: torch.Tensor,
        total_duration_tol: int = 4,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_repr (torch.Tensor): Text representations to be expanded
                shape: (B, T_text, D)
            duration (torch.Tensor): Duration predictions for each text token
                shape: (B, T_text)
            text_mask (torch.BoolTensor): Mask of valid text positions
                shape: (B, T_text)
            total_duration (torch.Tensor): Total duration for each sequence
                shape: (B)
            max_len (int): Maximum length to pad sequences to. If None, uses maximum sequence length
        Returns:
            Tuple containing:
                - Expanded text representations
                    shape: (B, T_speech, D)
                - Expanded sequence lengths
                    shape: (B)
        """
        duration = torch.clamp(duration, min=0) * text_mask

        # Scale duration to match total_duration
        sum_duration = torch.sum(duration, dim=1)
        diff = total_duration - sum_duration
        if diff.abs().max() > total_duration_tol:
            logger.warning(
                f"LengthRegulator: duration sum mismatch, max diff = {diff.abs().max().item():.1f} exceeds tolerance {total_duration_tol}"
            )
        scale = total_duration.to(duration.dtype) / (sum_duration + 1e-8)
        scaled_duration = duration * scale.unsqueeze(1)

        # Quantize to integers while preserving sum
        cum_duration = torch.cumsum(scaled_duration, dim=1)
        cum_duration_int = torch.round(cum_duration).long()
        duration = cum_duration_int - F.pad(cum_duration_int, (1, 0))[:, :-1]

        # Ensure durations are non-negative integers
        duration = torch.clamp(duration, min=0).long()

        # Calculate expanded sequence lengths
        expanded_lengths = torch.sum(duration, dim=1).long()

        # Determine maximum expanded length
        if max_len is None:
            max_expanded_len = expanded_lengths.max().item()
        else:
            max_expanded_len = max_len
            expanded_lengths = torch.clamp(expanded_lengths, max=max_len)

        # Create a binary mask of shape [B, T_text, max_expanded_len]
        # where path[b, t, :duration[b, t]] = 1
        path = self.generate_path(duration=duration, max_len=max_expanded_len, mask=text_mask, dtype=text_repr.dtype)

        # Apply the path to expand text_repr
        # [B, max_expanded_len, T_text] @ [B, T_text, D] -> [B, max_expanded_len, D]
        expanded_repr = torch.bmm(path.transpose(1, 2), text_repr)

        return expanded_repr, expanded_lengths

    def generate_path(self, duration, max_len, mask, dtype):
        """Generate a binary matrix mapping from text positions to expanded positions.

        Args:
            duration (torch.Tensor): Duration predictions for each text token
                shape: (B, T_text)
            max_len (int): Maximum length of expanded sequences
            mask (torch.BoolTensor): Binary mask indicating valid text positions
                shape: (B, T_text)

        Returns:
            Binary matrix where path[b, t, start:start+duration[b,t]] = 1
                shape: (B, T_text, max_len)
        """
        batch_size, text_len = duration.shape

        # Calculate cumulative durations
        cum_duration = torch.cumsum(duration, dim=1)

        # Flatten cumulative durations for batch processing
        cum_duration_flat = cum_duration.view(-1)  # [B*T_text]

        # Create mask where positions < cumulative duration are 1
        path_flat = sequence_mask(cum_duration_flat, max_len).to(dtype=dtype)  # (B*T_text, max_len)

        path = path_flat.view(batch_size, text_len, max_len)  # (B, T_text, max_len)

        # Convert cumulative mask to frame-level mask by subtracting shifted version
        # This creates a binary matrix where 1s indicate which expanded positions correspond to each text token
        path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]

        # Apply mask to zero-out invalid positions
        path = path * mask.unsqueeze(2)

        return path
