import torch
import torch.nn.functional as F


def deduplicate(feat_quantized, quant_indices, feat_mask):
    """
    Deduplicate the quantized features based on quantization indices and mask.
    Merges consecutive identical indices by averaging their features.

    Args:
        feat_quantized: (B, T_feat, D) Tensor of quantized features.
        quant_indices: (B, T_feat) Tensor of quantization indices.
        feat_mask: (B, T_feat) Boolean mask indicating valid feature positions (True=valid).
    Returns:
        token_embed: (B, T_token_max, D) Tensor of deduplicated token embeddings.
        token_embed_len: (B,) Tensor of lengths of deduplicated token embeddings.
        token_durations: (B, T_token_max) Tensor of durations for each token.
    """
    B, T, D = feat_quantized.shape
    device = feat_quantized.device

    # 1. Identify change points (boundaries)
    # Compare current index with previous index.
    # First element is always a change point if it's valid.
    # We shift right by padding the left side.
    indices_shifted = F.pad(quant_indices, (1, 0))[:, :-1]

    # A change occurs if the index is different OR if it's the very first element
    # We must enforce that the first element (t=0) is always treated as a change point.
    is_change = quant_indices != indices_shifted
    is_change[:, 0] = True

    # Apply mask: invalid positions cannot be change points (except we handle padding later)
    is_change = is_change & feat_mask

    # 2. Calculate Token Lengths (how many unique tokens per batch item)
    token_embed_len = is_change.sum(dim=1)
    T_token_max = token_embed_len.max().item()

    # 3. Create Group IDs for Scatter Reduction
    # We want a unique ID for every run of identical tokens within a batch.
    # cumsum gives us a running count of changes.
    # Example: [A, A, B, B, C] -> is_change [1, 0, 1, 0, 1] -> cumsum [1, 1, 2, 2, 3]
    # Use T_token_max (not T+1) as the per-batch stride — each batch item has at most
    # T_token_max segments, so local segment IDs are in [0, T_token_max) and never collide.
    # This keeps intermediate tensors at O(B*T_token_max*D) rather than O(B*T*D).
    batch_offsets = (torch.arange(B, device=device) * T_token_max).unsqueeze(1)

    # group_ids: (B, T). Unique ID for every segment across the whole batch.
    # We subtract 1 so indices start at 0.
    group_ids = torch.cumsum(is_change.long(), dim=1) - 1 + batch_offsets

    # Mask out invalid positions so they don't contribute to the scatter
    # We assign them to a "junk" bin. The max valid ID is B*T_token_max-1, so B*T_token_max
    # is a safe junk bin.
    junk_bin = B * T_token_max
    group_ids[~feat_mask] = junk_bin

    # 4. Flatten for Scatter Operations
    flat_feat = feat_quantized.reshape(-1, D)
    flat_group_ids = group_ids.view(-1)

    # Total number of bins: B*T_token_max valid slots + 1 junk bin
    num_groups = junk_bin + 1

    # 5. Calculate Sums and Counts
    # We use scatter_add to sum features and counts for averaging
    # out_tensor: (num_groups, D)

    # Initialize output tensors
    sum_features = torch.zeros((num_groups, D), dtype=feat_quantized.dtype, device=device)
    counts = torch.zeros((num_groups,), dtype=feat_quantized.dtype, device=device)  # float for division

    # Perform scatter add
    # index needs to be same dims as src for scatter in dim 0
    # index: (B*T, 1) broadcasted to (B*T, D) for features
    idx_expanded = flat_group_ids.unsqueeze(1).expand(-1, D)

    sum_features = sum_features.scatter_add(0, idx_expanded, flat_feat)
    counts = counts.scatter_add(0, flat_group_ids, torch.ones_like(flat_group_ids, dtype=feat_quantized.dtype))

    # 6. Average to get embeddings (Deduplication)
    # Avoid division by zero for the junk bin or empty groups
    avg_features = sum_features / counts.unsqueeze(1).clamp(min=1.0)

    # 7. Reshape back to (B, T_token_max, D)
    # We need to extract the valid groups and arrange them back into a batch tensor.

    # We know exactly which group IDs correspond to which Batch/Token index.
    # The group_ids were generated sequentially.
    # We need to construct a gather map.

    # Create a template for the output indices
    # shape: (B, T_token_max)
    # Values: batch_offset + 0, batch_offset + 1, ...
    out_indices = torch.arange(T_token_max, device=device).unsqueeze(0).expand(B, -1)
    out_indices = out_indices + batch_offsets

    # Mask for output: positions beyond the actual length of the deduplicated sequence
    # shape: (B, T_token_max)
    seq_range = torch.arange(T_token_max, device=device).unsqueeze(0)
    out_mask = seq_range < token_embed_len.unsqueeze(1)

    # Flatten out_indices to gather from avg_features
    flat_out_indices = out_indices.reshape(-1)

    # Gather the averaged features
    # (B*T_token_max, D)
    dedup_flat = torch.index_select(avg_features, 0, flat_out_indices)

    # Reshape to (B, T_token_max, D)
    token_embed = dedup_flat.view(B, T_token_max, D)

    # Zero out padding in the output
    token_embed = token_embed * out_mask.unsqueeze(-1)

    # 8. Get Durations
    # counts holds the duration for each group ID.
    # We gather counts using the same indices.
    durations_flat = torch.index_select(counts, 0, flat_out_indices)
    token_durations = durations_flat.view(B, T_token_max)

    # Zero out padding durations
    token_durations = token_durations * out_mask

    return token_embed, token_embed_len, token_durations


# --- Verification Helper ---
if __name__ == "__main__":
    # Small test case
    B, T, D = 2, 6, 4

    # Batch 1: A A B B B C (Length 6) -> A(2) B(3) C(1)
    # Batch 2: X Y Y Z Z (Length 5, 1 pad) -> X(1) Y(2) Z(2)

    quant_indices = torch.tensor(
        [[1, 1, 2, 2, 2, 3], [10, 20, 20, 20, 20, 0]]  # 0 is padding index here, but mask handles it
    )

    feat_mask = torch.tensor([[True, True, True, True, True, True], [True, True, True, True, True, False]])

    # Create dummy features corresponding to indices * 10 for easy checking
    # e.g., index 1 has feature [10, 10, 10, 10]
    feat_quantized = torch.zeros(B, T, D)
    for b in range(B):
        for t in range(T):
            if feat_mask[b, t]:
                val = quant_indices[b, t].float() * 10
                # Add small noise to verify averaging: + t
                feat_quantized[b, t] = val + t

    # Batch 0 features:
    # t=0 (idx 1): 10+0
    # t=1 (idx 1): 10+1 -> Avg should be 10.5
    # t=2 (idx 2): 20+2
    # t=3 (idx 2): 20+3
    # t=4 (idx 2): 20+4 -> Avg should be 20 + 3 = 23
    # t=5 (idx 3): 30+5 -> Avg 35

    emb, lens, durs = deduplicate(feat_quantized, quant_indices, feat_mask)

    print("Lengths:", lens)  # Should be [3, 3]
    print("Durations:\n", durs)
    # Batch 0: [2, 3, 1]
    # Batch 1: [1, 2, 2]

    print("Embeddings Batch 0:\n", emb[0])
    # Expected:
    # [10.5, ...], [23.0, ...], [35.0, ...]


def align_features(
    feat_quantized,
    quant_indices,
    feature_lengths,
    speech_feat,
    speech_feat_len,
):
    """
    Align features to have the same lengths based on feature_lengths and speech_feat_len.

    Args:
        feat_quantized: (B, T_feat, D) Tensor of quantized features.
        quant_indices: (B, T_feat) Tensor of quantization indices.
        feature_lengths: (B,) Tensor of lengths of quantized features.
        speech_feat: (B, T_speech, D_speech) Tensor of speech features.
        speech_feat_len: (B,) Tensor of lengths of speech features.
    Returns:
        feat_quantized: (B, T_aligned, D) Tensor of aligned quantized features.
        quant_indices: (B, T_aligned) Tensor of aligned quantization indices.
        feature_lengths: (B,) Tensor of aligned lengths of quantized features.
        speech_feat: (B, T_aligned, D_speech) Tensor of aligned speech features.
        speech_feat_len: (B,) Tensor of aligned lengths of speech features.
    """
    B, T_feat, D = feat_quantized.shape
    B, T_speech, D_speech = speech_feat.shape

    T = min(T_feat, T_speech)
    common_lengths = torch.min(feature_lengths, speech_feat_len)

    feat_quantized = feat_quantized[:, :T, :]
    quant_indices = quant_indices[:, :T]

    speech_feat = speech_feat[:, :T, :]

    return feat_quantized, quant_indices, speech_feat, common_lengths
