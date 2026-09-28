import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SslMiniBatchKMeans(nn.Module):
    def __init__(
        self,
        ssl_extractor,
        n_clusters,
        dim,
        init="k-means++",
        n_init=3,
        verbose=0,
        tol=0.0,
        max_no_improvement=10,
        num_dataset_samples=None,  # For EWA inertia calculation
        initial_inertia=0.0,  # Starting value for EWA inertia
        reassignment_ratio=0.01,
        random_state=None,
        input_key="speech",
        input_length_key="speech_len",
    ):
        super().__init__()
        self.ssl_extractor = ssl_extractor
        self.n_clusters = n_clusters
        self.dim = dim
        self.init_mode = init
        self.n_init = n_init
        self.verbose = verbose
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.num_dataset_samples = num_dataset_samples
        self.reassignment_ratio = reassignment_ratio
        self.random_state = random_state
        self.input_key = input_key
        self.input_length_key = input_length_key

        # Buffers
        self.register_buffer("cluster_centers", torch.zeros(n_clusters, dim))
        self.register_buffer("cluster_counts", torch.zeros(n_clusters, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.bool))
        self.register_buffer("n_iter", torch.tensor(0, dtype=torch.long))

        # Reassignment buffer
        self.register_buffer("n_since_last_reassign", torch.tensor(0, dtype=torch.long))

        # EWA / Convergence buffers
        self.register_buffer("ewa_inertia", torch.tensor(initial_inertia, dtype=torch.float32))
        self.register_buffer("ewa_inertia_min", torch.tensor(float("inf"), dtype=torch.float32))
        self.register_buffer("no_improvement", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_samples_seen", torch.tensor(0, dtype=torch.long))
        self.register_buffer("converged", torch.tensor(False, dtype=torch.bool))

    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized()

    def _kmeans_plus_plus(self, x):
        """
        Robust K-Means++ implementation for a single initialization.
        """
        n_samples, n_features = x.shape
        centers = torch.empty(self.n_clusters, n_features, device=x.device, dtype=x.dtype)

        # 1. Choose first center uniformly at random
        idx = torch.randint(0, n_samples, (1,), device=x.device).item()
        centers[0] = x[idx]

        # Keep track of the closest distance squared for every point to any existing center
        closest_dist_sq = torch.sum((x - centers[0]) ** 2, dim=1)

        # 2. Choose remaining centers
        for c in range(1, self.n_clusters):
            sum_dist = closest_dist_sq.sum()
            if sum_dist > 0:
                rand_idx = torch.multinomial(closest_dist_sq, 1).item()
            else:
                rand_idx = torch.randint(0, n_samples, (1,), device=x.device).item()

            centers[c] = x[rand_idx]

            new_dist_sq = torch.sum((x - centers[c]) ** 2, dim=1)
            closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)

        return centers

    def _initialize(self, x):
        """
        Initializes centroids using samples from ALL GPUs.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # DDP Gathering Logic
        if self._is_distributed():
            # 1. Gather batch sizes from all ranks
            local_size = torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
            all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
            dist.all_gather(all_sizes, local_size)

            max_size = max(s.item() for s in all_sizes)

            # 2. Pad local batch to max_size so we can use all_gather
            # Pad format for F.pad is (padding_left, padding_right, padding_top, padding_bottom)
            # x is (N, Dim), we want to pad the bottom (dim 0)
            pad_amount = max_size - x.size(0)
            if pad_amount > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_amount))
            else:
                x_padded = x

            # 3. Gather all padded batches
            # We create a list of tensors to hold the output
            gathered_x = [
                torch.zeros(max_size, self.dim, device=x.device, dtype=x.dtype) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_x, x_padded)

            # 4. Rank 0 reconstructs the full dataset (removing padding)
            if dist.get_rank() == 0:
                valid_batches = []
                for i, size_tensor in enumerate(all_sizes):
                    real_size = size_tensor.item()
                    if real_size > 0:
                        valid_batches.append(gathered_x[i][:real_size])

                if len(valid_batches) > 0:
                    x_full = torch.cat(valid_batches, dim=0)
                else:
                    x_full = x  # Fallback (shouldn't happen if sizes > 0)
            else:
                x_full = None
        else:
            # Single GPU case
            x_full = x

        # Initialization Logic (Only on Rank 0)
        if not self._is_distributed() or dist.get_rank() == 0:
            if x_full.shape[0] < self.n_clusters:
                raise ValueError(
                    f"Total batch size across all GPUs ({x_full.shape[0]}) is smaller than "
                    f"n_clusters ({self.n_clusters}). Cannot initialize."
                )

            best_inertia = None
            best_centers = None

            if self.verbose:
                print(f"Initialization: Running {self.n_init} attempts on global batch of size {x_full.shape[0]}...")

            for i in range(self.n_init):
                if self.init_mode == "random":
                    indices = torch.randperm(x_full.size(0))[: self.n_clusters]
                    candidates = x_full[indices]
                elif self.init_mode == "k-means++":
                    candidates = self._kmeans_plus_plus(x_full)
                else:
                    candidates = torch.randn(self.n_clusters, self.dim, device=x.device)

                # Compute inertia on the FULL gathered batch
                dists = torch.cdist(x_full, candidates, p=2)
                min_dists_sq = dists.min(dim=1).values ** 2
                inertia = min_dists_sq.sum().item()

                if self.verbose:
                    print(f"  Init {i + 1}/{self.n_init} - Inertia: {inertia:.4f}")

                if best_inertia is None or inertia < best_inertia:
                    best_inertia = inertia
                    best_centers = candidates.clone()

            self.cluster_centers.copy_(best_centers)
            if self.verbose:
                print(f"Initialization complete. Best inertia: {best_inertia:.4f}")

        # Broadcast Result
        if self._is_distributed():
            dist.broadcast(self.cluster_centers, src=0)

        self.initialized.fill_(True)

    def _check_convergence(self, current_batch_size, centers_squared_diff, batch_inertia):
        """
        Tracks convergence using Exponential Weighted Average (EWA) of inertia.
        """
        normalized_batch_inertia = batch_inertia / current_batch_size
        step = self.n_iter.item() + 1

        if step == 1:
            if self.verbose:
                print(f"Step {step}: Initial inertia {normalized_batch_inertia:.6f}")
            return False

        if self.ewa_inertia == 0:
            self.ewa_inertia.copy_(normalized_batch_inertia)
        else:
            total_samples = (
                self.num_dataset_samples if self.num_dataset_samples is not None else self.total_samples_seen.item()
            )
            alpha = current_batch_size * 2.0 / (total_samples + 1)
            alpha = min(alpha, 1.0)
            self.ewa_inertia.mul_(1 - alpha).add_(normalized_batch_inertia * alpha)

        if self.verbose:
            print(
                f"Step {step}: batch inertia: {normalized_batch_inertia:.6f}, "
                f"ewa inertia: {self.ewa_inertia.item():.6f}"
            )

        if self.tol > 0.0 and centers_squared_diff <= self.tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}")
            return True

        if self.ewa_inertia < self.ewa_inertia_min:
            self.no_improvement.zero_()
            self.ewa_inertia_min.copy_(self.ewa_inertia)
        else:
            self.no_improvement.add_(1)

        if self.max_no_improvement is not None and self.no_improvement >= self.max_no_improvement:
            if self.verbose:
                print(f"Converged (lack of improvement) at step {step}")
            return True

        return False

    def _reassign_clusters(self, x, current_batch_size):
        """
        Reassigns empty or very small clusters to new random positions from the current batch.
        """
        if self.reassignment_ratio <= 0:
            return

        # Increment counter by the global batch size
        self.n_since_last_reassign.add_(current_batch_size)

        # 1. Check Trigger Conditions
        has_empty = (self.cluster_counts == 0).any()
        threshold_reached = self.n_since_last_reassign >= (10 * self.n_clusters)

        if not (has_empty or threshold_reached):
            return

        # Reset counter
        self.n_since_last_reassign.zero_()

        # 2. Identify clusters to reassign
        max_count = self.cluster_counts.max()
        threshold = self.reassignment_ratio * max_count
        to_reassign_mask = self.cluster_counts < threshold

        # 3. Cap the number of reassignments
        n_reassigns = to_reassign_mask.sum()
        limit = int(0.5 * current_batch_size)

        if n_reassigns > limit:
            candidate_indices = torch.nonzero(to_reassign_mask, as_tuple=True)[0]
            candidate_counts = self.cluster_counts[candidate_indices]

            sorted_vals, sorted_idx = torch.sort(candidate_counts)
            indices_to_keep = candidate_indices[sorted_idx[limit:]]
            to_reassign_mask[indices_to_keep] = False

            n_reassigns = torch.tensor(limit, device=to_reassign_mask.device)

        if n_reassigns == 0:
            return

        # 4. Perform Reassignment
        if not self._is_distributed() or dist.get_rank() == 0:
            if self.verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.")

            n_local = x.size(0)
            if n_reassigns > n_local:
                new_indices = torch.randint(0, n_local, (n_reassigns,), device=x.device)
            else:
                new_indices = torch.randperm(n_local, device=x.device)[:n_reassigns]

            self.cluster_centers[to_reassign_mask] = x[new_indices].to(self.cluster_centers.dtype)

            non_reassigned_counts = self.cluster_counts[~to_reassign_mask]
            if len(non_reassigned_counts) > 0:
                new_count_val = non_reassigned_counts.min()
            else:
                new_count_val = torch.tensor(1, device=self.cluster_counts.device)

            self.cluster_counts[to_reassign_mask] = new_count_val

        # 5. Synchronize updates
        if self._is_distributed():
            dist.broadcast(self.cluster_centers, src=0)
            dist.broadcast(self.cluster_counts, src=0)

    def forward(self, batch, device):
        if self.converged:
            return {"converged": True, "inertia": 0.0, "center_shift": 0.0, "n_iter": self.n_iter.item()}

        waveforms_or_specs = batch[self.input_key].to(device)
        lengths = batch[self.input_length_key].to(device)

        with torch.no_grad():
            features, feature_lengths = self.ssl_extractor(waveforms_or_specs, lengths)

        T_feat = features.size(1)
        feat_mask = torch.arange(T_feat, device=lengths.device).unsqueeze(0) < feature_lengths.unsqueeze(1)
        x = features[feat_mask]  # Flattened: (Total_Samples_In_Batch, Dim)

        # Initialization
        if not self.initialized:
            self._initialize(x)

        B_current = x.shape[0]

        # 1. Compute distances & Assign Labels
        dists = torch.cdist(x, self.cluster_centers, p=2) ** 2
        min_dists, labels = torch.min(dists, dim=1)
        batch_inertia = min_dists.sum()

        # 2. Compute stats for centroid update
        one_hot = F.one_hot(labels, self.n_clusters).type(x.dtype)
        batch_counts = one_hot.sum(dim=0)
        batch_sums = one_hot.T @ x

        # 3. DDP Sync
        if self._is_distributed():
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_inertia, op=dist.ReduceOp.SUM)

            # Sync batch size for normalization
            batch_size_tensor = torch.tensor(B_current, device=device, dtype=torch.long)
            dist.all_reduce(batch_size_tensor, op=dist.ReduceOp.SUM)
            B_current = batch_size_tensor.item()

        # Update cumulative count
        self.total_samples_seen.add_(B_current)

        # 4. Update Centroids (Standard MiniBatch Update)
        old_centers = self.cluster_centers.clone()

        self.cluster_counts += batch_counts.long()
        lr = 1.0 / (self.cluster_counts.float() + 1e-6).unsqueeze(1)

        update_direction = batch_sums - (batch_counts.unsqueeze(1) * old_centers)
        self.cluster_centers = old_centers + (update_direction * lr)

        # 5. Reassign Clusters
        self._reassign_clusters(x, B_current)

        # 6. Check Convergence
        centers_diff = self.cluster_centers - old_centers
        centers_squared_diff = (centers_diff**2).sum(dim=1).mean()

        has_converged = self._check_convergence(B_current, centers_squared_diff, batch_inertia)

        if has_converged:
            self.converged.fill_(True)

        # 7. Count current active codes
        num_active_codes = torch.sum(batch_counts >= 1)

        self.n_iter += 1

        return {
            "inertia": batch_inertia,
            "center_shift": centers_squared_diff,
            "n_iter": self.n_iter,
            "ewa_inertia": self.ewa_inertia,
            "num_active_codes": num_active_codes,
            "converged": has_converged,
            "no_improvement": self.no_improvement,
        }

    def inference(self, waveforms, lengths):
        """
        waveforms: (B, T_wav)
        lengths: (B,)
        """
        features, feature_lengths = self.ssl_extractor.inference(waveforms, lengths)

        B, T_feat, D = features.size()

        # Compute distances
        dists = torch.cdist(features, self.cluster_centers, p=2) ** 2  # (B, T, C)

        # Assign labels
        min_dists, labels = torch.min(dists, dim=2)  # (B, T), (B, T)
        centroids = F.one_hot(labels, self.n_clusters).type(features.dtype) @ self.cluster_centers  # (B, T, D)

        return centroids, labels, feature_lengths

    def state_dict(self, *args, **kwargs):
        # Exclude ssl_extractor from state dict
        state_dict = super().state_dict(*args, **kwargs)
        ssl_keys = [key for key in state_dict.keys() if key.startswith("ssl_extractor.")]
        for key in ssl_keys:
            del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Load state dict while keeping ssl_extractor
        ssl_state_dict = {f"ssl_extractor.{k}": v for k, v in self.ssl_extractor.state_dict().items()}
        state_dict.update(ssl_state_dict)
        super().load_state_dict(state_dict, *args, **kwargs)
