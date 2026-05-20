import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import prenormalize_showers, load_config
from dataset1_preprocess import resample_alpha, TARGET_ALPHA
from dataset1_preprocess import (
    parse_binning_xml,
    build_weight_mats,
    build_mask_info,
    get_all_voxel_counts
)


def resample_alpha_torch(layer: torch.Tensor, src_alpha: int, dst_alpha: int) -> torch.Tensor:
    """Angular resampling (torch version), layer shape: [..., src_alpha, R] → [..., dst_alpha, R]"""
    if src_alpha == dst_alpha:
        return layer

    src_edges = torch.linspace(0, 1, src_alpha + 1)
    dst_edges = torch.linspace(0, 1, dst_alpha + 1)

    W_a = torch.zeros(dst_alpha, src_alpha)
    for src in range(src_alpha):
        a_lo = src_edges[src]
        a_hi = src_edges[src + 1]
        da_src = a_hi - a_lo
        for dst in range(dst_alpha):
            lo = dst_edges[dst]
            hi = dst_edges[dst + 1]
            overlap = max(0.0, min(a_hi, hi) - max(a_lo, lo))
            if overlap > 0:
                W_a[dst, src] = overlap / da_src

    return torch.einsum('...sr,...ds->...dr', layer, W_a)


def resample_alpha_inverse_torch(layer: torch.Tensor, src_alpha: int, dst_alpha: int) -> torch.Tensor:
    """Inverse angular resampling (torch version), layer shape: [..., src_alpha, R] → [..., dst_alpha, R]"""
    if src_alpha == dst_alpha:
        return layer

    src_edges = torch.linspace(0, 1, src_alpha + 1)
    dst_edges = torch.linspace(0, 1, dst_alpha + 1)

    W_a = torch.zeros(dst_alpha, src_alpha)
    for dst in range(dst_alpha):
        a_lo = dst_edges[dst]
        a_hi = dst_edges[dst + 1]
        da_dst = a_hi - a_lo
        for src in range(src_alpha):
            lo = src_edges[src]
            hi = src_edges[src + 1]
            overlap = max(0.0, min(a_hi, hi) - max(a_lo, lo))
            if overlap > 0:
                W_a[dst, src] = overlap / da_dst

    return torch.einsum('...sr,...ds->...dr', layer, W_a)


ENERGY_SCALE = 1  # mirrors utils.energy_scale


def normalize_volume_torch(volume, energies, stats, alpha=1e-6):
    """Torch version of prenormalize_showers using precomputed stats.

    Args:
        volume:   [N, 1, L, A, R] encoded volume
        energies: [N, 1] energy values
        stats:    dict with prenormalize_method, normalize_method, vmax,
                  log_mean/log_std or logit_mean/logit_std
        alpha:    small offset for numerical stability

    Returns:
        [N, 1, L, A, R] normalized volume
    """
    prenormalize_method = stats['prenormalize_method']
    vmax = stats['vmax']
    normalize_method = stats['normalize_method']
    e = energies.view(-1, 1, 1, 1, 1)

    if prenormalize_method == 'log10':
        volume = torch.log10(volume + 1e-9)
        e_log = torch.log10(e + 1e-9)
        volume = volume / e_log * ENERGY_SCALE
        x = alpha + (1 - 2 * alpha) * volume / vmax
    elif prenormalize_method == 'log1p':
        volume = volume / (e + 1e-9)
        volume_log = torch.log1p(volume)
        x = alpha + (1 - 2 * alpha) * volume_log / vmax
    elif prenormalize_method == 'norm':
        volume = volume / (e + 1e-9)
        x = alpha + (1 - 2 * alpha) * volume / vmax
    elif prenormalize_method == 'sqrt':
        volume = volume / (e + 1e-9)
        volume_sqrt = torch.sqrt(volume)
        x = alpha + (1 - 2 * alpha) * volume_sqrt / vmax
    else:
        raise ValueError(f"Unknown prenormalize_method: {prenormalize_method}")

    if normalize_method == 'log':
        volume = x * 2 - 1
        volume = (volume - stats['log_mean']) / stats['log_std']
    elif normalize_method == 'logit':
        x = torch.clamp(x, 1e-9, 1 - 1e-9)
        volume = torch.log(x / (1 - x))
        volume = (volume - stats['logit_mean']) / stats['logit_std']
    else:
        raise ValueError(f"Unknown normalize_method: {normalize_method}")

    return volume


class NNConverter(nn.Module):
    """Trainable geometric conversion for irregular → regular radial binning.

    Initialized from fixed weight_mats (area-overlap weights), then learns
    improved mappings during training.  The decoder learns the pseudo-inverse,
    enabling reconstruction back to the original detector geometry at inference.

    Args:
        weight_mats: list of weight matrices [M_i × n_r_i] per layer
        lay_r_edges: list of r-edge arrays per layer
        lay_alphas:  list of alpha counts per layer
        dim_r_out:   uniform radial output dimension (M)
        alpha_out:   uniform angular output dimension (TARGET_ALPHA)
        eps:         noise scale for weight initialization
    """

    def __init__(self, weight_mats, lay_r_edges, lay_alphas, dim_r_out, alpha_out, eps=1e-5):
        super().__init__()
        self.lay_r_edges = lay_r_edges
        self.lay_alphas = lay_alphas
        self.dim_r_out = dim_r_out
        self.alpha_out = alpha_out
        self.num_layers = len(weight_mats)

        self.encs = nn.ModuleList([])
        self.decs = nn.ModuleList([])

        for i in range(self.num_layers):
            rdim_in = len(lay_r_edges[i]) - 1

            enc = nn.Linear(rdim_in, dim_r_out, bias=False)
            noise = torch.randn_like(weight_mats[i])
            enc.weight.data = weight_mats[i] + eps * noise
            self.encs.append(enc)

            dec = nn.Linear(dim_r_out, rdim_in, bias=False)
            inv_init = torch.linalg.pinv(weight_mats[i])
            noise2 = torch.randn_like(inv_init)
            dec.weight.data = inv_init + eps * noise2
            self.decs.append(dec)

    def enc(self, layers):
        """Encode per-layer tensors to uniform geometry.

        Args:
            layers: list of [N, n_alpha_i, n_r_i] tensors

        Returns:
            [N, 1, num_layers, alpha_out, dim_r_out]
        """
        n_shower = layers[0].shape[0]
        device = layers[0].device

        out = torch.zeros((n_shower, 1, self.num_layers, self.alpha_out, self.dim_r_out),
                          device=device)
        for i in range(len(layers)):
            o = self.encs[i](layers[i])  # [N, n_alpha_i, dim_r_out]

            if self.lay_alphas[i] == 1:
                o = torch.repeat_interleave(o, self.alpha_out, dim=-2) / self.alpha_out
            elif self.lay_alphas[i] != self.alpha_out:
                o = resample_alpha_torch(o, self.lay_alphas[i], self.alpha_out)

            out[:, 0, i] = o
        return out

    def dec(self, x):
        """Decode uniform geometry back to per-layer tensors.

        Args:
            x: [N, 1, num_layers, alpha_out, dim_r_out]
               or [N, num_layers, alpha_out, dim_r_out]

        Returns:
            list of [N, n_alpha_i, n_r_i] tensors
        """
        if x.dim() == 5:
            x = x.squeeze(1)

        out = []
        for i in range(self.num_layers):
            o = self.decs[i](x[:, i])  # [N, alpha_out, n_r_i]

            if self.lay_alphas[i] == 1:
                o = torch.sum(o, dim=-2, keepdim=True)
            elif self.lay_alphas[i] != self.alpha_out:
                o = resample_alpha_inverse_torch(o, self.alpha_out, self.lay_alphas[i])

            out.append(o)
        return out

    def forward(self, x):
        return self.enc(x)

    def sync_decoder(self):
        """Update decoder weights to pseudo-inverse of trained encoder weights.

        Call this after training before using dec() for inference-time
        reverse mapping.  This ensures the decoder reflects whatever the
        encoder learned during training.
        """
        with torch.no_grad():
            for i in range(self.num_layers):
                self.decs[i].weight.data = torch.linalg.pinv(self.encs[i].weight.data)

    def decode_to_flat(self, x, bin_starts, bin_ends, valid_layer_indices, all_counts):
        """Decode and flatten back to original shower voxel format.

        Args:
            x:                    [N, 1, num_layers, alpha_out, dim_r_out] model output
            bin_starts:           list of start indices per layer in original shower
            bin_ends:             list of end indices per layer in original shower
            valid_layer_indices:  list of absolute layer indices for valid layers
            all_counts:           total voxel count per layer (n_alpha * n_r)

        Returns:
            [N, total_voxels] tensor in original detector geometry
        """
        layers = self.dec(x)  # list of [N, n_alpha_i, n_r_i], one per valid layer
        n_shower = layers[0].shape[0]
        total_voxels = sum(all_counts)
        device = layers[0].device

        out = torch.zeros((n_shower, total_voxels), device=device)
        for out_idx, lid in enumerate(valid_layer_indices):
            n_a = self.lay_alphas[out_idx]
            n_r = len(self.lay_r_edges[out_idx]) - 1
            flat = layers[out_idx].reshape(n_shower, n_a * n_r)
            out[:, bin_starts[lid]:bin_ends[lid]] = flat
        return out


def load_nnconverter_from_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load NNConverter from a training checkpoint for inference-time decoding.

    Returns (nn_converter, nn_binning_info) or (None, None) if the checkpoint
    was trained without nnconverter.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    nc = ckpt.get('nn_converter', None)
    bi = ckpt.get('nn_binning_info', None)
    if nc is None:
        print("[NNConverter] nn_converter not found in checkpoint, no geometric inverse conversion needed")
        return None, None
    nc.eval()
    print(f"[NNConverter] Loaded from checkpoint, {nc.num_layers} layers total")
    return nc, bi


def decode_samples_with_checkpoint(samples, checkpoint_path: str, device: str = 'cpu'):
    """Decode model samples back to original detector geometry using NNConverter.

    Args:
        samples:         [N, 1, L, A, R] model output in uniform geometry
        checkpoint_path: path to training checkpoint containing nn_converter
        device:          torch device

    Returns:
        [N, total_voxels] tensor in original irregular geometry, or None if
        the checkpoint has no nn_converter.
    """
    nc, bi = load_nnconverter_from_checkpoint(checkpoint_path, device)
    if nc is None or bi is None:
        return None
    nc = nc.to(device)
    return nc.decode_to_flat(
        samples.to(device),
        bi['bin_starts'], bi['bin_ends'],
        bi['valid_layer_indices'], bi['all_counts']
    )


class CaloDataset(Dataset):

    def __init__(
        self,
        hdf5_path: str,
        dataset_name: str = 'dataset2',
        max_samples: int = None,
        normalize_method: str = 'log',
        xml_path: str = None,
        particle: str = None,
        weight_cache: str = None,
        ecut: float = 0.0,
        compute_cold_stats: bool = False,
        num_energy_bins: int = 10,
        alpha:float = 1e-6,
        fine_tune: bool = False,
        reshape_method: str = 'weight',
        prenormalize_method: str = 'log10',
        config_path: str = None,
    ):
        self.dataset_name = dataset_name
        self.normalize_method = normalize_method
        self.compute_cold_stats = compute_cold_stats
        self.num_energy_bins = num_energy_bins
        self.alpha = alpha
        self.fine_tune = fine_tune
        self.reshape_method = reshape_method
        self.vmax = 1.0
        self.E_bins = None
        self.avg_showers = None
        self.std_showers = None
        self.prenormalize_method = prenormalize_method
        self.cfg = load_config(config_path, dataset_name, particle)
        
        # mask related
        self.masks = None  # per-sample mask, shape: [N, 1, L, A, R]

        # nnconverter
        self.nn_converter = None
        self.nn_binning_info = None  # {bin_starts, bin_ends, lay_ids, all_counts}
        self.raw_layers = None       # list of [N, n_a_i, n_r_i] tensors (nnconverter mode)
        self.raw_energies = None     # [N, 1] raw energy values in MeV (nnconverter mode)
        self._norm_stats = None      # normalization stats dict (nnconverter mode)

        if dataset_name == 'dataset1':
            self._load_dataset1(
                hdf5_path, xml_path, particle,
                max_samples, weight_cache, ecut
            )
        else:
            self._load_dataset_standard(hdf5_path, max_samples)
         
        if self.fine_tune:
            self._compute_cold_statistics()
            self._continue_fine_tune()

    def _load_dataset1(
        self, hdf5_path, xml_path, particle,
        max_samples, weight_cache, ecut
    ):
        print(f"[Dataset1] Using {self.reshape_method} method for radial processing")

        lay_ids, lay_r_edges, lay_alphas = parse_binning_xml(xml_path, particle)

        valid_layer_indices = [
            i for i, r in enumerate(lay_r_edges) if len(r) > 1
        ]

        # Determine which method to use
        if self.reshape_method in ('weight', 'nnconverter'):
            all_r_edges, weight_mats = build_weight_mats(
                lay_r_edges, cache_path=weight_cache
            )
            M = len(all_r_edges) - 1
            mask_list = None

            if self.reshape_method == 'nnconverter':
                self.nn_converter = NNConverter(
                    weight_mats=[torch.from_numpy(weight_mats[i]) for i in valid_layer_indices],
                    lay_r_edges=[lay_r_edges[i] for i in valid_layer_indices],
                    lay_alphas=[lay_alphas[i] for i in valid_layer_indices],
                    dim_r_out=M,
                    alpha_out=TARGET_ALPHA,
                )
                print(f"[Dataset1] NNConverter created, {self.nn_converter.num_layers} layers "
                      f"({len(lay_r_edges) - len(valid_layer_indices)} invalid layers filtered out)")
        elif self.reshape_method == 'mask':
            M, mask_list = build_mask_info(lay_r_edges)
            weight_mats = None
            all_r_edges = None
        else:
            raise ValueError(f"reshape_method must be 'weight', 'mask' or 'nnconverter', got '{self.reshape_method}'")

        all_counts = get_all_voxel_counts(xml_path, particle)

        bin_starts = [sum(all_counts[:lid]) for lid in lay_ids]
        bin_ends = [
            bin_starts[i] + lay_alphas[i] * (len(lay_r_edges[i]) - 1)
            for i in range(len(lay_ids))
        ]

        with h5py.File(hdf5_path, 'r') as f:
            showers = f['showers'][:max_samples].astype(np.float32)
            energies = f['incident_energies'][:max_samples].astype(np.float32)

        if ecut > 0:
            showers[showers < ecut] = 0.0

        if self.reshape_method == 'nnconverter':
            self.nn_binning_info = {
                'bin_starts': bin_starts,
                'bin_ends': bin_ends,
                'lay_ids': lay_ids,
                'all_counts': all_counts,
                'valid_layer_indices': valid_layer_indices,
            }

        N = showers.shape[0]
        L_valid = len(valid_layer_indices)

        self.volume_size = (L_valid, TARGET_ALPHA, M)

        # ================================================================
        #  nnconverter branch: keep irregular geometry, normalize directly then feed to network
        #  enc/dec participate in the computation graph inside the model
        # ================================================================
        if self.reshape_method == 'nnconverter':
            self.raw_layers = []
            irreg_parts = []
            self.irreg_shapes = []  # [(n_a, n_r)] per valid layer
            for out_idx, i in enumerate(valid_layer_indices):
                n_a = lay_alphas[i]
                n_r = len(lay_r_edges[i]) - 1
                layer = showers[:, bin_starts[i]:bin_ends[i]].reshape(N, n_a, n_r)
                self.raw_layers.append(torch.from_numpy(layer.copy()).float())
                self.irreg_shapes.append((n_a, n_r))
                irreg_parts.append(layer.reshape(N, n_a * n_r))

            irreg_flat = np.concatenate(irreg_parts, axis=1)  # [N, total_irreg]
            energies_raw = energies.copy()
            energies_4d = energies.reshape(-1, 1, 1, 1)

            # Normalization on irregular data
            volume_for_norm = irreg_flat.reshape(N, 1, 1, -1)
            volume, stats = prenormalize_showers(
                volume_for_norm, energies_4d, self.alpha,
                self.prenormalize_method, self.normalize_method
            )
            if 'log_mean' in stats:
                self.log_mean = stats['log_mean']
                self.log_std = stats['log_std']
            if 'logit_mean' in stats:
                self.logit_mean = stats['logit_mean']
                self.logit_std = stats['logit_std']
            self.vmax = stats.get('vmax', 1.0)
            self.prenormalize_method = stats.get('prenormalize_method', self.prenormalize_method)

            self.raw_energies = torch.from_numpy(energies_raw).float()
            self._norm_stats = {
                'normalize_method': self.normalize_method,
                'prenormalize_method': self.prenormalize_method,
                'vmax': self.vmax,
            }
            if 'log_mean' in stats:
                self._norm_stats['log_mean'] = stats['log_mean']
                self._norm_stats['log_std'] = stats['log_std']
            if 'logit_mean' in stats:
                self._norm_stats['logit_mean'] = stats['logit_mean']
                self._norm_stats['logit_std'] = stats['logit_std']

            energies = np.log10(energies.reshape(-1, 1))
            e_min, e_max = self.cfg['energy_range']
            log_emin, log_emax = np.log10(e_min), np.log10(e_max)
            energies = (energies - log_emin) / (log_emax - log_emin)

            # [N, 1, 1, total_irreg]
            self.showers = torch.from_numpy(volume).float()
            self.energies = torch.from_numpy(energies).float()
            self.masks = torch.ones_like(self.showers)

        else:
            # ============================================================
            #  weight / mask branch: existing logic unchanged
            # ============================================================
            volume = np.zeros((N, L_valid, TARGET_ALPHA, M), dtype=np.float32)

            if self.reshape_method == 'mask':
                masks = np.zeros((N, L_valid, TARGET_ALPHA, M), dtype=np.float32)

            for out_idx, i in enumerate(valid_layer_indices):
                n_a = lay_alphas[i]
                n_r = len(lay_r_edges[i]) - 1

                layer = showers[:, bin_starts[i]:bin_ends[i]].reshape(N, n_a, n_r)

                if self.reshape_method == 'weight':
                    layer = np.einsum('nar,mr->nam', layer, weight_mats[i])
                elif self.reshape_method == 'mask':
                    layer_padded = np.zeros((N, n_a, M), dtype=np.float32)
                    layer_padded[:, :, :n_r] = layer
                    layer = layer_padded
                    layer_mask = np.zeros((N, TARGET_ALPHA, M), dtype=np.float32)
                    layer_mask[:, :, :] = mask_list[i][None, None, :]

                layer = resample_alpha(layer, n_a, TARGET_ALPHA)
                volume[:, out_idx] = layer

                if self.reshape_method == 'mask':
                    masks[:, out_idx] = layer_mask

            energies_4d = energies.reshape(-1, 1, 1, 1)
            volume, stats = prenormalize_showers(
                volume, energies_4d, self.alpha,
                self.prenormalize_method, self.normalize_method
            )
            if 'log_mean' in stats:
                self.log_mean = stats['log_mean']
                self.log_std = stats['log_std']
            if 'logit_mean' in stats:
                self.logit_mean = stats['logit_mean']
                self.logit_std = stats['logit_std']
            self.vmax = stats.get('vmax', 1.0)
            self.prenormalize_method = stats.get('prenormalize_method', self.prenormalize_method)

            energies = np.log10(energies.reshape(-1, 1))
            e_min, e_max = self.cfg['energy_range']
            log_emin, log_emax = np.log10(e_min), np.log10(e_max)
            energies = (energies - log_emin) / (log_emax - log_emin)

            self.showers = torch.from_numpy(volume[:, None]).float()
            self.energies = torch.from_numpy(energies).float()

            if self.reshape_method == 'mask':
                self.masks = torch.from_numpy(masks[:, None]).float()
                print(f"[Dataset1] masks shape: {self.masks.shape}")
            else:
                self.masks = torch.ones_like(self.showers)

    def _load_dataset_standard(self, hdf5_path, max_samples):
        D, H, W = self.cfg['volume_size']
        e_min, e_max = self.cfg['energy_range']

        with h5py.File(hdf5_path, 'r') as f:
            showers = f['showers'][:max_samples].astype(np.float32)
            energies = f['incident_energies'][:max_samples].astype(np.float32)

        energies = energies.reshape(-1,1)
        showers, stats = prenormalize_showers(showers, energies, self.alpha, self.prenormalize_method, self.normalize_method)
        if 'log_mean' in stats:
            self.log_mean = stats['log_mean']
            self.log_std = stats['log_std']
        if 'logit_mean' in stats:
            self.logit_mean = stats['logit_mean']
            self.logit_std = stats['logit_std']
        self.vmax = stats.get('vmax', 1.0)
        self.prenormalize_method = stats.get('prenormalize_method', self.prenormalize_method)
        energies = np.log10(energies)
        energies = (energies-np.log10(e_min))/(np.log10(e_max)-np.log10(e_min))
        self.showers = torch.from_numpy(showers.reshape(-1, 1, D, H, W)).float()
        self.energies = torch.from_numpy(energies).float()

        # dataset2/3 use all-ones mask (all voxels valid)
        self.masks = torch.ones_like(self.showers)
        
    def _continue_fine_tune(self):
        showers = self._bins_normalize_showers(self.showers, self.energies)
        self.showers = showers
        
    def lookup_avg_std_shower(self, energies: torch.Tensor):
        """Look up avg and std shower by energy"""
        # energies: [B, 1]
        energies_flat = energies.squeeze()
        
        idxs = torch.bucketize(energies_flat, self.E_bins.to(energies.device)) - 1
        
        idxs = torch.clamp(idxs, 0, len(self.avg_showers) - 1)
        
        avg = self.avg_showers[idxs].to(energies.device)
        std = self.std_showers[idxs].to(energies.device)
        
        return avg, std
        
    def _bins_normalize_showers(self, showers, energies):

        avg,std = self.lookup_avg_std_shower(energies)
        showers = (showers - avg) / (std + self.alpha)
        return showers

    def _normalize_energies(self, energies, e_min, e_max):
        log_e = np.log10(energies)
        log_min = np.log10(e_min)
        log_max = np.log10(e_max)
        return (log_e - log_min) / (log_max - log_min)

    def _normalize_energies_dataset1(self, energies, particle):
        from dataset1_preprocess import ENERGY_RANGE as DS1_ENERGY_RANGE
        e_min, e_max = DS1_ENERGY_RANGE[particle]
        return self._normalize_energies(energies, e_min, e_max)
    

    def _compute_cold_statistics(self):
        print(f"\nComputing statistics (num energy bins: {self.num_energy_bins})")
        energies_flat = self.energies.squeeze()
        
        e_min, e_max = energies_flat.min(), energies_flat.max()
        self.E_bins = torch.linspace(e_min, e_max, self.num_energy_bins + 1)
        
        avg_showers_list = []
        std_showers_list = []
        
        for i in range(self.num_energy_bins):
            mask = (energies_flat >= self.E_bins[i]) & (energies_flat < self.E_bins[i+1])
            
            if i == self.num_energy_bins - 1:  # Last bin includes upper bound
                mask = (energies_flat >= self.E_bins[i]) & (energies_flat <= self.E_bins[i+1])
            
            num_samples_in_bin = mask.sum()
            
            if num_samples_in_bin > 1:
                showers_in_bin = self.showers[mask]
                avg_shower = showers_in_bin.mean(dim=0, keepdim=True)
                std_shower = showers_in_bin.std(dim=0, keepdim=True)
                
                print(f"  Bin {i}: [{self.E_bins[i]:.4f}, {self.E_bins[i+1]:.4f}] - {num_samples_in_bin} samples")
            else:
                # If no samples in this bin, use global statistics
                avg_shower = self.showers.mean(dim=0, keepdim=True)
                std_shower = self.showers.std(dim=0, keepdim=True)
                print(f"  Bin {i}: [{self.E_bins[i]:.4f}, {self.E_bins[i+1]:.4f}] - 0/1 samples (using global stats)")
            
            avg_showers_list.append(avg_shower)
            std_showers_list.append(std_shower)
        
        self.avg_showers = torch.cat(avg_showers_list, dim=0)
        self.std_showers = torch.cat(std_showers_list, dim=0)
        
        print(f"  avg_showers: {self.avg_showers.shape}")
        print(f"  std_showers: {self.std_showers.shape}")
        print(f"  E_bins: {self.E_bins.shape} - range [{self.E_bins[0]:.4f}, {self.E_bins[-1]:.4f}]")

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, idx):
        return self.showers[idx], self.energies[idx], self.masks[idx]


def get_calo_dataloader(
    hdf5_path: str,
    dataset_name: str = 'dataset2',
    batch_size: int = None,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: int = None,
    normalize_method: str = None,
    xml_path: str = None,
    particle: str = None,
    weight_cache: str = None,
    ecut: float = 0.0,
    compute_cold_stats: bool = None,
    num_energy_bins: int = None,
    alpha: float = None,
    fine_tune: bool = None,
    reshape_method: str = None,
    prenormalize_method: str = None,
    config_path: str = None,
):
    cfg = load_config(config_path, dataset_name, particle)

    if batch_size is None:
        batch_size = cfg.get('batch_size', 16)
    if normalize_method is None:
        normalize_method = cfg.get('normalize_method', 'log')
    if weight_cache is None:
        weight_cache = cfg.get('weight_cache', 'data/weight_mats.pkl')
    if compute_cold_stats is None:
        compute_cold_stats = cfg.get('cold_diffusion', False)
    if num_energy_bins is None:
        num_energy_bins = cfg.get('num_energy_bins', 10)
    if alpha is None:
        alpha = cfg.get('alpha', 1e-6)
    if fine_tune is None:
        fine_tune = cfg.get('fine_tune', False)
    if reshape_method is None:
        reshape_method = cfg.get('reshape_method', 'weight')
    if prenormalize_method is None:
        prenormalize_method = cfg.get('prenormalize_method', 'log10')

    dataset = CaloDataset(
        hdf5_path=hdf5_path,
        dataset_name=dataset_name,
        max_samples=max_samples,
        normalize_method=normalize_method,
        xml_path=xml_path,
        particle=particle,
        weight_cache=weight_cache,
        ecut=ecut,
        compute_cold_stats=compute_cold_stats,
        num_energy_bins=num_energy_bins,
        alpha=alpha,
        fine_tune=fine_tune,
        reshape_method=reshape_method,
        prenormalize_method=prenormalize_method,
        config_path=config_path,
    )

    if reshape_method == 'nnconverter' and num_workers > 0:
        print("[NNConverter] Warning: num_workers forced to 0 in nnconverter mode "
              "(trainable modules cannot be shared across processes)")
        num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )