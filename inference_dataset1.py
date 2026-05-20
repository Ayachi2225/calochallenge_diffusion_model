import os
import argparse
import numpy as np
import h5py
import torch
from model import DDIMModel3D
from utils import generate_energies, normalize_energies, denormalize_showers, decontinue_fine_tune, load_config
from dataset1_preprocess import (
    parse_binning_xml,
    build_weight_mats,
    build_mask_info,
    get_all_voxel_counts,
    resample_alpha,
    TARGET_ALPHA,
)


def inverse_resample_alpha(layer: np.ndarray, src_alpha: int, dst_alpha: int) -> np.ndarray:
    if src_alpha == dst_alpha:
        return layer
    
    src_edges = np.linspace(0, 1, src_alpha + 1)
    dst_edges = np.linspace(0, 1, dst_alpha + 1)
    
    W_a = np.zeros((dst_alpha, src_alpha), dtype=np.float32)
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
    
    return np.einsum('nsm,ds->ndm', layer, W_a).astype(np.float32)


def build_conservative_reverse_weight_mats(lay_r_edges: list, all_r_edges: list):
    """Build conservative inverse resampling matrices (for weight method)"""
    print("[INFO] Building conservative inverse resampling matrices...")
    M = len(all_r_edges) - 1
    reverse_weight_mats = []
    
    for r_edges in lay_r_edges:
        n_r = len(r_edges) - 1
        W_back = np.zeros((n_r, M), dtype=np.float32)
        
        for dst_orig in range(n_r):
            r_lo = r_edges[dst_orig]
            r_hi = r_edges[dst_orig + 1]
            
            for src_uni in range(M):
                lo = all_r_edges[src_uni]
                hi = all_r_edges[src_uni + 1]
                
                overlap = max(0.0, min(r_hi, hi) - max(r_lo, lo))
                
                if overlap > 0:
                    W_back[dst_orig, src_uni] = overlap / (hi - lo)
                    
        reverse_weight_mats.append(W_back)
        
    return reverse_weight_mats


def reconstruct_original_format_weight(
    volume: np.ndarray,
    lay_ids: list,
    lay_r_edges: list,
    lay_alphas: list,
    reverse_weight_mats: list,
) -> np.ndarray:
    """Reconstruct original format using weight method"""
    N = volume.shape[0]
    
    valid_layer_indices = [
        i for i, r in enumerate(lay_r_edges) if len(r) > 1
    ]
    
    all_counts = [
        lay_alphas[i] * (len(lay_r_edges[i]) - 1)
        for i in range(len(lay_ids))
    ]
    total_voxels = sum(all_counts)
    
    bin_starts = [sum(all_counts[:lid]) for lid in lay_ids]
    bin_ends = [
        bin_starts[i] + all_counts[i]
        for i in range(len(lay_ids))
    ]
    
    showers = np.zeros((N, total_voxels), dtype=np.float32)
    
    for out_idx, i in enumerate(valid_layer_indices):
        n_a = lay_alphas[i]
        n_r = len(lay_r_edges[i]) - 1
        
        layer = volume[:, out_idx, :, :]
        layer = inverse_resample_alpha(layer, TARGET_ALPHA, n_a)
        
        W_back = reverse_weight_mats[i]
        layer_reconstructed = np.einsum('nam,rm->nar', layer, W_back)
        
        layer_flat = layer_reconstructed.reshape(N, -1)
        showers[:, bin_starts[i]:bin_ends[i]] = layer_flat
    
    return showers


def reconstruct_original_format_mask(
    volume: np.ndarray,
    lay_ids: list,
    lay_r_edges: list,
    lay_alphas: list,
) -> np.ndarray:
    """Reconstruct original format using mask method (direct crop)"""
    N = volume.shape[0]
    
    valid_layer_indices = [
        i for i, r in enumerate(lay_r_edges) if len(r) > 1
    ]
    
    all_counts = [
        lay_alphas[i] * (len(lay_r_edges[i]) - 1)
        for i in range(len(lay_ids))
    ]
    total_voxels = sum(all_counts)
    
    bin_starts = [sum(all_counts[:lid]) for lid in lay_ids]
    bin_ends = [
        bin_starts[i] + all_counts[i]
        for i in range(len(lay_ids))
    ]
    
    showers = np.zeros((N, total_voxels), dtype=np.float32)
    
    for out_idx, i in enumerate(valid_layer_indices):
        n_a = lay_alphas[i]
        n_r = len(lay_r_edges[i]) - 1
        
        # [N, TARGET_ALPHA, max_r_bins]
        layer = volume[:, out_idx, :, :]
        
        # Inverse angular resample to original n_a
        layer = inverse_resample_alpha(layer, TARGET_ALPHA, n_a)

        # Direct radial crop to first n_r bins (mask method)
        layer_reconstructed = layer[:, :, :n_r]  # [N, n_a, n_r]
        
        layer_flat = layer_reconstructed.reshape(N, -1)
        showers[:, bin_starts[i]:bin_ends[i]] = layer_flat
    
    return showers


def inference(
    checkpoint_path: str,
    output_path: str,
    xml_path: str,
    particle: str = 'photon',
    num_samples: int = 1000,
    batch_size: int = 16,
    num_steps: int = 50,
    energy_distribution: str = 'uniform',
    weight_cache: str = 'data/weight_mats.pkl',
    seed: int = None,
    device: str = None,
    constant_energy: float = None,
    energy_file_dir: str = None,
    cold_noise_scale: float = 1.0,
    sample_method: str = None,
    sample_eta: float = None,
    n_correct: int = 1,
    delta: float = 0.17,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"{'='*70}")
    print(f"Inference config:")
    print(f"  Model checkpoint: {checkpoint_path}")
    print(f"  Output file: {output_path}")
    print(f"  Dataset: dataset1")
    print(f"  Particle type: {particle}")
    print(f"  XML config: {xml_path}")
    print(f"  Device: {device}")
    print(f"  Num samples: {num_samples}")
    print(f"  Energy distribution: {energy_distribution}")
    if seed is not None:
        print(f"  Random seed: {seed}")
    print(f"{'='*70}\n")
    
    print("Parsing XML geometry config...")
    lay_ids, lay_r_edges, lay_alphas = parse_binning_xml(xml_path, particle)
    
    # Load checkpoint to determine which method to use
    print("Loading model checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_steps = ckpt.get('num_steps', num_steps)
    batch_size = ckpt.get('batch_size', batch_size)
    print(f"  Sampling steps: {num_steps}")
    print(f"  Batch size: {batch_size}")


    reshape_method = ckpt.get('reshape_method', 'weight')  # Default to weight for backward compatibility
    print(f"  Radial processing method: {reshape_method}")

    # Choose processing flow based on method
    nn_converter = None
    nn_binning_info = None
    if reshape_method == 'nnconverter':
        nn_converter = ckpt.get('nn_converter')
        nn_binning_info = ckpt.get('nn_binning_info')
        if nn_converter is None:
            raise ValueError("nn_converter not found in checkpoint, cannot use nnconverter method")
        M = nn_converter.dim_r_out
        all_r_edges = None
        print(f"  NNConverter: {nn_converter.num_layers} layers, "
              f"alpha_out={nn_converter.alpha_out}, dim_r_out={M}")
    elif reshape_method == 'weight':
        all_r_edges, weight_mats = build_weight_mats(lay_r_edges, cache_path=weight_cache)
        M = len(all_r_edges) - 1
    elif reshape_method == 'mask':
        M, mask_list = build_mask_info(lay_r_edges)
        all_r_edges = None
    else:
        raise ValueError(f"Unknown reshape_method: {reshape_method}")
    
    valid_layer_indices = [
        i for i, r in enumerate(lay_r_edges) if len(r) > 1
    ]
    L_valid = len(valid_layer_indices)
    
    volume_size = (L_valid, TARGET_ALPHA, M)
    print(f"  Volume size: {L_valid} layers × {TARGET_ALPHA} angles × {M} radial")
    print(f"  Total layers: {len(lay_ids)}, Valid layers: {L_valid}\n")
    
    cfg = load_config(None, 'dataset1', particle)
    e_min, e_max = cfg['energy_range']
    print(f"Energy range: {e_min} - {e_max} MeV ({particle})\n")
    
    training_obj = ckpt.get('training_obj', 'noise_pred')
    cold_diffusion = ckpt.get('cold_diffusion', False)
    E_bins = ckpt.get('E_bins', None)
    avg_showers = ckpt.get('avg_showers', None)
    std_showers = ckpt.get('std_showers', None)
    use_mask = ckpt.get('use_mask', False)
    
    if sample_method is None:
        sample_method = ckpt.get('sample_method', 'pndm')
    if sample_eta is None:
        sample_eta = ckpt.get('sample_eta', 0.0)
    n_correct = ckpt.get('n_correct', n_correct)
    delta = ckpt.get('delta', delta)
    
    print(f"Model info:")
    print(f"  Training epoch: {ckpt['epoch']}")
    print(f"  Training objective: {training_obj}")
    print(f"  Using mask: {use_mask}")
    print(f"  Cold diffusion: {cold_diffusion}")
    if cold_diffusion:
        print(f"  Num energy bins: {len(E_bins) - 1 if E_bins is not None else 0}")
        print(f"  Noise scale: {cold_noise_scale}")
    print(f"  Sampling method: {sample_method}")
    if sample_method == 'ddim':
        print(f"  DDIM eta: {sample_eta}")
    if sample_method == 'pc':
        print(f"  PC correction steps: {n_correct}")
        print(f"  PC step size coefficient: {delta}")
    print()
    
    t_dim = cfg.get('t_dim', 128)
    e_dim = cfg.get('e_dim', 128)
    model = DDIMModel3D(
        t_dim=t_dim,
        e_dim=e_dim,
        training_obj=training_obj,
        cold_diffusion=cold_diffusion,
        E_bins=E_bins,
        avg_showers=avg_showers,
        std_showers=std_showers,
        cold_noise_scale=cold_noise_scale,
        use_mask=use_mask,
    ).to(device)
    
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    normalize_method = ckpt.get('normalize_method', None)
    log_mean = ckpt.get('log_mean', None)
    log_std = ckpt.get('log_std', None)
    logit_mean = ckpt.get('logit_mean', None)
    logit_std = ckpt.get('logit_std', None)
    
    print(f"Normalization parameters:")
    print(f"  Method: {normalize_method}")
    if normalize_method == 'log':
        print(f"  log_mean: {log_mean}, log_std: {log_std}")
    elif normalize_method == 'logit':
        print(f"  logit_mean: {logit_mean}, logit_std: {logit_std}")
    print()
    
    print(f"Generating {num_samples} energy samples (distribution: {energy_distribution})...")
    energies = generate_energies(
        num_samples, 
        energy_distribution, 
        e_min, 
        e_max, 
        seed, 
        constant_energy,
        energy_file_dir=energy_file_dir
    )
    
    print(f"Energy statistics:")
    print(f"  Min: {energies.min():.2f} MeV")
    print(f"  Max: {energies.max():.2f} MeV")
    print(f"  Mean: {energies.mean():.2f} MeV")
    print(f"  Median: {np.median(energies):.2f} MeV")
    print(f"  Std: {energies.std():.2f} MeV\n")
    
    energies_log = np.log10(energies.reshape(-1, 1, 1, 1))
    energies_norm = normalize_energies(energies_log, e_min, e_max)
    energies_norm = energies_norm.reshape(-1, 1)
    
    D, H, W = volume_size
    print(f"Starting shower generation (method: {sample_method}, steps: {num_steps})...")
    all_volumes = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            batch_energies = torch.from_numpy(
                energies_norm[start_idx:end_idx].reshape(-1, 1)
            ).float().to(device)
            
            samples = model.sample(
                shape=(current_batch_size, 1, D, H, W),
                energy=batch_energies,
                num_steps=num_steps,
                device=device,
                cold_noise_scale=cold_noise_scale,
                method=sample_method,
                eta=sample_eta,
                n_correct=n_correct,
                delta=delta,
            )
            
            all_volumes.append(samples.cpu().numpy())
            
            if (i + 1) % max(1, num_batches // 10) == 0 or (i + 1) == num_batches:
                print(f"  Batch {i+1}/{num_batches} complete "
                      f"({end_idx}/{num_samples} samples, "
                      f"{end_idx/num_samples*100:.1f}%)")
    
    all_volumes = np.concatenate(all_volumes, axis=0)
    all_volumes = all_volumes.squeeze(1)
    
    if ckpt.get('fine_tune', False):
        print("\nApplying fine-tune post-processing...")
        all_volumes = decontinue_fine_tune(
            all_volumes, 
            energies_norm, 
            E_bins, 
            avg_showers, 
            std_showers, 
            alpha=ckpt.get('alpha', 1e-6)
        )
    
    print("\nDenormalizing showers...")
    stats = {
        'normalize_method': normalize_method,
        'log_mean': log_mean,
        'log_std': log_std,
        'logit_mean': logit_mean,
        'logit_std': logit_std,
    }
    prenormalize_method = ckpt.get('prenormalize_method', 'log10')
    vmax = ckpt.get('vmax', None)
    all_volumes = denormalize_showers(
        all_volumes, energies, stats,
        alpha=ckpt.get('alpha', 1e-6),
        prenormalize_method=prenormalize_method,
        vmax=vmax,
    )
    
    print("Reconstructing to original HDF5 format...")
    if reshape_method == 'nnconverter':
        volume_t = torch.from_numpy(all_volumes[:, None]).float().to(device)
        nn_converter = nn_converter.to(device)
        nn_converter.eval()
        with torch.no_grad():
            all_showers = nn_converter.decode_to_flat(
                volume_t,
                nn_binning_info['bin_starts'],
                nn_binning_info['bin_ends'],
                nn_binning_info['valid_layer_indices'],
                nn_binning_info['all_counts'],
            ).cpu().numpy()
        print(f"  NNConverter decode complete: {all_showers.shape}")
    elif reshape_method == 'weight':
        reverse_weight_mats = build_conservative_reverse_weight_mats(lay_r_edges, all_r_edges)
        all_showers = reconstruct_original_format_weight(
            all_volumes,
            lay_ids,
            lay_r_edges,
            lay_alphas,
            reverse_weight_mats,
        )
    elif reshape_method == 'mask':
        all_showers = reconstruct_original_format_mask(
            all_volumes,
            lay_ids,
            lay_r_edges,
            lay_alphas,
        )
    
    energies = energies.reshape(-1, 1)
    
    print(f"\nShower statistics:")
    print(f"  Shape: {all_showers.shape}")
    print(f"  Min: {all_showers.min():.6f} MeV")
    print(f"  Max: {all_showers.max():.6f} MeV")
    print(f"  Mean: {all_showers.mean():.6f} MeV")
    print(f"  Median: {np.median(all_showers):.6f} MeV")
    print(f"  Non-zero ratio: {(all_showers > 0).sum() / all_showers.size * 100:.2f}%")
    
    total_deposited = all_showers.sum(axis=1)
    energy_ratio = total_deposited / energies.squeeze()
    print(f"\nEnergy conservation check:")
    print(f"  Deposited / Incident energy:")
    print(f"    Mean: {energy_ratio.mean():.4f}")
    print(f"    Std: {energy_ratio.std():.4f}")
    print(f"    Range: [{energy_ratio.min():.4f}, {energy_ratio.max():.4f}]")
    
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('showers', data=all_showers, compression='gzip')
        f.create_dataset('incident_energies', data=energies, compression='gzip')
        
        f.attrs['dataset_name'] = 'dataset1'
        f.attrs['particle'] = particle
        f.attrs['num_samples'] = num_samples
        f.attrs['energy_distribution'] = energy_distribution
        f.attrs['energy_min'] = e_min
        f.attrs['energy_max'] = e_max
        f.attrs['xml_path'] = xml_path
        
        f.attrs['checkpoint'] = checkpoint_path
        f.attrs['training_obj'] = training_obj
        f.attrs['model_epoch'] = ckpt['epoch']
        f.attrs['reshape_method'] = reshape_method
        
        f.attrs['sample_method'] = sample_method
        f.attrs['num_steps'] = num_steps
        if sample_method == 'ddim':
            f.attrs['sample_eta'] = sample_eta
        if sample_method == 'pc':
            f.attrs['n_correct'] = n_correct
            f.attrs['delta'] = delta
        
        f.attrs['cold_diffusion'] = cold_diffusion
        if cold_diffusion:
            f.attrs['cold_noise_scale'] = cold_noise_scale
            f.attrs['num_energy_bins'] = len(E_bins) - 1 if E_bins is not None else 0
        
        f.attrs['normalize_method'] = normalize_method if normalize_method else 'none'
        
        if seed is not None:
            f.attrs['seed'] = seed
        
        f.attrs['energy_mean'] = energies.mean()
        f.attrs['energy_std'] = energies.std()
        f.attrs['energy_ratio_mean'] = energy_ratio.mean()
        f.attrs['energy_ratio_std'] = energy_ratio.std()
    
    print(f"\n{'='*70}")
    print("Inference complete!")
    print(f"  Output file: {output_path}")
    print(f"  Generated samples: {num_samples}")
    print(f"  Particle type: {particle}")
    print(f"  Training objective: {training_obj}")
    print(f"  Radial processing: {reshape_method}")
    print(f"  Sampling method: {sample_method}")
    if sample_method == 'pc':
        print(f"  PC params: n_correct={n_correct}, delta={delta}")
    print(f"  Energy conservation: {energy_ratio.mean():.4f} ± {energy_ratio.std():.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dataset1 inference (supports multiple training objectives and sampling methods)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Inference with mask-method trained model:
   python inference_dataset1.py \
       --checkpoint models/ddim3d_dataset1_photon_mean_pred_mask_best.pt \
       --output generated_dataset1_photons_mask.hdf5 \
       --xml data/binning_dataset_1_photons.xml \
       --particle photon \
       --num_samples 10 \
       --distribution load --energy_file_dir data/dataset_1_photons_2.hdf5 \
       --seed 42

   python inference_dataset1.py \
       --checkpoint models/ddim3d_dataset1_photon_mean_pred_best.pt \
       --output generated_dataset1_photons_mean_pred_nn.hdf5 \
       --xml data/binning_dataset_1_photons.xml \
       --particle photon \
       --num_samples 10 \
       --distribution load --energy_file_dir data/dataset_1_photons_2.hdf5 \
       --seed 42

   python inference_dataset1.py \
       --checkpoint models/ddim3d_dataset1_photon_hybrid_best.pt \
       --output generated_dataset1_photons_hybrid_nn.hdf5 \
       --xml data/binning_dataset_1_photons.xml \
       --particle photon \
       --num_samples 10 \
       --distribution load --energy_file_dir data/dataset_1_photons_2.hdf5 \
       --seed 42

2. Inference with weight-method trained model (backward compatible):
   python inference_dataset1.py \
       --checkpoint models/ddim3d_dataset1_photon_mean_pred_best.pt \
       --output generated_dataset1_photons_weight.hdf5 \
       --xml data/binning_dataset_1_photons.xml \
       --particle photon \
       --num_samples 10 \
       --distribution load --energy_file_dir data/dataset_1_photons_2.hdf5 \
       --seed 42

3. Pions inference:
   python inference_dataset1.py \\
       --checkpoint ddim3d_dataset1_pion.pt \\
       --output generated_dataset1_pions.hdf5 \\
       --xml data/binning_dataset_1_pions.xml \\
       --particle pion \\
       --num_samples 1000 \\
       --distribution normal \\
       --seed 42

4. Fast DDIM sampling:
   python inference_dataset1.py \\
       --checkpoint ddim3d_dataset1_photon_hybrid_mask.pt \\
       --output generated_fast.hdf5 \\
       --xml data/binning_dataset_1_photons.xml \\
       --particle photon \\
       --sample_method ddim \\
       --sample_eta 0.0 \\
       --num_steps 25 \\
       --num_samples 500

5. Score-based diffusion -- Euler-Maruyama stochastic sampling:
   python inference_dataset1.py \\
       --checkpoint ddim3d_dataset1_photon_hybrid_mask.pt \\
       --output generated_em.hdf5 \\
       --xml data/binning_dataset_1_photons.xml \\
       --particle photon \\
       --sample_method euler_maruyama \\
       --num_steps 200 \\
       --num_samples 500

6. Score-based diffusion -- Predictor-Corrector high quality sampling:
   python inference_dataset1.py \\
       --checkpoint ddim3d_dataset1_photon_hybrid_mask.pt \\
       --output generated_pc.hdf5 \\
       --xml data/binning_dataset_1_photons.xml \\
       --particle photon \\
       --sample_method pc \\
       --num_steps 100 --n_correct 2 --delta 0.17 \\
       --num_samples 500

7. Score-based diffusion -- Probability Flow ODE deterministic sampling:
   python inference_dataset1.py \\
       --checkpoint ddim3d_dataset1_photon_hybrid_mask.pt \\
       --output generated_pf.hdf5 \\
       --xml data/binning_dataset_1_photons.xml \\
       --particle photon \\
       --sample_method prob_flow \\
       --num_steps 100 \\
       --num_samples 500
        """
    )
    
    required = parser.add_argument_group('Required')
    required.add_argument('--checkpoint', type=str, required=True,
                         help='Model checkpoint path')
    required.add_argument('--output', type=str, required=True,
                         help='Output HDF5 file path')
    required.add_argument('--xml', type=str, required=True,
                         help='XML geometry config file path')

    data_group = parser.add_argument_group('Dataset params')
    data_group.add_argument('--particle', type=str, default='photon',
                           choices=['photon', 'pion'],
                           help='Particle type')
    data_group.add_argument('--num_samples', type=int, default=1000,
                           help='Number of generated samples')
    data_group.add_argument('--weight_cache', type=str, default='data/weight_mats.pkl',
                           help='Weight matrix cache path')

    energy_group = parser.add_argument_group('Energy params')
    energy_group.add_argument('--distribution', type=str, default='uniform',
                             choices=['uniform', 'normal', 'lognormal',
                                     'exponential', 'constant', 'load'],
                             help='Energy distribution type')
    energy_group.add_argument('--constant_energy', type=float, default=None,
                             help='Constant energy value (used when distribution=constant, unit: MeV)')
    energy_group.add_argument('--energy_file_dir', type=str, default=None,
                             help='Energy file path (used when distribution=load)')

    sample_group = parser.add_argument_group('Sampling params')
    sample_group.add_argument('--sample_method', type=str, default=None,
                             choices=['pndm', 'ddim', 'euler_maruyama', 'prob_flow', 'pc'],
                             help='Sampling method (None=use checkpoint setting)')
    sample_group.add_argument('--num_steps', type=int, default=50,
                             help='Sampling steps')
    sample_group.add_argument('--sample_eta', type=float, default=None,
                             help='DDIM randomness param (0=deterministic, 1=DDPM, None=use checkpoint setting)')
    sample_group.add_argument('--n_correct', type=int, default=1,
                             help='PC Langevin correction steps')
    sample_group.add_argument('--delta', type=float, default=0.17,
                             help='PC Langevin step size coefficient')
    sample_group.add_argument('--batch_size', type=int, default=16,
                             help='Batch size')

    cold_group = parser.add_argument_group('Cold diffusion params')
    cold_group.add_argument('--cold_noise_scale', type=float, default=1.0,
                           help='Cold diffusion noise scaling factor (cold diffusion models only)')

    other_group = parser.add_argument_group('Other params')
    other_group.add_argument('--seed', type=int, default=None,
                            help='Random seed (optional)')
    other_group.add_argument('--device', type=str, default=None,
                            choices=['cuda', 'cpu'],
                            help='Compute device (default: auto-detect)')
    
    args = parser.parse_args()
    
    if args.distribution == 'load' and args.energy_file_dir is None:
        parser.error("--distribution load requires --energy_file_dir")
    
    if args.sample_eta is not None and (args.sample_eta < 0 or args.sample_eta > 1):
        parser.error("--sample_eta must be in [0, 1] range")
    
    inference(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        xml_path=args.xml,
        particle=args.particle,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        energy_distribution=args.distribution,
        weight_cache=args.weight_cache,
        seed=args.seed,
        device=args.device,
        constant_energy=args.constant_energy,
        energy_file_dir=args.energy_file_dir,
        cold_noise_scale=args.cold_noise_scale,
        sample_method=args.sample_method,
        sample_eta=args.sample_eta,
        n_correct=args.n_correct,
        delta=args.delta,
    )