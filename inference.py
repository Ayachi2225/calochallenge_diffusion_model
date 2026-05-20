import os
import argparse
import numpy as np
import h5py
import torch
from model import DDIMModel3D
from utils import generate_energies, normalize_energies, denormalize_showers, decontinue_fine_tune, load_config


def inference(
    checkpoint_path: str,
    output_path: str,
    dataset_name: str = 'dataset2',
    num_samples: int = None,
    batch_size: int = 16,
    num_steps: int = 50,
    energy_distribution: str = 'uniform',
    seed: int = None,
    device: str = None,
    constant_energy: float = None,
    cold_noise_scale: float = 1.0,
    energy_file_dir: str = None,
    sample_method: str = None,
    sample_eta: float = None,
    n_correct: int = 1,
    delta: float = 0.17,
    particle: str = None,
    config_path: str = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(config_path, dataset_name, particle)
    D, H, W = cfg['volume_size']
    e_min, e_max = cfg['energy_range']

    print(f"{'='*70}")
    print(f"Inference config:")
    print(f"  Model checkpoint: {checkpoint_path}")
    print(f"  Output file: {output_path}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Device: {device}")
    print(f"  Num samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sampling steps: {num_steps}")
    print(f"  Energy distribution: {energy_distribution}")
    if seed is not None:
        print(f"  Random seed: {seed}")
    print(f"{'='*70}\n")
    
    print(f"Dataset dimensions: (layers={D}, alphas={H}, rbins={W})")
    print(f"Energy range: {e_min} - {e_max} MeV\n")
    
    print("Loading model checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    training_obj = ckpt.get('training_obj', 'noise_pred')
    cold_diffusion = ckpt.get('cold_diffusion', False)
    E_bins = ckpt.get('E_bins', None)
    avg_showers = ckpt.get('avg_showers', None)
    std_showers = ckpt.get('std_showers', None)
    
    if sample_method is None:
        sample_method = ckpt.get('sample_method', 'pndm')
    if sample_eta is None:
        sample_eta = ckpt.get('sample_eta', 0.0)
    n_correct = ckpt.get('n_correct', n_correct)
    delta = ckpt.get('delta', delta)
    
    print(f"Model info:")
    print(f"  Training epoch: {ckpt['epoch']}")
    print(f"  Training objective: {training_obj}")
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
    use_mask = ckpt.get('use_mask', False)
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
    
    print(f"Starting shower generation (method: {sample_method}, steps: {num_steps})...")
    all_showers = []
    
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
            
            all_showers.append(samples.cpu().numpy())
            
            if (i + 1) % max(1, num_batches // 10) == 0 or (i + 1) == num_batches:
                print(f"  Batch {i+1}/{num_batches} complete "
                      f"({end_idx}/{num_samples} samples, "
                      f"{end_idx/num_samples*100:.1f}%)")
    
    all_showers = np.concatenate(all_showers, axis=0)
    all_showers = all_showers.squeeze(1)
    
    if ckpt.get('fine_tune', False):
        print("\nApplying fine-tune post-processing...")
        all_showers = decontinue_fine_tune(
            all_showers, 
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
    all_showers = denormalize_showers(
        all_showers, energies.reshape(-1,1,1,1), stats,
        alpha=ckpt.get('alpha', 1e-6),
        prenormalize_method=prenormalize_method,
        vmax=vmax,
    )
    all_showers = all_showers.reshape(num_samples, -1)
    
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
    print(f"  Deposited energy / Incident energy:")
    print(f"    Mean: {energy_ratio.mean():.4f}")
    print(f"    Std: {energy_ratio.std():.4f}")
    print(f"    Range: [{energy_ratio.min():.4f}, {energy_ratio.max():.4f}]")
    
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('showers', data=all_showers, compression='gzip')
        f.create_dataset('incident_energies', data=energies, compression='gzip')
        
        f.attrs['dataset_name'] = dataset_name
        f.attrs['num_samples'] = num_samples
        f.attrs['energy_distribution'] = energy_distribution
        f.attrs['energy_min'] = e_min
        f.attrs['energy_max'] = e_max
        f.attrs['shape'] = f"{D}x{H}x{W}"
        
        f.attrs['checkpoint'] = checkpoint_path
        f.attrs['training_obj'] = training_obj
        f.attrs['model_epoch'] = ckpt['epoch']
        
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
    print(f"  Training objective: {training_obj}")
    print(f"  Sampling method: {sample_method}")
    if sample_method == 'pc':
        print(f"  PC params: n_correct={n_correct}, delta={delta}")
    print(f"  Energy conservation: {energy_ratio.mean():.4f} +/- {energy_ratio.std():.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dataset2/3 inference (supports multiple training objectives and sampling methods)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Basic inference (using settings from checkpoint):
   python inference.py --checkpoint models/ddim3d_dataset2_electron_mean_pred_best.pt --output generated_dataset2.hdf5 --dataset dataset2 --num_samples 10 --distribution load --energy_file_dir data/dataset_2_1.hdf5 --seed 42

2. Load energy from file:
   python inference.py --checkpoint models/ddim3d_dataset2_electron_mean_pred_best.pt --output dataset2_results/generated_dataset2.hdf5 --dataset dataset2 --distribution load --energy_file_dir data/test_data.hdf5 --seed 42 --batch_size 32
   python inference.py --checkpoint models/ddim3d_dataset2_electron_mean_pred_E0.1_best.pt --output generated_dataset2_energykeep.hdf5 --dataset dataset2 --num_samples 10 --distribution load --energy_file_dir data/dataset_2_1.hdf5 --seed 42

3. Fast DDIM sampling:
   python inference.py \\
       --checkpoint ddim3d_dataset2_electron_hybrid.pt \\
       --output generated_dataset2_fast.hdf5 \\
       --dataset dataset2 \\
       --sample_method ddim \\
       --sample_eta 0.0 \\
       --num_steps 25 \\
       --num_samples 500

4. High quality PNDM sampling:
   python inference.py \\
       --checkpoint ddim3d_dataset2_electron_hybrid_cold.pt \\
       --output generated_dataset2_hq.hdf5 \\
       --dataset dataset2 \\
       --sample_method pndm \\
       --num_steps 100 \\
       --num_samples 500

5. Dataset3 inference:
   python inference.py \\
       --checkpoint ddim3d_dataset3_electron.pt \\
       --output generated_dataset3.hdf5 \\
       --dataset dataset3 \\
       --num_samples 500 \\
       --distribution normal \\
       --seed 42

6. Fixed energy generation:
   python inference.py \\
       --checkpoint ddim3d_dataset2_electron.pt \\
       --output generated_50GeV.hdf5 \\
       --dataset dataset2 \\
       --distribution constant \\
       --constant_energy 50000.0 \\
       --num_samples 100

7. Cold diffusion high quality generation:
   python inference.py \\
       --checkpoint ddim3d_dataset2_electron_hybrid_cold.pt \\
       --output generated_cold.hdf5 \\
       --dataset dataset2 \\
       --cold_noise_scale 0.8 \\
       --sample_method pndm \\
       --num_steps 100 \\
       --num_samples 1000

8. Score-based diffusion - Euler-Maruyama stochastic sampling:
   python inference.py \\
       --checkpoint ddim3d_dataset2_electron_hybrid.pt \\
       --output generated_em.hdf5 \\
       --dataset dataset2 \\
       --sample_method euler_maruyama \\
       --num_steps 200 \\
       --num_samples 1000

9. Score-based diffusion - Predictor-Corrector high quality sampling:
   python inference.py \\
       --checkpoint ddim3d_dataset2_electron_hybrid.pt \\
       --output generated_pc.hdf5 \\
       --dataset dataset2 \\
       --sample_method pc \\
       --num_steps 100 --n_correct 2 --delta 0.17 \\
       --num_samples 1000

10. Score-based diffusion - Probability Flow ODE deterministic sampling:
    python inference.py \\
        --checkpoint ddim3d_dataset2_electron_hybrid.pt \\
        --output generated_pf.hdf5 \\
        --dataset dataset2 \\
        --sample_method prob_flow \\
        --num_steps 100 \\
        --num_samples 1000
        """
    )
    
    required = parser.add_argument_group('Required')
    required.add_argument('--checkpoint', type=str, required=True,
                         help='Model checkpoint path')
    required.add_argument('--output', type=str, required=True,
                         help='Output HDF5 file path')
    
    data_group = parser.add_argument_group('Dataset params')
    data_group.add_argument('--dataset', type=str, default='dataset2',
                           choices=['dataset2', 'dataset3'],
                           help='Dataset name')
    data_group.add_argument('--particle', type=str, default='electron',
                           choices=['electron'], help='Particle type')
    data_group.add_argument('--num_samples', type=int, default=200,
                           help='Num samples to generate')
    
    energy_group = parser.add_argument_group('Energy params')
    energy_group.add_argument('--distribution', type=str, default='uniform',
                             choices=['uniform', 'normal', 'load', 'lognormal',
                                     'exponential', 'constant'],
                             help='Energy distribution type')
    energy_group.add_argument('--constant_energy', type=float, default=50000.0,
                             help='Constant energy value (used when distribution=constant, unit: MeV)')
    energy_group.add_argument('--energy_file_dir', type=str, default=None,
                             help='Energy file path (used when distribution=load)')
    
    sample_group = parser.add_argument_group('Sampling params')
    sample_group.add_argument('--sample_method', type=str, default=None,
                             choices=['pndm', 'ddim', 'euler_maruyama', 'prob_flow', 'pc'],
                             help='Sampling method (None=use checkpoint settings)')
    sample_group.add_argument('--num_steps', type=int, default=50,
                             help='Sampling steps')
    sample_group.add_argument('--sample_eta', type=float, default=None,
                             help='DDIM randomness param (0=deterministic, 1=DDPM, None=use checkpoint settings)')
    sample_group.add_argument('--n_correct', type=int, default=1,
                             help='PC Langevin correction steps')
    sample_group.add_argument('--delta', type=float, default=0.17,
                             help='PC Langevin step size coefficient')
    sample_group.add_argument('--batch_size', type=int, default=16,
                             help='Batch size')
    
    cold_group = parser.add_argument_group('Cold diffusion params')
    cold_group.add_argument('--cold_noise_scale', type=float, default=1.0,
                           help='Cold diffusion noise scale factor (cold diffusion models only)')
    
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
        dataset_name=args.dataset,
        particle=args.particle,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        energy_distribution=args.distribution,
        seed=args.seed,
        device=args.device,
        constant_energy=args.constant_energy,
        cold_noise_scale=args.cold_noise_scale,
        energy_file_dir=args.energy_file_dir,
        sample_method=args.sample_method,
        sample_eta=args.sample_eta,
        n_correct=args.n_correct,
        delta=args.delta,
    )