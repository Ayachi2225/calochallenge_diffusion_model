import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import DDIMModel3D
from dataprocess import get_calo_dataloader
from utils import load_config


CONFIG = {
    'volume_size':          None,
    'batch_size':           8,
    'num_epochs':           40,
    'lr':                   4e-4,
    'num_steps':            50,
    'save_every':           20,
    't_dim':                128,
    'e_dim':                128,
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint':           None,
    
    'training_obj':         'hybrid',
    'energy_loss_scale':    0.0,

    'cold_diffusion':       False,
    'num_energy_bins':      10,
    'cold_noise_scale':     1.0,

    'sample_method':        'pndm',
    'sample_eta':           0.0,
    'n_correct':            1,
    'delta':                0.17,

    'reshape_method':       'weight',
    'prenormalize_method':  'log10',

    'loss_every':           10,
    'normalize_method':     'logit',
    'alpha':                1e-6,
    'fine_tune':            False,
}


class RandomVolumeDataset(Dataset):
    def __init__(self, volume_size):
        self.volume_size = volume_size
        self.num_samples = 10
        self.vmax        = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        D, H, W = self.volume_size
        # Return (shower, energy, mask)
        return torch.randn(1, D, H, W), torch.rand(1), torch.ones(1, D, H, W)


def train(config: dict, hdf5_path: str = None, max_samples: int = None,
          dataset_name: str = 'dataset2', xml_path: str = None,
          particle: str = 'electron', weight_cache: str = None):
    device = config['device']

    cfg = load_config(None, dataset_name, particle)

    # ====== Data loading ======
    if hdf5_path:
        print(f"Loading data: {hdf5_path}")
        loader = get_calo_dataloader(
            hdf5_path,
            dataset_name=dataset_name,
            batch_size=config['batch_size'],
            max_samples=max_samples,
            xml_path=xml_path,
            particle=particle,
            weight_cache=weight_cache,
            compute_cold_stats=config['cold_diffusion'],
            num_energy_bins=config['num_energy_bins'],
            normalize_method=config['normalize_method'],
            alpha=config['alpha'],
            fine_tune=config['fine_tune'],
            reshape_method=config['reshape_method'],
            prenormalize_method=config['prenormalize_method'],
        )
        if dataset_name == 'dataset1':
            config['volume_size'] = loader.dataset.volume_size
        else:
            config['volume_size'] = tuple(cfg['volume_size'])
    else:
        if config['volume_size'] is None:
            config['volume_size'] = (45, 9, 16)
        print("No data file provided, using random data for smoke test")
        ds     = RandomVolumeDataset(config['volume_size'])
        loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    D, H, W = config['volume_size']

    # ====== Print training config ======
    print(f"{'='*70}")
    print(f"Training config:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Device: {device}")
    print(f"  Volume size: (1, layers:{D}, anglebins:{H}, rbins:{W})")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Num epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  " + "-" * 60)
    print(f"  Training objective: {config['training_obj']}")
    if config['energy_loss_scale'] > 0:
        print(f"  Energy loss weight: {config['energy_loss_scale']}")
    print(f"  " + "-" * 60)
    print(f"  Cold diffusion: {config['cold_diffusion']}")
    if config['cold_diffusion']:
        print(f"  Num energy bins: {config['num_energy_bins']}")
        print(f"  Noise scale: {config['cold_noise_scale']}")
    print(f"  " + "-" * 60)
    print(f"  Sampling method: {config['sample_method']}")
    print(f"  Sampling steps: {config['num_steps']}")
    if config['sample_method'] == 'ddim':
        print(f"  DDIM eta: {config['sample_eta']}")
    if config['sample_method'] == 'pc':
        print(f"  PC correction steps: {config['n_correct']}")
        print(f"  PC step size coefficient: {config['delta']}")
    print(f"  " + "-" * 60)
    print(f"  Radial processing method: {config['reshape_method']}")  # New
    if config['reshape_method'] == 'mask':
        print(f"  Using mask: Yes")
    if config['reshape_method'] == 'nnconverter':
        nc = loader.dataset.nn_converter
        if nc is not None:
            print(f"  NNConverter layers: {nc.num_layers}, output dims: alpha={nc.alpha_out}, r={nc.dim_r_out}")
    print(f"{'='*70}\n")

    E_bins = loader.dataset.E_bins
    avg_showers = loader.dataset.avg_showers
    std_showers = loader.dataset.std_showers

    # ====== Create model ======
    use_mask = (config['reshape_method'] == 'mask')
    nn_converter = getattr(loader.dataset, 'nn_converter', None)
    nn_binning_info = getattr(loader.dataset, 'nn_binning_info', None)
    irreg_shapes = getattr(loader.dataset, 'irreg_shapes', None)

    model = DDIMModel3D(
        t_dim=config['t_dim'],
        e_dim=config['e_dim'],
        training_obj=config['training_obj'],
        cold_diffusion=config['cold_diffusion'],
        E_bins=E_bins,
        avg_showers=avg_showers,
        std_showers=std_showers,
        cold_noise_scale=config['cold_noise_scale'],
        use_mask=use_mask,
        nn_converter=nn_converter,
        nn_binning_info=nn_binning_info,
        irreg_shapes=irreg_shapes,
    ).to(device)

    if nn_converter is not None:
        print(f"[NNConverter] Encoder+decoder params integrated into model (total "
              f"{sum(p.numel() for p in nn_converter.parameters())}), "
              f"both ends participate in training")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs']
    )

    start_epoch = 0
    
    # ====== Checkpoint management ======
    obj_suffix = f"_{config['training_obj']}"
    cold_suffix = "_cold" if config['cold_diffusion'] else ""
    energy_suffix = f"_E{config['energy_loss_scale']}" if config['energy_loss_scale'] > 0 else ""
    mask_suffix = "_mask" if config['reshape_method'] == 'mask' else ""  # New
    
    config['checkpoint'] = f"models/ddim3d_{dataset_name}_{particle}{obj_suffix}{cold_suffix}{energy_suffix}{mask_suffix}.pt"
    
    if os.path.exists(config['checkpoint']):
        print(f"Found checkpoint: {config['checkpoint']}")
        try:
            ckpt = torch.load(config['checkpoint'], map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'], strict=False)
            optimizer.load_state_dict(ckpt['optim'])
            if 'scheduler' in ckpt:
                last_epoch = ckpt.get('epoch', 0)+1
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=config['num_epochs'], 
                    last_epoch=last_epoch
                )
            start_epoch = ckpt['epoch'] + 1
            
            saved_obj = ckpt.get('training_obj', 'unknown')
            if saved_obj != config['training_obj']:
                print(f"  Warning: checkpoint training objective ({saved_obj}) does not match current ({config['training_obj']})!")
            
            print(f"Resuming from epoch {start_epoch}\n")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch\n")
            start_epoch = 0

    print(f"\nStarting training...")
    print(f"{'='*70}\n")

    os.makedirs('samples', exist_ok=True)
    
    best_loss = float('inf')
    best_epoch = -1
    loss_history = []

    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_data in loader:
            # Unpack data: (showers, energies, masks)
            if len(batch_data) == 3:
                x0, energies, mask = batch_data
                x0 = x0.to(device)
                energies = energies.to(device)
                mask = mask.to(device)
            else:
                # Backward compatible with old format
                x0, energies = batch_data
                x0 = x0.to(device)
                energies = energies.to(device)
                mask = None

            loss = model.get_loss(
                x0, 
                energies,
                mask=mask,  # New
                energy_loss_scale=config['energy_loss_scale']
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss   = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
        
        if (epoch + 1) % config['loss_every'] == 0:
            print(f"Epoch {epoch+1:4d}/{config['num_epochs']}  |  "
                  f"Loss: {avg_loss:.5f}  |  "
                  f"Best: {best_loss:.5f} (Epoch {best_epoch})  |  "
                  f"LR: {current_lr:.2e}")

        # ====== Periodic sampling and saving ======
        if epoch>=49 and ((epoch + 1) % config['save_every'] == 0 or (epoch + 1) == best_epoch):
            model.eval()
            '''
            num_energy_samples = 4
            if config['cold_diffusion'] and hdf5_path and E_bins is not None:
                num_energy_samples = min(4, len(E_bins) - 1)
                sample_energies_list = []
                for i in range(num_energy_samples):
                    bin_idx = i * (len(E_bins) - 1) // num_energy_samples
                    mid_energy = (E_bins[bin_idx] + E_bins[bin_idx + 1]) / 2
                    sample_energies_list.append(mid_energy.item())
                sample_energies_tensor = torch.tensor(
                    sample_energies_list,
                    device=device
                ).reshape(-1, 1)
                print(f"\n  Sample energies: {sample_energies_list}")
            else:
                sample_energies_tensor = torch.ones(num_energy_samples, 1, device=device)

            with torch.no_grad():
                samples = model.sample(
                    shape=(num_energy_samples, 1, D, H, W),
                    energy=sample_energies_tensor,
                    num_steps=config['num_steps'],
                    device=device,
                    cold_noise_scale=config['cold_noise_scale'],
                    method=config['sample_method'],
                    eta=config['sample_eta'],
                    n_correct=config['n_correct'],
                    delta=config['delta'],
                )

            samples = (samples.clamp(-1, 1) + 1) / 2

            sample_path = f"samples/samples_epoch{epoch+1}_{dataset_name}{obj_suffix}{cold_suffix}.pt"
            torch.save({
                'samples': samples.cpu(),
                'energies': sample_energies_tensor.cpu(),
                'training_obj': config['training_obj'],
                'sample_method': config['sample_method'],
            }, sample_path)
            print(f"  -> Samples saved to: {sample_path}")
            '''
            vmax = loader.dataset.vmax if hdf5_path else None

            checkpoint_data = {
                'model':                model.state_dict(),
                'optim':                optimizer.state_dict(),
                'scheduler':            scheduler.state_dict(),
                'epoch':                epoch,
                'loss':                 avg_loss,
                'best_loss':            best_loss,
                'best_epoch':           best_epoch,
                'training_obj':         config['training_obj'],
                'energy_loss_scale':    config['energy_loss_scale'],
                'sample_method':        config['sample_method'],
                'sample_eta':           config['sample_eta'],
                'n_correct':            config['n_correct'],
                'delta':                config['delta'],
                'vmax':                 vmax,
                'num_steps':            config['num_steps'],
                'batch_size':           config['batch_size'],
                'dataset_name':         dataset_name,
                'volume_size':          config['volume_size'],
                'normalize_method':     config['normalize_method'],
                'alpha':                config['alpha'],
                'fine_tune':            config['fine_tune'],
                'log_mean':             getattr(loader.dataset, 'log_mean', None),
                'log_std':              getattr(loader.dataset, 'log_std', None),
                'logit_mean':           getattr(loader.dataset, 'logit_mean', None),
                'logit_std':            getattr(loader.dataset, 'logit_std', None),
                'prenormalize_method':  getattr(loader.dataset, 'prenormalize_method', 'log10'),
                'cold_diffusion':       config['cold_diffusion'],
                'num_energy_bins':      config['num_energy_bins'],
                'cold_noise_scale':     config['cold_noise_scale'],
                'E_bins':               E_bins,
                'avg_showers':          avg_showers,
                'std_showers':          std_showers,
                'reshape_method':       config['reshape_method'],
                'use_mask':             use_mask,
                'nn_converter':         model.nn_converter,
                'nn_binning_info':      model.nn_binning_info,
            }
            
            torch.save(checkpoint_data, config['checkpoint'])
            print(f"  -> Checkpoint saved to: {config['checkpoint']}")
            
            if (epoch+1) == best_epoch:
                best_checkpoint = config['checkpoint'].replace('.pt', '_best.pt')
                torch.save(checkpoint_data, best_checkpoint)
                print(f"  -> Best model saved to: {best_checkpoint}")
            
            print()

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"  Best loss: {best_loss:.5f} (Epoch {best_epoch})")
    print(f"  Final checkpoint: {config['checkpoint']}")
    print(f"{'='*70}")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', markersize=3, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss — {dataset_name}_{particle} ({config["training_obj"]})')
    plt.grid(True, alpha=0.3)
    if best_epoch > 0:
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'loss_curve_{dataset_name}_{particle}{obj_suffix}{cold_suffix}.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument('--dataset', type=str, default='dataset2',
                            choices=['dataset1', 'dataset2', 'dataset3'])
    pre_parser.add_argument('--particle', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.particle is None:
        pre_args.particle = 'photon' if pre_args.dataset == 'dataset1' else 'electron'

    cfg = load_config(None, pre_args.dataset, pre_args.particle)

    # ---- Main parser: defaults from config.json, user CLI args can override ----
    parser = argparse.ArgumentParser(
        description='Train 3D DDIM model — defaults from config.json based on --dataset / --particle',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    # Data params
    data_group = parser.add_argument_group('Data params')
    data_group.add_argument('--data', type=str,
                            default=cfg.get('default_data', None),
                            help='HDF5 data file path')
    data_group.add_argument('--dataset', type=str, default=pre_args.dataset,
                           choices=['dataset1', 'dataset2', 'dataset3'], help='Dataset type')
    data_group.add_argument('--max_samples', type=int, default=None, help='Max samples')
    data_group.add_argument('--xml', type=str,
                            default=cfg.get('default_xml', None),
                            help='XML geometry file path')
    data_group.add_argument('--particle', type=str, default=pre_args.particle,
                           choices=['photon', 'pion', 'electron'], help='Particle type')
    data_group.add_argument('--weight_cache', type=str,
                            default=cfg.get('weight_cache', 'data/weight_mats.pkl'))
    data_group.add_argument('--reshape_method', type=str,
                            default=cfg.get('reshape_method', 'weight'),
                            choices=['weight', 'mask', 'nnconverter'],
                            help='Radial processing method: weight=area-weight interpolation, mask=zero-pad+mask, nnconverter=trainable geometric conversion')
    data_group.add_argument('--prenormalize_method', type=str,
                            default=cfg.get('prenormalize_method', 'log10'),
                            choices=['log10', 'log1p', 'sqrt'],
                            help='Prenormalize method')

    # Training params
    train_group = parser.add_argument_group('Training params')
    train_group.add_argument('--epochs', type=int, default=cfg.get('num_epochs', 40))
    train_group.add_argument('--batch', type=int, default=cfg.get('batch_size', 8))
    train_group.add_argument('--lr', type=float, default=cfg.get('lr', 4e-4))

    # Training objective params
    objective_group = parser.add_argument_group('Training objective params')
    objective_group.add_argument('--training_obj', type=str,
                                default=cfg.get('training_obj', 'hybrid'),
                                choices=['noise_pred', 'mean_pred', 'hybrid', 'score_pred'],
                                help='Training objective')
    objective_group.add_argument('--energy_loss_scale', type=float,
                                default=cfg.get('energy_loss_scale', 0.0),
                                help='Energy conservation loss weight (recommended 0.001-0.01)')

    # Cold diffusion params
    cold_group = parser.add_argument_group('Cold diffusion params')
    cold_group.add_argument('--cold_diffusion', action='store_true',
                            default=cfg.get('cold_diffusion', False))
    cold_group.add_argument('--num_energy_bins', type=int,
                            default=cfg.get('num_energy_bins', 10))
    cold_group.add_argument('--cold_noise_scale', type=float,
                            default=cfg.get('cold_noise_scale', 1.0))

    # Sampling params
    sample_group = parser.add_argument_group('Sampling params')
    sample_group.add_argument('--sample_method', type=str,
                              default=cfg.get('sample_method', 'pndm'),
                              choices=['pndm', 'ddim', 'euler_maruyama', 'prob_flow', 'pc'])
    sample_group.add_argument('--sample_eta', type=float,
                              default=cfg.get('sample_eta', 0.0))
    sample_group.add_argument('--n_correct', type=int,
                              default=cfg.get('n_correct', 1),
                              help='PC Langevin correction steps')
    sample_group.add_argument('--delta', type=float,
                              default=cfg.get('delta', 0.17),
                              help='PC Langevin step size coefficient')
    sample_group.add_argument('--num_steps', type=int,
                              default=cfg.get('num_steps', 50))

    args = parser.parse_args()

    if args.dataset == 'dataset1' and args.xml is None:
        parser.error("dataset1 requires --xml argument")

    # If user changed dataset/particle via CLI (different from pre-parsed), reload corresponding config
    if args.dataset != pre_args.dataset or args.particle != pre_args.particle:
        cfg = load_config(None, args.dataset, args.particle)

    CONFIG.update({
        'num_epochs':         args.epochs,
        'batch_size':         args.batch,
        'lr':                 args.lr,
        'num_steps':          args.num_steps,
        'training_obj':       args.training_obj,
        'energy_loss_scale':  args.energy_loss_scale,
        'cold_diffusion':     args.cold_diffusion,
        'num_energy_bins':    args.num_energy_bins,
        'cold_noise_scale':   args.cold_noise_scale,
        'sample_method':      args.sample_method,
        'sample_eta':         args.sample_eta,
        'n_correct':          args.n_correct,
        'delta':              args.delta,
        'reshape_method':     args.reshape_method,
        'prenormalize_method': args.prenormalize_method,
    })

    train(CONFIG, args.data, args.max_samples, args.dataset, args.xml,
          args.particle, args.weight_cache)


# ============================================================================
# Usage examples
# ============================================================================
'''
# 1. Train Dataset1 with mask method
python train.py --data data/dataset_1_photons_1.hdf5 --dataset dataset1 \
    --xml data/binning_dataset_1_photons.xml --particle photon \
    --reshape_method mask --training_obj mean_pred --epochs 200 --max_samples 200 \

python train.py --data data/dataset_1_photons_1.hdf5 --dataset dataset1 \
    --xml data/binning_dataset_1_photons.xml --particle photon \
    --reshape_method nnconverter --training_obj mean_pred --epochs 200 --max_samples 200 \
    --sample_method ddim --sample_eta 0.0 --num_steps 200

python train.py --data data/dataset_1_photons_1.hdf5 --dataset dataset1 \
    --xml data/binning_dataset_1_photons.xml --particle photon \
    --reshape_method mask --training_obj mean_pred --epochs 200 --max_samples 1000

python train.py --data data/dataset_1_photons_1.hdf5 --dataset dataset1 \
    --xml data/binning_dataset_1_photons.xml --particle photon \
    --reshape_method mask --training_obj hybrid --epochs 200 --max_samples 200

# 2. Train with weight method (default, backward compatible)
python train.py --data data/dataset_1_photons_1.hdf5 --dataset dataset1 \
    --xml data/binning_dataset_1_photons.xml --particle photon \
    --reshape_method weight --training_obj mean_pred --epochs 200 --max_samples 200

# 3. Mask method + cold diffusion
python train.py --data data/dataset_1_photons_1.hdf5 --dataset dataset1 \
    --xml data/binning_dataset_1_photons.xml --particle photon \
    --reshape_method mask --training_obj hybrid \
    --cold_diffusion --num_energy_bins 10 --epochs 200

# 4. Dataset2 (unaffected, default all-ones mask)
python train.py  --dataset dataset2 --training_obj mean_pred --epochs 150
'''