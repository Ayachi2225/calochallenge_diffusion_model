import os
import argparse
import json
import numpy as np
import h5py
import torch


def load_config(config_path: str = None, dataset_name: str = None, particle: str = None) -> dict:
    """Load config.json and return merged config for given dataset + particle.

    Merge order: global < dataset < particle (later overrides earlier).
    If dataset_name is None, returns the raw full config.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    if dataset_name is None:
        return cfg

    merged = dict(cfg.get('global', {}))

    ds_cfg = cfg.get('datasets', {}).get(dataset_name, {})
    for k, v in ds_cfg.items():
        if k != 'particles':
            merged[k] = v

    if particle is not None:
        particle_cfg = ds_cfg.get('particles', {}).get(particle, {})
        merged.update(particle_cfg)

    return merged


def generate_energies(num_samples: int , distribution: str, 
                     e_min: float, e_max: float, 
                     seed: int = None,constant_energy: float = None,energy_file_dir:str =None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    if distribution == 'uniform':
        log_min = np.log10(e_min)
        log_max = np.log10(e_max)
        log_energies = np.random.uniform(log_min, log_max, num_samples)
        energies = 10 ** log_energies
    elif distribution == 'constant':
        energies = np.full(num_samples, constant_energy, dtype=np.float32)
    elif distribution == 'normal':
        log_min = np.log10(e_min)
        log_max = np.log10(e_max)
        log_mean = (log_min + log_max) / 2
        log_std = (log_max - log_min) / 6
        log_energies = np.random.normal(log_mean, log_std, num_samples)
        log_energies = np.clip(log_energies, log_min, log_max)
        energies = 10 ** log_energies
    elif distribution == 'load':
        partile_file = h5py.File(energy_file_dir, 'r')
        energies = partile_file['incident_energies'][:num_samples]
        
    elif distribution == 'lognormal':
        mean = (e_min + e_max) / 2
        sigma = (e_max - e_min) / 6
        mu = np.log(mean**2 / np.sqrt(mean**2 + sigma**2))
        s = np.sqrt(np.log(1 + sigma**2 / mean**2))
        energies = np.random.lognormal(mu, s, num_samples)
        energies = np.clip(energies, e_min, e_max)
        
    elif distribution == 'exponential':
        scale = (e_max - e_min) / 3
        energies = np.random.exponential(scale, num_samples) + e_min
        energies = np.clip(energies, e_min, e_max)
        
    else:
        raise ValueError(f"不支持的分布类型: {distribution}")
    
    return energies.astype(np.float32)

energy_scale = 1
def normalize_energies(energies: np.ndarray, e_min: float, e_max: float) -> np.ndarray:
    log_min = np.log10(e_min)
    log_max = np.log10(e_max)
    return (energies - log_min) / (log_max - log_min + 1e-9)

def decontinue_fine_tune(showers: np.ndarray, energies: np.ndarray, E_bins: np.ndarray, avg_showers: np.ndarray, std_showers: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    E_bins = np.array(E_bins)
    avg_showers = np.array(avg_showers)
    std_showers = np.array(std_showers)
    avg_showers = np.squeeze(avg_showers, axis=1)
    std_showers = np.squeeze(std_showers, axis=1)
    idxs = np.digitize(energies.squeeze(), E_bins) - 1
    idxs = np.clip(idxs, 0, len(avg_showers) - 1)
    
    avg = avg_showers[idxs]
    std = std_showers[idxs]
    
    showers = showers * (std + alpha) + avg
    return showers
    
def prenormalize_showers(showers, energies, alpha=1e-6, prenormalize_method='log10', normalize_method='log'):
    if prenormalize_method == 'log10':
        showers = np.log10(showers + 1e-9)
        energies = np.log10(energies + 1e-9)
        showers = showers / energies * energy_scale
        vmax = showers.max()
        x = alpha + (1 - 2*alpha) * showers / vmax
    elif prenormalize_method == 'log1p':
        showers = showers/(energies + 1e-9)
        showers_log = np.log1p(showers)
        vmax = showers_log.max()
        x = alpha + (1 - 2*alpha) * showers_log / vmax
    elif prenormalize_method == 'norm':
        showers = showers/(energies + 1e-9)
        vmax = showers.max()
        x = alpha + (1 - 2*alpha) * showers / vmax
    elif prenormalize_method == 'sqrt':
        showers = showers/(energies + 1e-9)
        showers_sqrt = np.sqrt(showers)
        vmax = showers_sqrt.max()
        x = alpha + (1 - 2*alpha) * showers_sqrt / vmax

    stats = {
        'normalize_method': normalize_method,
        'prenormalize_method': prenormalize_method,
        'vmax': vmax,
    }

    if normalize_method == 'log':
        showers = x * 2 - 1
        mean = showers.mean()
        std = showers.std()
        print(f"[Normalization] log方法: mean={mean:.4f}, std={std:.4f}")
        stats['log_mean'] = mean
        stats['log_std'] = std
        showers = (showers - mean) / std

    if normalize_method == 'logit':
        showers = np.ma.log(x / (1 - x)).filled(0)
        mean = showers.mean()
        std = showers.std()
        print(f"[Normalization] logit方法: mean={mean:.4f}, std={std:.4f}")
        stats['logit_mean'] = mean
        stats['logit_std'] = std
        showers = (showers - mean) / std

    return showers, stats


def denormalize_showers(showers, energies, stats, alpha=1e-6, prenormalize_method='log10', vmax=None):
    normalize_method = stats.get('normalize_method', None)

    if vmax is None:
        vmax = stats.get('vmax', 1.0)

    if normalize_method == 'log':
        showers = showers * stats['log_std'] + stats['log_mean']
        x = (showers + 1) / 2
    elif normalize_method == 'logit':
        showers = showers * stats['logit_std'] + stats['logit_mean']
        x = 1 / (1 + np.exp(-showers))
    else:
        raise ValueError(f"Unknown normalize_method: {normalize_method}")

    x = np.clip(x, 1e-9, 1 - 1e-9)

    if prenormalize_method == 'log10':
        energies_log = np.log10(energies + 1e-9)
        showers = (x - alpha) / (1 - 2*alpha) * vmax * energies_log / energy_scale
        showers = np.power(10, showers) - 1e-9
    elif prenormalize_method == 'log1p':
        s_log = (x - alpha) / (1 - 2*alpha) * vmax
        s = np.expm1(s_log)
        showers = s * (energies + 1e-9)
    elif prenormalize_method == 'norm':
        s_norm = (x - alpha) / (1 - 2*alpha) * vmax
        showers = s_norm * (energies + 1e-9)
    elif prenormalize_method == 'sqrt':
        s_sqrt = (x - alpha) / (1 - 2*alpha) * vmax
        s = s_sqrt ** 2
        showers = s * (energies + 1e-9)
    else:
        raise ValueError(f"Unknown prenormalize_method: {prenormalize_method}")

    return np.maximum(showers, 0.0)