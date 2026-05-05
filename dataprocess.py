import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import prenormalize_showers, load_config
from dataset1_preprocess import resample_alpha, TARGET_ALPHA
from dataset1_preprocess import (
    parse_binning_xml,
    build_weight_mats,
    build_mask_info,
    get_all_voxel_counts
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
        
        # mask相关
        self.masks = None  # 每个样本的mask，shape: [N, 1, L, A, R]

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
        print(f"[Dataset1] 使用 {self.reshape_method} 方法进行径向处理")

        lay_ids, lay_r_edges, lay_alphas = parse_binning_xml(xml_path, particle)

        # 判断使用哪种方法
        if self.reshape_method == 'weight':
            all_r_edges, weight_mats = build_weight_mats(
                lay_r_edges, cache_path=weight_cache
            )
            M = len(all_r_edges) - 1
            mask_list = None
        elif self.reshape_method == 'mask':
            M, mask_list = build_mask_info(lay_r_edges)
            weight_mats = None
            all_r_edges = None
        else:
            raise ValueError(f"reshape_method 必须是 'weight' 或 'mask'，但得到 '{self.reshape_method}'")

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

        valid_layer_indices = [
            i for i, r in enumerate(lay_r_edges) if len(r) > 1
        ]

        N = showers.shape[0]
        L_valid = len(valid_layer_indices)

        self.volume_size = (L_valid, TARGET_ALPHA, M)

        volume = np.zeros((N, L_valid, TARGET_ALPHA, M), dtype=np.float32)
        
        # 创建mask数组
        if self.reshape_method == 'mask':
            masks = np.zeros((N, L_valid, TARGET_ALPHA, M), dtype=np.float32)

        for out_idx, i in enumerate(valid_layer_indices):
            n_a = lay_alphas[i]
            n_r = len(lay_r_edges[i]) - 1

            # [N, n_a, n_r]
            layer = showers[:, bin_starts[i]:bin_ends[i]].reshape(N, n_a, n_r)

            if self.reshape_method == 'weight':
                layer = np.einsum('nar,mr->nam', layer, weight_mats[i])
                
            elif self.reshape_method == 'mask':
                layer_padded = np.zeros((N, n_a, M), dtype=np.float32)
                layer_padded[:, :, :n_r] = layer  # 前n_r个bin填充数据
                layer = layer_padded
                
                layer_mask = np.zeros((N, TARGET_ALPHA, M), dtype=np.float32)
                layer_mask[:, :, :] = mask_list[i][None, None, :]  # broadcast到[N, TARGET_ALPHA, M]

            # 角度重采样到 TARGET_ALPHA：[N, TARGET_ALPHA, M]
            layer = resample_alpha(layer, n_a, TARGET_ALPHA)

            volume[:, out_idx] = layer
            
            if self.reshape_method == 'mask':
                masks[:, out_idx] = layer_mask
        energies = energies.reshape(-1,1,1,1)
        volume, stats = prenormalize_showers(volume, energies, self.alpha, self.prenormalize_method, self.normalize_method)
        if 'log_mean' in stats:
            self.log_mean = stats['log_mean']
            self.log_std = stats['log_std']
        if 'logit_mean' in stats:
            self.logit_mean = stats['logit_mean']
            self.logit_std = stats['logit_std']
        self.vmax = stats.get('vmax', 1.0)
        self.prenormalize_method = stats.get('prenormalize_method', self.prenormalize_method)
        energies = np.log10(energies.reshape(-1,1))
        e_min, e_max = self.cfg['energy_range']
        log_emin, log_emax = np.log10(e_min), np.log10(e_max)
        energies = (energies-log_emin)/(log_emax-log_emin)

        self.showers = torch.from_numpy(volume[:, None]).float()
        self.energies = torch.from_numpy(energies).float()

        if self.reshape_method == 'mask':
            self.masks = torch.from_numpy(masks[:, None]).float()
            print(f"[Dataset1] masks shape: {self.masks.shape}")
        else:
            # weight方法下，全部区域有效
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

        # dataset2/3使用全1 mask（全部有效）
        self.masks = torch.ones_like(self.showers)
        
    def _continue_fine_tune(self):
        showers = self._bins_normalize_showers(self.showers, self.energies)
        self.showers = showers
        
    def lookup_avg_std_shower(self, energies: torch.Tensor):
        """根据能量查找对应的平均和标准差 shower"""
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
        print(f"\n计算统计数据 (能量区间数: {self.num_energy_bins})")
        energies_flat = self.energies.squeeze()
        
        e_min, e_max = energies_flat.min(), energies_flat.max()
        self.E_bins = torch.linspace(e_min, e_max, self.num_energy_bins + 1)
        
        avg_showers_list = []
        std_showers_list = []
        
        for i in range(self.num_energy_bins):
            mask = (energies_flat >= self.E_bins[i]) & (energies_flat < self.E_bins[i+1])
            
            if i == self.num_energy_bins - 1:  # 最后一个区间包含上界
                mask = (energies_flat >= self.E_bins[i]) & (energies_flat <= self.E_bins[i+1])
            
            num_samples_in_bin = mask.sum()
            
            if num_samples_in_bin > 1:
                showers_in_bin = self.showers[mask]
                avg_shower = showers_in_bin.mean(dim=0, keepdim=True)
                std_shower = showers_in_bin.std(dim=0, keepdim=True)
                
                print(f"  区间 {i}: [{self.E_bins[i]:.4f}, {self.E_bins[i+1]:.4f}] - {num_samples_in_bin} 个样本")
            else:
                # 如果该区间没有样本，使用全局统计
                avg_shower = self.showers.mean(dim=0, keepdim=True)
                std_shower = self.showers.std(dim=0, keepdim=True)
                print(f"  区间 {i}: [{self.E_bins[i]:.4f}, {self.E_bins[i+1]:.4f}] - 0/1 个样本 (使用全局统计)")
            
            avg_showers_list.append(avg_shower)
            std_showers_list.append(std_shower)
        
        self.avg_showers = torch.cat(avg_showers_list, dim=0)
        self.std_showers = torch.cat(std_showers_list, dim=0)
        
        print(f"  avg_showers: {self.avg_showers.shape}")
        print(f"  std_showers: {self.std_showers.shape}")
        print(f"  E_bins: {self.E_bins.shape} - 范围 [{self.E_bins[0]:.4f}, {self.E_bins[-1]:.4f}]")

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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )