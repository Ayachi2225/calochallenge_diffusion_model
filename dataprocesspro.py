import h5py
import numpy as np
import torch
import pickle
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from dataset1_preprocess import resample_alpha, TARGET_ALPHA
from dataset1_preprocess import (
    parse_binning_xml,
    build_weight_mats,
    get_all_voxel_counts
)


DATASET_CONFIGS = {
    'dataset2': (45, 9, 16), # layers,alphas,redges
    'dataset3': (45, 18, 50),
}

ENERGY_RANGE = {
    'dataset2': (1000, 1000000),
    'dataset3': (1000, 1000000),
}


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
        alpha: float = 1e-6,
        cold_stats_cache_dir: str = './cold_stats_cache',
        force_recompute: bool = False,
    ):
        self.dataset_name = dataset_name
        self.normalize_method = normalize_method
        self.compute_cold_stats = compute_cold_stats
        self.num_energy_bins = num_energy_bins
        self.alpha = alpha
        self.cold_stats_cache_dir = cold_stats_cache_dir
        self.force_recompute = force_recompute
        
        # 冷扩散统计数据
        self.E_bins = None
        self.avg_showers = None
        self.std_showers = None

        if dataset_name == 'dataset1':
            self._load_dataset1(
                hdf5_path, xml_path, particle,
                max_samples, weight_cache, ecut
            )
        else:
            self._load_dataset_standard(hdf5_path, max_samples)
        
        # 计算或加载冷扩散所需的统计数据
        if self.compute_cold_stats:
            self._compute_or_load_cold_statistics(hdf5_path)

    def _load_dataset1(
        self, hdf5_path, xml_path, particle,
        max_samples, weight_cache, ecut
    ):
        print("[Dataset1] 使用XML几何预处理（对齐 preprocess）")

        lay_ids, lay_r_edges, lay_alphas = parse_binning_xml(xml_path, particle)

        all_r_edges, weight_mats = build_weight_mats(
            lay_r_edges, cache_path=weight_cache
        )

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
        M = len(all_r_edges) - 1
        L_valid = len(valid_layer_indices)

        self.volume_size = (L_valid, TARGET_ALPHA, M)

        volume = np.zeros((N, L_valid, TARGET_ALPHA, M), dtype=np.float32)

        for out_idx, i in enumerate(valid_layer_indices):
            n_a = lay_alphas[i]
            n_r = len(lay_r_edges[i]) - 1

            # [N, n_a, n_r]
            layer = showers[:, bin_starts[i]:bin_ends[i]].reshape(N, n_a, n_r)

            layer = np.einsum('nar,mr->nam', layer, weight_mats[i])

            layer = resample_alpha(layer, n_a, TARGET_ALPHA)

            volume[:, out_idx] = layer

        volume, self.vmax = self._normalize_showers(volume)

        energies = self._normalize_energies_dataset1(
            energies.reshape(-1, 1), particle
        )

        self.showers = torch.from_numpy(volume[:, None]).float()
        self.energies = torch.from_numpy(energies).float()

    def _load_dataset_standard(self, hdf5_path, max_samples):
        D, H, W = DATASET_CONFIGS[self.dataset_name]
        e_min, e_max = ENERGY_RANGE[self.dataset_name]

        with h5py.File(hdf5_path, 'r') as f:
            showers = f['showers'][:max_samples].astype(np.float32)
            energies = f['incident_energies'][:max_samples].astype(np.float32)

        showers, self.vmax = self._normalize_showers(showers)
        energies = self._normalize_energies(energies.reshape(-1, 1), e_min, e_max)

        self.showers = torch.from_numpy(showers.reshape(-1, 1, D, H, W)).float()
        self.energies = torch.from_numpy(energies).float()

    def _normalize_showers(self, showers):
        alpha = self.alpha
        showers = np.log10(showers+1e-9)
        vmax = showers.max()
        x = alpha +(1-2*alpha)*showers / vmax
        if self.normalize_method == 'log':
            showers = x*2-1
            mean = showers.mean(axis=1,keepdims=True)
            std = showers.std(axis=1,keepdims=True)
            self.log_mean = mean
            self.log_std = std
            showers = (showers-mean)/std

        if self.normalize_method == 'logit':
            showers = np.ma.log(x/(1-x)).filled(0)
            mean = showers.mean(axis=1,keepdims=True)
            std = showers.std(axis=1,keepdims=True)
            self.logit_mean = mean
            self.logit_std = std
            showers = (showers-mean)/std

        return showers, vmax

    def _normalize_energies(self, energies, e_min, e_max):
        log_e = np.log10(energies)
        log_min = np.log10(e_min)
        log_max = np.log10(e_max)
        return (log_e - log_min) / (log_max - log_min)

    def _normalize_energies_dataset1(self, energies, particle):
        from dataset1_preprocess import ENERGY_RANGE as DS1_ENERGY_RANGE
        e_min, e_max = DS1_ENERGY_RANGE[particle]
        return self._normalize_energies(energies, e_min, e_max)
    
    def _get_cold_stats_cache_path(self, hdf5_path):
        """
        生成冷扩散统计数据的缓存文件路径
        
        基于数据集名称、归一化方法、能量区间数等参数生成唯一的缓存文件名
        """
        # 创建缓存目录
        os.makedirs(self.cold_stats_cache_dir, exist_ok=True)
        
        # 从hdf5路径提取文件名（不含扩展名）
        hdf5_basename = Path(hdf5_path).stem
        
        # 生成缓存文件名（包含关键参数以确保唯一性）
        cache_filename = (
            f"cold_stats_{hdf5_basename}_"
            f"{self.dataset_name}_"
            f"{self.normalize_method}_"
            f"bins{self.num_energy_bins}_"
            f"alpha{self.alpha:.0e}.pkl"
        )
        
        return os.path.join(self.cold_stats_cache_dir, cache_filename)
    
    def _save_cold_statistics(self, cache_path):
        """
        保存冷扩散统计数据到pickle文件
        """
        cold_stats = {
            'E_bins': self.E_bins.cpu() if self.E_bins is not None else None,
            'avg_showers': self.avg_showers.cpu() if self.avg_showers is not None else None,
            'std_showers': self.std_showers.cpu() if self.std_showers is not None else None,
            'num_energy_bins': self.num_energy_bins,
            'dataset_name': self.dataset_name,
            'normalize_method': self.normalize_method,
            'alpha': self.alpha,
            'shower_shape': self.showers.shape[1:],  # 保存shower的形状信息
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cold_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n[Cold Diffusion] 统计数据已保存到: {cache_path}")
        print(f"  文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")
    
    def _load_cold_statistics(self, cache_path):
        """
        从pickle文件加载冷扩散统计数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(cache_path, 'rb') as f:
                cold_stats = pickle.load(f)
            
            # 验证统计数据的参数是否匹配
            if (cold_stats['num_energy_bins'] != self.num_energy_bins or
                cold_stats['dataset_name'] != self.dataset_name or
                cold_stats['normalize_method'] != self.normalize_method or
                abs(cold_stats['alpha'] - self.alpha) > 1e-10):
                print(f"\n[Cold Diffusion] 警告: 缓存文件参数不匹配，将重新计算")
                print(f"  缓存参数: bins={cold_stats['num_energy_bins']}, "
                      f"dataset={cold_stats['dataset_name']}, "
                      f"norm={cold_stats['normalize_method']}, "
                      f"alpha={cold_stats['alpha']}")
                print(f"  当前参数: bins={self.num_energy_bins}, "
                      f"dataset={self.dataset_name}, "
                      f"norm={self.normalize_method}, "
                      f"alpha={self.alpha}")
                return False
            
            # 验证shower形状是否匹配
            expected_shape = self.showers.shape[1:]
            if cold_stats['shower_shape'] != expected_shape:
                print(f"\n[Cold Diffusion] 警告: 缓存文件shower形状不匹配，将重新计算")
                print(f"  缓存形状: {cold_stats['shower_shape']}")
                print(f"  当前形状: {expected_shape}")
                return False
            
            # 加载统计数据（移动到GPU如果可用）
            device = self.showers.device
            self.E_bins = cold_stats['E_bins'].to(device) if cold_stats['E_bins'] is not None else None
            self.avg_showers = cold_stats['avg_showers'].to(device) if cold_stats['avg_showers'] is not None else None
            self.std_showers = cold_stats['std_showers'].to(device) if cold_stats['std_showers'] is not None else None
            
            print(f"\n[Cold Diffusion] 从缓存加载统计数据: {cache_path}")
            print(f"  avg_showers: {self.avg_showers.shape}")
            print(f"  std_showers: {self.std_showers.shape}")
            print(f"  E_bins: {self.E_bins.shape} - 范围 [{self.E_bins[0]:.4f}, {self.E_bins[-1]:.4f}]")
            
            return True
            
        except FileNotFoundError:
            print(f"\n[Cold Diffusion] 缓存文件不存在: {cache_path}")
            return False
        except Exception as e:
            print(f"\n[Cold Diffusion] 加载缓存文件时出错: {e}")
            print(f"  将重新计算统计数据")
            return False
    
    def _compute_or_load_cold_statistics(self, hdf5_path):
        """
        计算或加载冷扩散统计数据
        
        如果缓存文件存在且参数匹配，则加载；否则重新计算并保存
        """
        cache_path = self._get_cold_stats_cache_path(hdf5_path)
        
        # 如果强制重新计算，跳过加载
        if self.force_recompute:
            print(f"\n[Cold Diffusion] 强制重新计算统计数据（忽略缓存）")
            self._compute_cold_statistics()
            self._save_cold_statistics(cache_path)
            return
        
        # 尝试加载缓存
        if self._load_cold_statistics(cache_path):
            return  # 加载成功，直接返回
        
        # 加载失败或不存在，重新计算
        print(f"\n[Cold Diffusion] 开始计算统计数据...")
        self._compute_cold_statistics()
        self._save_cold_statistics(cache_path)

    def _compute_cold_statistics(self):
        """
        计算冷扩散统计数据（内部实现）
        """
        print(f"\n[Cold Diffusion] 计算统计数据 (能量区间数: {self.num_energy_bins})")
        energies_flat = self.energies.squeeze()
        
        # 创建能量区间
        e_min, e_max = energies_flat.min(), energies_flat.max()
        self.E_bins = torch.linspace(e_min, e_max, self.num_energy_bins + 1)
        
        # 为每个能量区间计算平均和标准差
        avg_showers_list = []
        std_showers_list = []
        
        for i in range(self.num_energy_bins):
            # 找到属于这个能量区间的样本
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
        
        print(f"\n[Cold Diffusion] 统计数据形状:")
        print(f"  avg_showers: {self.avg_showers.shape}")
        print(f"  std_showers: {self.std_showers.shape}")
        print(f"  E_bins: {self.E_bins.shape} - 范围 [{self.E_bins[0]:.4f}, {self.E_bins[-1]:.4f}]")

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, idx):
        return self.showers[idx], self.energies[idx]


def get_calo_dataloader(
    hdf5_path: str,
    dataset_name: str = 'dataset2',
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: int = None,
    normalize_method: str = 'log',
    xml_path: str = None,
    particle: str = None,
    weight_cache: str = None,
    ecut: float = 0.0,
    compute_cold_stats: bool = False,
    num_energy_bins: int = 10,
    cold_stats_cache_dir: str = './cold_stats_cache',
    force_recompute: bool = False,
):
    """
    创建Calorimeter数据加载器
    
    Args:
        hdf5_path: HDF5数据文件路径
        dataset_name: 数据集名称 ('dataset1', 'dataset2', 'dataset3')
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
        max_samples: 最大样本数（None表示加载全部）
        normalize_method: 归一化方法 ('log', 'logit')
        xml_path: Dataset1的XML配置文件路径
        particle: Dataset1的粒子类型
        weight_cache: 权重矩阵缓存路径
        ecut: 能量截断阈值
        compute_cold_stats: 是否计算/加载冷扩散统计数据
        num_energy_bins: 能量区间数量
        cold_stats_cache_dir: 冷扩散统计数据缓存目录
        force_recompute: 是否强制重新计算统计数据（忽略缓存）
    
    Returns:
        DataLoader对象和Dataset对象（包含冷扩散统计数据）
    """
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
        cold_stats_cache_dir=cold_stats_cache_dir,
        force_recompute=force_recompute,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 返回dataloader和dataset，方便访问冷扩散统计数据
    return dataloader, dataset


# 工具函数：单独加载冷扩散统计数据（用于推理）
def load_cold_statistics(
    cache_path: str,
    device: torch.device = None
):
    """
    从缓存文件加载冷扩散统计数据（用于推理）
    
    Args:
        cache_path: 缓存文件路径
        device: 目标设备（None表示自动选择）
    
    Returns:
        dict: 包含 'E_bins', 'avg_showers', 'std_showers' 的字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        with open(cache_path, 'rb') as f:
            cold_stats = pickle.load(f)
        
        # 移动到指定设备
        result = {
            'E_bins': cold_stats['E_bins'].to(device),
            'avg_showers': cold_stats['avg_showers'].to(device),
            'std_showers': cold_stats['std_showers'].to(device),
            'num_energy_bins': cold_stats['num_energy_bins'],
            'dataset_name': cold_stats['dataset_name'],
            'normalize_method': cold_stats['normalize_method'],
            'shower_shape': cold_stats['shower_shape'],
        }
        
        print(f"\n[Cold Diffusion] 成功加载统计数据: {cache_path}")
        print(f"  Energy bins: {result['num_energy_bins']}")
        print(f"  Dataset: {result['dataset_name']}")
        print(f"  Normalize: {result['normalize_method']}")
        print(f"  Shower shape: {result['shower_shape']}")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"加载冷扩散统计数据失败: {e}")